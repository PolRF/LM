from dataclasses import dataclass
from enum import Enum
import math
import random
import time
from typing import Literal
import tiktoken
import torch
import inspect

from hellaswag import get_most_likely_row, iterate_examples, render_example
from loaders.model_loader import (
    from_pretrained_rope_gpt2,
    resume_from_checkpoints,
)
from huggingface_hub import HfApi, Repository
from model import ModelConfig, GPTLM
import os

from loaders.data_loader import FineWebEduDataLoader
from torch.distributed import init_process_group, destroy_process_group
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

BASE_DATA_PATH = "./data/"
BASE_CHECKPOINT_PATH = "./checkpoints/"
BASE_PROFILER_PATH = "./profiler/"


@dataclass
class TrainConfig:
    batch_size: int  # if gradient_accumulation_steps is >1, then this is the mini-batch size
    block_size: int
    init_lr: float
    lr: float
    min_lr: float
    lr_decay_iters: int
    warmup_iters: int
    weight_decay: float
    device: torch.device
    # dtype: str
    checkpoint_output_dir: str
    gradient_accumulation_steps: int = 5 * 8
    always_save_checkpoint: bool = False
    compile: bool = False
    ddp: bool = False
    grad_clip: float = 1.0
    loading_mode: Literal[
        "from_scratch", "from_pretrained", "resume_from_checkpoint"
    ] = "from_scratch"
    profile: bool = False

    # logging
    wandb_project: str = "gpt2"
    wandb_name: str = "gpt2-gqa-0.5M"


class Dataset(Enum):
    FINEWEB_EDU = "fineweb_edu"


class TrainGPTM:
    def __init__(self, tr_config: TrainConfig, model_config: ModelConfig):
        self.tr_config = tr_config
        self.model_config = model_config
        # Initial cuda config
        torch.manual_seed(1337)
        torch.set_float32_matmul_precision("high")

        # Initial vars
        self.max_iters = 225_000
        self.eval_interval = 500
        self.iter_num = 0
        self.best_val_loss = 1e9
        self.checkpoint = None
        self.lr = None
        self.loss_accum = 0.0
        self.time_0 = 0.0

        # Distributed Data Parallel
        if tr_config.ddp:
            assert torch.cuda.is_available()
            init_process_group(backend="nccl")
            self.ddp_rank = int(os.environ["RANK"])
            self.ddp_local_rank = int(os.environ["LOCAL_RANK"])
            self.ddp_world_size = int(os.environ["WORLD_SIZE"])
            device = f"cuda:{self.ddp_local_rank}"
            self.tr_config.device = device  # type: ignore
            tr_config.device = device  # type: ignore
            torch.cuda.set_device(self.tr_config.device)
            self.master_process = (
                self.ddp_rank == 0
            )  # master process will log and save the checkpoints
        else:
            self.ddp_rank = 0
            self.ddp_local_rank = 0
            self.ddp_world_size = 1
            self.master_process = True

        # Checkpoints
        if tr_config.checkpoint_output_dir and not os.path.exists(
            tr_config.checkpoint_output_dir
        ):
            if self.master_process:
                print(f"Creating directory: {tr_config.checkpoint_output_dir}")
                os.makedirs(tr_config.checkpoint_output_dir)

        # Maybe we can remove this outside of the class and pass it as a parameter
        self.model = GPTLM(model_config)
        self.model_config = model_config
        if tr_config.loading_mode == "from_pretrained":
            self.model, self.model_config = from_pretrained_rope_gpt2(
                tr_config.device
            )
        elif tr_config.loading_mode == "resume_from_checkpoint":
            self.model, self.iter_num, self.best_val_loss = (
                resume_from_checkpoints(tr_config, model_config)
            )

        self.model.to(tr_config.device)
        if tr_config.compile:
            self.model: torch.nn.Module = torch.compile(self.model)  # type: ignore
        if self.master_process:
            print(
                sum(p.numel() for p in self.model.parameters()) / 1e6,
                "M parameters",
            )

        # optimizer
        self.betas = (0.9, 0.95)
        self.optimizer = self._configure_optimizer()

        if tr_config.ddp:
            self.model = DDP(self.model, device_ids=[self.ddp_local_rank])
        # Logging
        if self.master_process:
            wandb.init(
                project=tr_config.wandb_project,
                name=tr_config.wandb_name,
                config={**tr_config.__dict__, **model_config.__dict__},
            )
            wandb.watch(self.model)
        if tr_config.profile:
            self._init_profiler()

        # Data loader
        self.train_loader = FineWebEduDataLoader(
            B=tr_config.batch_size,
            T=tr_config.block_size,
            process_rank=self.ddp_rank,
            num_processes=self.ddp_world_size,
            split="train",
        )
        self.val_loader = FineWebEduDataLoader(
            B=tr_config.batch_size,
            T=tr_config.block_size,
            process_rank=self.ddp_rank,
            num_processes=self.ddp_world_size,
            split="val",
        )

    def _init_profiler(self):
        self.profile_dir = os.path.join(
            BASE_PROFILER_PATH,
            self.tr_config.wandb_project,
            self.tr_config.wandb_name,
        )
        self.profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=1, warmup=1, active=3, repeat=2
            ),
            record_shapes=False,
            with_stack=True,
            with_flops=False,
        )
        self.profiler.start()

    def _configure_optimizer(self):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.model.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {
                "params": decay_params,
                "weight_decay": self.tr_config.weight_decay,
            },
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if self.master_process:
            print(
                f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
            )
            print(
                f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
            )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = (
            "fused" in inspect.signature(torch.optim.AdamW).parameters
        )
        use_fused = fused_available  # and self.tr_config.device == "cuda"

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.tr_config.lr,
            betas=self.betas,
            fused=use_fused,
        )
        if self.master_process:
            print(f"using fused AdamW: {use_fused}")

        return optimizer

    def _get_current_lr(self):
        """
        Get the current learning following the learning rate scheduler.
        The learning rate scheduler is a cosine with warmup.
        """
        # 1) linear warmup for warmup_iters steps
        if self.iter_num < self.tr_config.warmup_iters:
            return (
                self.tr_config.lr * self.iter_num / self.tr_config.warmup_iters
            )
        # 2) if it > lr_decay_iters, return min learning rate
        if self.iter_num > self.tr_config.lr_decay_iters:
            return self.tr_config.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (self.iter_num - self.tr_config.warmup_iters) / (
            self.tr_config.lr_decay_iters - self.tr_config.warmup_iters
        )
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (
            1.0 + math.cos(math.pi * decay_ratio)
        )  # coeff ranges 0..1
        return self.tr_config.min_lr + coeff * (
            self.tr_config.lr - self.tr_config.min_lr
        )

    @torch.no_grad()
    def _estimate_loss(self):
        val_loss_accum = 0.0
        val_loss_steps = 20
        for _ in range(val_loss_steps):
            x, y = self.val_loader.next_batch()
            x, y = x.to(self.tr_config.device), y.to(self.tr_config.device)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):
                logits, loss = self.model(x, y)
            loss = loss / val_loss_steps
            val_loss_accum += loss.detach()
        return val_loss_accum

    def _validate_and_save_checkpoint(self):
        self.val_loader.reset()
        val_loss_accum = self._estimate_loss()
        if self.tr_config.ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if not self.master_process:
            return
        val_loss_accum = val_loss_accum.item()  # type: ignore
        wandb.log(
            {
                "iter": self.iter_num,
                "val/loss": val_loss_accum,
                "val_perplexity": math.exp(val_loss_accum),
                "lr": self.lr,
                "time": time.time() - self.time_0,
            }
        )
        print(
            f"step {self.iter_num}: val loss {val_loss_accum:.4f}, time (s): {time.time()-self.time_0:3f}"
        )
        if (
            val_loss_accum < self.best_val_loss
            or self.tr_config.always_save_checkpoint
        ):
            self.best_val_loss = val_loss_accum
            if self.iter_num > 0:
                # TODO: Type this
                checkpoint = {
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "model_args": self.model_config,
                    "iter_num": self.iter_num,
                    "best_val_loss": self.best_val_loss,
                    "training_config": self.tr_config,
                }
                print(
                    f"saving checkpoint to {self.tr_config.checkpoint_output_dir}"
                )
                torch.save(
                    checkpoint,
                    os.path.join(
                        self.tr_config.checkpoint_output_dir, "ckpt.pt"
                    ),
                )

    def _microstep_training(self):
        self.loss_accum = 0.0
        for micro_step in range(self.tr_config.gradient_accumulation_steps):
            x, y = self.train_loader.next_batch()
            x, y = x.to(self.tr_config.device), y.to(self.tr_config.device)
            if self.tr_config.ddp:
                self.model.require_backward_grad_sync = (
                    micro_step
                    == self.tr_config.gradient_accumulation_steps - 1
                )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, loss = self.model(x, y)

            loss = loss / self.tr_config.gradient_accumulation_steps
            self.loss_accum += loss.detach()
            # There is no need of using scaler is we are not using float16
            loss.backward()
        if self.tr_config.ddp:
            dist.all_reduce(self.loss_accum, op=dist.ReduceOp.AVG)

    def _log_iteration(self):
        time_elapsed = time.time() - self.time_0
        tokens_processed = (
            self.tr_config.batch_size
            * self.tr_config.block_size
            * self.tr_config.gradient_accumulation_steps
        )
        tokens_per_sec = tokens_processed / time_elapsed
        wandb.log(
            {
                "iter": self.iter_num,
                "train/loss": self.loss_accum.item(),
                "train_perplexity": math.exp(self.loss_accum.item()),
                "lr": self.lr,
                "time": time_elapsed,
                "tokens_per_sec": tokens_per_sec,
            }
        )
        print(
            f"step {self.iter_num}: train loss {self.loss_accum.item():.4f}, time (s): {time_elapsed:.4f}, lr: {self.lr:.7f}, tok/sec: {tokens_per_sec:.2f}"
        )

    def train(self):
        for i in range(self.max_iters - self.iter_num):
            self.iter_num += 1
            if self.tr_config.profile:
                assert self.profiler
                self.profiler.step()
            self.time_0 = time.time()
            self.lr = self._get_current_lr()
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.lr
            # every once in a while evaluate the loss on train and val sets
            if (
                self.iter_num % self.eval_interval == 0
                or self.iter_num == self.max_iters - 1
            ):
                self.model.eval()
                self._validate_and_save_checkpoint()

            self.model.train()
            # TODO: compare optimizer.zero_grad(set_to_none=True) vs.
            # for param in model.parameters():
            #     param.grad = None
            self.optimizer.zero_grad(set_to_none=True)
            self._microstep_training()
            # This is done in case there is a bad batch that causes the gradients to explode.
            norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.tr_config.grad_clip
            )
            self.optimizer.step()
            # Wait for the GPU to finish
            torch.cuda.synchronize()
            if self.master_process:
                self._log_iteration()

            if self.tr_config.profile and self.iter_num % 10 == 0:
                assert self.profiler
                assert self.profile_dir
                self.profiler.stop()
                self.profiler.export_chrome_trace(self.profile_dir)
                break
        if self.tr_config.ddp:
            destroy_process_group()


def test_generation(tr_config: TrainConfig, model_config: ModelConfig):
    model, _, __ = resume_from_checkpoints(tr_config, model_config)
    model = model.to("cuda")
    model.eval()
    print("Model loaded")
    enc = tiktoken.get_encoding("gpt2")
    context = "The quick brown fox jumps over the lazy dog"
    context = enc.encode_ordinary(context)
    context = torch.tensor(context, dtype=torch.long).unsqueeze(0)
    context = context.to("cuda")
    out = model.generate(context, 100)
    print(enc.decode(out[0].cpu().numpy()))


def evaluate(tr_config: TrainConfig, model_config: ModelConfig):
    model, _, __ = resume_from_checkpoints(tr_config, model_config)
    model = model.to("cuda")
    print("Model loaded")

    num_correct_norm = 0
    num_total = 0
    for i, example in enumerate(iterate_examples("val")):
        # render the example into tokens and labels
        _, tokens, mask, label = render_example(example)
        tokens = tokens.to("cuda")
        mask = mask.to("cuda")
        # get the logits
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, loss = model(tokens)
            pred_norm = get_most_likely_row(tokens, mask, logits)
        num_total += 1
        num_correct_norm += int(pred_norm == label)

    acc_norm = num_correct_norm / num_total
    print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")


if __name__ == "__main__":
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    tr_config = TrainConfig(
        batch_size=4,
        block_size=1024,
        init_lr=6e-4,  # for lr decay
        lr=6e-4,
        min_lr=6e-5,
        warmup_iters=20_000,
        lr_decay_iters=220_000,
        weight_decay=1e-1,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        # dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16',
        gradient_accumulation_steps=16,
        loading_mode="from_scratch",
        checkpoint_output_dir=BASE_CHECKPOINT_PATH,
        always_save_checkpoint=False,
        ddp=True,
        compile=True,
        grad_clip=1.0,
        profile=False,
        wandb_name="GPT-500M",
    )
    model_config = ModelConfig(
        vocab_size=50304,
        block_size=1024,
        n_embd=768,
        n_layer=12,
        device=tr_config.device,
        dropout=0.1,
        n_head=16,
        n_kv_heads=4,
        pos_emb="rope",
        num_experts=1,
        num_experts_per_token=None,
    )

    TrainGPTM(tr_config, model_config).train()
    # evaluate(tr_config, model_config)
    # test_generation(tr_config, model_config)
