from contextlib import nullcontext
from dataclasses import dataclass
from enum import Enum
import math
import time
from typing import Literal
import tiktoken
import torch
import inspect

from load import from_pretrained_rope_gpt2
from model import ModelConfig, GPTLM
import numpy as np
import os
from torch.utils.tensorboard.writer import SummaryWriter
import torch.nn as nn

from training.data_loader import FineWebEduDataLoader
# Tensorboard logs writers
writer = SummaryWriter("runs/gqa")
import wandb

BASE_DATA_PATH = './data/'
BASE_CHECKPOINT_PATH = './checkpoints_with_training_gqa/'
@dataclass
class TrainConfig:
    model: Literal["RoPeGPT2","GQAGPT2"]
    batch_size: int # if gradient_accumulation_steps is >1, then this is the mini-batch size
    block_size: int
    eval_iters: int
    init_lr: float
    lr: float
    min_lr: float
    lr_decay_iters:int
    warmup_iters:int
    weight_decay: float
    device: torch.device
    # dtype: str
    checkpoint_output_dir: str
    gradient_accumulation_steps: int = 5 * 8
    from_pretrained: bool = False
    resume_from_checkpoint: bool = False
    always_save_checkpoint: bool = False
    compile: bool = False
    grad_clip: float = 1.0



class Dataset(Enum):
    FINEWEB_EDU = 'fineweb_edu'



@torch.no_grad()
def estimate_loss(model, config: TrainConfig,dataset:Dataset,train_loader,validation_loader ):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            if split=="train":
                X, Y = train_loader.next_batch()
            else:
                X, Y = validation_loader.next_batch()
            X, Y = X.to(config.device), Y.to(config.device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(config: TrainConfig,iteration):
    # 1) linear warmup for warmup_iters steps
    if iteration < config.warmup_iters:
        return config.lr * iteration / config.warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if iteration > config.lr_decay_iters:
        return config.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (iteration - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return config.min_lr + coeff * (config.lr - config.min_lr)

def configure_optimizers(model: nn.Module,weight_decay, learning_rate, betas, device_type):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == 'cuda'

    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas,fused=use_fused)
    print(f"using fused AdamW: {use_fused}")

    return optimizer

def train(dataset:Dataset):
    
    tr_config = TrainConfig(
        model="RoPeGPT2",
        batch_size=16,
        block_size=1024,
        eval_iters=200,
        init_lr = 6e-4, # for lr decay (TODO need a lower lr????)
        lr = 6e-4,
        min_lr=6e-5,
        warmup_iters=10_000,
        lr_decay_iters=100_000,
        weight_decay=1e-1,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        # dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16',
        gradient_accumulation_steps=32,
        from_pretrained=False,
        checkpoint_output_dir=BASE_CHECKPOINT_PATH,
        always_save_checkpoint=False,
        resume_from_checkpoint=False,
        compile=False,
        grad_clip=1.0
    )
    torch.manual_seed(1337)
    torch.set_float32_matmul_precision('high')


    if tr_config.checkpoint_output_dir and not os.path.exists(tr_config.checkpoint_output_dir):
        print(f"Creating directory: {tr_config.checkpoint_output_dir}")
        os.makedirs(tr_config.checkpoint_output_dir)
    # Override device for Macbook pro m2 chip
    # tr_config.device=torch.device("mps")
    max_iters = 100_000
    eval_interval = 500
    model_config = ModelConfig(
        vocab_size=50304,
        block_size=1024,
        device=tr_config.device,
        dropout=0.1,
        n_head=16,
        n_kv_heads=4,
    )

    # Iterator only
    iter_num = 0
    best_val_loss = 1e9
    model = GPTLM(model_config)
    # TODO: Move to other files
    # if tr_config.from_pretrained and not tr_config.resume_from_checkpoint:
    #     if tr_config.model == "RoPeGPT2":
    #         model, config = from_pretrained_rope_gpt2(tr_config.device)
    #     else:
    #         raise ValueError("Loading from pretrained is only supported for RoPeGPT2 model.")
    #     model_config=config
    # if tr_config.from_pretrained and tr_config.resume_from_checkpoint:
    #     print("Ignoring from_pretrained flag since we are resuming from a checkpoint")
    # if tr_config.resume_from_checkpoint:
    #     # This is a copy-paste from the Andrej Karpathy code, should be refined later
    #     print(f"Resuming training from {tr_config.checkpoint_output_dir}")
    #     # resume training from a checkpoint.
    #     ckpt_path = os.path.join(tr_config.checkpoint_output_dir, 'ckpt.pt')
    #     checkpoint = torch.load(ckpt_path, map_location=tr_config.device)
    #     checkpoint_model_args = checkpoint['model_args']
    #     # force these config attributes to be equal otherwise we can't even resume training
    #     # the rest of the attributes (e.g. dropout) can stay as desired from command line
    #     # for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
    #     #     setattr(dict_config, k, checkpoint_model_args[k])
    #     # create the model

    #     model = GPTLM(model_config)
    #     print(f"Model args: {checkpoint_model_args}")
    #     state_dict = checkpoint['model']
    #     # fix the keys of the state dictionary :(
    #     # this prefix is present when saving compiled model
    #     unwanted_prefix = '_orig_mod.'
    #     for k,v in list(state_dict.items()):
    #         if k.startswith(unwanted_prefix):
    #             state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    #     model.load_state_dict(state_dict)
    #     iter_num = checkpoint['iter_num']
    #     best_val_loss = checkpoint['best_val_loss']
    checkpoint=None
    assert model
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
    model.to(tr_config.device)
    if tr_config.compile:
        model = torch.compile(model)
    optimizer = configure_optimizers(model, tr_config.weight_decay, tr_config.lr, (0.9, 0.95), 'cuda') 
    print(f"We are using device: {tr_config.device}")
    wandb_project = "gpt2"
    wandb.init(project=wandb_project, name="gpt2-gqa-0.5M", config=tr_config.__dict__)
    # Init the first batch
    train_loader = FineWebEduDataLoader(B=tr_config.batch_size, T=tr_config.block_size, process_rank=0, num_processes=1, split="train")
    val_loader = FineWebEduDataLoader(B=tr_config.batch_size, T=tr_config.block_size, process_rank=0, num_processes=1, split="val")
    while True:
        time_0 = time.time()
        lr = get_lr(tr_config, iter_num)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # every once in a while evaluate the loss on train and val sets
        if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
            model.eval()
            val_loader.reset()
            losses = estimate_loss(model,tr_config,dataset,train_loader,val_loader)
            wandb.log({
                "iter": iter_num,
                "val/loss": losses['val'],
                "lr": lr,
                "time": time.time() - time_0,
            })
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, time (s): {time.time()-time_0:3f}, full time: {round(time.time()-start_time, 5)}" )
            if losses['val'] < best_val_loss or tr_config.always_save_checkpoint:
                best_val_loss = losses['val']
                if iter_num > 0:
                    # TODO: Type this
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_config,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'training_config': tr_config,
                    }
                    print(f"saving checkpoint to {tr_config.checkpoint_output_dir}")
                    torch.save(checkpoint, os.path.join(tr_config.checkpoint_output_dir, 'ckpt.pt'))
        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(tr_config.gradient_accumulation_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(tr_config.device), y.to(tr_config.device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, loss = model(x, y)
            
            loss = loss / tr_config.gradient_accumulation_steps
            loss_accum += loss.detach()
            # There is no need of using scaler is we are not using float16
            loss.backward()

        # This is done in case there is a bad batch that causes the gradients to explode.
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tr_config.grad_clip)
        optimizer.step()
        # Wait for the GPU to finish
        torch.cuda.synchronize()
        time_elapsed = time.time() - time_0
        tokens_processed = tr_config.batch_size * tr_config.block_size * tr_config.gradient_accumulation_steps
        tokens_per_sec = tokens_processed / time_elapsed
        wandb.log({
                "iter": iter_num,
                "train/loss": loss_accum.item(),
                "lr": lr,
                "time": time_elapsed,
                "tokens_per_sec": tokens_per_sec,
            })
        print(f"step {iter_num}: train loss {loss_accum.item():.4f}, time (s): {time_elapsed:.4f}, lr: {lr:.7f}, tok/sec: {tokens_per_sec:.2f}")

        iter_num += 1
        if iter_num >= max_iters:
            break
        
    writer.close()

def test_generation():
    model, _ = from_pretrained_rope_gpt2(torch.device("cuda"))
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

if __name__ == '__main__':
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
    os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
    train(Dataset.FINEWEB_EDU)
    # test_generation()