## Create a scheduler function to trigger different trainings based on hyperparam modification


import os
import torch
from typing import List, Literal
from model import GPTConfig, from_gptconfig_to_modelconfig
from train import TrainConfig, TrainGPTM
from huggingface_hub import create_repo, upload_file


def get_config_from_model_class(
    model_class: Literal["gpt-small", "gpt-medium", "gpt-xl"]
) -> GPTConfig:
    shared_config = {
        "vocab_size": 50304,
        "block_size": 1024,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "dropout": 0.0,
        "pos_emb": "rope",
        "num_experts": 1,
        "num_experts_per_token": None,
    }
    match model_class:
        case "gpt-small":
            return GPTConfig(
                n_embd=768,
                n_layer=12,
                n_head=16,
                n_kv_heads=4,
                **shared_config,
            )
        case "gpt-medium":
            return GPTConfig(
                n_embd=1024,
                n_layer=24,
                n_head=16,
                n_kv_heads=8,
                **shared_config,
            )
        case "gpt-xl":
            return GPTConfig(
                n_embd=2048,
                n_layer=24,
                n_head=32,
                n_kv_heads=8,
                **shared_config,
            )


def scheduler():
    seq_len = [
        1024,
        8192,
        32_768,
    ]
    theta = [
        10_000,
        100_000,
        500_000,
        2_000_000,
    ]
    models: List[
        Literal[
            "gpt-small",
            "gpt-medium",
            "gpt-xl",
        ]
    ] = [
        "gpt-small",  # 117M
        "gpt-medium",  # 345M
        "gpt-xl",  # 1.5B
    ]
    ddp_rank = int(os.environ["RANK"])
    print("ddp_rank", ddp_rank)
    for model_class in models:
        repo_name = f"GQA-{model_class}-RoPE"
        if ddp_rank == 0:
            create_repo(repo_name, private=True, exist_ok=True)
        for seq in seq_len:
            for th in theta:
                # Already trained
                if (
                    seq == 1024
                    and th in [10_000]
                    and model_class == "gpt-small"
                ):
                    continue
                checkpoint_output_dir = (
                    f"./checkpoints/{model_class}/seq_len_{seq}/theta_{th}"
                )
                config_output_dir = (
                    f"./configs/{model_class}/seq_len_{seq}/theta_{th}"
                )
                # tr_config = TrainConfig(
                #     batch_size=64,
                #     block_size=seq,
                #     init_lr=6e-4,  # for lr decay (TODO need a lower lr????)
                #     lr=6e-4,
                #     min_lr=6e-5,
                #     warmup_iters=10_000,
                #     lr_decay_iters=100_000,
                #     weight_decay=1e-1,
                #     device=torch.device(
                #         "cuda" if torch.cuda.is_available() else "cpu"
                #     ),
                #     # dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16',
                #     gradient_accumulation_steps=1,
                #     loading_mode="from_scratch",
                #     checkpoint_output_dir=checkpoint_output_dir,
                #     always_save_checkpoint=False,
                #     ddp=True,
                #     compile=True,
                #     grad_clip=1.0,
                #     profile=False,
                #     wandb_name=f"{repo_name}_seq_{seq}_theta_{th}",
                # )
                hf_conf = get_config_from_model_class(model_class)
                hf_conf.max_seq_len = seq
                hf_conf.block_size = seq
                hf_conf.theta = th
                if ddp_rank == 0:
                    print("we are here")
                    hf_conf.save_pretrained(config_output_dir)

                model_config = from_gptconfig_to_modelconfig(hf_conf)
                # TrainGPTM(tr_config, model_config).train()

                if ddp_rank == 0:
                    print("Uploading to Huggingface Hub")
                    # Upload configs
                    upload_file(
                        path_or_fileobj=f"{config_output_dir}/config.json",
                        path_in_repo="configs",
                        repo_id=f"polrf/{repo_name}",
                        run_as_future=False,
                    )

                    # Upload checkpoints
                    upload_file(
                        path_or_fileobj=f"{checkpoint_output_dir}/ckpt.pt",
                        path_in_repo="checkpoints",
                        repo_id=f"polrf/{repo_name}",
                        run_as_future=False,
                    )
                break
            break
        break


if __name__ == "__main__":
    scheduler()
