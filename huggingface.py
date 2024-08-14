import torch
from loaders.model_loader import remove_prefix
from model import GPTConfig, GPTModel
import tiktoken
from transformers import AutoModelForCausalLM, AutoConfig


def push_pytorch_checkpoints_to_hf(checkpoint_path, huggingface_repo):
    config = GPTConfig.from_pretrained(
        pretrained_model_name_or_path=huggingface_repo,
    )

    # Create a new model instance
    # resume training from a checkpoint.
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )
    checkpoint_model_args = checkpoint["model_args"]

    model = GPTModel(config)
    print(f"Model args: {checkpoint_model_args}")
    state_dict = checkpoint["model"]
    # fix the keys of the state dictionary :(
    # this prefix is present when saving compiled model
    unwanted_prefix = "_orig_mod."
    state_dict = remove_prefix(state_dict)
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

    # rename keys to add "model." prefix
    state_dict = {f"model.{k}": v for k, v in state_dict.items()}

    # Handle shared weights
    if "model.lm_head.weight" in state_dict:
        del state_dict["model.lm_head.weight"]
    model.load_state_dict(state_dict, strict=False)
    # Manually tie the weights
    model.tie_weights()
    model.push_to_hub(huggingface_repo, safe_serialization=False)


def test_generation():
    model = GPTModel.from_pretrained(
        "polrf/GQA-RoPE-MoE-500M", trust_remote_code=True
    )
    model.eval()
    print("Model loaded")
    enc = tiktoken.get_encoding("gpt2")
    context = "I'm a large language model, "
    context = enc.encode_ordinary(context)
    context = torch.tensor(context, dtype=torch.long).unsqueeze(0)
    context = context.to("cpu")
    out = model.generate(context, 100)
    print(enc.decode(out[0].cpu().numpy()))
