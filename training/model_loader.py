

import os
from typing import Literal

import torch

from load import from_pretrained_rope_gpt2
from model import GPTLM, ModelConfig
from train import TrainConfig


from typing import Literal
import torch
from transformers import GPT2LMHeadModel

from model import GPTLM, ModelConfig

# Map the names to gpt2
def check_mapping(gpt2_key_name:str):
    gpt2_key_name = gpt2_key_name.replace("transformer.","")
    gpt2_key_name = gpt2_key_name.replace("wte","token_embedding")
    gpt2_key_name = gpt2_key_name.replace("h.","blocks.")
    gpt2_key_name = gpt2_key_name.replace("ln_1","ln1")
    gpt2_key_name = gpt2_key_name.replace("ln_2","ln2")
    gpt2_key_name = gpt2_key_name.replace("c_attn","att")
    gpt2_key_name = gpt2_key_name.replace("mlp.c_fc","ffn.net.0")
    gpt2_key_name = gpt2_key_name.replace("mlp.c_proj","ffn.net.2")
    gpt2_key_name = gpt2_key_name.replace("c_proj","linear_projection")
    gpt2_key_name = gpt2_key_name.replace("ln_f","ln_f")
    return gpt2_key_name
     

def from_pretrained_rope_gpt2(device:torch.device) -> tuple[GPTLM, ModelConfig]:
    print("Loading GPT-2 weights")
    model_config = ModelConfig(
        vocab_size=50257,
        block_size=1024,
        device=device,
        dropout=0.1,
    )

    model = GPTLM(model_config)
    sd = model.state_dict()
    my_model_keys = sd.keys()
    my_model_keys = [k for k in my_model_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

    model_hf = GPT2LMHeadModel.from_pretrained("gpt2")
    sd_hf = model_hf.state_dict()
    sd_keys_hf = sd_hf.keys()
    # We want to skip teh positional embedding weights since we use RoPE
    sd_keys_hf = [k for k in sd_keys_hf if 'wpe' not in k]
    sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
    sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('attn.bias')] # same, just the mask (buffer)

    # We can see that GPT-2 uses Conv1D instead of linear layers. We have to transpose the weights for the linear layers
    transposed = ["attn.c_attn.weight","attn.c_proj.weight","mlp.c_fc.weight","mlp.c_proj.weight"]

    assert len(sd_keys_hf) == len(my_model_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(my_model_keys)}"

    for k in sd_keys_hf:
        k_mine = check_mapping(k)
        if any(k.endswith(w) for w in transposed):
            # special treatment for the Conv1D weights we need to transpose
            assert sd_hf[k].shape[::-1] == sd[k_mine].shape
            with torch.no_grad():
                sd[k_mine].copy_(sd_hf[k].t())
        else:
            # vanilla copy over the other parameters
            assert sd_hf[k].shape == sd[k_mine].shape
            with torch.no_grad():
                sd[k_mine].copy_(sd_hf[k])
    
    return model, model_config

def resume_from_checkpoints(config:TrainConfig,model_config:ModelConfig) -> tuple[GPTLM, int, float]:
    # This is a copy-paste from the Andrej Karpathy code, should be refined later
    print(f"Resuming training from {config.checkpoint_output_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(config.checkpoint_output_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=config.device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    # for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
    #     setattr(dict_config, k, checkpoint_model_args[k])
    # create the model

    model = GPTLM(model_config)
    print(f"Model args: {checkpoint_model_args}")
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # this prefix is present when saving compiled model
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
    return model, iter_num, best_val_loss