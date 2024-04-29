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
     

def from_pretrained_gpt2():

    model_config = ModelConfig(
        vocab_size=50257,
        block_size=1024,
        # n_head=4,
        # n_layer=4,
        # n_embd=384,
        device=torch.device("cpu"),
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
    
    return model