from transformers import GPT2LMHeadModel


def from_pretrained_gpt2():
    model_hf = GPT2LMHeadModel.from_pretrained("gpt2")
    sd_hf = model_hf.state_dict()
    
    print(model_hf)
    print(sum(p.numel() for p in model_hf.parameters())/1e6, 'M parameters')
    pass

if __name__ == '__main__':
    from_pretrained_gpt2()