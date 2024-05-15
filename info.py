
from model import GPTLM, ModelConfig
import torch
from collections import OrderedDict
def total_params(model:GPTLM)->int:
    return sum(p.numel() for p in model.parameters())

def percentage_of_gpu_usage(total_params:int, gpu_size:int)->float:
    params_bytes = total_params*4
    params_and_buffers_bytes = params_bytes + 2*params_bytes
    print(f"memory ratio taken up just for parameters: {params_and_buffers_bytes / gpu_size * 100:.2f}%")

def flops(config: ModelConfig):
    # we only count Weight FLOPs, all other layers (LayerNorm, Softmax, etc) are effectively irrelevant
    # we count actual FLOPs, not MACs. Hence 2* all over the place
    # basically for any matrix multiply A (BxC) @ B (CxD) -> (BxD) flops are 2*B*C*D
    n_embd = config.n_embd
    n_head = config.n_head
    n_layer = config.n_layer
    block_size = config.block_size
    vocab_size = config.vocab_size

    out = OrderedDict()
    head_size = n_embd // n_head

    # attention blocks
    # 1) the projection to key, query, values
    out['attention/kqv'] = 2 * block_size * (n_embd * 3*n_embd)
    # 2) calculating the attention scores
    out['attention/scores'] = 2 * block_size * block_size * n_embd
    # 3) the reduction of the values (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
    out['attention/reduce'] = 2 * n_head * (block_size * block_size * head_size)
    # 4) the final linear projection
    out['attention/proj'] = 2 * block_size * (n_embd * n_embd)
    out['attention'] = sum(out['attention/'+k] for k in ['kqv', 'scores', 'reduce', 'proj'])

    # MLP blocks
    ffw_size = 4*n_embd # feed forward size
    out['mlp/ffw1'] = 2 * block_size * (n_embd * ffw_size)
    out['mlp/ffw2'] = 2 * block_size * (ffw_size * n_embd)
    out['mlp'] = out['mlp/ffw1'] + out['mlp/ffw2']

    # the transformer and the rest of it
    out['block'] = out['attention'] + out['mlp']
    out['transformer'] = n_layer * out['block']
    out['dense'] = 2 * block_size * (n_embd * vocab_size)

    # forward,backward,total
    out['forward_total'] = out['transformer'] + out['dense']
    out['backward_total'] = 2 * out['forward_total'] # use common estimate of bwd = 2*fwd
    out['total'] = out['forward_total'] + out['backward_total']

    return out

# now here is an estimate copy pasted from the PaLM paper
# this formula is often used to calculate MFU (model flops utilization)
def palm_flops(model_config: ModelConfig):
    """estimate of the model flops following PaLM paper formula"""
    n_embd = model_config.n_embd
    n_head = model_config.n_head
    n_layer = model_config.n_layer
    block_size = model_config.block_size

    # non-embedding model parameters. note that we do not subtract the
    # embedding/token params because those are tied and get used in the last layer.
    N = total_params(GPTLM(model_config))
    L, H, Q, T = n_layer, n_head, n_embd//n_head, block_size
    mf_per_token = 6*N + 12*L*H*Q*T
    mf = mf_per_token * block_size
    return mf

if __name__ == "__main__":
    model_config = ModelConfig(
        vocab_size=50257,
        block_size=1024,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        dropout=0.1,
    )
    model = GPTLM(model_config)
    total_pms = total_params(model)
    print(f"Total parameters: {total_pms}")
    gpu_memory = 24e9 # 24GB
    percentage_of_gpu_usage(total_pms, gpu_memory)
    f = flops(model_config)
    flops_total = f['forward_total']
    print(f"{'name':20s} {'flops':14s} {'ratio (%)':10s}")
    for k,v in f.items():
        print(f"{k:20s} {v:14d} {v/flops_total*100:10.4f}")
    print(f"palm_flops: {palm_flops(model_config):d}, flops: {flops(model_config)['total']:d}, ratio: {palm_flops(model_config)/flops(model_config)['total']:.4f}")
    batch_size = 20 * 5 # 5 is grad_accum, so total batch size is 100
    measured_time = 0.755 # in seconds per iteration
    measured_throughput = batch_size / measured_time
    flops_achieved = f['total'] * measured_throughput

    # L4 is cited to be 125 TFLOPS
    l4_flops_promised = 125e12

    # the fraction of the A100 that we are using:
    print(f"fraction of L4 used: {flops_achieved / l4_flops_promised * 100:.2f}%")






    ### days
    model_size = total_params(model) # this is number of parameters, N
    tokens_num = 300e9 # 300B tokens, this is dataset size in tokens, D
    l4_flops = 125e12 # 125 TFLOPS
    assumed_mfu = 0.92
    flops_throughput = l4_flops * assumed_mfu 
    flops_needed = 6 * model_size * tokens_num # 6ND
    time_needed_s = flops_needed / flops_throughput # in seconds
    print(f"time needed to train the model: {time_needed_s/3600/24:.2f} days")