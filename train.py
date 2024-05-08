from dataclasses import dataclass
from enum import Enum
import math
import time
import tiktoken
import torch

from load import from_pretrained_gpt2
from model import ModelConfig, GPTLM
import numpy as np
import os
from torch.utils.tensorboard.writer import SummaryWriter
from data.openwebtext import prepare as prepare_openwebtext
# Tensorboard logs writers
writer = SummaryWriter("runs/ler_decay")

BASE_DATA_PATH = './data/'
BASE_CHECKPOINT_PATH = './checkpoints/'
@dataclass
class TrainConfig:
    batch_size: int # if gradient_accumulation_steps is >1, then this is the mini-batch size
    block_size: int
    eval_iters: int
    init_lr: float
    lr: float
    min_lr: float
    lr_decay_iters:int
    warmup_iters:int
    device: torch.device
    checkpoint_output_dir: str
    gradient_accumulation_steps: int = 5 * 8
    from_pretrained: bool = False
    resume_from_checkpoint: bool = False
    always_save_checkpoint: bool = False



class Dataset(Enum):
    BIBLE = 'bible'
    SHAKESPEARE = 'shakespeare'
    OPENWEBTEXT = 'openwebtext'


def prepare_data(dataset:Dataset):
    print(f"Preparing data for {dataset.name}")
    path = None
    encoding = None
    match dataset:
        case Dataset.BIBLE:
            path ='bible_es.txt'
            encoding = "iso-8859-1"
        case Dataset.SHAKESPEARE:
            path = 'shakespeare.txt'
            encoding = "utf-8"
        case Dataset.OPENWEBTEXT:
            prepare_openwebtext.prepare()
            return

    data = open(BASE_DATA_PATH+path,encoding=encoding).read()
    n = len(data)
    train_data = data[:int(n*0.9)]
    val_data = data[int(n*0.9):]
    enc = tiktoken.get_encoding("gpt2")
    train_ids = enc.encode_ordinary(train_data)
    val_ids = enc.encode_ordinary(val_data)
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    # export to bin files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    os.makedirs(BASE_DATA_PATH+dataset.name, exist_ok=True)
    train_ids.tofile(os.path.join(BASE_DATA_PATH+dataset.name+"/", 'train.bin'))
    val_ids.tofile(os.path.join(BASE_DATA_PATH+dataset.name+"/", 'val.bin'))
    print("Data prepared")

def get_batch(split, config: TrainConfig,dataset:Dataset):
    data_path = BASE_DATA_PATH+dataset.name.lower()+"/"
    if not os.path.isfile(data_path+"train.bin") or not os.path.isfile(data_path+"val.bin"):
        prepare_data(dataset)
    
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_path, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_path, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+config.block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+config.block_size]).astype(np.int64)) for i in ix])
    if config.device == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(config.device, non_blocking=True), y.pin_memory().to(config.device, non_blocking=True)
    else:
        x, y = x.to(config.device), y.to(config.device)
    return x, y

@torch.no_grad()
def estimate_loss(model, config: TrainConfig,dataset:Dataset):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = get_batch(split,config,dataset)
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
def train(dataset:Dataset):
    
    tr_config = TrainConfig(
        batch_size=12,
        block_size=1024,
        eval_iters=200,
        init_lr = 6e-4, # for lr decay
        lr = 6e-4,
        min_lr=6e-5,
        warmup_iters=200,
        lr_decay_iters=5_000,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        gradient_accumulation_steps=5*8,
        from_pretrained=True,
        checkpoint_output_dir=BASE_CHECKPOINT_PATH
    )
    # Override device for Macbook pro m2 chip
    # tr_config.device=torch.device("mps")
    max_iters = 5_000
    eval_interval = 500
    model_config = ModelConfig(
        vocab_size=50304,
        block_size=tr_config.block_size,
        device=tr_config.device
    )

    # Iterator only
    start_time = time.time() 
    durations = []
    iter_num = 0
    best_val_loss = 1e9
    model = GPTLM(model_config)
    if tr_config.from_pretrained and not tr_config.resume_from_checkpoint:
        model = from_pretrained_gpt2(tr_config.device)
    if tr_config.from_pretrained and tr_config.resume_from_checkpoint:
        print("Ignoring from_pretrained flag since we are resuming from a checkpoint")
    if tr_config.resume_from_checkpoint:
        # This is a copy-paste from the Andrej Karpathy code, should be refined later
        print(f"Resuming training from {tr_config.checkpoint_output_dir}")
        # resume training from a checkpoint.
        ckpt_path = os.path.join(tr_config.checkpoint_output_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=tr_config.device)
        checkpoint_model_args = checkpoint['model_args']
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            setattr(model_config, k, checkpoint_model_args[k])
        # create the model
        model = GPTLM(model_config)
        state_dict = checkpoint['model']
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
    m = model.to(tr_config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=tr_config.lr)
    print(f"We are using device: {tr_config.device}")

    # Init the first batch
    xb, yb = get_batch('train', tr_config,dataset)
    while True:
        time_0 = time.time()
        lr = get_lr(tr_config, iter_num)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # every once in a while evaluate the loss on train and val sets
        if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
            losses = estimate_loss(m,tr_config,dataset)
            writer.add_scalar("Loss/test", losses['val'], iter_num)
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, time (s): {np.mean(durations).round(5) if len(durations)>0 else 0}, full time: {round(time.time()-start_time, 5)}" )
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
        last_loss = None
        for micro_step in range(tr_config.gradient_accumulation_steps):
            logits, loss = model(xb, yb)
            last_loss = loss
            loss = loss / tr_config.gradient_accumulation_steps
            xb, yb = get_batch('train', tr_config,dataset)
            loss.backward()
        writer.add_scalar("Loss/train", last_loss, iter_num)
        print(f"step {iter_num}: train loss {last_loss:.4f}, time (s): {np.mean(durations).round(5) if len(durations)>0 else 0}, lr: {lr}" )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        time_elapsed = time.time() - time_0
        writer.add_scalar("time_elapsed", time_elapsed, iter_num)
        durations.append(time_elapsed)
        iter_num += 1
        if iter_num >= max_iters:
            break
        
    writer.close()

def test_generation():
    model = from_pretrained_gpt2(torch.device("cuda"))
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
    train(Dataset.OPENWEBTEXT)
    # test_generation()