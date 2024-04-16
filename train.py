from dataclasses import dataclass
from enum import Enum
from typing import Literal
import tiktoken
import torch

from model import ModelConfig, GPTLM
import numpy as np
import os

BASE_DATA_PATH = './data/'

@dataclass
class TrainConfig:
    batch_size: int
    block_size: int
    eval_iters: int
    lr: float
    device: torch.device



class Dataset(Enum):
    BIBLE = 'bible'
    SHAKESPEARE = 'shakespeare'


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
    data_path = BASE_DATA_PATH+dataset.name+"/"
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


def train(dataset:Dataset):
    
    tr_config = TrainConfig(
        batch_size=64,
        block_size=256,
        eval_iters=200,
        lr=6e-5,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )
    # Override device for Macbook pro m2 chip
    # tr_config.device=torch.device("mps")
    max_iters = 60_000
    eval_interval = 200

    model_config = ModelConfig(
        vocab_size=50304,
        block_size=tr_config.block_size,
        n_head=2,
        n_layer=2,
        n_embd=384,
    )

    model = GPTLM(model_config)
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
    m = model.to(tr_config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=tr_config.lr)
    print(f"We are using device: {tr_config.device}")
    # Iterator only
    for iter in range(max_iters):
        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(m,tr_config,dataset)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch('train', tr_config,dataset)

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
    os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
    train(Dataset.BIBLE)