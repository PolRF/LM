from abc import ABC, abstractmethod
import numpy as np
import torch
import os


def _load_tokens(filename):
    npt = np.load(filename, mmap_mode="r")
    npt = npt.astype(np.int32, copy=False)  # added after video
    ptt = torch.from_numpy(npt) 
    return ptt


class DataLoader(ABC):

    def __init__(self, B, T, process_rank, num_processes, split):
        pass

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @abstractmethod
    def next_batch(self):
        raise NotImplementedError


class FineWebEduDataLoader(DataLoader):
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {"train", "val"}
        # get the shard filenames
        data_root = "data/fineweb_edu/edu_fineweb10B"
        if not os.path.exists(data_root):
            print(
                "Please download the dataset first by running 'python data/prepare.py'"
            )
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = _load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[
            self.current_position : self.current_position + B * T + 1
        ]
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)  # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(
            self.tokens
        ):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = _load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y
