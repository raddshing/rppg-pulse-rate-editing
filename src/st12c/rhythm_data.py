"""Stored RhythmFormer tensors with unchanged paired source-index streams."""
import numpy as np
import torch
from src.st9t.data import FixedBatches
from src.st9s.streams import seed_worker


class RhythmTensors(torch.utils.data.Dataset):
    def __init__(self,original,pool):self.original=original;self.pool=pool
    def __len__(self):return len(self.original)+len(self.pool)
    def __getitem__(self,index):
        kind,number=index;r=(self.original if kind==0 else self.pool)[number]
        x=np.ascontiguousarray(np.load(r['input_path'],allow_pickle=False).transpose(0,3,1,2),dtype=np.float32)
        y=np.ascontiguousarray(np.load(r['label_path'],allow_pickle=False),dtype=np.float32)
        high=float(r['reference_HR'] if kind==0 else r['target_HR'])>=100
        return torch.from_numpy(x),torch.from_numpy(y),kind,number,int(high)


def loader(original,pool,seed,arm,steps,workers):
    from functools import partial
    generator=torch.Generator();generator.manual_seed(seed)
    return torch.utils.data.DataLoader(RhythmTensors(original,pool),
        batch_sampler=FixedBatches(len(original),len(pool),seed,None,arm,steps),num_workers=workers,
        worker_init_fn=partial(seed_worker,seed=seed),generator=generator,pin_memory=True,
        prefetch_factor=2,persistent_workers=True)
