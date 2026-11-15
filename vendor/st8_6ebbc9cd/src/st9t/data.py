"""Prescribed parent-owned index streams and CPU-only stored tensor loading."""
import numpy as np
import torch
from src.st9s.streams import SourceStreams,seed_worker


class StoredTensors(torch.utils.data.Dataset):
    def __init__(self,original,pool):self.original=original;self.pool=pool
    def __len__(self):return len(self.original)+len(self.pool)
    def __getitem__(self,index):
        kind,number=index;r=(self.original if kind==0 else self.pool)[number]
        x=np.load(r['input_path'],allow_pickle=False);y=np.load(r['label_path'],allow_pickle=False)
        x=np.ascontiguousarray(x.transpose(3,0,1,2),dtype=np.float32)
        y=np.ascontiguousarray(y,dtype=np.float32)
        high=float(r['reference_HR'] if kind==0 else r['target_HR'])>=100
        return torch.from_numpy(x),torch.from_numpy(y),kind,number,int(high)


class FixedBatches:
    def __init__(self,n,pool_size,seed,weights,arm,steps,batch_size=4):
        self.stream=SourceStreams(n,pool_size,seed,weights);self.arm=arm;self.steps=steps;self.batch_size=batch_size
    def __iter__(self):
        for _ in range(self.steps):
            original,augmented,_=self.stream.batch(self.arm,self.batch_size)
            yield [(0,int(i)) for i in original]+[(1,int(i)) for i in augmented]
    def __len__(self):return self.steps


def loader(original,pool,seed,arm,steps,workers):
    from functools import partial
    stream=FixedBatches(len(original),len(pool),seed,[float(r['LDS_weight']) for r in original],arm,steps)
    generator=torch.Generator();generator.manual_seed(seed)
    return torch.utils.data.DataLoader(StoredTensors(original,pool),batch_sampler=stream,num_workers=workers,
        worker_init_fn=partial(seed_worker,seed=seed),generator=generator,pin_memory=True,prefetch_factor=2,
        persistent_workers=True)
