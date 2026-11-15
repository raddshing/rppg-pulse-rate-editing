"""Unchanged official RhythmFormer model/loss/optimizer, without validation."""
from pathlib import Path
import sys
import torch
from src.st9s.streams import seed_globals

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'third_party/rPPG-Toolbox'))


def initialize(settings,seed,device='cuda:0'):
    from neural_methods.model.RhythmFormer import RhythmFormer
    from src.st12c.rhythm_loss_author import RhythmFormer_Loss
    seed_globals(seed,torch);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    model=RhythmFormer().to(device)
    model=torch.nn.DataParallel(model,device_ids=[0])
    criterion=RhythmFormer_Loss()
    optimizer=torch.optim.AdamW(model.parameters(),lr=float(settings['learning_rate']),weight_decay=0)
    scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=float(settings['learning_rate']),
        epochs=int(settings['epochs']),steps_per_epoch=int(settings['steps_per_epoch']))
    model.train();return model,criterion,optimizer,scheduler


def step(model,criterion,optimizer,scheduler,x,y,epoch):
    optimizer.zero_grad();prediction=model(x)
    prediction=(prediction-torch.mean(prediction,axis=-1).view(-1,1))/torch.std(prediction,axis=-1).view(-1,1)
    loss=sum(criterion(prediction[i],y[i],epoch,30,0) for i in range(len(x)))/len(x)
    if not torch.isfinite(loss):return loss
    loss.backward();optimizer.step();scheduler.step();return loss
