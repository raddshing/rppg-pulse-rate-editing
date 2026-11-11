"""Official model/loss/optimizer steps without validation or evaluation calls."""
from pathlib import Path
import sys
import hashlib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TOOLBOX = ROOT / 'third_party/rPPG-Toolbox'
sys.path.insert(0, str(TOOLBOX))
from dataset.data_loader.BaseLoader import BaseLoader
from src.st9s.streams import seed_globals


def preprocess_context(view, label, model, length):
    if model == 'PhysNet':
        x = BaseLoader.diff_normalize_data(view)
        y = BaseLoader.diff_normalize_label(label)
    elif model == 'FactorizePhys':
        x = view
        y = BaseLoader.standardized_label(label)
    else:
        x = BaseLoader.standardized_data(view)
        y = BaseLoader.diff_normalize_label(label)
    offset = (len(view) - length) // 2
    return np.ascontiguousarray(x[offset:offset + length].transpose(3, 0, 1, 2), dtype=np.float32), np.ascontiguousarray(y[offset:offset + length], dtype=np.float32)


def initialize(model_name, settings, seed, device='cpu'):
    seed_globals(seed, torch)
    # Frozen official main.py runtime settings apply to every B backbone run.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(device)
    if model_name == 'PhysNet':
        from neural_methods.model.PhysNet import PhysNet_padding_Encoder_Decoder_MAX
        from neural_methods.loss.PhysNetNegPearsonLoss import Neg_Pearson
        model = PhysNet_padding_Encoder_Decoder_MAX(frames=128).to(device)
    else:
        from config import _C
        from neural_methods.model.FactorizePhys.FactorizePhys import FactorizePhys
        from neural_methods.loss.NegPearsonLoss import Neg_Pearson
        cfg = _C.clone()
        cfg.merge_from_file(str(ROOT / settings['official_config']))
        md = {k:cfg.MODEL.FactorizePhys[k] for k in ['FRAME_NUM','MD_TYPE','MD_FSAM','MD_TRANSFORM',
            'MD_S','MD_R','MD_STEPS','MD_INFERENCE','MD_RESIDUAL']}
        model = FactorizePhys(frames=160, md_config=md, in_channels=3, dropout=cfg.MODEL.DROP_RATE, device=device).to(device)
    criterion = Neg_Pearson()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(settings['learning_rate']))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=float(settings['learning_rate']),
        epochs=settings['epochs'], steps_per_epoch=settings['steps_per_epoch'])
    model.train()
    return model, criterion, optimizer, scheduler


def state_digest(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def step(model_name, model, criterion, optimizer, scheduler, x, y):
    optimizer.zero_grad()
    y = (y - torch.mean(y)) / torch.std(y)
    if model_name == 'FactorizePhys':
        x = torch.cat([x, x[:, :, -1:, :, :]], dim=2)
    pred = model(x)[0]
    pred = (pred - torch.mean(pred)) / torch.std(pred)
    loss = criterion(pred, y)
    if not torch.isfinite(loss):
        return loss
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss
