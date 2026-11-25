"""Brief 02 7.04a 다 (A6 item 4): compare the OPS2 retraining of arms (0), (i), (ii) (PhysNet, FactorizePhys) with the
confirmatory checkpoints trained by the October code.

Per run: initial-weight digest (OPS2 result.json vs the October record), order of exposures (October records keep, per
step, the number of originals and pool items and of high-HR items, not the item indices: these counts are compared with
the OPS2 exposures.csv; the item order itself is 'none' in the October records), per-step loss (OPS2 steps.csv vs the
October steps.csv: equal steps and largest absolute difference), checkpoint SHA-256 (OPS2 vs docs/PREREGISTRATION_STAGE5_
MANIFEST.csv) and, if they differ, the largest absolute difference of any tensor of the two state dicts.
Writes outputs/704/trainer_check/{runs,steps_by_epoch}.csv. Usage (repro environment): trainer_check_704.py [SEED]
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parents[1]
CFG = json.loads((Path(__file__).resolve().parents[1] / 'config' / 'paths.json').read_text())  # 7.08: machine paths, see config/paths.example.json
OPS1 = Path(CFG['october_project']) / 'reports'  # read-only
BULK = Path(CFG['october_bulk'])  # read-only October checkpoints
OUT = WORK / 'outputs' / '704' / 'trainer_check'
V = WORK / 'vendor' / 'st8_6ebbc9cd'
SRC_NEW, SRC_OCT = '7.04 new training (OPS2 trainer)', 'October records (OPS1, read only)'


def rows(p):
    with open(p, newline='') as f:
        return list(csv.DictReader(f))


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def runs(seed):
    s = str(seed)
    return [
        dict(backbone='PhysNet', arm='(0)', ops2=f'PhysNet_0_s{s}', oct=f'st9t_PhysNet_0_{s}', steps=OPS1 / f'st9t/loss/st9t_PhysNet_0_{s}_steps.csv',
             ckpt=BULK / f'st9t/st9t_PhysNet_0_{s}/final_epoch.pth', init=('PhysNet', s)),
        dict(backbone='PhysNet', arm='(i)', ops2=f'PhysNet_i_s{s}', oct=f'st9t_PhysNet_i_{s}', steps=OPS1 / f'st9t/loss/st9t_PhysNet_i_{s}_steps.csv',
             ckpt=BULK / f'st9t/st9t_PhysNet_i_{s}/final_epoch.pth', init=('PhysNet', s)),
        dict(backbone='PhysNet', arm='(ii)', ops2=f'PhysNet_ii_s{s}', oct=f'st9t_PhysNet_ii-h_{s}', steps=OPS1 / f'st9t/loss/st9t_PhysNet_ii-h_{s}_steps.csv',
             ckpt=BULK / f'st9t/st9t_PhysNet_ii-h_{s}/final_epoch.pth', init=('PhysNet', s)),
        dict(backbone='FactorizePhys-matched', arm='(0)', ops2=f'FPm_0_s{s}', oct=f'st30t_FPm_PURE_0_s{s}', steps=OPS1 / f'st30t/loss/st30t_FPm_PURE_0_s{s}_steps.csv',
             ckpt=BULK / f'st30t/checkpoints/st30t_FPm_PURE_0_s{s}/final_epoch.pth', init=('FactorizePhys', s)),
        dict(backbone='FactorizePhys-matched', arm='(i)', ops2=f'FPm_i_s{s}', oct=f'st30t_FPm_PURE_i_s{s}', steps=OPS1 / f'st30t/loss/st30t_FPm_PURE_i_s{s}_steps.csv',
             ckpt=BULK / f'st30t/checkpoints/st30t_FPm_PURE_i_s{s}/final_epoch.pth', init=('FactorizePhys', s)),
        dict(backbone='FactorizePhys-matched', arm='(ii)', ops2=f'FPm_ii_s{s}', oct=f'st30t_FPm_PURE_ii_s{s}', steps=OPS1 / f'st30t/loss/st30t_FPm_PURE_ii_s{s}_steps.csv',
             ckpt=BULK / f'st30t/checkpoints/st30t_FPm_PURE_ii_s{s}/final_epoch.pth', init=('FactorizePhys', s)),
    ]


def october_counts(r):
    """Per step: (originals, pool items, high-HR items) from the October steps.csv (column names differ by namespace)."""
    o = r.get('original', r.get('original_exposures'))
    a = r.get('augmented', r.get('pool_exposures'))
    if 'high_original' in r and 'high_augmented' in r:  # PhysNet: high items by reference / target HR
        h = int(r['high_original']) + int(r['high_augmented'])
    elif 'manifest_high' in r:  # FactorizePhys (i), (ii): high items by the manifest target HR
        h = int(r['manifest_high'])
    else:  # FactorizePhys (0): high_original (reference HR, as the OPS2 count); high_actual is another definition
        h = int(r['high_original'])
    return int(o), int(a), h


HIGH_COLUMN = {'PhysNet': 'high_original + high_augmented', '(0)': 'high_original', 'other': 'manifest_high'}


def same(x, y):
    import torch
    if isinstance(x, torch.Tensor):
        return isinstance(y, torch.Tensor) and x.dtype == y.dtype and x.shape == y.shape and torch.equal(x, y)
    if isinstance(x, dict):
        return isinstance(y, dict) and set(x) == set(y) and all(same(x[k], y[k]) for k in x)
    if isinstance(x, (list, tuple)):
        return isinstance(y, (list, tuple)) and len(x) == len(y) and all(same(u, v) for u, v in zip(x, y))
    return x == y


def tensor_diff(a, b):
    """Model tensors (largest absolute difference), optimizer and scheduler states (equal or not) and the other
    (metadata) entries of the two checkpoint files."""
    import torch
    ca, cb = torch.load(a, map_location='cpu'), torch.load(b, map_location='cpu')
    sa, sb = ca['model'], cb['model']
    keys = sorted(set(sa) | set(sb))
    m = max(float((sa[k].float() - sb[k].float()).abs().max()) if k in sa and k in sb and sa[k].numel() else 0.0 for k in keys)
    meta = sorted(k for k in set(ca) | set(cb) if k not in ('model', 'optimizer', 'scheduler') and (k not in ca or k not in cb or ca[k] != cb[k]))
    return dict(state_dict_same_keys=set(sa) == set(sb), tensor_max_abs_difference=m,
                model_state_identical=same(sa, sb), optimizer_state_identical=same(ca.get('optimizer'), cb.get('optimizer')),
                scheduler_state_identical=same(ca.get('scheduler'), cb.get('scheduler')),
                other_entries_differing=' '.join(meta))


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 20261131
    manifest = {(r['backbone'], r['arm'], r['seed']): r['sha256'] for r in rows(WORK / 'pub/docs/PREREGISTRATION_STAGE5_MANIFEST.csv') if r['kind'] == 'checkpoint'}
    init = json.loads((V / 'reports/st9t/st9t_initial_weights.json').read_text())
    out, by_epoch = [], []
    for c in runs(seed):
        d = WORK / 'cache' / 'train704' / c['ops2']
        res = json.loads((d / 'result.json').read_text()) if (d / 'result.json').exists() else {}
        rec = dict(backbone=c['backbone'], arm=c['arm'], seed=seed, ops2_status=res.get('status', 'absent'))
        rec['initial_weights_october'] = init[f"{c['init'][0]}|{c['init'][1]}"]['sha256']
        rec['initial_weights_ops2'] = res.get('initial_weights_sha256', '')
        rec['initial_weights_equal'] = rec['initial_weights_october'] == rec['initial_weights_ops2']
        if not res:
            out.append(rec)
            continue
        old = rows(c['steps'])
        new = rows(d / 'steps.csv')
        ex = rows(d / 'exposures.csv')
        per = {}
        for e in ex:
            k = int(e['step'])
            o, a, h = per.get(k, (0, 0, 0))
            per[k] = (o + (e['kind'] == '0'), a + (e['kind'] == '1'), h + (float(e['label_HR']) >= 100))
        cnt_eq = sum(october_counts(r) == per.get(int(r['step'])) for r in old)
        rec['october_high_column'] = HIGH_COLUMN['PhysNet'] if c['backbone'] == 'PhysNet' else HIGH_COLUMN['(0)' if c['arm'] == '(0)' else 'other']
        for alt in ('high_actual', 'stored_high'):
            if alt in old[0]:  # other high definition kept in the October record, compared for information only
                rec[f'steps_with_equal_{alt}'] = sum(int(r[alt]) == per.get(int(r['step']), (0, 0, -1))[2] for r in old)
        lo = np.array([float(r['loss']) for r in old])
        ln = np.array([float(r['loss']) for r in new])
        n = min(len(lo), len(ln))
        eq = lo[:n] == ln[:n]
        first_diff = int(np.argmin(eq)) + 1 if not eq.all() else ''
        rec.update(steps_october=len(old), steps_ops2=len(new), exposure_order_october='none (only per-step counts are kept)',
                   steps_with_equal_counts=cnt_eq, loss_equal_steps=int(eq.sum()), loss_first_different_step=first_diff,
                   loss_max_abs_difference=float(np.abs(lo[:n] - ln[:n]).max()),
                   final_loss_october=float(lo[-1]), final_loss_ops2=float(ln[-1]))
        rec['checkpoint_sha256_confirmatory'] = manifest.get((c['backbone'], c['arm'], str(seed)), '')
        rec['checkpoint_sha256_ops2'] = res.get('checkpoint_sha256', '')
        rec['checkpoint_identical'] = rec['checkpoint_sha256_confirmatory'] == rec['checkpoint_sha256_ops2']
        rec['october_file_sha256_equals_manifest'] = sha(c['ckpt']) == rec['checkpoint_sha256_confirmatory'] if c['ckpt'].exists() else 'file absent'
        if not rec['checkpoint_identical'] and c['ckpt'].exists() and res.get('status') == 'complete':
            rec.update(tensor_diff(d / 'final_epoch.pth', c['ckpt']))
        rec.update(source_ops2=SRC_NEW, source_october=SRC_OCT)
        out.append(rec)
        epoch = [int(r['epoch']) for r in old][:n]
        for e in sorted(set(epoch)):
            idx = [i for i, x in enumerate(epoch) if x == e]
            by_epoch.append(dict(backbone=c['backbone'], arm=c['arm'], seed=seed, epoch=e, steps=len(idx),
                                 equal_steps=int(eq[idx].sum()), max_abs_difference=float(np.abs(lo[idx] - ln[idx]).max()),
                                 mean_loss_october=float(lo[idx].mean()), mean_loss_ops2=float(ln[idx].mean()), source=SRC_NEW))
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rr in (('runs.csv', out), ('steps_by_epoch.csv', by_epoch)):
        if rr:
            keys = list(dict.fromkeys(k for r in rr for k in r))
            with open(OUT / f'{name[:-4]}_s{seed}.csv', 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(rr)
    print(json.dumps(out, indent=1, default=str))


if __name__ == '__main__':
    main()
