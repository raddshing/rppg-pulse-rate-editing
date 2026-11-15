"""Model groups for MCD-dev/MMPD-A inference order (PI 11-10 evening) and the trained (ii-pc) models.

g1 matched FactorizePhys (0)/(i)/(ii), g2 PhysNet (0)/(i)/(ii), g3 RhythmFormer (0)/(i)/(ii) -- seeds
20261131-33 (the 1st-group contrast seeds); g4 the remaining 1b models (incl. PhysNet 5-seed extras).
iipc_<backbone>: the three (ii-pc) runs, available once all three seeds completed.
"""
import json
from pathlib import Path

from roster import roster

WORK = Path(__file__).resolve().parents[1]
MAIN_SEEDS = (20261131, 20261132, 20261133)
FIRST = {'g1': 'FactorizePhys (matched)', 'g2': 'PhysNet', 'g3': 'RhythmFormer'}
IIPC_VIEW = {'FPm': 'FPm', 'PhysNet': 'PhysNet', 'RhythmFormer': 'RhythmFormer'}


def groups():
    out = {k: [] for k in ['g1', 'g2', 'g3', 'g4']}
    for r in roster():
        g = next((k for k, b in FIRST.items() if r['backbone'] == b and r['strategy'] in ('(0)', '(i)', '(ii)')
                  and r['seed'] in MAIN_SEEDS), 'g4')
        out[g].append(r)
    assert [len(out[k]) for k in ['g1', 'g2', 'g3']] == [9, 9, 9] and len(out['g4']) == 61
    return out


def iipc(backbone):
    """The three trained (ii-pc) runs of one backbone, or None until all three completed."""
    return trained(backbone, 'ii-pc')


def ipc(backbone):
    """The three trained (i-pc) runs of one backbone (brief 02 6.12), or None until all three completed."""
    return trained(backbone, 'i-pc')


def iinv(backbone):
    """The three trained (ii-inv) runs of one backbone (brief 02 6.18), or None until all three completed."""
    return trained(backbone, 'ii-inv')


def ilag(backbone):
    """The three trained (ii-lag) runs of one backbone (brief 02 6.19), or None until all three completed."""
    return trained(backbone, 'ii-lag')


def rfl(backbone='PhysNet'):
    """Brief 02 6.19 C / 6.29 loss swap: PhysNet (ii) and (ii-pc) trained with the RhythmFormer author loss, 3 seeds
    each (cache/train/PhysNet_ii-rfl_s*, PhysNet_ii-pc-rfl_s*; strategies '(ii-rfl)', '(ii-pc-rfl)'), or None until
    all six completed."""
    a, b = trained(backbone, 'ii-rfl'), trained(backbone, 'ii-pc-rfl')
    return a + b if a and b else None


def rfli(backbone='PhysNet'):
    """Brief 02 6.37 loss swap, resampling pair: PhysNet (i) and (i-pc) with the RhythmFormer author loss, 3 seeds each
    (cache/train/PhysNet_i-rfl_s*, PhysNet_i-pc-rfl_s*; strategies '(i-rfl)', '(i-pc-rfl)'), or None until all six completed."""
    a, b = trained(backbone, 'i-rfl'), trained(backbone, 'i-pc-rfl')
    return a + b if a and b else None


LAB_ARMS = ('ii-pc-g', 'ii-pc-g80', 'ii-pc-g2', 'i-pc-g80', 'sham-pc', 'roi-pc',  # brief 02 6.43 라
            'ii-std', 'ii-lag-std', 'ii-pc-std')  # brief 02 6.45 다 3 PhysNet-Std (Standardized labels; readout without diff)


def lab(backbone, arm):
    """Brief 02 6.43 라: the three trained runs of one label/processing arm (cache/train/<B>_<arm>_s*), or None."""
    return trained(backbone, arm)


def isimplerep(backbone):
    """Brief 02 6.67 마 (diagnostic rule 6): the (i-simple) replicate runs of one backbone (learning rate x (1 + 1e-6), otherwise
    the original run; cache/train/<B>_i-simple-rep_s*), or None until all three completed. Group 'isimplerep_<B>' only:
    not in all_trained() and not in b1/analyze.py runs(), so no development analysis reads them."""
    return trained(backbone, 'i-simple-rep')


def isimple(backbone):
    """Brief 02 6.43 (i-simple): the three trained runs of one backbone (cache/train/<B>_i-simple_s*), or None until all
    three completed."""
    return trained(backbone, 'i-simple')


S5F_ARMS = {'0F': '(0-F)', 'TRF': '(TR-F)', 'EDF': '(ED-F)'}
S5F_DIR = Path('<MACHINE_PATH>')  # read-only October checkpoints (ST4 instr. 07 R7)


def s5f():
    """Brief 02 6.19 B4: the PhysNet + frequency-loss runs 0-F, TR-F, ED-F (3 seeds). Not part of all_trained()."""
    out = []
    for arm, st in S5F_ARMS.items():
        for seed in MAIN_SEEDS:
            ck = S5F_DIR / f'st14t_PhysNet_{arm}_s{seed}' / 'attempt1' / 'final_epoch.pth'
            assert ck.exists(), ck
            out.append(dict(run_id=f'st14t_PhysNet_{arm}_s{seed}', backbone='PhysNet', view='PhysNet', strategy=st,
                            seed=seed, five_seed_extra=False, checkpoint=str(ck), namespace='st14t'))
    return out


B8_RNG = (101, 102, 103, 104, 105)


def b8():
    """Brief 02 6.19 B8: matched FactorizePhys (0)/(i)/(ii)/(ii-pc), seeds 20261131-33, re-inferred with the inference
    RNG seeded 101-105 (instead of 100) and once with FSAM off at inference (MD_INFERENCE False, seed 100).
    Keys '<run_id>__rng<seed>' / '<run_id>__fsamoff'; not part of all_trained(). base_run_id = the stored run's run_id
    (analysis); checkpoint_run_id = the run_id stored in the checkpoint (server check; differs for OPS2-trained runs)."""
    base = [r for r in groups()['g1']] + (iipc('FPm') or [])
    assert len(base) == 12, len(base)
    out = []
    for r in base:
        ck = r.get('checkpoint_run_id', r['run_id'])
        for s in B8_RNG:
            out.append(dict(r, run_id=f"{r['run_id']}__rng{s}", base_run_id=r['run_id'], checkpoint_run_id=ck, rng_seed=s, variant=f'rng{s}'))
        out.append(dict(r, run_id=f"{r['run_id']}__fsamoff", base_run_id=r['run_id'], checkpoint_run_id=ck, md_inference=False, variant='fsamoff'))
    return out


def b8ipc():
    """Brief 02 6.34 item 1: matched FactorizePhys (i-pc), seeds 20261131-33, FSAM off at inference (MD_INFERENCE False,
    seed 100), completing the B8 FSAM-off table. Keys '<run_id>__fsamoff'."""
    base = ipc('FPm') or []
    assert len(base) == 3, len(base)
    return [dict(r, run_id=f"{r['run_id']}__fsamoff", base_run_id=r['run_id'], checkpoint_run_id=r.get('checkpoint_run_id', r['run_id']),
                 md_inference=False, variant='fsamoff') for r in base]


def by_group(g):
    """'iipc_<B>' / 'ipc_<B>' / 'iinv_<B>' / 'ilag_<B>' trained OPS2 runs (None until complete); otherwise None."""
    if g.startswith('iipc_'):
        return iipc(g[5:])
    if g.startswith('ipc_'):
        return ipc(g[4:])
    if g.startswith('iinv_'):
        return iinv(g[5:])
    if g.startswith('ilag_'):
        return ilag(g[5:])
    if g.startswith('isimple_'):
        return isimple(g[8:])
    if g.startswith('isimplerep_'):
        return isimplerep(g[11:])
    if g.startswith('lab-'):  # 'lab-<arm>_<backbone>'
        arm, backbone = g[4:].rsplit('_', 1)
        assert arm in LAB_ARMS, g
        return lab(backbone, arm)
    if g == 's5f_PhysNet':
        return s5f()
    if g == 'rfl_PhysNet':
        return rfl('PhysNet')
    if g == 'rfli_PhysNet':
        return rfli('PhysNet')
    if g == 'b8_FPm':
        return b8()
    if g == 'b8ipc_FPm':
        return b8ipc()
    return None


def all_trained():
    out = []
    for b in ('FPm', 'PhysNet', 'RhythmFormer'):
        for arm in ('ii-pc', 'i-pc', 'ii-inv', 'ii-lag', 'i-simple') + LAB_ARMS:
            out += trained(b, arm) or []
    return out


def trained(backbone, arm):
    runs = []
    for seed in MAIN_SEEDS:
        d = WORK / 'cache' / 'train' / f'{backbone}_{arm}_s{seed}'
        try:
            res = json.loads((d / 'result.json').read_text())
        except (OSError, ValueError):
            return None
        if res.get('status') != 'complete':
            return None
        runs.append(dict(run_id=f'ops2_{backbone}_{arm}_s{seed}', backbone=backbone, view=IIPC_VIEW[backbone],
                         strategy=f'({arm})', seed=seed, five_seed_extra=False, checkpoint=str(d / 'final_epoch.pth'),
                         namespace='ops2', checkpoint_run_id=d.name))  # run_id stored in the checkpoint
    return runs
