"""OPS2 7.19 B4: the UBFC-rPPG 360-window editor check of B02 (audit/a2_validity_ubfc.py, unchanged) with d = 1 frame.

Everything is that of the f0/2f0 run of B02 (outputs/2u: the same 360 windows of the October G-U sample, PhysNet,
FactorizePhys and RhythmFormer views, notch at f0 and 2f0, POS and CHROM readouts, ED2W-S editor, s = -1); the only change
is the delay of the (ii-pc) injection, D_FRAMES = 1 instead of 2 (set in every worker before its first window). Mode (ii)
does not use d; its readouts must equal those of outputs/2u. Reads the raw UBFC-rPPG videos and contact PPG (no stored
pool). CPU only; no model; no test data.
Usage: a2_validity_ubfc_d1.py OUT_DIR   (OPS2_WORKERS). Writes the four files of a2_validity_ubfc.py in OUT_DIR.
"""
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ['OPS2_UBFC_NOTCH'] = 'source'  # f0 and 2f0, as outputs/2u (read at import, also by the spawned workers)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'audit'))  # 7.21: published in tools/review4/; a2_validity_ubfc.py is in audit/
import a2_validity_ubfc as V  # noqa: E402

D1 = 1


def init_worker():
    V.D_FRAMES = D1
    V.init_worker()
    print(f'worker {os.getpid()} D_FRAMES {V.D_FRAMES} notch {V.NOTCH}', flush=True)


def main():
    out = Path(sys.argv[1])
    names = ['readouts', 'baselines', 'diagnostics', 'summary']
    for n in names:
        if (out / f'validity360_ubfc_{n}.csv').exists():
            sys.exit(f'{n} exists')
    out.mkdir(parents=True, exist_ok=True)
    rids = sorted(V.cases().recording_id.unique())
    started = time.time()
    result, baselines, diags = [], [], []
    with ProcessPoolExecutor(max_workers=int(os.environ.get('OPS2_WORKERS', '3')), mp_context=mp.get_context('spawn'),
                             initializer=init_worker) as ex:
        for n, (r, b, d) in enumerate(ex.map(V.job, rids), 1):
            result += r
            baselines += b
            diags += d
            print(f'validity ubfc d=1 {n}/{len(rids)} recordings, {time.time() - started:.0f}s', flush=True)
    V.write_rows(out / 'validity360_ubfc_readouts.csv', [dict(r, d_pc=D1) for r in result])
    V.write_rows(out / 'validity360_ubfc_baselines.csv', baselines)
    V.write_rows(out / 'validity360_ubfc_diagnostics.csv', [dict(r, d_pc=D1) for r in diags])
    s = V.summary(V.pd.DataFrame(result), V.pd.DataFrame(baselines))
    s.to_csv(out / 'validity360_ubfc_summary.csv', index=False)
    print('done', len(result), len(baselines), f'{time.time() - started:.0f}s', flush=True)


if __name__ == '__main__':
    main()
