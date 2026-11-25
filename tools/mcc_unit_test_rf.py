"""Brief 02 A6 item 9 / PI 7.07i item 4: unit tests a and d of the MCC term inside the RhythmFormer loss
(train/rhythm_mcc.RhythmFormer_Loss_MCC), before any conditional RhythmFormer MCC run. Clip length 160 (RhythmFormer).

The archived frequency and HR terms build tensors on CUDA (TorchLossComputer), so the full loss is tested on CUDA only;
the MCC time term on CPU and CUDA.
a. Correctness, 100 random pairs (Gaussian white noise, numpy default_rng(7049)), float32:
   a1. the time term of RhythmFormer_Loss_MCC (criterion_Pearson(pred.view(1, -1), label.view(1, -1))) equals -MCC of the
       direct time-domain value of tools/mcc_unit_test.direct_mcc (|difference| <= 1e-5);
   a2. the full loss equals 0.2 x MCCLoss + frequency cross-entropy + HR term, the last two from the archived
       TorchLossComputer with the archived arguments (diff_flag 0, Fs 30, std 3.0) (|difference| <= 1e-5);
   a3. the full MCC loss minus the archived full loss equals 0.2 x (MCC term - negative Pearson term): only the time term
       differs (|difference| <= 1e-5).
d. Gradient, as in tools/mcc_unit_test.py d: random, prediction constant, label constant, both constant, one constant
   clip in a batch of 4 (summed per sample and divided by 4, as in src/st12c/rhythm_backend_v2.step).
   For the MCC time term alone and for the full loss, MCC and archived versions. Pass: the MCC time term is finite with a
   finite gradient in every case, and the full MCC loss is finite with a finite gradient in every case where the archived
   full loss is.
Writes outputs/704/mcc_unit_rf/{a_correctness,d_gradient}.csv and summary.json. Usage (repro, one visible GPU): mcc_unit_test_rf.py
"""
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np  # noqa: E402
import torch  # noqa: E402

WORK = Path(__file__).resolve().parents[1]
V = WORK / 'vendor' / 'st8_6ebbc9cd'
for p in (str(WORK / 'train'), str(V), str(WORK / 'tools')):
    sys.path.insert(0, p)
import mcc_unit_test as MU  # noqa: E402  (direct_mcc; nothing runs on import)
from mcc_loss import MCCLoss  # noqa: E402
from rhythm_mcc import RhythmFormer_Loss_MCC  # noqa: E402
from src.st12c.rhythm_loss_author import RhythmFormer_Loss, TorchLossComputer  # noqa: E402

OUT = WORK / 'outputs' / '704' / 'mcc_unit_rf'
N, FS = 160, 30


def write(name, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / name, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def parts(p, l):
    ce, _ = TorchLossComputer.Frequency_loss(p.squeeze(-1), l.squeeze(-1), diff_flag=0, Fs=FS, std=3.0)
    hr = TorchLossComputer.HR_loss(p.squeeze(-1), l.squeeze(-1), diff_flag=0, Fs=FS, std=3.0)
    return ce, hr


def test_a(devices):
    rng = np.random.default_rng(7049)
    rows = []
    for i in range(100):
        p, l = rng.standard_normal(N), rng.standard_normal(N)
        direct, _ = MU.direct_mcc(p, l)
        for device in devices:
            pt = torch.tensor(p, dtype=torch.float32, device=device)
            lt = torch.tensor(l, dtype=torch.float32, device=device)
            lm, lo = RhythmFormer_Loss_MCC(), RhythmFormer_Loss()
            term = float(lm.criterion_Pearson(pt.view(1, -1), lt.view(1, -1)))
            r = dict(pair=i, device=device, mcc_direct=direct, term=term, a1_abs_diff=abs(term + direct), a2_abs_diff=np.nan, a3_abs_diff=np.nan)
            if device == 'cuda':
                pear = float(lo.criterion_Pearson(pt.view(1, -1), lt.view(1, -1)))
                full_m, full_o = float(lm(pt, lt, 0, FS, 0)), float(lo(pt, lt, 0, FS, 0))
                ce, hr = parts(pt, lt)
                mcc = float(MCCLoss(Fs=FS)(pt.view(1, -1), lt.view(1, -1)))
                r.update(a2_abs_diff=abs(full_m - (0.2 * mcc + float(ce) + float(hr))), a3_abs_diff=abs((full_m - full_o) - 0.2 * (term - pear)))
            rows.append(r)
    write('a_correctness.csv', rows)
    out = {}
    for k in ('a1_abs_diff', 'a2_abs_diff', 'a3_abs_diff'):
        v = [r[k] for r in rows if np.isfinite(r[k])]
        out[k] = dict(n=len(v), max=max(v), passed=len(v) > 0 and all(x <= 1e-5 for x in v))
    return out


def test_d(devices):
    rng = np.random.default_rng(7049 + 1)
    rnd_p, rnd_l = rng.standard_normal((4, N)), rng.standard_normal((4, N))
    const = np.full((4, N), 0.7)
    mixed = rnd_p.copy()
    mixed[1] = 0.7
    cases = {'random': (rnd_p, rnd_l), 'pred constant': (const, rnd_l), 'label constant': (rnd_p, const),
             'both constant': (const, const), 'one constant clip in batch': (mixed, rnd_l)}
    rows = []
    for case, (p, l) in cases.items():
        for device in devices:
            for what in ('MCC time term', 'full loss MCC', 'full loss archived'):
                if what != 'MCC time term' and device != 'cuda':
                    continue
                pt = torch.tensor(p, dtype=torch.float32, device=device, requires_grad=True)
                lt = torch.tensor(l, dtype=torch.float32, device=device)
                if what == 'MCC time term':
                    c = RhythmFormer_Loss_MCC().criterion_Pearson
                    loss = sum(c(pt[i].view(1, -1), lt[i].view(1, -1)) for i in range(4)) / 4
                else:
                    c = RhythmFormer_Loss_MCC() if what == 'full loss MCC' else RhythmFormer_Loss()
                    loss = sum(c(pt[i], lt[i], 0, FS, 0) for i in range(4)) / 4
                graph = torch.is_tensor(loss) and loss.requires_grad
                if graph:
                    loss.backward()
                    g = pt.grad.detach().cpu().numpy() if pt.grad is not None else np.zeros(p.shape)
                else:
                    g = np.zeros(p.shape)
                lv = float(loss.detach()) if torch.is_tensor(loss) else float(loss)
                rows.append(dict(case=case, device=device, what=what, loss_has_graph=graph, loss=lv, loss_finite=bool(np.isfinite(lv)),
                                 grad_finite=bool(np.isfinite(g).all()), grad_nan=int(np.isnan(g).sum())))
    write('d_gradient.csv', rows)
    ok = lambda r: r['loss_finite'] and r['grad_finite']  # noqa: E731
    term = [r for r in rows if r['what'] == 'MCC time term']
    pairs = {(r['case'], r['device']): r for r in rows if r['what'] == 'full loss archived'}
    full = [r for r in rows if r['what'] == 'full loss MCC']
    return dict(cases=len(term), time_term_passed=all(ok(r) for r in term),
                full_loss_passed=all(ok(r) or not ok(pairs[(r['case'], r['device'])]) for r in full),
                full_loss_nonfinite_cases=[(r['case'], r['device']) for r in full if not ok(r)],
                archived_nonfinite_cases=[k for k, r in pairs.items() if not ok(r)])


def main():
    assert torch.cuda.is_available(), 'the archived frequency term needs CUDA'
    devices = ['cpu', 'cuda']
    a, d = test_a(devices), test_d(devices)
    s = dict(devices=devices, torch=torch.__version__, a=a, d=d,
             passed=all(v['passed'] for v in a.values()) and d['time_term_passed'] and d['full_loss_passed'])
    (OUT / 'summary.json').write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))
    sys.exit(0 if s['passed'] else 1)


if __name__ == '__main__':
    main()
