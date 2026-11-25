"""Brief 02 7.04 나 3: unit tests of the MCC loss (train/mcc_loss.py) before any MCC training.

a. Correctness: 100 random pairs per clip length (128 = PhysNet, 160 = FactorizePhys), Gaussian white noise,
   numpy default_rng(704). Direct time-domain value: both signals mean-removed and zero-padded to 2N, each passed through
   the same ideal band-pass (rfft bins 0.6-3.3 Hz kept, irfft), circular cross-correlation of the two padded band-passed
   signals summed in the time domain for every lag, divided by (N - 1), by the two sample standard deviations of the
   unpadded inputs and by the author's energy ratio (in-band over total sum of |P conj(L)|), maximum over lags.
   Implementation value = -NegativeMaxCrossCorr. Pass: |difference| <= 1e-5 for every pair (CPU float32, CPU float64,
   CUDA float32). The verbatim author code (loops, torch.std) is also run and compared with the implementation.
b. Shift recovery: real labels. For each of the 47 PURE training recordings (subjects 01-08) the contact PPG at the frame
   times (the stage-1 PURE loader interpolation, src/st9e/native.contact) is cut at frames [0, N) (label g) and
   [k, k + N) (prediction, k = 0..30); each window is turned into the label of the setting as in training:
   PhysNet differenced (diff_normalize_label, N 128), PhysNet standardized (cumulative sum of that, standardized; N 128),
   FactorizePhys standardized (standardized_label, N 160). The lag of the maximum is d with pred[t] = label[t + d].
   Pass: |d - k| <= 1 for every recording and k. MCC values per k are reported.
c. MCC(g, g) for the 47 recordings and three settings.
d. Gradient: loss and gradient with respect to the prediction finite for random and constant inputs (pred constant,
   label constant, both constant, one constant clip in a random batch), implementation and author code, CPU and CUDA.
e. Values only: MCC(-g, g); MCC(g + a sin(2 pi 0.3 t) + a sin(2 pi 5 t), g) and each component alone, a = std(g).
Writes outputs/704/mcc_unit/{a_correctness,b_shift,c_self,d_gradient,e_values}.csv and summary.json.
Usage (repro environment): mcc_unit_test.py
"""
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
import torch
import torch.nn.functional as F

WORK = Path(__file__).resolve().parents[1]
V = WORK / 'vendor' / 'st8_6ebbc9cd'
sys.path.insert(0, str(WORK / 'train'))
sys.path.insert(0, str(V))
import mcc_loss  # noqa: E402
from mcc_loss import NegativeMaxCrossCorr  # noqa: E402

OUT = WORK / 'outputs' / '704' / 'mcc_unit'
FS, LO, HI = 30, 36.0, 198.0
TOL = 1e-5
torch.set_num_threads(1)


# ---- verbatim author code (RemotePPG 38cfde2, MIT), band passed as arguments ----
class AuthorCov(torch.nn.Module):
    def __init__(self, Fs, high_pass, low_pass):
        super().__init__()
        self.Fs, self.high_pass, self.low_pass = Fs, high_pass, low_pass

    def forward(self, preds, labels):
        tr = torch
        preds_norm = preds - torch.mean(preds, dim=-1, keepdim=True)
        labels_norm = labels - torch.mean(labels, dim=-1, keepdim=True)
        min_N = min(preds.shape[-1], labels.shape[-1])
        padded_N = max(preds.shape[-1], labels.shape[-1]) * 2
        preds_pad = F.pad(preds_norm, (0, padded_N - preds.shape[-1]))
        labels_pad = F.pad(labels_norm, (0, padded_N - labels.shape[-1]))
        preds_fft = torch.fft.rfft(preds_pad, dim=-1)
        labels_fft = torch.fft.rfft(labels_pad, dim=-1)
        X = preds_fft * torch.conj(labels_fft)
        X_real = tr.view_as_real(X)
        Fn = self.Fs / 2
        freqs = torch.linspace(0, Fn, X.shape[-1])
        use_freqs = torch.logical_and(freqs >= self.high_pass / 60, freqs <= self.low_pass / 60)
        zero_freqs = torch.logical_not(use_freqs)
        use_energy = tr.sum(tr.linalg.norm(X_real[:, use_freqs], dim=-1), dim=-1)
        zero_energy = tr.sum(tr.linalg.norm(X_real[:, zero_freqs], dim=-1), dim=-1)
        denom = use_energy + zero_energy
        energy_ratio = tr.ones_like(denom)
        for ii in range(len(denom)):
            if denom[ii] > 0:
                energy_ratio[ii] = use_energy[ii] / denom[ii]
        X[:, zero_freqs] = 0.
        cc = torch.fft.irfft(X, dim=-1) / (min_N - 1)
        max_cc = torch.max(cc, dim=-1)[0] / energy_ratio
        return -max_cc


class AuthorCorr(torch.nn.Module):
    def __init__(self, Fs, high_pass, low_pass):
        super().__init__()
        self.cross_cov = AuthorCov(Fs, high_pass, low_pass)

    def forward(self, preds, labels):
        tr = torch
        denom = tr.std(preds, dim=-1) * tr.std(labels, dim=-1)
        cov = self.cross_cov(preds, labels)
        output = torch.zeros_like(cov)
        for ii in range(len(denom)):
            if denom[ii] > 0:
                output[ii] = cov[ii] / denom[ii]
        return output


def write(name, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / name, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def band_mask(npad):
    freqs = np.fft.rfftfreq(npad, 1 / FS)
    return (freqs >= LO / 60) & (freqs <= HI / 60)


def direct_mcc(p, l):
    """Time-domain value (float64): see a. in the module docstring."""
    n = len(p)
    npad = 2 * n
    pm, lm = np.zeros(npad), np.zeros(npad)
    pm[:n], lm[:n] = p - p.mean(), l - l.mean()
    mask = band_mask(npad)
    P, L = np.fft.rfft(pm), np.fft.rfft(lm)
    pf = np.fft.irfft(P * mask, npad)
    lf = np.fft.irfft(L * mask, npad)
    cc = np.array([np.dot(np.roll(pf, -tau), lf) for tau in range(npad)])  # sum_t pf[t + tau] lf[t], circular
    X = P * np.conj(L)
    total = np.abs(X).sum()
    ratio = np.abs(X[mask]).sum() / total if total > 0 else 1.0
    m = cc / (n - 1) / ratio / (np.std(p, ddof=1) * np.std(l, ddof=1))
    k = int(np.argmax(m))
    tau = k - npad if k >= npad // 2 else k
    return float(m[k]), -tau


def impl(p, l, dtype=torch.float32, device='cpu', author=False, lag=False):
    pt = torch.as_tensor(np.atleast_2d(p), dtype=dtype, device=device)
    lt = torch.as_tensor(np.atleast_2d(l), dtype=dtype, device=device)
    if author:
        return (-AuthorCorr(FS, LO, HI)(pt, lt)).detach().cpu().double().numpy()
    out, d = NegativeMaxCrossCorr(FS, LO, HI)(pt, lt, return_lag=True)
    out = (-out).detach().cpu().double().numpy()
    return (out, d.cpu().numpy()) if lag else out


def test_a(cuda):
    rng = np.random.default_rng(704)
    rows = []
    for n in (128, 160):
        for i in range(100):
            p, l = rng.standard_normal(n), rng.standard_normal(n)
            ref, ref_lag = direct_mcc(p, l)
            r = dict(length=n, pair=i, direct=ref, direct_lag=ref_lag)
            r['cpu_float32'] = float(impl(p, l)[0])
            r['cpu_float64'] = float(impl(p, l, torch.float64)[0])
            r['cuda_float32'] = float(impl(p, l, device='cuda')[0]) if cuda else ''
            r['author_cpu_float32'] = float(impl(p, l, author=True)[0])
            _, lag = impl(p, l, torch.float64, lag=True)
            r['cpu_float64_lag'] = int(lag[0])
            for k in ('cpu_float32', 'cpu_float64', 'cuda_float32'):
                r[f'abs_diff_{k}'] = abs(r[k] - ref) if r[k] != '' else ''
            r['abs_diff_impl_vs_author_float32'] = abs(r['cpu_float32'] - r['author_cpu_float32'])
            rows.append(r)
    write('a_correctness.csv', rows)
    keys = ['cpu_float32', 'cpu_float64'] + (['cuda_float32'] if cuda else [])
    summary = {k: max(r[f'abs_diff_{k}'] for r in rows) for k in keys}
    return dict(pairs=len(rows), tolerance=TOL, max_abs_diff=summary,
                max_abs_diff_impl_vs_author=max(r['abs_diff_impl_vs_author_float32'] for r in rows),
                lag_equal_direct=sum(r['cpu_float64_lag'] == r['direct_lag'] for r in rows),
                passed=all(v <= TOL for v in summary.values()))


def diff_normalize_label(label):  # src/st9g/preprocess.py (unchanged official transform)
    diff_label = np.diff(label, axis=0)
    out = diff_label / np.std(diff_label)
    out = np.append(out, np.zeros(1), axis=0)
    out[np.isnan(out)] = 0
    return out


def standardized_label(label):  # src/st9g/preprocess.py
    label = label - np.mean(label)
    label = label / np.std(label)
    label[np.isnan(label)] = 0
    return label


def std_cumsum(label):  # train/u8_store.py OPS2_LABEL_REPR=std_cumsum, applied to the float32 DiffNormalized label
    c = np.cumsum(diff_normalize_label(label).astype(np.float32).astype(np.float64))
    return (c - c.mean()) / c.std()


def mcc_curve(p, l):
    """MCC as a function of the lag index tau (float64, frequency-domain form of the direct value of a.)."""
    n = len(p)
    npad = 2 * n
    pm, lm = np.zeros(npad), np.zeros(npad)
    pm[:n], lm[:n] = p - p.mean(), l - l.mean()
    mask = band_mask(npad)
    X = np.fft.rfft(pm) * np.conj(np.fft.rfft(lm))
    ratio = np.abs(X[mask]).sum() / np.abs(X).sum()
    return np.fft.irfft(X * mask, npad) / (n - 1) / ratio / (np.std(p, ddof=1) * np.std(l, ddof=1))


def readout_bpm(raw):
    """HR of the raw label window: highest 512-point periodogram bin within 0.6-3.3 Hz after mean removal (bpm)."""
    x = np.asarray(raw, dtype=float) - np.mean(raw)
    pw = np.abs(np.fft.rfft(x, 512)) ** 2
    f = np.fft.rfftfreq(512, 1 / FS)
    m = (f >= .6) & (f <= 3.3)
    return float(f[m][np.argmax(pw[m])] * 60)


SETTINGS = {'PhysNet differenced': (128, lambda x: diff_normalize_label(x)),
            'PhysNet standardized': (128, std_cumsum),
            'FactorizePhys standardized': (160, lambda x: standardized_label(x))}


def real_labels():
    from src.st9e.native import contact
    pure = {r['recording_id']: r for r in csv.DictReader(open(V / 'data/index/pure_index_st1.csv'))}
    orig = [r for r in csv.DictReader(open(V / 'outputs/st9g_original_manifest.csv')) if r['model'] == 'PhysNet']
    recs = sorted({r['recording_id'] for r in orig})
    assert len(recs) == 47 and all(r[:2] in {f'{s:02d}' for s in range(1, 9)} for r in recs)
    return {rid: np.asarray(contact(pure[rid], 'PURE', None), dtype=float) for rid in recs}


def test_bce(cuda):
    cwd = os.getcwd()
    os.chdir(V)
    try:
        raw = real_labels()
    finally:
        os.chdir(cwd)
    b_rows, c_rows, e_rows = [], [], []
    for setting, (n, f) in SETTINGS.items():
        for rid, x in raw.items():
            g = np.asarray(f(x[0:n]), dtype=np.float32)
            preds = np.stack([np.asarray(f(x[k:k + n]), dtype=np.float32) for k in range(31)])
            labels = np.repeat(g[None], 31, axis=0)
            val, lag = impl(preds, labels, lag=True)
            f0 = readout_bpm(x[0:n])
            for k in range(31):
                curve = mcc_curve(preds[k].astype(float), g.astype(float))
                b_rows.append(dict(setting=setting, recording=rid, k=k, lag=int(lag[k]), abs_lag_error=abs(int(lag[k]) - k),
                                   passed=abs(int(lag[k]) - k) <= 1, MCC=float(val[k]), MCC_at_k=float(curve[-k % len(curve)]),
                                   label_window_HR_bpm=f0, period_frames=60 * FS / f0,
                                   lag_minus_k_in_periods=(int(lag[k]) - k) / (60 * FS / f0)))
            c_rows.append(dict(setting=setting, recording=rid, MCC_g_g=float(impl(g, g)[0]),
                               MCC_g_g_float64=float(impl(g.astype(float), g.astype(float), torch.float64)[0])))
            t = np.arange(n) / FS
            a = float(np.std(g, ddof=1))
            for name, pred in [('-g', -g), ('g + 0.3 Hz', g + a * np.sin(2 * np.pi * .3 * t)),
                               ('g + 5 Hz', g + a * np.sin(2 * np.pi * 5 * t)),
                               ('g + 0.3 Hz + 5 Hz', g + a * np.sin(2 * np.pi * .3 * t) + a * np.sin(2 * np.pi * 5 * t))]:
                e_rows.append(dict(setting=setting, recording=rid, prediction=name, MCC=float(impl(pred.astype(np.float32), g)[0])))
    write('b_shift.csv', b_rows)
    write('c_self.csv', c_rows)
    write('e_values.csv', e_rows)
    b_summary = {}
    for setting in SETTINGS:
        rs = [r for r in b_rows if r['setting'] == setting]
        b_summary[setting] = dict(cases=len(rs), passed=sum(r['passed'] for r in rs),
                                  first_failing_k=min([r['k'] for r in rs if not r['passed']], default=None))
    return dict(b=dict(recordings=len(raw), k=list(range(31)), per_setting=b_summary,
                       passed=all(r['passed'] for r in b_rows)),
                c={s: dict(min=min(r['MCC_g_g'] for r in c_rows if r['setting'] == s),
                           max=max(r['MCC_g_g'] for r in c_rows if r['setting'] == s)) for s in SETTINGS})


def test_d(cuda):
    rng = np.random.default_rng(7041)
    rows = []
    devices = ['cpu'] + (['cuda'] if cuda else [])
    for n in (128, 160):
        rnd_p, rnd_l = rng.standard_normal((4, n)), rng.standard_normal((4, n))
        const = np.full((4, n), 0.7)
        mixed = rnd_p.copy()
        mixed[1] = 0.7
        cases = {'random': (rnd_p, rnd_l), 'pred constant': (const, rnd_l), 'label constant': (rnd_p, const),
                 'both constant': (const, const), 'one constant clip in batch': (mixed, rnd_l)}
        for case, (p, l) in cases.items():
            for device in devices:
                for code in ('implementation', 'author'):
                    pt = torch.tensor(p, dtype=torch.float32, device=device, requires_grad=True)
                    lt = torch.tensor(l, dtype=torch.float32, device=device)
                    loss = (mcc_loss.MCCLoss().to(device)(pt, lt) if code == 'implementation'
                            else torch.mean(AuthorCorr(FS, LO, HI)(pt, lt)))
                    graph = loss.requires_grad  # the author code returns a constant 0 without a graph when no clip has std > 0
                    if graph:
                        loss.backward()
                        g = pt.grad.detach().cpu().numpy()
                    else:
                        g = np.zeros(p.shape)
                    rows.append(dict(length=n, case=case, device=device, code=code, loss_has_graph=graph, loss=float(loss.detach()),
                                     loss_finite=bool(np.isfinite(float(loss.detach()))),
                                     grad_finite=bool(np.isfinite(g).all()), grad_nan=int(np.isnan(g).sum()),
                                     grad_abs_max=float(np.nanmax(np.abs(g))) if np.isfinite(g).any() else ''))
    write('d_gradient.csv', rows)
    impl_rows = [r for r in rows if r['code'] == 'implementation']
    return dict(cases=len(impl_rows), passed=all(r['loss_finite'] and r['grad_finite'] for r in impl_rows),
                author_nonfinite_cases=[(r['length'], r['case'], r['device']) for r in rows
                                        if r['code'] == 'author' and not (r['loss_finite'] and r['grad_finite'])])


def main():
    cuda = torch.cuda.is_available()
    s = dict(torch=torch.__version__, cuda=cuda, band_Hz=[LO / 60, HI / 60], fs=FS)
    s['a'] = test_a(cuda)
    s.update(test_bce(cuda))
    s['d'] = test_d(cuda)
    s['training_allowed'] = bool(s['a']['passed'] and s['b']['passed'] and s['d']['passed'])
    (OUT / 'summary.json').write_text(json.dumps(s, indent=1, default=str))
    print(json.dumps(s, indent=1, default=str))


if __name__ == '__main__':
    main()
