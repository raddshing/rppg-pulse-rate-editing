"""Brief 02 7.03 다 (part 2): harmonic phase audit of the cross-corpus audit recordings (Table VI; 6.97 나), no model.

Input: the per-frame face-mask weighted RGB and contact PPG of every complete 300-frame context, kept by
p703_da_extract.py (audit tool unchanged). Per recording: v = G / mean(G) - 1 over the recording's contexts (all contexts
are complete and contiguous from frame 0; checked), p = contact PPG; definitions in p703_da_common.py (Welch 10 s Hann,
50% overlap; S = E[V conj(P)]; f0 = PPG spectral maximum in 0.7-3.0 Hz; phi1, phi2, gamma2; Delta = wrap(phi2 - 2 phi1);
d_h = wrap(phi1 - phi2) / (2 pi f0); decision fixed before the data). Correlation sign: the audit's recording sign
(abs_max_sign: sign of the maximum |correlation| of the mean context lag curve, lags -30..+30; audit_cross_recordings.csv).
Checks: the context lag curves recomputed from the kept signals equal the stored audit curves; the recording signs
recomputed from them equal the audit table; the negative counts equal Table VI; the local copy of the audit filter equals
the archived editor_band; filtered and unfiltered decisions (the filter is common to v and p).
Writes outputs/posthoc703/da/{da_corpus_counts,da_by_decision,da_agreement,da_pure_positive,da_checks}.csv and a
server-only per-recording table in cache/posthoc703/da/.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(WORK / 'audit'))
from p703_da_common import G1, G2, NEG, POS, editor_band, harmonic_phase, wrap  # noqa: E402
import a1_common as A  # noqa: E402  (archived editor_band, LAGS, summarize)

TRACES = WORK / 'cache' / 'posthoc703' / 'da_traces'
AUD = WORK / 'outputs' / 'audit_cross'
OUT = WORK / 'outputs' / 'posthoc703' / 'da'
CACHE = WORK / 'cache' / 'posthoc703' / 'da'
CORPORA = [('PURE', 'PURE'), ('UBFC', 'UBFC-rPPG'), ('MMPD', 'MMPD-A'), ('MCD', 'MCD-rPPG dev')]
TABLE_VI_NEG = {'PURE': (49, 59), 'UBFC-rPPG': (40, 42), 'MMPD-A': (74, 220), 'MCD-rPPG dev': (16, 89)}  # 6.97 나 report
SRC = 'training/development data (audit recordings, Table VI)'
DEC = ['negative', 'positive', 'undecidable (intermediate Delta)', 'undecidable (coherence)']


def q(x):
    x = np.asarray(x, float)
    if not len(x):
        return dict(median=np.nan, q25=np.nan, q75=np.nan)
    return dict(median=float(np.median(x)), q25=float(np.percentile(x, 25)), q75=float(np.percentile(x, 75)))


def recording(corpus, rid, ref_curves):
    z = np.load(TRACES / corpus / (rid.replace('/', '__') + '.npz'))
    fps, n = float(z['fps']), int(z['n_frames'])
    starts = z['starts']
    contiguous = bool(len(starts) == n // 300 and (starts == 300 * np.arange(len(starts))).all())
    curves = z['curves']
    dmax, nan_eq = 0.0, True
    for k, c in zip(z['keys'], curves):
        r = ref_curves[str(k)]
        nan_eq &= bool((np.isnan(c) == np.isnan(r)).all())
        dd = np.abs(c - r)
        if np.isfinite(dd).any():
            dmax = max(dmax, float(np.nanmax(dd)))
    sign = int(A.summarize(np.nanmean(curves, axis=0))['sign'])
    G = z['rgb'][..., 1].reshape(-1).astype(np.float64)
    v = G / G.mean() - 1
    p = z['ppg'].reshape(-1).astype(np.float64)
    filt_diff = float(max(np.abs(editor_band(v) - A.editor_band(v)).max(), np.abs(editor_band(p) - A.editor_band(p)).max()))
    hf = harmonic_phase(v, p, fps, filtered=True)
    hu = harmonic_phase(v, p, fps, filtered=False)
    return dict(corpus_code=corpus, recording_id=rid, fps=fps, frames_used=len(v), contexts=len(starts), contiguous=contiguous,
                contexts_curve_max_abs_diff=dmax, contexts_curve_nan_equal=nan_eq, sign_recomputed=sign,
                filter_copy_max_abs_diff=filt_diff, **hf,
                unfiltered_decision=hu['decision'], unfiltered_Delta_deg=hu['Delta_deg'], unfiltered_d_h_frames=hu['d_h_frames'],
                unfiltered_gamma2_f0=hu['gamma2_f0'], unfiltered_gamma2_2f0=hu['gamma2_2f0'], unfiltered_f0_Hz=hu['f0_Hz'])


def summary_rows(g, label, extra):
    rows = []
    for dec in DEC + ['all']:
        x = g if dec == 'all' else g[g.decision.eq(dec)]
        dev = np.where(x.decision.eq('negative'), [np.degrees(wrap(np.radians(a) - np.pi)) for a in x.Delta_deg], x.Delta_deg) \
            if len(x) else np.array([])
        row = dict(extra, group=label, decision=dec, recordings=len(x), Delta_negative=int((x.Delta_deg < 0).sum()))
        for name, vals in (('abs_Delta_deg', x.Delta_deg.abs()), ('Delta_deg', x.Delta_deg), ('Delta_deviation_deg', dev),
                           ('d_h_frames', x.d_h_frames),
                           ('d_h_s', x.d_h_s), ('gamma2_f0', x.gamma2_f0), ('gamma2_2f0', x.gamma2_2f0), ('f0_bpm', x.f0_bpm)):
            for k, val in q(vals).items():
                row[f'{name}_{k}'] = val
        rows.append(dict(row, source=SRC))
    return rows


def main():
    tab = pd.read_csv(AUD / 'audit_cross_recordings.csv', dtype={'recording': str})
    serial = json.loads((AUD / 'serial_to_recording_server_only.json').read_text())
    recs, chk = [], []
    for code, label in CORPORA:
        zc = np.load(AUD / f'{code}_curves.npz')
        ref = dict(zip((str(k) for k in zc['keys']), zc['curves']))
        rid_of = serial[label]
        t = tab[tab.corpus.eq(label)]
        rows = []
        for pub in t.recording:
            r = recording(code, rid_of[pub], ref)
            r.update(corpus=label, recording_public=pub, abs_max_sign=int(t[t.recording.eq(pub)].abs_max_sign.iloc[0]))
            rows.append(r)
            print(label, len(rows), r['decision'], flush=True)
        g = pd.DataFrame(rows)
        recs.append(g)
        neg, tot = TABLE_VI_NEG[label]
        chk += [dict(corpus=label, check='context lag curves recomputed from the kept signals vs stored audit curves',
                     items=int(g.contexts.sum()), equal=bool((g.contexts_curve_max_abs_diff == 0).all() and g.contexts_curve_nan_equal.all()),
                     max_abs_difference=float(g.contexts_curve_max_abs_diff.max())),
                dict(corpus=label, check='recording sign recomputed vs audit table abs_max_sign', items=len(g),
                     equal=bool((g.sign_recomputed == g.abs_max_sign).all()), max_abs_difference=float((g.sign_recomputed != g.abs_max_sign).sum())),
                dict(corpus=label, check=f'negative sign count vs Table VI ({neg}/{tot})', items=len(g),
                     equal=bool(int((g.abs_max_sign == -1).sum()) == neg and len(g) == tot), max_abs_difference=float(abs(int((g.abs_max_sign == -1).sum()) - neg))),
                dict(corpus=label, check='all contexts complete and contiguous from frame 0', items=len(g), equal=bool(g.contiguous.all()),
                     max_abs_difference=float((~g.contiguous).sum())),
                dict(corpus=label, check='local filter copy vs archived editor_band (v and p)', items=len(g),
                     equal=bool((g.filter_copy_max_abs_diff == 0).all()), max_abs_difference=float(g.filter_copy_max_abs_diff.max())),
                dict(corpus=label, check='decision with the filter vs without (filter common to v and p)', items=len(g),
                     equal=bool((g.decision == g.unfiltered_decision).all()), max_abs_difference=float((g.decision != g.unfiltered_decision).sum()),
                     f0_different=int((g.f0_Hz != g.unfiltered_f0_Hz).sum()),
                     decisions_different_with_equal_f0=int(((g.decision != g.unfiltered_decision) & (g.f0_Hz == g.unfiltered_f0_Hz)).sum()),
                     max_abs_Delta_difference_deg_equal_f0=float(np.abs([np.degrees(wrap(np.radians(a - b))) for a, b, e in
                                                                          zip(g.Delta_deg, g.unfiltered_Delta_deg, g.f0_Hz == g.unfiltered_f0_Hz) if e]).max()))]
    R = pd.concat(recs, ignore_index=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    R.to_csv(CACHE / 'recordings_server_only.csv', index=False)
    counts, bydec, agree = [], [], []
    for label in R.corpus.unique():
        g = R[R.corpus.eq(label)]
        und = g.decision.str.startswith('undecidable')
        counts.append(dict(corpus=label, recordings=len(g), negative=int(g.decision.eq('negative').sum()),
                           positive=int(g.decision.eq('positive').sum()), undecidable=int(und.sum()),
                           undecidable_coherence=int(g.decision.eq('undecidable (coherence)').sum()),
                           undecidable_intermediate_Delta=int(g.decision.eq('undecidable (intermediate Delta)').sum()),
                           coherence_f0_below=int((g.gamma2_f0 < G1).sum()),
                           coherence_2f0_below_with_f0_ok=int(((g.gamma2_f0 >= G1) & (g.gamma2_2f0 < G2)).sum()),
                           correlation_sign_negative=int((g.abs_max_sign == -1).sum()), correlation_sign_positive=int((g.abs_max_sign == 1).sum()),
                           fps_min=float(g.fps.min()), fps_max=float(g.fps.max()), segment_s=10.0, source=SRC))
        bydec += summary_rows(g, 'all recordings', dict(corpus=label))
        for dec in DEC:
            for s in (-1, 1):
                agree.append(dict(corpus=label, harmonic_decision=dec, correlation_sign='negative' if s < 0 else 'positive',
                                  recordings=int((g.decision.eq(dec) & g.abs_max_sign.eq(s)).sum()), source=SRC))
    pp = R[R.corpus.eq('PURE') & R.abs_max_sign.eq(1)]
    pure_pos = summary_rows(pp, 'PURE, correlation sign positive', dict(corpus='PURE'))
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(counts).to_csv(OUT / 'da_corpus_counts.csv', index=False)
    pd.DataFrame(bydec).to_csv(OUT / 'da_by_decision.csv', index=False)
    pd.DataFrame(agree).to_csv(OUT / 'da_agreement.csv', index=False)
    pd.DataFrame(pure_pos).to_csv(OUT / 'da_pure_positive.csv', index=False)
    u = pd.read_csv(OUT / 'da_unit_test.csv')
    chk.append(dict(corpus='synthetic', check='unit test: decision equals the known polarity (all cases)', items=len(u),
                    equal=bool(u.decision_ok.all()), max_abs_difference=float((~u.decision_ok).sum())))
    inside = u[u.d_frames < u.fs / (2 * u.f0_true_Hz)]
    chk.append(dict(corpus='synthetic', check='unit test: |d_h - d| (frames) for d inside the unambiguous range fs/(2 f0)',
                    items=len(inside), equal=bool(inside.d_h_error_frames.abs().max() <= 1), max_abs_difference=float(inside.d_h_error_frames.abs().max())))
    pd.DataFrame(chk).assign(source=SRC).to_csv(OUT / 'da_checks.csv', index=False)
    print(pd.DataFrame(counts).to_string())
    print(pd.DataFrame(chk).to_string())
    (OUT / 'run.json').write_text(json.dumps(dict(recordings=len(R), thresholds=dict(gamma2_f0=G1, gamma2_2f0=G2, negative_deg=NEG,
                                                                                     positive_deg=POS)), indent=1))


if __name__ == '__main__':
    main()
