"""OPS2 7.15 2: posthoc descriptive table (not part of A9) of the f* phase error of the E1 UBFC-rPPG pool items of arms (ii)
and (ii-lag), per backbone, from manifest values only (no pixels): |wrap(2 pi f* (tau + T*/2 - d) / 30)| with f* = h*/60,
T* = 1800/h* frames, d = 1 (UBFC-rPPG), tau = tau_rec of the recording; (ii) +g*(t - tau) and (ii-lag) -g*(t - tau - T*/2)
have the same fundamental phase error against the aligned -g*(t - d). Summary: mean and median (all items, target 70-100 and
100-140 bpm) and r = h*/h0 (h0 = reference HR of the chunk): median and quartiles. Beside it, the same on the PURE first pool
of FactorizePhys (2,384 items, d = 2, tau of tools/a9_eval E7 = audit/a1_common.tau_table), which must equal the alpha = 1 row
of E7_phase_error.csv.
Item lists: the fixed UBFC-rPPG pool plans (cache/ubfc709/plan/<backbone>_items.csv, written before any item was generated);
each is compared with the (ii) and (ii-lag) pool manifests that exist (chunk, augmentation index, target, tau, d).
Writes outputs/715/ubfc_phase_error.csv and ubfc_phase_checks.csv (refuses to overwrite).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

W = Path(__file__).resolve().parents[2]
PLAN = W / 'cache' / 'ubfc709' / 'plan'
POOLS = W / 'cache' / 'ubfc709' / 'pools'
OUT = W / 'outputs' / '715'
BB = {'PhysNet': 'PhysNet', 'FPm': 'FactorizePhys', 'RhythmFormer': 'RhythmFormer'}


def phase_deg(h, tau, d):
    delay = tau + (1800.0 / h) / 2 - d
    return np.degrees(np.abs(np.angle(np.exp(1j * 2 * np.pi * (h / 60) * delay / 30))))


def summary(corpus, bb, pool, d, h, h0, tau):
    ph, r = phase_deg(h, tau, d), h / h0
    rows = []
    for name, m in (('all', np.ones(len(h), bool)), ('target 70-100 bpm', h < 100), ('target 100-140 bpm', h >= 100)):
        rows.append(dict(corpus=corpus, backbone=bb, pool=pool, d=d, targets=name, items=int(m.sum()),
                         target_min=float(h[m].min()), target_max=float(h[m].max()),
                         mean_abs_phase_error_deg=float(ph[m].mean()), median_abs_phase_error_deg=float(np.median(ph[m])),
                         r_q25=float(np.quantile(r[m], .25)), r_median=float(np.median(r[m])), r_q75=float(np.quantile(r[m], .75))))
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    outs = [OUT / 'ubfc_phase_error.csv', OUT / 'ubfc_phase_checks.csv']
    for o in outs:
        if o.exists():
            sys.exit(f'{o.name} exists')
    rows, checks = [], []
    key = ['chunk_id', 'augmentation_index']
    for b, bb in BB.items():
        p = pd.read_csv(PLAN / f'{b}_items.csv', low_memory=False)
        assert p.d_pc.eq(1).all(), b
        for arm, f in (('(ii)', f'{b}_ii_source_all_manifest.csv'), ('(ii-lag)', f'{b}_ii-lag_source_all_manifest.csv')):
            if not (POOLS / f).exists():
                checks.append(dict(backbone=bb, arm=arm, check='pool manifest vs plan', value='manifest not yet written', passed=''))
                continue
            m = pd.read_csv(POOLS / f, low_memory=False)
            j = p.merge(m, on=key, suffixes=('', '_m'), how='outer', indicator=True)
            same = (j._merge.eq('both').all() and len(m) == len(p)
                    and np.allclose(j.target_HR, j.target_HR_m, rtol=0, atol=1e-9)
                    and (j.tau_rec == j.tau_rec_m).all() and (j.reference_HR == j.reference_HR_m).all())
            checks.append(dict(backbone=bb, arm=arm, check='pool manifest vs plan: items, target, tau, reference HR equal',
                               value=f'{len(m)} of {len(p)}', passed=bool(same)))
        rows += summary('UBFC-rPPG', bb, 'E1 (ii) and (ii-lag), key 20261159', 1, p.target_HR.to_numpy(float),
                        p.reference_HR.to_numpy(float), p.tau_rec.to_numpy(float))
    sys.path.insert(0, str(W / 'audit'))
    import a1_common as AC
    taus = AC.tau_table()
    m = pd.read_csv(W / 'cache' / 'pools' / 'FPm_polarity_explicit_all_manifest.csv', low_memory=False)
    assert m.d.eq(2).all()
    tau = m.recording_id.map(lambda r: float(taus[r]['tau_rec'])).to_numpy()
    checks.append(dict(backbone='FactorizePhys', arm='(ii-pc) PURE first pool', check='tau of the E7 table vs manifest tau_rec',
                       value=f'{int((tau == m.tau_rec.to_numpy(float)).sum())} of {len(m)} equal',
                       passed=bool((tau == m.tau_rec.to_numpy(float)).all())))
    pure = summary('PURE', 'FactorizePhys', 'first pool (ii-pc) items, as in the E7 table', 2, m.target_HR.to_numpy(float),
                   m.reference_HR.to_numpy(float), tau)
    rows += pure
    e7 = pd.read_csv(W / 'outputs' / 'a9_eval' / 'E467' / 'run' / 'E7_phase_error.csv')
    e7 = e7[e7.alpha.eq(1.0)].iloc[0]
    dm = abs(pure[0]['mean_abs_phase_error_deg'] - e7.mean_abs_phase_error_deg)
    dd = abs(pure[0]['median_abs_phase_error_deg'] - e7.median_abs_phase_error_deg)
    checks.append(dict(backbone='FactorizePhys', arm='PURE first pool', check='equal to the alpha = 1 row of E7_phase_error.csv '
                       '(items, mean, median; largest absolute difference in degrees)',
                       value=f"items {pure[0]['items']} vs {int(e7['items'])}; {max(dm, dd):.1e}",
                       passed=bool(pure[0]['items'] == e7['items'] and max(dm, dd) < 1e-9)))
    pd.DataFrame(rows).to_csv(outs[0], index=False)
    pd.DataFrame(checks).to_csv(outs[1], index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print(pd.DataFrame(checks).to_string(index=False))


if __name__ == '__main__':
    main()
