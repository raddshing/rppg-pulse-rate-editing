"""Diagnostic rule 4 (docs/STAGE5_DEV_DIAGNOSTICS.md; brief 02 6.89 다 2, 6.90 다): judgement numbers of the re-encoding transplant.

Inputs (read only): outputs/rule4/preds_<condition>.csv.gz (tools/rule4_windows.py) for gop10, gop12, gop15, nokey and
distributed; windows and references of the existing UBFC/PURE evaluation (main study outputs/st15u_windows.csv: probe_dataset
U = UBFC-rPPG, PH = PURE 09-10, reference_HR, in_P, in_Qstar, high). Runs: 21 checkpoints (seeds 20261131-33).
Support: per layer, the windows available in all 21 runs in all five conditions (the same windows everywhere).
Populations: Q* (judgement) and P (reported); layers HAR (judgement) and L0 (reported). Values are seed means unless stated.
Bins: readout grid g = 3.515625 bpm. 'Within one bin of fs/GOP' = read on the grid value nearest 1800/GOP bpm or on one of its
two neighbours (GOP 10: 180 -> bins 50-52; 12: 150 -> bins 42-44, the MCD line 147.66-154.69; 15: 120 -> bins 33-35).
Fractions (UBFC and PURE together) exclude the windows whose reference HR lies in the same three bins.
One-bin hit: |predicted - reference| <= g. High-HR MAE and contrasts: UBFC only, high = 'high' of the window list.
Judgement (fixed before results, 6.90 다):
1. Usability: FactorizePhys (FPm) (0) hit rate in each of gop10, gop12, gop15 >= 0.5 x its hit rate on distributed.
2. Concentration (only interpreted if 1 holds): for every GOP, RhythmFormer (0) fraction at 1800/GOP >= 0.2 in that GOP's
   condition, and <= 0.05 in nokey and distributed, and PhysNet (0) and FPm (0) <= 0.05 in that GOP's condition.
3. Aligned arms (only interpreted if 1 holds): RhythmFormer (ii-pc)-(ii) and (i-pc)-(i) high-HR MAE, paired by seed,
   95% subject x seed crossed bootstrap 10,000 (b1/boot.py): each lower bound > 0 in >= 2 of the 3 key-frame conditions and
   each interval including 0 in both nokey and distributed.
Also: the fractions at 120 and 180 bpm beside the MMPD-A fractions read in bins 33-35 and 50-52 (same 21 checkpoints, b1 dev
readouts, 760 validation windows, same exclusion of reference in the bins).
Usage: rule4_analysis.py   Output: outputs/rule4/analysis/
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK / 'b1'))
import analyze  # noqa: E402
import boot  # noqa: E402

G = 3.515625
CONDS = ['gop10', 'gop12', 'gop15', 'nokey', 'distributed']
GOPS = {'gop10': 10, 'gop12': 12, 'gop15': 15}
SEEDS = [20261131, 20261132, 20261133]
OUT = WORK / 'outputs' / 'rule4' / 'analysis'
ARM = {('RhythmFormer', s): 'RhythmFormer ' + s for s in ('(0)', '(i)', '(ii)', '(i-pc)', '(ii-pc)')}
ARM.update({('PhysNet', '(0)'): 'PhysNet (0)', ('FactorizePhys (matched)', '(0)'): 'FPm (0)'})


def bins_of(target):
    k = int(round(target / G))
    return k - 1, k + 1


def in_bins(hr, lo, hi):
    k = np.round(np.asarray(hr, float) / G)
    return (k >= lo) & (k <= hi)


def mae_high(W, r, p, s):
    h = s == 'high'
    return dict(MAE_high=boot.mae(W[:, h], r[h], p[h]))


def load():
    meta = analyze.runs()
    w = pd.read_csv(Path(json.loads((WORK / 'config' / 'paths.json').read_text())['october_project']) / 'outputs' / 'st15u_windows.csv', dtype={'subject_id': str})  # 7.08: machine path, see config/paths.example.json
    w = w[['probe_dataset', 'recording_id', 'subject_id', 'window_id', 'layer', 'reference_HR', 'in_P', 'in_Qstar', 'high']]
    parts = []
    for c in CONDS:
        p = pd.read_csv(WORK / 'outputs' / 'rule4' / f'preds_{c}.csv.gz')
        parts.append(p)
    d = pd.concat(parts, ignore_index=True)
    d['arm'] = d.run_id.map(lambda r: ARM[(meta[r]['backbone'], meta[r]['strategy'])])
    d['seed'] = d.run_id.map(lambda r: int(meta[r]['seed']))
    n = d.run_id.nunique() * len(CONDS)
    ok = d[d.available.astype(bool)].groupby(['window_id', 'layer']).size()
    keep = set(ok[ok == n].index)
    counts = dict(runs=d.run_id.nunique(), planned=d.groupby('layer').window_id.nunique().to_dict())
    d = d[[k in keep for k in zip(d.window_id, d.layer)]].merge(w, on=['window_id', 'layer'], how='left')
    assert d.reference_HR.notna().all()
    d['set'] = np.where(d.high.astype(bool), 'high', 'low')
    return d, counts, meta


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d, counts, meta = load()
    rows_n = []
    for (layer, pop), g in [((l, p), d[d.layer.eq(l) & d[{'Qstar': 'in_Qstar', 'P': 'in_P'}[p]].astype(bool)]) for l in ('HAR', 'L0') for p in ('Qstar', 'P')]:
        one = g[(g.run_id == g.run_id.iloc[0]) & g.condition.eq('distributed')]
        rows_n.append(dict(layer=layer, population=pop, windows=len(one), UBFC=int(one.probe_dataset.eq('U').sum()),
                           PURE=int(one.probe_dataset.eq('PH').sum()), UBFC_high=int((one.probe_dataset.eq('U') & one.high).sum()),
                           subjects_UBFC=one[one.probe_dataset.eq('U')].subject_id.nunique()))
    pd.DataFrame(rows_n).to_csv(OUT / 'counts.csv', index=False)
    frac, hit, mae = [], [], []
    for (layer, pop) in [(l, p) for l in ('HAR', 'L0') for p in ('Qstar', 'P')]:
        g = d[d.layer.eq(layer) & d[{'Qstar': 'in_Qstar', 'P': 'in_P'}[pop]].astype(bool)]
        for (c, arm, seed), x in g.groupby(['condition', 'arm', 'seed']):
            base = dict(layer=layer, population=pop, condition=c, arm=arm, seed=seed)
            hit.append(dict(base, windows=len(x), hit=float((np.abs(x.predicted_HR - x.reference_HR) <= G + 1e-6).mean())))
            u = x[x.probe_dataset.eq('U') & x.high.astype(bool)]
            mae.append(dict(base, windows=len(u), MAE_high=float(np.abs(u.predicted_HR - u.reference_HR).mean())))
            for gop in (10, 12, 15):
                lo, hi = bins_of(1800 / gop)
                y = x[~in_bins(x.reference_HR, lo, hi)]
                frac.append(dict(base, target_bpm=1800 / gop, bins=f'{lo}-{hi}', windows=len(y),
                                 read_in_bins=int(in_bins(y.predicted_HR, lo, hi).sum()),
                                 fraction=float(in_bins(y.predicted_HR, lo, hi).mean())))
    frac, hit, mae = pd.DataFrame(frac), pd.DataFrame(hit), pd.DataFrame(mae)
    for t, name in ((frac, 'fractions'), (hit, 'hit'), (mae, 'mae_high')):
        t.to_csv(OUT / f'{name}_per_seed.csv', index=False)
    fm = frac.groupby(['layer', 'population', 'condition', 'arm', 'target_bpm', 'bins']).agg(windows=('windows', 'first'), fraction=('fraction', 'mean'),
                                                                                         fraction_min=('fraction', 'min'), fraction_max=('fraction', 'max')).reset_index()
    hm = hit.groupby(['layer', 'population', 'condition', 'arm']).agg(windows=('windows', 'first'), hit=('hit', 'mean')).reset_index()
    mm = mae.groupby(['layer', 'population', 'condition', 'arm']).agg(windows=('windows', 'first'), MAE_high=('MAE_high', 'mean')).reset_index()
    fm.to_csv(OUT / 'fractions_seed_mean.csv', index=False)
    hm.to_csv(OUT / 'hit_seed_mean.csv', index=False)
    mm.to_csv(OUT / 'mae_high_seed_mean.csv', index=False)
    # contrasts (UBFC, high-HR MAE)
    con = []
    for (layer, pop) in [(l, p) for l in ('HAR', 'L0') for p in ('Qstar', 'P')]:
        g = d[d.layer.eq(layer) & d[{'Qstar': 'in_Qstar', 'P': 'in_P'}[pop]].astype(bool) & d.probe_dataset.eq('U')]
        for c in CONDS:
            x = g[g.condition.eq(c)]
            for a, b in (('RhythmFormer (ii-pc)', 'RhythmFormer (ii)'), ('RhythmFormer (i-pc)', 'RhythmFormer (i)')):
                pt, ci = boot.contrast_ci(x[x.arm.eq(a)], x[x.arm.eq(b)], SEEDS, mae_high, boot.B)
                con.append(dict(layer=layer, population=pop, condition=c, contrast=f'{a} - {b}', windows_high=int((x[x.run_id.eq(x[x.arm.eq(a)].run_id.iloc[0])].set == 'high').sum()),
                                point=pt['MAE_high'], ci_lo=ci['MAE_high'][0], ci_hi=ci['MAE_high'][1]))
    con = pd.DataFrame(con)
    con.to_csv(OUT / 'contrasts.csv', index=False)
    # MMPD-A background bins beside
    mmeta = {k: v for k, v in meta.items() if (v['backbone'], v['strategy']) in ARM and int(v['seed']) in SEEDS}
    mm_a = analyze.load_mmpd(mmeta)
    bg = []
    for (layer, arm, seed), x in mm_a.assign(arm=mm_a.run_id.map(lambda r: ARM[(meta[r]['backbone'], meta[r]['strategy'])])).groupby(['layer', 'arm', 'seed']):
        for target, (lo, hi) in ((120.0, (33, 35)), (180.0, (50, 52))):
            y = x[~in_bins(x.reference_HR, lo, hi)]
            bg.append(dict(layer=layer, arm=arm, seed=seed, target_bpm=target, bins=f'{lo}-{hi}', windows=len(y), fraction=float(in_bins(y.predicted_HR, lo, hi).mean())))
    bg = pd.DataFrame(bg)
    bg.to_csv(OUT / 'mmpd_a_background_bins_per_seed.csv', index=False)
    bg.groupby(['layer', 'arm', 'target_bpm', 'bins']).agg(windows=('windows', 'first'), fraction=('fraction', 'mean')).reset_index().to_csv(OUT / 'mmpd_a_background_bins_seed_mean.csv', index=False)
    # judgement (Q*, HAR)
    J = dict(support=counts, judgement_population='Qstar', judgement_layer='HAR')
    h = hm[hm.layer.eq('HAR') & hm.population.eq('Qstar') & hm.arm.eq('FPm (0)')].set_index('condition').hit
    J['usability'] = {c: dict(hit=float(h[c]), ratio_to_distributed=float(h[c] / h['distributed'])) for c in CONDS}
    usable = bool(all(h[c] >= 0.5 * h['distributed'] for c in GOPS))
    J['usable'] = usable
    f = fm[fm.layer.eq('HAR') & fm.population.eq('Qstar')].set_index(['condition', 'arm', 'target_bpm']).fraction
    conc = {}
    for c, gop in GOPS.items():
        t = 1800 / gop
        v = dict(RF0_in_condition=float(f[(c, 'RhythmFormer (0)', t)]), RF0_nokey=float(f[('nokey', 'RhythmFormer (0)', t)]),
                 RF0_distributed=float(f[('distributed', 'RhythmFormer (0)', t)]), PhysNet0_in_condition=float(f[(c, 'PhysNet (0)', t)]),
                 FPm0_in_condition=float(f[(c, 'FPm (0)', t)]))
        v['holds'] = bool(v['RF0_in_condition'] >= 0.2 and v['RF0_nokey'] <= 0.05 and v['RF0_distributed'] <= 0.05
                          and v['PhysNet0_in_condition'] <= 0.05 and v['FPm0_in_condition'] <= 0.05)
        conc[c] = v
    J['concentration'] = conc
    q = con[con.layer.eq('HAR') & con.population.eq('Qstar')]
    al = {}
    for ctr, x in q.groupby('contrast'):
        x = x.set_index('condition')
        al[ctr] = dict(lower_above_0_in_keyframe=int(sum(x.loc[c, 'ci_lo'] > 0 for c in GOPS)),
                       includes_0_nokey=bool(x.loc['nokey', 'ci_lo'] <= 0 <= x.loc['nokey', 'ci_hi']),
                       includes_0_distributed=bool(x.loc['distributed', 'ci_lo'] <= 0 <= x.loc['distributed', 'ci_hi']))
    aligned = bool(all(v['lower_above_0_in_keyframe'] >= 2 and v['includes_0_nokey'] and v['includes_0_distributed'] for v in al.values()))
    J['aligned'] = al
    if not usable:
        J['statement'] = 'not interpreted (usability condition not met)'
    else:
        J['concentration_induced_by_periodic_key_frames'] = bool(all(v['holds'] for v in conc.values()))
        J['aligned_arm_worsening_consistent_with_compression_interaction'] = aligned
    (OUT / 'judgement.json').write_text(json.dumps(J, indent=1))
    print(json.dumps(J, indent=1))


if __name__ == '__main__':
    main()
