"""Diagnostic rule 1 (docs/STAGE5_DEV_DIAGNOSTICS.md; brief 02 6.89 다 3, 6.90 라): RhythmFormer trained with the negative
Pearson loss only, arms (ii) and (ii-pc), seeds 20261131-33 (cache/train/RhythmFormer_ii-pearson_s*, _ii-pc-pearson_s*).

Runs compared: Pearson-only (ii-pearson), (ii-pc-pearson) and the standard RhythmFormer (ii) (archived October runs) and
(ii-pc) (OPS2 runs), seeds 20261131-33. Readouts (read only): outputs/1b, loaded with b1/analyze.py load_mmpd / load_mcd
(MMPD-A: 760 validation windows, high = draft 'high'; MCD-rPPG development half: draft-rule windows, high = agreement and
reference >= 100). The two rule-1 arms are added to b1/model_sets.LAB_ARMS in this process only (model_sets.py unchanged).
Support: per dataset and layer, the windows available in all twelve runs. Estimates: seed mean and 95% subject x seed crossed
bootstrap (b1/boot.py, 10,000 replicates). Judgement layer HAR on both sets; L0 is reported beside it.
1. Convergence: no judgement if MAE_high (ii-pc-pearson) > 1.5 x MAE_high of the standard (ii-pc) on either set.
2. Alignment response, C2 = (ii-pc) - (ii) high-HR MAE between the Pearson runs, paired by seed:
   'gained' if C2 upper bound < 0 on both sets; 'does not depend on the loss' if the MCD-dev point is >= +3 bpm;
   otherwise 'open'. The standard C2 is reported beside it.
3. Key-frame line, MCD-dev high-HR windows: line window = predicted HR within 147.65625-154.6875 bpm (tools/keyframe_line.py).
   'did not occur without the frequency loss' if the (ii-pc-pearson) fraction is <= 0.05 in all three seeds;
   'does not depend on the loss' if it is within or above the seed range of the standard (ii-pc) runs (>= their minimum)
   for at least two seeds; otherwise 'open'.
Usage: rule1_pearson.py <out_dir>
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
import model_sets  # noqa: E402

model_sets.LAB_ARMS = tuple(model_sets.LAB_ARMS) + ('ii-pearson', 'ii-pc-pearson')
SEEDS = [20261131, 20261132, 20261133]
STRATS = ['(ii)', '(ii-pc)', '(ii-pearson)', '(ii-pc-pearson)']
GRID = 3.515625
LO, HI = 42 * GRID, 44 * GRID  # 147.65625-154.6875 bpm (centre 43 bins = 151.171875)


def mae_high(W, r, p, s):
    h = s == 'high'
    return dict(MAE_high=boot.mae(W[:, h], r[h], p[h]))


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    meta = {k: v for k, v in analyze.runs().items()
            if v['backbone'] == 'RhythmFormer' and v['strategy'] in STRATS and int(v['seed']) in SEEDS}
    assert sorted((v['strategy'], int(v['seed'])) for v in meta.values()) == sorted((s, d) for s in STRATS for d in SEEDS), \
        sorted((v['strategy'], v['seed']) for v in meta.values())
    mm = analyze.load_mmpd(meta)
    mc, _ = analyze.load_mcd(meta)
    d = pd.concat([mm, mc], ignore_index=True)
    pd.DataFrame(analyze.counts(d)).to_csv(out / 'counts.csv', index=False)
    rows, contr = [], []
    for (ds, layer), g in d.groupby(['dataset', 'layer']):
        for st in STRATS:
            x = g[g.strategy.eq(st)]
            assert sorted(set(x.seed)) == SEEDS, (ds, layer, st)
            pt, ci = boot.point_ci(x, SEEDS, mae_high)
            per = {sd: float(np.mean(np.abs(x[(x.seed == sd) & x.set.eq('high')].predicted_HR
                                            - x[(x.seed == sd) & x.set.eq('high')].reference_HR))) for sd in SEEDS}
            rows.append(dict(dataset=ds, layer=layer, strategy=st, MAE_high=pt['MAE_high'], ci_lo=ci['MAE_high'][0],
                             ci_hi=ci['MAE_high'][1], **{f's{sd}': per[sd] for sd in SEEDS}))
        subjects = boot.numeric_subject_order(g.subject_id)
        smp = {st: boot.Sample(g[g.strategy.eq(st)], subjects, SEEDS) for st in STRATS}

        def f(W):
            v = {st: boot.per_seed(smp[st], mae_high, W) for st in STRATS}
            return {sd: dict(C2_pearson=v['(ii-pc-pearson)'][sd]['MAE_high'] - v['(ii-pearson)'][sd]['MAE_high'],
                             C2_standard=v['(ii-pc)'][sd]['MAE_high'] - v['(ii)'][sd]['MAE_high']) for sd in SEEDS}
        pt, ci = boot.summarize(f, subjects, SEEDS)
        for k in ('C2_pearson', 'C2_standard'):
            contr.append(dict(dataset=ds, layer=layer, quantity=k, point=pt[k], ci_lo=ci[k][0], ci_hi=ci[k][1]))
    t = pd.DataFrame(rows)
    t.to_csv(out / 'mae_high.csv', index=False)
    c = pd.DataFrame(contr)
    c.to_csv(out / 'contrasts.csv', index=False)
    # key-frame line on MCD-dev high-HR windows (HAR), per seed, common support of the twelve runs
    k = d[d.dataset.eq('MCD-dev') & d.layer.eq('HAR') & d.set.eq('high')].copy()
    k['line'] = (k.predicted_HR >= LO - 1e-6) & (k.predicted_HR <= HI + 1e-6)
    line = (k.groupby(['strategy', 'seed']).agg(windows=('line', 'size'), line_windows=('line', 'sum'), line_fraction=('line', 'mean'))
            .reset_index())
    line.to_csv(out / 'keyframe_line_mcd_high.csv', index=False)
    j = dict(judgement_layer='HAR', loss=dict(pearson='negative Pearson only (train/train_pc_pearson.py)',
                                              standard='RhythmFormer_Loss (author loss)'))
    conv = {}
    for ds in ('MMPD-A', 'MCD-dev'):
        q = t[t.dataset.eq(ds) & t.layer.eq('HAR')].set_index('strategy').MAE_high
        conv[ds] = dict(pearson=q['(ii-pc-pearson)'], standard=q['(ii-pc)'], ratio=q['(ii-pc-pearson)'] / q['(ii-pc)'])
    j['convergence'] = conv
    j['converged'] = bool(all(v['ratio'] <= 1.5 for v in conv.values()))
    if not j['converged']:
        j['judgement'] = 'no judgement (convergence condition not met)'
    else:
        q = c[c.layer.eq('HAR') & c.quantity.eq('C2_pearson')].set_index('dataset')
        gained = bool(q.loc['MMPD-A'].ci_hi < 0 and q.loc['MCD-dev'].ci_hi < 0)
        nodep = bool(q.loc['MCD-dev'].point >= 3.0)
        j['alignment'] = dict(condition_gained=gained, condition_does_not_depend=nodep,
                              judgement='gained' if gained and not nodep else 'does not depend' if nodep and not gained else 'open')
        p = line[line.strategy.eq('(ii-pc-pearson)')].set_index('seed').line_fraction
        s = line[line.strategy.eq('(ii-pc)')].set_index('seed').line_fraction
        none = bool((p <= 0.05).all())
        within = int((p >= s.min()).sum())
        j['keyframe_line'] = dict(pearson={int(a): float(b) for a, b in p.items()}, standard={int(a): float(b) for a, b in s.items()},
                                  standard_range=[float(s.min()), float(s.max())], pearson_seeds_within_or_above=within,
                                  condition_did_not_occur=none, condition_does_not_depend=within >= 2,
                                  judgement='did not occur' if none and within < 2 else 'does not depend' if within >= 2 and not none else 'open')
    (out / 'judgement.json').write_text(json.dumps(j, indent=1))
    print(json.dumps(j, indent=1))


if __name__ == '__main__':
    main()
