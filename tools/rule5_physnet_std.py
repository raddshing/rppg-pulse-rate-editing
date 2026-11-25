"""Diagnostic rule 5 (docs/STAGE5_DEV_DIAGNOSTICS.md; brief 02 6.89 다 1): PhysNet trained with standardized labels.

Runs: PhysNet standard (ii), (ii-pc), (ii-lag) and PhysNet-Std (ii-std), (ii-pc-std), (ii-lag-std), seeds 20261131-33.
Readouts (read only): outputs/1b (the PhysNet-Std inference of run_std_infer.sh and the standard 1b readouts of the same
b1 inference/readout code), loaded with b1/analyze.py load_mmpd / load_mcd (MMPD-A: 760 validation windows, high = draft
'high'; MCD-rPPG development half: draft-rule windows, high = gap-free, agreement and reference >= 100).
Support: per dataset and layer, the windows available in every run loaded here (all runs of the strategies above that exist).
Estimates: seed mean and 95% subject x seed crossed bootstrap (b1/boot.py, 10,000 replicates), high-HR MAE.
Judgement layer: HAR (wide band) on both sets; L0 is reported beside it.
1. Convergence: no judgement if MAE_high (ii-pc-std) > 1.5 x MAE_high (ii-pc) on either development set.
2. Only if all three PhysNet-Std arms exist: lag_std = (ii-lag-std) - (ii-std); lag_std_ref = (ii-lag) - (ii);
   D = lag_std_ref - lag_std, paired by seed (same seed in all four runs), same windows.
   'depended' if lag_std point <= +1.0 on MMPD-A, D lower bound > 0 on MMPD-A and D point > 0 on MCD-dev;
   'did not depend' if lag_std lower bound > 0 and point >= +3 on MMPD-A; otherwise 'open'.
Usage: rule5_physnet_std.py <out_dir>
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

SEEDS = [20261131, 20261132, 20261133]
STRATS = ['(ii)', '(ii-pc)', '(ii-lag)', '(ii-std)', '(ii-pc-std)', '(ii-lag-std)']


def mae_high(W, r, p, s):
    h = s == 'high'
    return dict(MAE_high=boot.mae(W[:, h], r[h], p[h]))


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    meta = {k: v for k, v in analyze.runs().items()
            if v['backbone'] == 'PhysNet' and v['strategy'] in STRATS and int(v['seed']) in SEEDS}
    have = sorted({v['strategy'] for v in meta.values()})
    mm = analyze.load_mmpd(meta)
    mc, _ = analyze.load_mcd(meta)
    d = pd.concat([mm, mc], ignore_index=True)
    counts = pd.DataFrame(analyze.counts(d))
    counts.to_csv(out / 'counts.csv', index=False)
    rows, contr = [], []
    for (ds, layer), g in d.groupby(['dataset', 'layer']):
        for st in have:
            x = g[g.strategy.eq(st)]
            assert sorted(set(x.seed)) == SEEDS, (ds, layer, st)
            pt, ci = boot.point_ci(x, SEEDS, mae_high)
            per = {sd: float(np.mean(np.abs(x[(x.seed == sd) & x.set.eq('high')].predicted_HR
                                            - x[(x.seed == sd) & x.set.eq('high')].reference_HR))) for sd in SEEDS}
            rows.append(dict(dataset=ds, layer=layer, strategy=st, MAE_high=pt['MAE_high'], ci_lo=ci['MAE_high'][0],
                             ci_hi=ci['MAE_high'][1], **{f's{sd}': per[sd] for sd in SEEDS}))
        if '(ii-lag-std)' in have:
            subjects = boot.numeric_subject_order(g.subject_id)
            smp = {st: boot.Sample(g[g.strategy.eq(st)], subjects, SEEDS) for st in STRATS}

            def f(W):
                v = {st: boot.per_seed(smp[st], mae_high, W) for st in STRATS}
                res = {}
                for sd in SEEDS:
                    m = {st: v[st][sd]['MAE_high'] for st in STRATS}
                    ls, lr = m['(ii-lag-std)'] - m['(ii-std)'], m['(ii-lag)'] - m['(ii)']
                    res[sd] = dict(lag_std=ls, lag_std_ref=lr, D=lr - ls)
                return res
            pt, ci = boot.summarize(f, subjects, SEEDS)
            for k in ('lag_std', 'lag_std_ref', 'D'):
                contr.append(dict(dataset=ds, layer=layer, quantity=k, point=pt[k], ci_lo=ci[k][0], ci_hi=ci[k][1]))
    t = pd.DataFrame(rows)
    t.to_csv(out / 'mae_high.csv', index=False)
    j = dict(strategies_present=have, judgement_layer='HAR')
    conv = {}
    for ds in ('MMPD-A', 'MCD-dev'):
        q = t[t.dataset.eq(ds) & t.layer.eq('HAR')].set_index('strategy').MAE_high
        conv[ds] = dict(std=q['(ii-pc-std)'], standard=q['(ii-pc)'], ratio=q['(ii-pc-std)'] / q['(ii-pc)'])
    j['convergence'] = conv
    j['converged'] = bool(all(v['ratio'] <= 1.5 for v in conv.values()))
    if not j['converged']:
        j['judgement'] = 'no judgement (convergence condition not met)'
    elif contr:
        c = pd.DataFrame(contr)
        c.to_csv(out / 'contrasts.csv', index=False)
        q = c[c.layer.eq('HAR')].set_index(['dataset', 'quantity'])
        ls, dm, dc = q.loc[('MMPD-A', 'lag_std')], q.loc[('MMPD-A', 'D')], q.loc[('MCD-dev', 'D')]
        a = bool(ls.point <= 1.0 and dm.ci_lo > 0 and dc.point > 0)
        b = bool(ls.ci_lo > 0 and ls.point >= 3.0)
        j.update(condition_depended=a, condition_did_not_depend=b,
                 judgement='depended' if a and not b else 'did not depend' if b and not a else 'open')
    else:
        j['judgement'] = 'pending: (ii-lag-std) not trained'
    (out / 'judgement.json').write_text(json.dumps(j, indent=1))
    print(json.dumps(j, indent=1))


if __name__ == '__main__':
    main()
