"""Row-count check of a b1/analyze.py output folder (brief 02 6.63 가 5; also for rehearsal part 2 and test step (c)).

The expected combinations come from the readouts the analysis reads (OPS2_OUT1B, b1/analyze.py loaders unchanged, run
registry analyze.runs()): every (dataset, population, backbone, arm) with the three main seeds present = the manifest of
this analysis. For the requested backbones (display names):
  arm tables m123, m4, m6, m7, m9: every expected (dataset, population, backbone, arm) of a dataset/population that the table
    covers at all must have rows (m7: MCD-dev M7_both_states).
  m5 (constant predictors, no backbone): rows for every dataset.
  contrast tables (c25, cipc, cinv, clag, csimple, clab, crfl, cstd, closs, m8): every contrast of the family's pair list
    (analyze.C25 ...) whose two arms are both expected for that backbone and dataset/population must have rows, seed set '3'.
Writes <folder>/row_check.csv (family, dataset, population, backbone, kind, item, found) and prints the missing ones.
Exit 0 if nothing is missing, 3 otherwise. b1/analyze.py is not changed.
Usage: OPS2_OUT1B=<1b folder> check_rows.py OUTPUT_TAG FAMILIES BACKBONE_KEYS     e.g. guarded_v1 m123,...,csimple FPm,PhysNet,RhythmFormer
"""
import sys
from pathlib import Path

import pandas as pd

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK / 'b1'))
import analyze  # noqa: E402

PAIRS = {'c25': analyze.C25, 'cipc': analyze.CIPC, 'cinv': analyze.CINV, 'clag': analyze.CLAG, 'csimple': analyze.CSIMPLE,
         'clab': analyze.CLAB, 'crfl': analyze.CRFL, 'cstd': analyze.CSTD, 'closs': analyze.CLOSS, 'm8': analyze.M8}
ARMS = ('m123', 'm4', 'm6', 'm7', 'm9')


def main():
    tag, fams, keys = sys.argv[1], sys.argv[2].split(','), sys.argv[3].split(',')
    bbs = [analyze.BACKBONE[k] for k in keys]
    out = WORK / 'outputs' / tag
    meta = analyze.runs()
    mm = analyze.load_mmpd(meta)
    mc, m7 = analyze.load_mcd(meta)
    d = pd.concat([mm, mc, m7], ignore_index=True)
    d = d[d.backbone.isin(bbs)]
    have = {}
    for (ds, pop, bb, st), g in d.groupby(['dataset', 'population', 'backbone', 'strategy']):
        if set(analyze.MAIN) <= set(g.seed):
            have.setdefault((ds, pop, bb), set()).add(st)
    rows = []
    for fam in fams:
        f = out / f'{fam}.csv'
        t = pd.read_csv(f) if f.exists() else pd.DataFrame(columns=['dataset', 'population', 'backbone'])
        if fam in ARMS:
            covered = set(zip(t.dataset, t.population)) if len(t) else set()
            for (ds, pop, bb), sts in sorted(have.items()):
                if fam == 'm7' and pop != 'M7_both_states':
                    continue
                if fam != 'm7' and (ds, pop) not in covered:
                    continue
                got = set(t[(t.dataset == ds) & (t.population == pop) & (t.backbone == bb)].strategy)
                for st in sorted(sts):
                    rows.append(dict(family=fam, dataset=ds, population=pop, backbone=bb, kind='arm', item=st, found=st in got))
        elif fam in PAIRS:
            for (ds, pop, bb), sts in sorted(have.items()):
                got = set(t[(t.dataset == ds) & (t.population == pop) & (t.backbone == bb) & (t.seeds.astype(str) == '3')].contrast) if len(t) else set()
                for a, c in PAIRS[fam]:
                    if a in sts and c in sts:
                        rows.append(dict(family=fam, dataset=ds, population=pop, backbone=bb, kind='contrast', item=f'{a}-{c}',
                                         found=f'{a}-{c}' in got))
        elif fam == 'm5':  # constant predictors, no backbone: rows for every dataset of the analysis
            for ds in sorted({k[0] for k in have}):
                rows.append(dict(family=fam, dataset=ds, population='', backbone='(constant predictors)', kind='table', item='rows',
                                 found=bool(len(t)) and bool((t.dataset == ds).any())))
        else:
            rows.append(dict(family=fam, dataset='', population='', backbone='', kind='unknown family', item='', found=False))
    r = pd.DataFrame(rows)
    r.to_csv(out / 'row_check.csv', index=False)
    miss = r[~r.found.astype(bool)]
    by = r.groupby(['family', 'backbone']).found.agg(['size', 'sum'])
    print(by.to_string())
    print('expected', len(r), 'found', int(r.found.sum()), 'missing', len(miss))
    if len(miss):
        print(miss.to_string(index=False))
        sys.exit(3)


if __name__ == '__main__':
    main()
