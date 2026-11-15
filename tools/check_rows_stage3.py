"""Row-count check of an eval5/stage3_analyze.py output folder against the manifests (brief 02 6.63 가 5, A4 item 4(c);
for rehearsal part 2 and test step (c)). eval5/stage3_analyze.py is not changed.

Manifest arms: every (backbone, arm) with the three main seeds among the checkpoint rows of the given manifest files
(confirmatory manifests and the exploratory manifest; backbone names mapped as eval5/stage3_analyze.NAME).
Expected rows (analysis 'primary', the population folder given):
  arms.csv          every manifest (backbone, arm)
  confirmatory.csv  C1-C4 (stage3_analyze.CONF) whose two arms are confirmatory-manifest arms of the backbone
  secondary.csv     every stage3_analyze.SECONDARY pair whose two arms are manifest arms of the backbone
  benchmark.csv     every manifest (backbone, arm), when the folder has a benchmark.csv (stage 1 with contact signals)
  MAE_verified      (6.70 나, A4 item 5) a row of metric MAE_verified for every expected arm, confirmatory and secondary row
Writes ROW_CHECK_CSV (backbone, table, item, found); prints the missing; exit 0 if none, 3 otherwise.
Usage: check_rows_stage3.py ANALYSIS_FOLDER ROW_CHECK_CSV --confirmatory M1.csv [M2.csv ...] --exploratory M3.csv [...]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK / 'eval5'))
import stage3_analyze as S  # noqa: E402


def arms(files):
    out = {}
    for f in files:
        m = pd.read_csv(f)
        m = m[m.kind.eq('checkpoint')]
        for (b, a), g in m.groupby(['backbone', 'arm']):
            if set(S.SEEDS) <= set(g.seed.astype(int)):
                out.setdefault(S.NAME.get(b, b), set()).add(a)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('folder'); ap.add_argument('out')
    ap.add_argument('--confirmatory', nargs='+', required=True); ap.add_argument('--exploratory', nargs='*', default=[])
    a = ap.parse_args()
    d = Path(a.folder)
    conf, expl = arms(a.confirmatory), arms(a.exploratory)
    allarms = {b: conf.get(b, set()) | expl.get(b, set()) for b in set(conf) | set(expl)}
    t = {n: pd.read_csv(d / f'{n}.csv') for n in ('arms', 'confirmatory', 'secondary', 'benchmark') if (d / f'{n}.csv').exists()}
    rows = []
    for b, sts in sorted(allarms.items()):
        ab = t['arms'][t['arms'].analysis.eq('primary') & t['arms'].backbone.eq(b)]
        got = set(ab.strategy)
        gotv = set(ab[ab.metric.eq('MAE_verified')].strategy)
        rows += [dict(backbone=b, table='arms', item=s, found=s in got) for s in sorted(sts)]
        rows += [dict(backbone=b, table='arms MAE_verified', item=s, found=s in gotv) for s in sorted(sts)]
        if 'benchmark' in t:
            gb = set(t['benchmark'][t['benchmark'].backbone.eq(b)].arm)
            rows += [dict(backbone=b, table='benchmark', item=s, found=s in gb) for s in sorted(sts)]
        cb = t['confirmatory'][t['confirmatory'].analysis.eq('primary') & t['confirmatory'].backbone.eq(b)]
        gc, gcv = set(cb.contrast), set(cb[cb.metric.eq('MAE_verified')].contrast)
        for name, x, y in S.CONF:
            if x in conf.get(b, set()) and y in conf.get(b, set()):
                rows.append(dict(backbone=b, table='confirmatory', item=f'{name} {x}-{y}', found=f'{x}-{y}' in gc))
                rows.append(dict(backbone=b, table='confirmatory MAE_verified', item=f'{name} {x}-{y}', found=f'{x}-{y}' in gcv))
        sb = t['secondary'][t['secondary'].analysis.eq('primary') & t['secondary'].backbone.eq(b)]
        gs, gsv = set(sb.contrast), set(sb[sb.metric.eq('MAE_verified')].contrast)
        for x, y, _ in S.SECONDARY:
            if x in sts and y in sts:
                rows.append(dict(backbone=b, table='secondary', item=f'{x}-{y}', found=f'{x}-{y}' in gs))
                rows.append(dict(backbone=b, table='secondary MAE_verified', item=f'{x}-{y}', found=f'{x}-{y}' in gsv))
    r = pd.DataFrame(rows)
    r.to_csv(a.out, index=False)
    print(r.groupby(['table', 'backbone']).found.agg(['size', 'sum']).to_string())
    miss = r[~r.found.astype(bool)]
    print('expected', len(r), 'found', int(r.found.sum()), 'missing', len(miss))
    if len(miss):
        print(miss.to_string(index=False))
        sys.exit(3)


if __name__ == '__main__':
    main()
