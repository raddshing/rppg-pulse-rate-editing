"""OPS2 A9 item 8b (7.13 1): VIPL-HR by scene (v1-v9): (ii-pc) - (i-simple), C2 = (ii-pc) - (ii) and (ii-lag) - (ii-pc), with
window and subject counts.

Keys: the checkpoints evaluated on VIPL-HR when A9 was committed, arms (ii-pc), (i-simple), (ii), (ii-lag) of the three
backbones, seeds 20261131-20261133, primary variants. Windows: the primary analysis of eval5/vipl_c.py (primary set, HAR,
verified, no 200-ms flag, reference = gt_HR mean, confirmatory support and valid windows) of one scene; per contrast the
windows valid for every key of both arms (stage3_analyze.pair_windows). Estimates and intervals: stage3_analyze.contrast_rows
unchanged (same-seed pairs, crossed subject x seed bootstrap B = 10,000, default_rng(20261110), every metric of the
contrast family); intervals, no tiers. The same contrasts on all scenes together are given as the reference rows.
Scenes v8 and v9 exist only in the secondary set (source2, no frame times; vipl_c population 'secondary', verified windows):
with the argument 'secondary' the same contrasts are computed there (8b_scene_contrasts_secondary.csv).
Writes outputs/a9_8/8b/8b_scene_contrasts.csv.
"""
import sys
import time

import a9_common as C
from a9_common import B, SA, SEEDS, pd

PAIRS = [('', '(ii-pc)', '(i-simple)', 'values only'), ('C2', '(ii-pc)', '(ii)', 'values only'), ('', '(ii-lag)', '(ii-pc)', 'values only')]


def main(pop='primary'):
    t0 = time.time()
    d, refs, choice, bad = C.vipl_frame()
    prim = C.primary(d, population=pop)
    prim = prim[prim.seed.isin(SEEDS) & prim.arm.isin({a for _, a, c, _ in PAIRS} | {c for _, a, c, _ in PAIRS})]
    out = []
    for scope, z in [('all scenes', prim)] + [(f'scene {s}', x) for s, x in sorted(prim.groupby('scene'))]:
        rows = SA.contrast_rows(z, PAIRS, B, scope)
        r = SA.restrict(z)
        counts = {}
        for bb in r.backbone.unique():
            for _, a, c, _ in PAIRS:
                ga, gc = r[r.backbone.eq(bb) & r.strategy.eq(a)], r[r.backbone.eq(bb) & r.strategy.eq(c)]
                ga, gc, nw = SA.pair_windows(ga, gc)
                one = ga[ga.seed.eq(SEEDS[0])]
                counts[(bb, f'{a}-{c}')] = dict(windows_high=int(one.set.eq('high').sum()), windows_low=int(one.set.eq('low').sum()),
                                                subjects_high=one[one.set.eq('high')].subject_id.nunique(),
                                                recordings=one.recording_id.nunique())
        for x in rows:
            x = dict(x)
            x.pop('grade', None)
            x['scope'] = scope
            out.append(dict(x, **counts[(x['backbone'], x['contrast'])], source=C.SRC_VIPL))
        print(scope, round(time.time() - t0), flush=True)
    C.write(pd.DataFrame(out).assign(population=pop), '8b_scene_contrasts.csv' if pop == 'primary' else f'8b_scene_contrasts_{pop}.csv', '8b')
    C.stamp('8b' if pop == 'primary' else f'8b_{pop}', seconds=time.time() - t0)
    print('done', round(time.time() - t0), flush=True)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'primary')
