"""OPS2 7.19 A: pool manifests (plans) of B1 for the candidate selected by tools/b1_719/b1_design.py, and the chunk lists.

Selected candidate: outputs/719/A/b1_selection.csv (selected = True); refuses to run if none is selected.
Plans (A9 format, train/gen_ubfc.py columns): cache/b1_719/plan/{PhysNet,RhythmFormer,FPm}_items.csv, one row per item
(4 per U-low chunk), copied from the A9 item 1 plan row of the same chunk and k, with these columns replaced:
  target_HR   = numpy.random.default_rng([20261160, v, c, k]).uniform(lo, hi); v = 0 PhysNet, 2 RhythmFormer and FPm
                (shared targets); c = chunk order in the A9 list; target_key '[20261160,v,c,k]'
  rounding_h  = int(first 8 hex digits of SHA-256('OPS2-719|UBFC|<model>|<chunk_id>|<k>'), 16); rounding_key
                '[20261160,j,h]', j = 1 PhysNet and RhythmFormer, 2 FPm (the generator of gen_ubfc.job_family_i)
  pool_index  = 0, 1, ... in (chunk order, k) order.
Chunk lists: cache/b1_719/plan/{PhysNet,RhythmFormer}_{U_low,U_full_m}_chunks.csv (chunk order, chunk id, recording,
subject, reference HR); U-full-m = sorted(numpy.random.default_rng([20261160, 9, v]).choice(N, n, replace=False)), n = U-low
count of the list. FactorizePhys uses the RhythmFormer lists. Checks: targets equal those of the design table's
calculation; no item excluded by the generator rule; A9 columns other than the replaced ones equal.
Writes outputs/719/A/b1_plan_sha256.csv and b1_keys.txt (refuses to overwrite).
"""
import csv
import hashlib
import sys
from pathlib import Path

import numpy as np

W = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(W / 'tools' / 'a10'))
import b1_design as D  # noqa: E402

E1PLAN = W / 'cache' / 'ubfc709' / 'plan'
PLAN = W / 'cache' / 'b1_719' / 'plan'
OUT = W / 'outputs' / '719' / 'A'
K = D.K
MODELS = {'PhysNet': ('PhysNet', 0, 1), 'RhythmFormer': ('RhythmFormer', 2, 1), 'FPm': ('RhythmFormer', 2, 2)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, rows, fields=None):
    fields = fields or list(rows[0])
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    sel = [r for r in D.read(OUT / 'b1_selection.csv') if r['selected'] == 'True']
    if len(sel) != 1:
        sys.exit('no selected candidate')
    name = sel[0]['candidate']
    cutoff, (lo, hi) = [c for c in D.CANDIDATES if f'cutoff {c[0]}, U({c[1][0]}, {c[1][1]})' == name][0]
    PLAN.mkdir(parents=True, exist_ok=True)
    outs = [OUT / 'b1_plan_sha256.csv', OUT / 'b1_keys.txt']
    for o in outs + [PLAN / f'{m}_items.csv' for m in MODELS]:
        if o.exists():
            sys.exit(f'{o} exists')
    shas, lists = [], {}
    for lname, v in (('PhysNet', 0), ('RhythmFormer', 2)):
        rows = D.chunks(lname)
        low = [c for c, r in enumerate(rows) if float(r['reference_HR']) < cutoff]
        fm = sorted(int(x) for x in np.random.default_rng([K, 9, v]).choice(len(rows), len(low), replace=False))
        lists[lname] = low
        for tag, cs in (('U_low', low), ('U_full_m', fm)):
            p = PLAN / f'{lname}_{tag}_chunks.csv'
            if p.exists():
                sys.exit(f'{p} exists')
            write(p, [dict(chunk_order=c, chunk_id=rows[c]['chunk_id'], recording_id=rows[c]['recording_id'],
                           subject_id=rows[c]['subject_id'], reference_HR=rows[c]['reference_HR']) for c in cs])
            shas.append(dict(file=p.name, rows=len(cs), sha256=sha(p),
                             content=f'{lname} list ({"FactorizePhys uses it" if lname == "RhythmFormer" else "PhysNet"}): '
                                     + ('U-low, reference HR < %d bpm' % cutoff if tag == 'U_low' else
                                        f'U-full-m, default_rng([{K}, 9, {v}]).choice({len(rows)}, {len(low)}, replace=False)')))
    for model, (lname, v, j) in MODELS.items():
        e1 = {(int(r['chunk_order']), int(r['augmentation_index'])): r for r in D.read(E1PLAN / f'{model}_items.csv')}
        rows = D.chunks(lname)
        out = []
        for c in lists[lname]:
            for k in range(4):
                r = dict(e1[(c, k)])
                h = int(hashlib.sha256(f"OPS2-719|UBFC|{model}|{r['chunk_id']}|{k}".encode()).hexdigest()[:8], 16)
                r.update(target_HR=repr(D.target(K, v, c, k, lo, hi)), target_key=f'[{K},{v},c,k]', rounding_h=h,
                         rounding_key=f'[{K},{j},h]', pool_index=len(out))
                assert r['chunk_id'] == rows[c]['chunk_id'] and r['tau_rec'] == rows[c]['tau_rec']
                out.append(r)
        p = PLAN / f'{model}_items.csv'
        write(p, out, list(next(iter(e1.values()))))
        # checks: generator exclusion rule, target range
        its = D.items_of(rows, lists[lname], {(int(r['chunk_order']), int(r['augmentation_index'])): float(r['target_HR'])
                                             for r in out})
        assert not any(i['excluded'] for i in its), model
        assert all(lo <= float(r['target_HR']) < hi for r in out)
        shas.append(dict(file=p.name, rows=len(out), sha256=sha(p), content=f'{model} pool plan, U-low items, U({lo}, {hi})'))
    write(OUT / 'b1_plan_sha256.csv', shas)
    txt = [f'B1 (7.19 A): selected candidate: {name} (rule of 7.19; outputs/719/A/b1_selection.csv).',
           '', f'Target HR of item k (k = 0, 1, 2, 3) of chunk c:',
           f'  numpy.random.default_rng([{K}, v, c, k]).uniform({lo}, {hi})',
           '  v = 0 for PhysNet; v = 2 for RhythmFormer and FactorizePhys, which share the targets; c = chunk order in the A9 item 1 list (from 0).',
           'Rounding generator of item k of chunk c (arms (i), (i-pc), (i-simple) share one generated clip per item):',
           f'  numpy.random.default_rng([{K}, j, h]); j = 1 for PhysNet and RhythmFormer, j = 2 for FactorizePhys;',
           "  h = int(first 8 hexadecimal digits of SHA-256 of the string 'OPS2-719|UBFC|<model>|<chunk>|<k>', 16), <model> PhysNet, RhythmFormer or FPm, <chunk> '<recording>|<chunk index>'.",
           f'U-low: chunks of the A9 item 1 list with reference HR < {cutoff} bpm.',
           f'U-full-m: sorted(numpy.random.default_rng([{K}, 9, v]).choice(N, n, replace=False)), N = chunks of the list, n = U-low chunks of the list.',
           '', 'SHA-256:'] + [f"{s['sha256']}  {s['file']}  ({s['rows']} rows; {s['content']})" for s in shas]
    (OUT / 'b1_keys.txt').write_text('\n'.join(txt) + '\n')
    print('\n'.join(txt))


if __name__ == '__main__':
    main()
