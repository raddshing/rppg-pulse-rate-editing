"""OPS2 7.07 (P second pools), 11-24: verdict on the batch preparation of a P batch (tools/batch_prep_704.py P_<backbone>).

Replaces the first criterion of run_707_P*.sh ("every pool item's label differs from the first pool"), which did not hold
for the (ii) stream by the generator's own label rule: the (ii) label is the contact PPG resampled with the rational
up/down = Fraction(1/scale).limit_denominator(64), scale = target_HR / reference_HR (src/st9e/native.parameters, through
src/st9e/generator.raw_target_label), so two targets of the same chunk with the same rational give the same label (first
PhysNet run of this check, 11-23 23:32: 7 of 2,972 second-pool items equal to their first-pool item, target gaps
0.014-0.582 bpm; the confirmatory first pool itself has 15 such groups).

Conditions of the instruction (7.07 / A6 item 3: settings equal but the pool, originals' labels equal); exit 1 if broken,
which stops the backbone's P pool arms:
  1. every configuration difference is allowed;
  2. every original's label equals the reference.
Checks added by OPS2 (PI 7.07i item 2, 11-24: when one fails, an alert is written to STATUS_ALERT.md and the work goes on,
exit 0):
  3. (ii) stream arms: every pool item's target_HR differs from the first-pool target_HR of the same item, with the same
     reference HR and context start;
  4. (i) stream arms: every pool item's label differs from the first pool;
     (ii) stream arms: a pool item's label equals its first-pool label exactly when the two rationals are identical, and
     within each pool (second, first) the groups of equal labels are exactly the groups of (chunk, rational).
(The loader's label of every pool item equals its manifest SHA: tools/pool_tensor_check_704.py, before this step.)
Writes outputs/704/batch_prep/P_<backbone>_label_rule_check.csv (aggregates; columns registered_pass and ops2_check_pass;
the first criterion's counts too). Usage (main environment): p2_label_check.py P_PhysNet | P_FPm | P_RhythmFormer [PART]
(PART: the output suffix of a partial batch preparation, batch_prep_704.py BATCH ARMS PART; 11-24, RhythmFormer (i) and
(ii) families prepared at different times.)
"""
import collections
import csv
import json
import os
import sys
import time
from fractions import Fraction

W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V = W + '/vendor/st8_6ebbc9cd'
OUT = W + '/outputs/704/batch_prep'
SRV = W + '/cache/train704/batch_prep_server_only'
FIRST_II = {'PhysNet': V + '/outputs/st9g_pool_PhysNet_ii-h_manifest.csv', 'FPm': V + '/outputs/st30t_FPm_A_PURE_ii_pool.csv',
            'RhythmFormer': V + '/outputs/st12c_a_pool_manifest.csv'}
II_ARMS = ('(ii)', '(ii-pc)')


def rows(p):
    with open(p, newline='') as f:
        return list(csv.DictReader(f))


def rational(target, reference):  # src/st9e/native.parameters(target / reference)
    r = Fraction(1 / float(float(target) / float(reference))).limit_denominator(64)
    return r.numerator, r.denominator


def by_item(rs):
    d = {(r['chunk_id'], r['augmentation_index']): r for r in rs}
    assert len(d) == len(rs), 'chunk_id, augmentation_index not unique'
    return d


def main():
    batch = sys.argv[1]
    part = sys.argv[2] if len(sys.argv) > 2 else ''
    bname = f'{batch}_{part}' if part else batch
    b = batch.split('_', 1)[1]
    diff = f'{OUT}/{bname}_config_diff.csv'
    bad = [r for r in rows(diff) if r['allowed'] != 'True'] if os.path.exists(diff) else []  # no file: no difference
    first_ii = by_item(rows(FIRST_II[b]))
    out, ok, own_ok = [], not bad, True
    for s in rows(f'{OUT}/{bname}_label_sha_summary.csv'):
        arm = s['arm']
        lab = rows(f'{SRV}/{batch}_{arm.strip("()")}_labels.csv')
        orig = [r for r in lab if r['kind'] == 'original']
        pool = by_item([r for r in lab if r['kind'] == 'pool'])
        rec = dict(batch=batch, arm=arm, originals=len(orig), originals_equal_reference=sum(r['label_used_sha256_new'] == r['label_used_sha256_reference'] for r in orig),
                   pool_items=len(pool), pool_equal_first=sum(r['label_used_sha256_new'] == r['label_used_sha256_reference'] for r in pool.values()),
                   old_criterion_pass=int(s['labels_differ_reference']) == int(s['pool_items']) and int(s['labels_equal_reference']) == int(s['originals']))
        rec['pool_equal_first_explained'] = rec['pool_equal_first_unexplained'] = rec['rational_equal_label_differs'] = ''
        rec['second_pool_equal_label_groups'] = rec['first_pool_equal_label_groups'] = rec['groups_match_rational'] = ''
        reg_ok = rec['originals_equal_reference'] == rec['originals'] > 0
        arm_ok = True
        rec['target_differs_first'] = rec['items_matched_first'] = ''
        if arm == '(0)':
            arm_ok = rec['pool_items'] == 0
        else:
            arm_ok = len(pool) > 0
            if arm in II_ARMS:
                p2 = by_item(rows(f'{W}/cache/pools_p2/{b}_p2_{arm.strip("()")}_manifest.csv'))
                rec['items_matched_first'] = sum(k in first_ii and p2[k]['reference_HR'] == first_ii[k]['reference_HR']
                                                 and p2[k]['context_start'] == first_ii[k]['context_start'] for k in pool if k in p2)
                arm_ok = arm_ok and set(p2) == set(pool) and rec['items_matched_first'] == len(pool)
                pool = {k: v for k, v in pool.items() if k in p2 and k in first_ii}
                rec['target_differs_first'] = sum(p2[k]['target_HR'] != first_ii[k]['target_HR'] for k in pool)
                arm_ok = arm_ok and rec['target_differs_first'] == len(pool)
                r2 = {k: rational(p2[k]['target_HR'], p2[k]['reference_HR']) for k in pool}
                r1 = {k: rational(first_ii[k]['target_HR'], first_ii[k]['reference_HR']) for k in pool}
                eq = {k for k, r in pool.items() if r['label_used_sha256_new'] == r['label_used_sha256_reference']}
                same = {k for k in pool if r1[k] == r2[k]}
                rec['pool_equal_first_explained'] = len(eq & same)
                rec['pool_equal_first_unexplained'] = len(eq - same)
                rec['rational_equal_label_differs'] = len(same - eq)
                match = True
                for col, rat, name in (('label_used_sha256_new', r2, 'second'), ('label_used_sha256_reference', r1, 'first')):
                    gl, gr = collections.defaultdict(set), collections.defaultdict(set)
                    for k, r in pool.items():
                        gl[r[col]].add(k)
                        gr[(k[0], rat[k])].add(k)
                    rec[f'{name}_pool_equal_label_groups'] = sum(len(v) > 1 for v in gl.values())
                    match = match and sorted(map(sorted, gl.values())) == sorted(map(sorted, gr.values()))
                rec['groups_match_rational'] = match
                arm_ok = arm_ok and not (eq - same) and not (same - eq) and match
            else:
                arm_ok = arm_ok and rec['pool_equal_first'] == 0
        rec['registered_pass'] = reg_ok and not bad  # originals' labels equal and no disallowed configuration difference
        rec['ops2_check_pass'] = arm_ok
        ok = ok and reg_ok
        own_ok = own_ok and arm_ok
        out.append(rec)
        print(json.dumps(rec), flush=True)
    with open(f'{OUT}/{bname}_label_rule_check.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    msg = (f'{batch}: not allowed differences {len(bad)}, arms with originals\' labels unequal {sum(r["originals_equal_reference"] != r["originals"] for r in out)}'
           f' (instruction conditions); arms failing OPS2 label checks {sum(not r["ops2_check_pass"] for r in out)} (alert only)')
    print(msg)
    if ok and not own_ok:
        with open(os.environ.get('OPS2_ALERT', W + '/STATUS_ALERT.md'), 'a') as f:
            f.write(f'{time.strftime("%Y-%m-%dT%H:%M:%S%z")} P batch prep {batch}: an OPS2-added label check failed '
                    f'(outputs/704/batch_prep/{batch}_label_rule_check.csv); instruction conditions hold, training goes on (7.07i item 2)\n')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
