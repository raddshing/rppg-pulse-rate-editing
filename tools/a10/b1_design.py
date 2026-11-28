"""OPS2 7.19 A: B1 design calculation (UBFC-rPPG training data only; no pixels, no model).

Inputs: the A9 item 1 (E1) chunk lists with their reference HR (frame units, as A9 item 1), tau_rec and frame counts:
cache/ubfc709/plan/{PhysNet,RhythmFormer}_originals.csv (555 and 436 chunks; FactorizePhys uses the RhythmFormer list),
and the E1 pool plans cache/ubfc709/plan/{PhysNet,RhythmFormer}_items.csv (targets of key 20261159).

New key K = 20261160 (unused before 7.19), in the A9 key format:
  target of item k (0-3) of chunk c: numpy.random.default_rng([K, v, c, k]).uniform(lo, hi)
     v = 0 PhysNet list, v = 2 RhythmFormer/FactorizePhys list (shared targets, as A9); c = chunk order in the A9 list.
  U-full-m: numpy.random.default_rng([K, 9, v]).choice(N, n, replace=False), sorted (n = U-low chunk count of the list).
Candidates: cutoff 100 or 90 bpm x targets U(70, 140) or U(100, 140). U-low = chunks with reference HR < cutoff.
Per candidate and list:
  1. U-low chunks, subjects, reference HR median and quartiles;
  2. phase error of (i) against (i-pc) at f*: |wrap(2 pi (tau - d)(r - 1) / T0)|, T0 = 1800 / h0, r = h* / h0, d = 1;
     mean and median, all items and by target interval 70-100, 100-120, 120-140 bpm; items that the generator would
     exclude (src.st9e.temporal.shifted_start returns None, the rule of train/gen_ubfc.py) are counted and left out;
  3. expected high-HR (>= 100 bpm) share of a batch: augmented arms 2 originals (epoch permutation of U-low) + 2 pool
     items (uniform over the generated items; high = target >= 100, as train/u8_store.U8Tensors); U-full-m arm (0):
     4 originals, high share = share of high-HR chunks; with U-full-m subject counts.
Reference row: the same on the E1 pool (whole list, U(70, 140), key 20261159; targets from the E1 plans).
Selection (fixed by 7.19 before the table): candidates with mean phase error >= 55 deg on both lists; among them the one
with most U-low chunks; ties: cutoff 100, then U(70, 140). None: no selection.
Check: the A9 targets are reproduced from the key formula (all items of both E1 plans).
Writes outputs/719/A/{b1_candidates.csv, b1_phase_bins.csv, b1_batch_share.csv, b1_selection.csv, b1_checks.csv}
(refuses to overwrite).
"""
import csv
import sys
from pathlib import Path

import numpy as np

W = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(W / 'vendor' / 'st8_6ebbc9cd'))
from src.st9e.temporal import shifted_start  # noqa: E402

PLAN = W / 'cache' / 'ubfc709' / 'plan'
OUT = W / 'outputs' / '719' / 'A'
K, K_A9, D = 20261160, 20261159, 1
LISTS = {'PhysNet': ('PhysNet', 0), 'RhythmFormer/FactorizePhys': ('RhythmFormer', 2)}
CANDIDATES = [(100, (70, 140)), (100, (100, 140)), (90, (70, 140)), (90, (100, 140))]
BINS = [('70-100', 70, 100), ('100-120', 100, 120), ('120-140', 120, 140.0001)]


def read(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def chunks(plan_name):
    rows = sorted(read(PLAN / f'{plan_name}_originals.csv'), key=lambda r: int(r['chunk_order']))
    assert [int(r['chunk_order']) for r in rows] == list(range(len(rows)))
    return rows


def target(key, v, c, k, lo, hi):
    return float(np.random.default_rng([key, v, c, k]).uniform(lo, hi))


def phase_deg(h, h0, tau, d=D):
    r = h / h0
    t0 = 1800.0 / h0
    return np.degrees(np.abs(np.angle(np.exp(1j * 2 * np.pi * (tau - d) * (r - 1) / t0))))


def items_of(rows, sel, tgt):
    """rows of the chunks in sel (chunk orders) with targets tgt[(c, k)]; generator exclusion by shifted_start."""
    out = []
    for c in sel:
        r = rows[c]
        h0, tau = float(r['reference_HR']), float(r['tau_rec'])
        for k in range(4):
            h = tgt[(c, k)]
            pos, _ = shifted_start(int(r['start_frame']), int(r['chunk_length']), int(r['frame_count']), h / h0, tau)
            out.append(dict(c=c, k=k, h=h, h0=h0, tau=tau, excluded=pos is None))
    return out


def describe(rows, sel):
    hr = np.array([float(rows[c]['reference_HR']) for c in sel])
    subj = {rows[c]['subject_id'] for c in sel}
    return dict(chunks=len(sel), subjects=len(subj), ref_HR_median=float(np.median(hr)), ref_HR_q25=float(np.quantile(hr, .25)),
                ref_HR_q75=float(np.quantile(hr, .75)), ref_HR_min=float(hr.min()), ref_HR_max=float(hr.max()),
                chunks_ref_ge100=int((hr >= 100).sum()))


def phase_rows(label, its):
    kept = [i for i in its if not i['excluded']]
    h = np.array([i['h'] for i in kept])
    ph = phase_deg(h, np.array([i['h0'] for i in kept]), np.array([i['tau'] for i in kept]))
    out = [dict(label, targets='all', items=len(kept), mean_phase_error_deg=float(ph.mean()),
                median_phase_error_deg=float(np.median(ph)))]
    for name, lo, hi in BINS:
        m = (h >= lo) & (h < hi)
        out.append(dict(label, targets=name, items=int(m.sum()),
                        mean_phase_error_deg=float(ph[m].mean()) if m.any() else '',
                        median_phase_error_deg=float(np.median(ph[m])) if m.any() else ''))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    names = ['b1_candidates.csv', 'b1_phase_bins.csv', 'b1_batch_share.csv', 'b1_selection.csv', 'b1_checks.csv']
    for n in names:
        if (OUT / n).exists():
            sys.exit(f'{n} exists')
    cand, bins, share, checks = [], [], [], []
    data = {}
    for lname, (pname, v) in LISTS.items():
        rows = chunks(pname)
        plan = read(PLAN / f'{pname}_items.csv')
        a9 = {(int(r['chunk_order']), int(r['augmentation_index'])): float(r['target_HR']) for r in plan}
        rep = sum(target(K_A9, v, c, k, 70, 140) == h for (c, k), h in a9.items())
        checks.append(dict(check=f'{lname}: A9 targets reproduced by default_rng([20261159, {v}, c, k]).uniform(70, 140)',
                           value=f'{rep} of {len(a9)} equal', passed=rep == len(a9) == 4 * len(rows)))
        tau_same = all(r['tau_rec'] == rows[int(r['chunk_order'])]['tau_rec'] and
                       r['reference_HR'] == rows[int(r['chunk_order'])]['reference_HR'] for r in plan)
        checks.append(dict(check=f'{lname}: plan items and chunk list agree on tau_rec and reference HR', value=len(plan),
                           passed=tau_same))
        data[lname] = (rows, v, a9)
        # E1 reference row
        sel = list(range(len(rows)))
        its = items_of(rows, sel, a9)
        label = dict(candidate='E1 (A9 item 1)', cutoff='none (whole list)', targets_dist='U(70, 140), key 20261159', list=lname)
        cand.append(dict(label, **describe(rows, sel), items_planned=len(its), items_excluded=sum(i['excluded'] for i in its)))
        bins += phase_rows(label, its)
        hi_pool = np.mean([i['h'] >= 100 for i in its if not i['excluded']])
        hi_orig = np.mean([float(rows[c]['reference_HR']) >= 100 for c in sel])
        share.append(dict(label, arm='augmented arms (2 originals + 2 pool items)', originals_high_share=float(hi_orig),
                          pool_high_share=float(hi_pool), expected_batch_high_share=float(.5 * hi_orig + .5 * hi_pool)))
        share.append(dict(label, arm='(0) (4 originals)', originals_high_share=float(hi_orig), pool_high_share='',
                          expected_batch_high_share=float(hi_orig)))
    for cutoff, (lo, hi) in CANDIDATES:
        for lname, (rows, v, _) in data.items():
            label = dict(candidate=f'cutoff {cutoff}, U({lo}, {hi})', cutoff=cutoff, targets_dist=f'U({lo}, {hi}), key {K}',
                         list=lname)
            low = [c for c, r in enumerate(rows) if float(r['reference_HR']) < cutoff]
            tgt = {(c, k): target(K, v, c, k, lo, hi) for c in low for k in range(4)}
            its = items_of(rows, low, tgt)
            cand.append(dict(label, **describe(rows, low), items_planned=len(its), items_excluded=sum(i['excluded'] for i in its)))
            bins += phase_rows(label, its)
            hi_pool = np.mean([i['h'] >= 100 for i in its if not i['excluded']])
            share.append(dict(label, arm='U-low augmented arms (2 originals + 2 pool items)', originals_high_share=0.0,
                              pool_high_share=float(hi_pool), expected_batch_high_share=float(.5 * hi_pool)))
            share.append(dict(label, arm='U-low (0) (4 originals)', originals_high_share=0.0, pool_high_share='',
                              expected_batch_high_share=0.0))
            fm = sorted(int(x) for x in np.random.default_rng([K, 9, v]).choice(len(rows), len(low), replace=False))
            d = describe(rows, fm)
            hs = {rows[c]['subject_id'] for c in fm if float(rows[c]['reference_HR']) >= 100}
            share.append(dict(label, arm='U-full-m (0) (4 originals)', originals_high_share=d['chunks_ref_ge100'] / len(fm),
                              pool_high_share='', expected_batch_high_share=d['chunks_ref_ge100'] / len(fm),
                              U_full_m_chunks=len(fm), U_full_m_subjects=d['subjects'], U_full_m_high_chunks=d['chunks_ref_ge100'],
                              U_full_m_subjects_with_high_chunks=len(hs), U_full_m_ref_HR_median=d['ref_HR_median']))
    # selection
    sel_rows, ok = [], []
    for cutoff, (lo, hi) in CANDIDATES:
        name = f'cutoff {cutoff}, U({lo}, {hi})'
        means = {b['list']: b['mean_phase_error_deg'] for b in bins if b['candidate'] == name and b['targets'] == 'all'}
        n = {c['list']: c['chunks'] for c in cand if c['candidate'] == name}
        passes = all(m >= 55 for m in means.values())
        sel_rows.append(dict(candidate=name, mean_phase_PhysNet=means['PhysNet'],
                             mean_phase_RhythmFormer_FactorizePhys=means['RhythmFormer/FactorizePhys'],
                             both_ge_55=passes, U_low_chunks_PhysNet=n['PhysNet'],
                             U_low_chunks_RhythmFormer_FactorizePhys=n['RhythmFormer/FactorizePhys']))
        if passes:
            ok.append((-(n['PhysNet'] + n['RhythmFormer/FactorizePhys']), 0 if cutoff == 100 else 1, 0 if lo == 70 else 1, name))
    chosen = sorted(ok)[0][3] if ok else ''
    # chunk-count ordering must agree on both lists for the rule to be unambiguous
    if ok:
        best = [x for x in ok if x[3] == chosen][0]
        same_cut = [x for x in ok]
        for x in same_cut:
            a = [s for s in sel_rows if s['candidate'] == x[3]][0]
            b = [s for s in sel_rows if s['candidate'] == chosen][0]
            assert (a['U_low_chunks_PhysNet'] <= b['U_low_chunks_PhysNet'] and
                    a['U_low_chunks_RhythmFormer_FactorizePhys'] <= b['U_low_chunks_RhythmFormer_FactorizePhys']), x
        del best
    for s in sel_rows:
        s['selected'] = s['candidate'] == chosen
    for n, rows in zip(names, (cand, bins, share, sel_rows, checks)):
        keys = []
        for r in rows:
            keys += [k for k in r if k not in keys]
        with open(OUT / n, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    print('selected:', chosen or 'none')
    for c in checks:
        print(c)


if __name__ == '__main__':
    main()
