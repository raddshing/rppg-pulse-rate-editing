"""OPS2 7.20 2 (7.20b): docs/A10_UBFC_B1.txt (no recording IDs) for the A10 commit; the only place of the SHA-256.

Reads: the B1 plans and chunk lists (cache/b1_719/plan; SHA-256 recomputed here), outputs/719/A/b1_candidates.csv and
b1_phase_bins.csv (design table), outputs/719/A10/u_full_m_v2.csv, outputs/719/A10/pure_first_pool_phase_i.csv,
the pool manifests written by train/gen_b1.py (items generated, excluded), outputs/719/A10/originals_check.csv and
outputs/719/A10/wrapper_trial.csv. Writes outputs/719/A10/A10_UBFC_B1.txt (refuses to overwrite).
"""
import csv
import hashlib
import sys
from pathlib import Path

W = Path(__file__).resolve().parents[2]
PLAN = W / 'cache' / 'b1_719' / 'plan'
POOLS = W / 'cache' / 'b1_719' / 'pools'
A, O = W / 'outputs' / '719' / 'A', W / 'outputs' / '719' / 'A10'


def rd(p):
    with open(p, newline='') as f:
        return list(csv.DictReader(f))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def f1(x):
    return '–' if x in ('', None) else f'{float(x):.1f}'


def main():
    out_txt = O / 'A10_UBFC_B1.txt'
    if out_txt.exists():
        sys.exit(f'{out_txt.name} exists')
    fm = {r['list']: r for r in rd(O / 'u_full_m_v2.csv')}
    files = [('pool_manifest', 'PhysNet', 'U-low', 'PhysNet_items.csv'),
             ('pool_manifest', 'RhythmFormer', 'U-low', 'RhythmFormer_items.csv'),
             ('pool_manifest', 'FactorizePhys', 'U-low', 'FPm_items.csv'),
             ('chunk_list', 'PhysNet list', 'U-low', 'PhysNet_U_low_chunks.csv'),
             ('chunk_list', 'RhythmFormer list (also FactorizePhys)', 'U-low', 'RhythmFormer_U_low_chunks.csv'),
             ('chunk_list', 'PhysNet list', 'U-full-m', 'PhysNet_U_full_m_chunks.csv'),
             ('chunk_list', 'RhythmFormer list (also FactorizePhys)', 'U-full-m', 'RhythmFormer_U_full_m_chunks.csv')]
    man = []
    for kind, bb, arm, name in files:
        n = len(rd(PLAN / name))
        man.append(dict(kind=kind, backbone=bb, set=arm, n=n, file=name, sha256=sha(PLAN / name)))
    assert man[5]['sha256'] == fm['PhysNet']['sha256'] and man[6]['sha256'] == fm['RhythmFormer']['sha256']
    L = []
    L += ['A10 (entry of 2026-11-27), item B1: keys, SHA-256, design table and checks of the UBFC-rPPG training sets.',
          'The pool manifests and chunk lists are not committed (they hold recording codes); their SHA-256 are below.',
          'Heart rates are in the frame units of item 1 of the entry of 2026-11-26.',
          '', '1. Keys and formats', '',
          'Target HR of pool item k (k = 0, 1, 2, 3) of chunk c:',
          '  numpy.random.default_rng([20261160, v, c, k]).uniform(100, 140)',
          '  v = 0 for PhysNet; v = 2 for RhythmFormer and FactorizePhys, which share the targets;',
          '  c = chunk order in the chunk list of item 1 of the entry of 2026-11-26 (from 0).',
          'Rounding generator of the resampled clip of item k of chunk c (shared by (i), (i-pc) and (i-simple)):',
          '  numpy.random.default_rng([20261160, j, h]); j = 1 for PhysNet and RhythmFormer, j = 2 for FactorizePhys;',
          "  h = int(first 8 hexadecimal digits of SHA-256 of the string 'OPS2-719|UBFC|<model>|<chunk>|<k>', 16), where",
          "  <model> is PhysNet, RhythmFormer or FPm (FactorizePhys) and <chunk> is '<recording>|<chunk index>'.",
          'U-low: the chunks of the list with reference HR < 100 bpm.',
          'U-full-m, per list (v as above):',
          '  S = the subjects of U-low; C = the chunk orders of all chunks of the list whose subject is in S, ascending;',
          '  n = the number of U-low chunks of the list;',
          '  U-full-m = sorted(C[i] for i in numpy.random.default_rng([20261160, 10, v]).choice(len(C), n, replace=False)).',
          'Replaced before this commit and before any training: a first U-full-m drawn from all chunks of the list',
          '  (all 42 subjects) with numpy.random.default_rng([20261160, 9, v]).choice(N, n, replace=False); SHA-256',
          f"  PhysNet list {fm['PhysNet']['superseded_list_sha256']}, RhythmFormer list {fm['RhythmFormer']['superseded_list_sha256']}; not used.",
          '', '2. SHA-256', '']
    L += [f"{m['sha256']}  {m['file']}  ({m['n']} rows; {m['kind'].replace('_', ' ')}, {m['backbone']}, {m['set']})" for m in man]
    L += ['', '3. Training sets', '',
          'List                          set       chunks  subjects  chunks >= 100 bpm  their subjects  ref. HR median [quartiles]']
    cand = rd(A / 'b1_candidates.csv')
    for lname, key in (('PhysNet', 'PhysNet'), ('RhythmFormer/FactorizePhys', 'RhythmFormer')):
        c = [r for r in cand if r['candidate'] == 'cutoff 100, U(100, 140)' and r['list'] == lname][0]
        f = fm[key]
        L.append(f"{lname:<29} U-low     {c['chunks']:>6}  {c['subjects']:>8}  {0:>17}  {0:>14}  "
                 f"{f1(c['ref_HR_median'])} [{f1(c['ref_HR_q25'])}, {f1(c['ref_HR_q75'])}]")
        L.append(f"{lname:<29} U-full-m  {f['U_full_m_chunks']:>6}  {f['U_full_m_subjects']:>8}  {f['U_full_m_high_chunks']:>17}  "
                 f"{f['U_full_m_high_subjects']:>14}  {f1(f['U_full_m_ref_HR_median'])} [{f1(f['U_full_m_ref_HR_q25'])}, {f1(f['U_full_m_ref_HR_q75'])}]")
        L.append(f"{'':<29} (U-full-m drawn from {f['candidate_chunks']} chunks of the {f['U_low_subjects']} subjects, "
                 f"{f['candidate_high_chunks']} of them >= 100 bpm)")
    L += ['', 'Expected share of labels >= 100 bpm in a training batch (the sampling of the training code: originals in a',
          'permutation per epoch, pool items drawn uniformly; a pool item counts by its target, an original by its reference HR):']
    for lname, key in (('PhysNet', 'PhysNet'), ('RhythmFormer/FactorizePhys', 'RhythmFormer')):
        its = rd(PLAN / f"{'PhysNet' if key == 'PhysNet' else 'RhythmFormer'}_items.csv")
        pool_hi = sum(float(r['target_HR']) >= 100 for r in its) / len(its)
        L.append(f"  {lname}: U-low augmented arms (2 originals + 2 pool items) {0.5 * pool_hi:.3f}; U-low (0) 0; "
                 f"U-full-m (0) {float(fm[key]['U_full_m_high_share']):.3f}")
    L += ['', '4. Design table (training data only; before this entry)', '',
          'Phase error of (i) against (i-pc) at f*: |wrap(2 pi (tau - d)(r - 1) / T0)|, T0 = 1800 / h0 frames, r = h* / h0,',
          'd = 1 frame, tau = per-recording tau of the October rule; targets drawn with the keys above (the U(70, 140)',
          'designs with the same key). Degrees, mean / median (items). No item is excluded by the generator rule.',
          'Rule (fixed before the computation): designs with mean >= 55 deg on both lists; most U-low chunks; ties: cutoff',
          '100, then targets 70-140. Selected: cutoff 100, targets 100-140.', '',
          'design (cutoff, targets) / list | U-low chunks | subjects | ref. HR median [quartiles] | all | 70-100 | 100-120 | 120-140']
    bins = rd(A / 'b1_phase_bins.csv')
    for c in cand:
        bb = {b['targets']: b for b in bins if b['candidate'] == c['candidate'] and b['list'] == c['list']}
        ph = ' | '.join(f"{f1(bb[t]['mean_phase_error_deg'])} / {f1(bb[t]['median_phase_error_deg'])} ({bb[t]['items']})"
                        for t in ('all', '70-100', '100-120', '120-140'))
        name = c['candidate'] if not c['candidate'].startswith('E1') else 'pools of item 1 of the entry of 2026-11-26 (all chunks, U(70, 140))'
        L.append(f"{name} / {c['list']} | {c['chunks']} | {c['subjects']} | {f1(c['ref_HR_median'])} "
                 f"[{f1(c['ref_HR_q25'])}, {f1(c['ref_HR_q75'])}] | {ph}")
    L += ['', 'The same measure on the first PURE pool (manifests only; items of (i-pc), d = 2 frames, tau of the PURE recordings):']
    for r in rd(O / 'pure_first_pool_phase_i.csv'):
        L.append(f"  {r['list']}, targets {r['targets']}: {f1(r['mean_phase_error_deg'])} / {f1(r['median_phase_error_deg'])} ({r['items']})")
    L += ['', '5. Checks before this commit', '']
    for b in ('PhysNet', 'RhythmFormer', 'FPm'):
        ms = {a: rd(POOLS / f'{b}_{a}_all_manifest.csv') for a in ('i', 'i-pc', 'i-simple')}
        n = {a: sum(str(r.get('generation_excluded')) != 'True' for r in v) for a, v in ms.items()}
        ex = {a: sum(str(r.get('generation_excluded')) == 'True' for r in v) for a, v in ms.items()}
        L.append(f"Pools, {'FactorizePhys' if b == 'FPm' else b}: items generated (i) {n['i']}, (i-pc) {n['i-pc']}, (i-simple) {n['i-simple']}; "
                 f"excluded {max(ex.values())}.")
    for r in rd(O / 'originals_check.csv'):
        L.append(f"Regenerated originals, {r['backbone']}: {r['equal']} of {r['chunks']} chunk tensors (input and label SHA-256) equal "
                 'to those of item 1 of the entry of 2026-11-26.')
    first = O / 'wrapper_trial_first_failed.csv'
    if first.exists():
        ff = rd(first)
        L.append(f"Training-wrapper trial, first attempt ({min(r['start'] for r in ff)} to {max(r['end'] for r in ff)}): "
                 'no case reached its first step; the data-loader workers re-read the wrapper\'s arguments at import and')
        L.append('stopped there. The wrapper was corrected (arguments read in the main process only; training code unchanged)'
                 ' and the trial repeated:')
    for r in rd(O / 'wrapper_trial.csv'):
        L.append(f"Training-wrapper trial (discarded), {r['backbone']} {r['set']} {r['pool']}: {r['steps']} steps, "
                 f"{r['start']} to {r['end']} (exit {r['exit']}).")
    out_txt.write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
