"""Second pool of A6 item 3 (7.04, 7.07 다): the code of the first pool with its random keys replaced.

Keys (A6 item 3): target HRs default_rng([20261158, v, c, k]).uniform(70, 140) in place of [20261127, v, c, k]
(v = 0 PhysNet, 2 FactorizePhys and RhythmFormer; c = chunk order, k = augmentation index; check: the old key reproduces
every first-pool target); per-item rounding generators [20261158, 1, h] in place of [20261137, h] (PhysNet, RhythmFormer)
and [20261158, 2, h] in place of [20261202, h] (FactorizePhys); h = the item's rounding_h, unchanged. Items, chunks and
pairing across arms are those of the first pool (the first-pool manifests are the item lists).

family i  (arms (i), (i-pc), (i-simple)): per item one input video, resampled at the (i) source positions with stochastic
          rounding (archived st9g/st13x/st30g (i) workers; the FactorizePhys branch of train/gen_i_pc.py), stored as uint8
          view frames (train/u8_store.py, chunk = whole view, ops2_view_offset 0), and three raw labels at the label
          positions of shift modes 'original' ((i)), 'd' ((i-pc)) and 'zero' ((i-simple)) (train/gen_i_pc.py rules:
          shifted_label_positions, edge hold, EdgeHoldStop). Model inputs at load time: u8_store.model_input.
family ii (arms (ii), (ii-pc)): train/gen_pool.py job unchanged, modes 'original' ((ii)) and 'polarity_explicit' ((ii-pc)),
          given the second-pool rows; its rounding generator is redirected to the new key by a numpy proxy in the worker
          (only default_rng([old seed, h]) calls are changed).
Check mode (--first-pool ITEM_LIST): the same code with the first-pool keys on listed first-pool items; the tensors are
compared with the first-pool manifests (family i: the October (i) manifest and the OPS2 (i-pc)/(i-simple) manifests).
Usage: gen_p2.py BACKBONE FAMILY OUT_ROOT [--first-pool ITEM_LIST.csv]   (OPS2_WORKERS, OPS2_POOL_COMPRESS as the others)
Writes OUT_ROOT/<BACKBONE>_p2_<arm>_manifest.csv (or <BACKBONE>_check_<arm>_manifest.csv).
"""
import atexit
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK / 'train'))
import gen_i_pc as GI  # noqa: E402  (a1_common, shifted_start, ViewWorker, PURE readers)
import gen_pool as GP  # noqa: E402  (editor_pc, RhythmView)
import u8_store  # noqa: E402
import numpy  # noqa: E402

NEW = 20261158
OLD_TARGET = 20261127
OLD_ROUND = {'PhysNet': 20261137, 'RhythmFormer': 20261137, 'FPm': 20261202}
J = {20261137: 1, 20261202: 2}
V = {'PhysNet': 0, 'RhythmFormer': 2, 'FPm': 2}
ARMS_I = {'original': '(i)', 'd': '(i-pc)', 'zero': '(i-simple)'}
ARMS_II = {'original': '(ii)', 'polarity_explicit': '(ii-pc)'}
FIRST_I_LABELS = {  # OPS2 first-pool (i-pc)/(i-simple) manifests (label SHA of the loader label)
    'd': {'PhysNet': 'cache/pools_i/PhysNet_i_d_all_manifest.csv', 'RhythmFormer': 'cache/pools_i/RhythmFormer_i_d_all_manifest.csv',
          'FPm': 'cache/pools_i/FPm_i_d_all_manifest.csv'},
    'zero': {'PhysNet': 'cache/pools_i/PhysNet_i_zero_all_manifest.csv', 'RhythmFormer': 'cache/pools_i/RhythmFormer_i_zero_all_manifest.csv',
             'FPm': 'cache/pools_i/FPm_i_zero_all_manifest.csv'}}
W = R = None
SECOND = True


def target(backbone, c, k, second):
    return float(numpy.random.default_rng([NEW if second else OLD_TARGET, V[backbone], c, k]).uniform(70, 140))


def rounding_rng(backbone, h, second):
    old = OLD_ROUND[backbone]
    return numpy.random.default_rng([NEW, J[old], h] if second else [old, h])


class _Random:
    """numpy.random with default_rng([old seed, h]) redirected to [20261158, j, h] (second pool only)."""
    def __getattr__(self, name):
        return getattr(numpy.random, name)

    def default_rng(self, seed=None):
        if SECOND and isinstance(seed, list) and len(seed) == 2 and seed[0] in J:
            seed = [NEW, J[seed[0]], seed[1]]
        return numpy.random.default_rng(seed)


class _Numpy:
    random = _Random()

    def __getattr__(self, name):
        return getattr(numpy, name)


def init_worker(backbone, family, second):
    global W, R, SECOND
    SECOND = second
    if family == 'ii':
        GP.init_worker(backbone)
        GP.np = _Numpy()  # only gen_pool.job's own rounding generator goes through this name
        return
    if backbone == 'RhythmFormer':
        R = GP.RhythmView()
        atexit.register(R.close)
    else:
        W = GI.ViewWorker()
        atexit.register(W.close)


def job_i(args):
    backbone, out_root, rid, items, second, tag = args
    np = numpy
    row = GI.pure_index()[rid]
    files = GI.frame_files(row)
    gt = GI.contact(row, 'PURE', None)
    tau = int(GI.tau_table()[rid]['tau_rec'])
    n, n_gt = len(files), len(gt)
    grid = np.arange(n_gt)
    first = GI.read_pure(files, 0, 1)
    if backbone == 'RhythmFormer':
        box = R.view(first)[1][0]
    else:
        physbox = W.view(first, 'PhysNet')[1]
    folder = Path(out_root) / backbone / f'i_{tag}' / rid
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for r in items:
        tick = time.time()
        c, k, length = int(r['chunk_order']), int(r['augmentation_index']), int(r['chunk_length'])
        scale = float(r['target_HR']) / float(r['reference_HR'])
        positions, plan = GI.shifted_start(int(r['start_frame']), length, n, scale, tau)
        if positions is None:
            out.append(dict(r, mode='', generation_excluded=True, exclusion_reason='source_or_label_interval_exceeds_recording'))
            continue
        sourcepos, labelpos_i = positions
        lo, hi = int(np.floor(sourcepos.min())), int(np.ceil(sourcepos.max()))
        src = GI.read_pure(files, lo, hi - lo + 1)
        rng = rounding_rng(backbone, int(r['rounding_h']), second)
        output = np.empty((length,) + src.shape[1:], np.uint8)
        for j, p in enumerate(sourcepos - lo):
            left, right = int(np.floor(p)), int(np.ceil(p))
            frac = p - left
            value = (1 - frac) * src[left].astype(np.float64) + frac * src[right]
            output[j] = np.floor(value + rng.random(value.shape)).astype(np.uint8)
        if backbone == 'RhythmFormer':
            view, _ = R.view(output, box)
        else:
            view, _ = W.view(output, 'PhysNet', reuse_boxes=physbox)
        vp = u8_store.save_view(folder / f'c{c}_k{k}_view', u8_store.to_u8(view))
        for mode in ('original', 'd', 'zero'):
            labelpos, shift = GI.shifted_label_positions(sourcepos, scale, tau, mode)
            outside = max(0.0, float(-labelpos.min()), float(labelpos.max() - (n_gt - 1)))
            if outside > GI.EDGE_HOLD_STOP_FRAMES:
                raise GI.EdgeHoldStop(f"{r['chunk_id']} k{k} {mode}: label positions {outside:.3f} frames outside")
            label = np.interp(np.clip(labelpos, 0, n_gt - 1), grid, gt)
            lp = u8_store.save_array(folder / f'c{c}_k{k}_rawlabel_{mode}', np.asarray(label, dtype=np.float64))
            x, y = u8_store.model_input(backbone, u8_store.read_array(vp), u8_store.read_array(lp), 0, length)
            ctx, _ = GI.shifted_start(plan['source_start'] + (length - 300) * scale / 2, 300, n_gt, scale, tau)
            cpos = ctx[0] + shift if ctx is not None else None
            achieved = GI.common_hr(np.interp(np.clip(cpos, 0, n_gt - 1), grid, gt), 'HAR') if cpos is not None else np.nan
            out.append(dict(r, mode=mode, arm=ARMS_I[mode], generation_excluded=False, exclusion_reason='', ops2_tau_rec=tau,
                            scale=scale, source_start=plan['source_start'], label_shift_source_frames_new=shift,
                            edge_hold_samples=int(((labelpos < 0) | (labelpos > n_gt - 1)).sum()), edge_hold_max_frames=outside,
                            achieved_context_HR_new=achieved, ops2_view_path=str(vp), ops2_rawlabel_path=str(lp),
                            ops2_stats_path='', ops2_view_offset=0, ops2_view_bytes=vp.stat().st_size,
                            ops2_rawlabel_bytes=lp.stat().st_size, ops2_input_sha256=u8_store.npy_sha(x),
                            ops2_label_sha256=u8_store.npy_sha(y), ops2_CPU_seconds=time.time() - tick))
        del src, output, view
    return out


def job_ii(args):
    backbone, out_root, rid, items, second, tag = args
    out = GP.job((backbone, 'original,polarity_explicit', out_root, rid, items))
    for r in out:
        r['arm'] = ARMS_II[r['mode']]
        r['generation_excluded'], r['exclusion_reason'] = False, ''
    return out


def plan_rows(backbone, family, second, keep=None):
    if family == 'i':
        rows = GI.manifest_rows(backbone)
    else:
        rows = [r for r in GI.read_csv(GP.VENDOR / GP.MANIFEST[backbone]) if str(r.get('generation_excluded', 'False')) != 'True']
    for r in rows:  # the old key reproduces every first-pool target (checked once more here)
        assert target(backbone, int(r['chunk_order']), int(r['augmentation_index']), False) == float(r['target_HR'])
    if keep is not None:
        rows = [r for r in rows if (r['chunk_id'], r['augmentation_index']) in keep]
        assert len(rows) == len(keep)
    if not second:
        return rows
    new = []
    for r in rows:
        t = target(backbone, int(r['chunk_order']), int(r['augmentation_index']), True)
        q = {k: v for k, v in r.items() if not (k.endswith('sha256') or k.endswith('_path') or k.endswith('_bytes'))}
        # target_HR_original: train_pc.main then records the second-pool target in exposures.csv (labels come from files)
        q.update(pool='B', target_HR=t, target_HR_original=r['target_HR'], rounding_key_vector=f'[{NEW},{J[OLD_ROUND[backbone]]},h]',
                 target_key_vector=f'[{NEW},{V[backbone]},c,k]', input_sha256='', label_sha256='')
        new.append(q)
    return new


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    backbone, family, out_root = args[0], args[1], Path(args[2])
    assert backbone in V and family in ('i', 'ii')
    check = '--first-pool' in sys.argv
    keep = None
    if check:
        keep = {(r['chunk_id'], r['augmentation_index']) for r in GI.read_csv(args[3])}
    second = not check
    tag = 'check' if check else 'p2'
    rows = plan_rows(backbone, family, second, keep)
    by_rec = {}
    for r in rows:
        by_rec.setdefault(r['recording_id'], []).append(r)
    workers = int(os.environ.get('OPS2_WORKERS', '4'))
    fn = job_i if family == 'i' else job_ii
    jobs = [(backbone, str(out_root), rid, its, second, tag) for rid, its in sorted(by_rec.items(), key=lambda kv: -len(kv[1]))]
    started, out = time.time(), []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'), initializer=init_worker,
                             initargs=(backbone, family, second)) as ex:
        for i, o in enumerate(ex.map(fn, jobs), 1):
            out += o
            print(f'p2 {backbone} {family} {tag} {i}/{len(jobs)} recordings, {len(out)} rows, {time.time() - started:.0f}s', flush=True)
    arms = ARMS_I if family == 'i' else ARMS_II
    for mode, arm in arms.items():
        mr = [r for r in out if r.get('mode') == mode]
        excl = [r for r in out if r.get('generation_excluded') is True]
        GI.write_rows(out_root / f'{backbone}_{tag}_{arm.strip("()")}_manifest.csv', mr + excl)
        line = f'done {backbone} {arm} items {len(mr)} excluded {len(excl)}'
        if check:
            if family == 'i':
                ref = {(q['chunk_id'], q['augmentation_index']): q for q in
                       (GI.manifest_rows(backbone) if mode == 'original' else GI.read_csv(WORK / FIRST_I_LABELS[mode][backbone]))}
                ki = 'input_sha256' if mode == 'original' else 'ops2_input_sha256'
                kl = 'label_sha256' if mode == 'original' else 'ops2_label_sha256'
                if backbone in ('PhysNet', 'RhythmFormer') and mode != 'original':
                    ki = 'input_sha256'  # (i-pc)/(i-simple) of PhysNet/RhythmFormer kept the stored (i) input
                inp = sum(r['ops2_input_sha256'] == ref[(r['chunk_id'], r['augmentation_index'])][ki] for r in mr)
                lab = sum(r['ops2_label_sha256'] == ref[(r['chunk_id'], r['augmentation_index'])][kl] for r in mr)
            else:
                inp = sum(r['input_sha_equals_original'] for r in mr) if mode == 'original' else ''
                lab = sum(r['label_sha_equals_original'] for r in mr) if mode == 'original' else ''
                if mode == 'polarity_explicit':
                    ref = {(q['chunk_id'], q['augmentation_index']): q for q in GI.read_csv(WORK / f'cache/pools/{backbone}_polarity_explicit_all_manifest.csv')}
                    inp = sum(r['ops2_input_sha256'] == ref[(r['chunk_id'], r['augmentation_index'])]['ops2_input_sha256'] for r in mr)
                    lab = sum(r['ops2_label_sha256'] == ref[(r['chunk_id'], r['augmentation_index'])]['ops2_label_sha256'] for r in mr)
            line += f' input_sha_equal_first_pool {inp} label_sha_equal_first_pool {lab}'
        print(line, flush=True)


if __name__ == '__main__':
    main()
