"""(i) / (i-pc) temporal-resampling items (brief 02 6.12): only the label shift differs.

Archived (i) generators (commit 6ebbc9cd), one per backbone:
  PhysNet  src/st9g/pool_worker.py arm i    rng [20261137,h]; PhysNet HC view; DiffNormalized input/label (128 frames)
  RF       src/st13x/pool_worker.py         rng [20261137,h]; RF HC 128 view; Standardized input/label (160 frames)
  FPm      src/st30g/pool_A_worker.py arm i rng [20261202,h]; PhysNet HC 72 view; Raw input, Standardized label (160)
All use shifted_start(start_frame, length, n, r, tau_rec): source positions s_j and label positions s_j + tau_rec*(r-1).
Here the source positions are always the (i) ones (same input video); the label positions are
  shift_mode 'original': s_j + tau_rec*(r-1)   (the archived (i) label; default)
  shift_mode 'd'       : s_j + d*(r-1), d = +2 frames (the (ii-pc) value, all items)
  shift_mode 'zero'    : s_j (brief 02 6.43 (i-simple): literature time resampling, video and label resampled at the same
                         positions, label shift 0)
Edge hold (brief 02 6.13): label positions outside the contact signal [0, n-1] take the nearest end value (the
positions are clipped to [0, n-1] before interpolation); only those samples are affected. An item whose positions
leave the signal by more than 3 frames stops the generator (EdgeHoldStop). The 300-frame achieved-HR context uses
the same rule and is flagged; achieved HR is recorded only (no item is filtered by it).
Storage: PhysNet/RhythmFormer keep the stored (i) input and get a new label file (the preprocessed float32 label);
matched FactorizePhys regenerates the (i) input as uint8 view frames (train/u8_store.py) plus the raw float64 label.
Per item it records: label positions vs the intended ones (independent recomputation), the shift difference against
(d - tau_rec)(r - 1), label/input SHA against the (i) manifest, and the achieved context HR with the new label.
Usage: gen_i_pc.py BACKBONE SHIFT_MODE OUT_ROOT [ITEM_LIST.csv] [--hash-inputs]
"""
import atexit
import hashlib
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK / 'audit'))
from a1_common import *  # noqa: F401,F403
from src.st9g.preprocess import diff_normalize_label, standardized_label  # noqa: E402
sys.path.insert(0, str(WORK / 'train'))
import u8_store  # noqa: E402

MAIN_OUT = Path('<MACHINE_PATH>')  # read-only
MANIFEST = {'PhysNet': 'st9g_pool_PhysNet_i_manifest.csv', 'RhythmFormer': 'st13x_rf_i_pool_manifest.csv',
            'FPm': 'st30t_FPm_A_PURE_i_pool.csv'}
SEED = {'PhysNet': 20261137, 'RhythmFormer': 20261137, 'FPm': 20261202}
D_FRAMES = 2
EDGE_HOLD_STOP_FRAMES = 3.0
W = None


class EdgeHoldStop(RuntimeError):
    pass


def init_worker(backbone):
    global W
    if backbone == 'FPm':
        W = ViewWorker()
        atexit.register(W.close)


def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def manifest_rows(backbone):
    rows = read_csv(MAIN_OUT / MANIFEST[backbone])
    if backbone == 'PhysNet':
        rows = [r for r in rows if r['model'] == 'PhysNet' and r['arm'] == 'i']
    assert not any(r['generation_excluded'] == 'True' for r in rows)
    return rows


def shifted_label_positions(sourcepos, scale, tau, mode):
    shift = {'original': float(tau), 'd': float(D_FRAMES), 'zero': 0.0}[mode] * (scale - 1)
    return sourcepos + shift, shift


def job(args):
    backbone, mode, out_root, rid, items, hash_inputs = args
    row = pure_index()[rid]
    files = frame_files(row)
    gt = contact(row, 'PURE', None)
    tau = int(tau_table()[rid]['tau_rec'])
    n = len(files)
    grid = np.arange(len(gt))
    folder = Path(out_root) / backbone / mode / rid
    folder.mkdir(parents=True, exist_ok=True)
    physbox = W.view(read_pure(files, 0, 1), 'PhysNet')[1] if backbone == 'FPm' else None
    out = []
    for r in items:
        tick = time.time()
        c, k, length = int(r['chunk_order']), int(r['augmentation_index']), int(r['chunk_length'])
        scale = float(r['target_HR']) / float(r['reference_HR'])
        positions, plan = shifted_start(int(r['start_frame']), length, n, scale, tau)
        assert positions is not None and int(r['tau_rec']) == tau
        sourcepos, labelpos_i = positions
        labelpos, shift = shifted_label_positions(sourcepos, scale, tau, mode)
        intended = (float(plan['source_start']) + np.arange(length) * scale) + (
            {'original': float(tau), 'd': float(D_FRAMES), 'zero': 0.0}[mode]) * (scale - 1)
        n_gt = len(gt)
        held = (labelpos < 0) | (labelpos > n_gt - 1)
        outside = max(0.0, float(-labelpos.min()), float(labelpos.max() - (n_gt - 1)))
        if outside > EDGE_HOLD_STOP_FRAMES:
            raise EdgeHoldStop(f"{r['chunk_id']} k{r['augmentation_index']}: label positions {outside:.3f} frames outside")
        rec = dict(r, shift_mode=mode, d=D_FRAMES if mode == 'd' else '', ops2_tau_rec=tau, scale_recomputed=scale,
                   edge_hold_samples=int(held.sum()), edge_hold_max_frames=outside,
                   source_start_matches_manifest=float(plan['source_start']) == float(r['source_start']),
                   label_shift_source_frames_new=shift,
                   shift_change_vs_i_max_abs_dev=float(np.max(np.abs((labelpos - labelpos_i) - (shift - float(tau) * (scale - 1))))),
                   shift_change_expected=(shift - float(tau) * (scale - 1)),
                   label_positions_vs_intended_max_abs=float(np.max(np.abs(labelpos - intended))),
                   label_in_range=bool(labelpos.min() >= 0 and labelpos.max() <= len(gt) - 1))
        label = np.interp(np.clip(labelpos, 0, n_gt - 1), grid, gt)  # edge hold only outside [0, n-1]
        rec['label_vs_intended_interp_max_abs'] = float(np.max(np.abs(
            label - np.interp(np.clip(intended, 0, n_gt - 1), grid, gt))))
        ctx, _ = shifted_start(plan['source_start'] + (length - 300) * scale / 2, 300, len(gt), scale, tau)
        cpos = ctx[0] + shift
        rec['ctx_label_in_range'] = bool(cpos.min() >= 0 and cpos.max() <= n_gt - 1)
        rec['ctx_edge_hold_samples'] = int(((cpos < 0) | (cpos > n_gt - 1)).sum())
        rec['ctx_edge_hold_max_frames'] = max(0.0, float(-cpos.min()), float(cpos.max() - (n_gt - 1)))
        achieved = common_hr(np.interp(np.clip(cpos, 0, n_gt - 1), grid, gt), 'HAR')
        rec.update(achieved_context_HR_new=achieved,
                   achieved_equals_i=bool(np.isclose(achieved, float(r['achieved_context_HR']), rtol=0, atol=1e-9)))
        if backbone in ('PhysNet', 'RhythmFormer'):
            norm = diff_normalize_label if backbone == 'PhysNet' else standardized_label
            y = np.ascontiguousarray(norm(label), dtype=np.float32)
            assert y.shape == (length,) and np.isfinite(y).all()
            lp = folder / f'c{c}_k{k}_label.npy'
            assert not lp.exists()
            np.save(lp, y)
            rec.update(ops2_label_path=str(lp), ops2_label_sha256=file_sha(lp),
                       label_sha_equals_i=file_sha(lp) == r['label_sha256'])
            if hash_inputs:
                rec['stored_input_sha_equals_manifest'] = file_sha(r['input_path']) == r['input_sha256']
        else:
            lo, hi = int(np.floor(sourcepos.min())), int(np.ceil(sourcepos.max()))
            src = read_pure(files, lo, hi - lo + 1)
            rng = np.random.default_rng([SEED[backbone], int(r['rounding_h'])])
            output = np.empty((length,) + src.shape[1:], np.uint8)
            for j, p in enumerate(sourcepos - lo):
                left, right = int(np.floor(p)), int(np.ceil(p))
                frac = p - left
                value = (1 - frac) * src[left].astype(np.float64) + frac * src[right]
                output[j] = np.floor(value + rng.random(value.shape)).astype(np.uint8)
            view, _ = W.view(output, 'PhysNet', reuse_boxes=physbox)
            u8 = u8_store.to_u8(view)
            vp = u8_store.save_view(folder / f'c{c}_k{k}_view', u8)
            lp = u8_store.save_array(folder / f'c{c}_k{k}_rawlabel', np.asarray(label, dtype=np.float64))
            x, yy = u8_store.model_input('FPm', u8_store.read_array(vp), u8_store.read_array(lp), 0, length)
            xs, ys = u8_store.npy_sha(x), u8_store.npy_sha(yy)
            rec.update(ops2_view_path=str(vp), ops2_rawlabel_path=str(lp), ops2_stats_path='',
                       ops2_view_bytes=vp.stat().st_size, ops2_rawlabel_bytes=lp.stat().st_size,
                       ops2_input_sha256=xs, ops2_label_sha256=ys,
                       input_sha_equals_i=xs == r['input_sha256'], label_sha_equals_i=ys == r['label_sha256'])
            del src, output, view, u8, x, yy
        out.append(dict(rec, ops2_CPU_seconds=time.time() - tick))
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    backbone, mode, out_root = args[0], args[1], args[2]
    assert mode in ('original', 'd', 'zero')
    hash_inputs = '--hash-inputs' in sys.argv
    items = manifest_rows(backbone)
    tag = 'all'
    if len(args) > 3:
        keep = {(r['chunk_id'], r['augmentation_index']) for r in read_csv(args[3])}
        items = [r for r in items if (r['chunk_id'], r['augmentation_index']) in keep]
        assert len(items) == len(keep)
        tag = Path(args[3]).stem
    by_rec = {}
    for r in items:
        by_rec.setdefault(r['recording_id'], []).append(r)
    workers = int(os.environ.get('OPS2_WORKERS', '4'))
    started = time.time()
    rows = []
    jobs = [(backbone, mode, out_root, rid, its, hash_inputs) for rid, its in sorted(by_rec.items(), key=lambda kv: -len(kv[1]))]
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'), initializer=init_worker,
                             initargs=(backbone,)) as ex:
        for i, out in enumerate(ex.map(job, jobs), 1):
            rows += out
            print(f'i-pc {backbone} {mode} {i}/{len(jobs)} recordings, {len(rows)} items, {time.time() - started:.0f}s', flush=True)
    write_rows(Path(out_root) / f'{backbone}_i_{mode}_{tag}_manifest.csv', rows)
    def count(k):
        return sum(1 for r in rows if r.get(k) is True)
    print('done', len(rows), 'label_in_range', count('label_in_range'), 'label_sha_equals_i', count('label_sha_equals_i'),
          'input_sha_equals_i', count('input_sha_equals_i'), 'stored_input_ok', count('stored_input_sha_equals_manifest'),
          'achieved_equals_i', count('achieved_equals_i'))


if __name__ == '__main__':
    main()
