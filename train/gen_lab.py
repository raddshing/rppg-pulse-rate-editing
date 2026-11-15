"""Brief 02 6.43 라 / 6.44 나: label and processing arms built on the (ii-pc) generator (train/gen_pool.py), one pass per run.

Everything is the (ii-pc) item of train/gen_pool.py (polarity_explicit: injection -g*(t - d), d = +2; same chunk, notch,
face masks, v, calibration, PhysNet/FPm view, uint8 storage, rounding generator [SEED, rounding_h], label rule) except:
  pc_base  the (ii-pc) item itself, all items (the reference for unchanged items; compared with the (ii-pc) manifest SHA)
  pc_g     target h* in [100, 120) -> h* + 20 (October (ii-g) mapping); only the changed items are written
  pc_g80   target h* < 100 -> 70 + (h* - 70)/3 (October (ii-g2) low-HR mapping); only changed items
  pc_g2    both mappings; only changed items
  pc_sham  identity edit: target = h0 (r = 1), injection -g*(t - d) with g* the original PPG, label = original PPG; all
           items (each with its own rounding key)
  pc_roi   (PhysNet) the S7a 'ROI only' editor (Stage 4 src/st4/editor_ops.py edit_float, ED2, route S; reproduced by
           audit/roi_s7a_check.py): notch at h0 and 2 h0 and additive injection mu A v w(t) only inside the forehead/cheek
           masks (src/st4/editor_inputs.py interpolate_masks on the 300 native frames of the chunk), A from the original
           target waveform g*(t - tau) as in the ROI editor, injected wave w = -g*(t - d) (the 2.1 injection argument);
           targets of (ii-pc); all items
The notch, face masks, calibration and ROI masks are computed once per chunk and shared by all modes of the pass.
Usage: gen_lab.py BACKBONE MODE[,MODE...] OUT_ROOT  (OPS2_POOL_COMPRESS selects the storage, as gen_pool.py)
Writes OUT_ROOT/BACKBONE/MODE/<rid>/c{c}_k{k}_{view,rawlabel}.* and OUT_ROOT/<BACKBONE>_<MODE>_all_manifest.csv.
"""
import atexit
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK / 'audit'))
sys.path.insert(0, str(WORK / 'train'))
from editor_pc import *  # noqa: F401,F403
import u8_store  # noqa: E402
import gen_pool  # noqa: E402
from scipy.signal import butter, sosfiltfilt  # noqa: E402

SEED = gen_pool.SEED
W = None


def target_for(mode, t, h0):
    if mode in ('pc_base', 'pc_roi'):
        return t
    if mode == 'pc_sham':
        return h0
    out = t
    if mode in ('pc_g', 'pc_g2') and 100 <= t < 120:
        out = t + 20
    if mode in ('pc_g80', 'pc_g2') and t < 100:
        out = 70 + (t - 70) / 3
    return out


def init_worker(backbone):
    global W
    W = ViewWorker()
    atexit.register(W.close)


def roi_prepare(clip, masks, h0):
    """Chunk part of the Stage 4 ED2 edit (edit_float): R of the ROI green signal and the ROI-pixel notch."""
    from src.st4 import editor_ops as E4
    from src.st4.signals import filter4
    means = E4.skin_means(clip, masks)
    normalized = E4.green(means)
    R = float(np.sqrt(np.mean(filter4(normalized, [h0 / 60 - .15, h0 / 60 + .15])**2)))
    filters = [butter(4, [m * h0 / 60 - .15, m * h0 / 60 + .15], btype='bandstop', fs=30, output='sos') for m in (1, 2)]
    mask_flat = masks.reshape(300, -1)
    idx = np.flatnonzero(mask_flat.any(axis=0))
    block = clip.reshape(300, -1, 3)[:, idx].astype(np.float64)
    filtered = block
    for sos in filters:
        filtered = sosfiltfilt(sos, filtered, axis=0)
    mu = clip.mean(axis=0, dtype=np.float64).reshape(-1, 3)[idx]
    return dict(R=R, idx=idx, active=mask_flat[:, idx, None], block=block, filtered=filtered, mu=mu)


def roi_apply(clip, prep, goal, inject, v):
    """Item part of edit_float(..., 'ED2'): A from goal; injected wave = inject (or goal when inject is None)."""
    v = np.asarray(v, float)
    A = prep['R'] / (v[1] * np.sqrt(np.mean(goal**2)))
    wave = goal if inject is None else np.asarray(inject, dtype=np.float64)
    out = clip.astype(np.float32)
    flat = out.reshape(300, -1, 3)
    injected = prep['filtered'] + prep['mu'][None, :, :] * A * v[None, None, :] * wave[:, None, None]
    flat[:, prep['idx']] = np.where(prep['active'], np.clip(injected, 0, 255), prep['block']).astype(np.float32)
    return out, float(A)


def job(args):
    backbone, modes, out_root, rid, items = args
    modes = modes.split(',')
    from src.st4.editor_inputs import interpolate_masks
    row = pure_index()[rid]
    files = frame_files(row)
    gt = contact(row, 'PURE', None)
    v = direction()
    tau = int(tau_table()[rid]['tau_rec'])
    native = read_pure(files, 0, len(files))
    phys_all, physboxes = W.view(native, 'PhysNet')
    folders = {m: Path(out_root) / backbone / m / rid for m in modes}
    for f in folders.values():
        f.mkdir(parents=True, exist_ok=True)
    out = []
    for c in sorted({int(r['chunk_order']) for r in items}):
        current = [r for r in items if int(r['chunk_order']) == c]
        ref = current[0]
        cs, off, length, h0 = int(ref['context_start']), int(ref['context_offset']), int(ref['chunk_length']), float(ref['reference_HR'])
        tick = time.time()
        clip = native[cs:cs + 300]
        face, roi, failure = face_masks(clip)
        notched = notch_all(clip, bands_for(h0))
        physview = phys_all[cs:cs + 300]
        roiprep = None
        if 'pc_roi' in modes:
            roimasks, roifail = interpolate_masks(np.ascontiguousarray(clip))
            roiprep = roi_prepare(clip, roimasks, h0)
        for r in current:
            k, t0 = int(r['augmentation_index']), float(r['target_HR'])
            g0, _, _ = injection_wave(gt, cs, h0, t0, tau, 'original')
            pulse, w, beta = calibration(physview, crop_native(notched, physboxes[0]), crop_native(face, physboxes[0]), g0, v)
            arm = 'ii' if backbone == 'FPm' else 'ii-h'
            for mode in modes:
                target = target_for(mode, t0, h0)
                if mode in ('pc_g', 'pc_g80', 'pc_g2') and target == t0:
                    continue
                rng = np.random.default_rng([SEED[backbone], int(r['rounding_h'])])
                g0m, gstar, inject = injection_wave(gt, cs, h0, target, tau, 'polarity_explicit')
                if mode == 'pc_roi':
                    value, A = roi_apply(clip, roiprep, gstar, inject, v)
                    output = np.empty(value.shape, np.uint8)
                    for i, frame in enumerate(value):
                        output[i] = np.floor(frame.astype(np.float64) + rng.random(frame.shape)).astype(np.uint8)
                    diag = dict(injection_A=A, roi_pixels=int(len(roiprep['idx'])), roi_landmark_failure=roifail)
                else:
                    output, diag = edited(clip, notched, face, roi, g0m, gstar, v, h0, 'ED2W-S', pulse, w, r['chunk_id'], arm,
                                          rounding_rng=rng, inject=inject)
                raw_label = np.asarray(raw_target_label(gt, cs, h0, target))
                assert raw_label.dtype == np.float64 and raw_label.shape == (300,)
                view, _ = W.view(output, 'PhysNet', reuse_boxes=physboxes)
                u8, stats = u8_store.pack(backbone, u8_store.to_u8(view), off, length)
                folder = folders[mode]
                vp = u8_store.save_view(folder / f'c{c}_k{k}_view', u8)
                lp = u8_store.save_array(folder / f'c{c}_k{k}_rawlabel', raw_label)
                sp = None if stats is None else u8_store.save_array(folder / f'c{c}_k{k}_stats', stats)
                x, y = u8_store.model_input(backbone, u8_store.read_array(vp), u8_store.read_array(lp), off, length,
                                            None if sp is None else u8_store.read_array(sp))
                out.append(dict(r, mode=mode, target_HR_original=t0, target_HR=target, target_changed=bool(target != t0),
                                ops2_view_path=str(vp), ops2_rawlabel_path=str(lp), ops2_stats_path='' if sp is None else str(sp),
                                ops2_view_bytes=vp.stat().st_size, ops2_rawlabel_bytes=lp.stat().st_size,
                                ops2_stats_bytes=0 if sp is None else sp.stat().st_size,
                                ops2_input_sha256=u8_store.npy_sha(x), ops2_label_sha256=u8_store.npy_sha(y),
                                ops2_u8_data_sha256=u8_store.data_sha(u8), ops2_input_data_sha256=u8_store.data_sha(x),
                                ops2_injection_A=diag['injection_A'], ops2_beta=beta, ops2_tau_rec=tau, d=D_FRAMES, s=-1,
                                ops2_roi_pixels=diag.get('roi_pixels', ''), ops2_roi_landmark_failure=diag.get('roi_landmark_failure', ''),
                                ops2_CPU_seconds=time.time() - tick))
                del output, view, u8, x, y
        del face, roi, notched, physview, roiprep
    return out


def main():
    backbone, mode, out_root = sys.argv[1], sys.argv[2], sys.argv[3]
    items = read_csv(VENDOR / gen_pool.MANIFEST[backbone])
    items = [r for r in items if str(r.get('generation_excluded', 'False')) != 'True']
    if os.environ.get('OPS2_LAB_ITEMS'):  # test subset: csv with chunk_id, augmentation_index
        keep = {(r['chunk_id'], r['augmentation_index']) for r in read_csv(Path(os.environ['OPS2_LAB_ITEMS']))}
        items = [r for r in items if (r['chunk_id'], r['augmentation_index']) in keep]
    by_rec = {}
    for r in items:
        by_rec.setdefault(r['recording_id'], []).append(r)
    workers = int(os.environ.get('OPS2_WORKERS', '6'))
    started = time.time()
    rows = []
    jobs = [(backbone, mode, out_root, rid, its) for rid, its in sorted(by_rec.items(), key=lambda kv: -len(kv[1]))]
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'), initializer=init_worker,
                             initargs=(backbone,)) as ex:
        for n, out in enumerate(ex.map(job, jobs), 1):
            rows += out
            print(f'lab {backbone} {mode} {n}/{len(jobs)} recordings, {len(rows)} items, {time.time() - started:.0f}s', flush=True)
    for m in mode.split(','):
        mr = [r for r in rows if r['mode'] == m]
        write_rows(Path(out_root) / f'{backbone}_{m}_all_manifest.csv', mr)
        print('done', m, len(mr), 'changed', sum(r['target_changed'] for r in mr))


if __name__ == '__main__':
    main()
