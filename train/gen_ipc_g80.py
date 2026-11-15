"""Brief 02 6.43 라: (i-pc-g80) PhysNet items: the (i-pc) arm with the (ii-pc-g80) target mapping (CPU).

(i-pc) = the archived PhysNet (i) item (src/st9g/pool_worker.py arm i: shifted_start source positions, frame interpolation
with stochastic rounding rng [20261137, rounding_h], PhysNet HC view, DiffNormalized input) with the label read at the
source positions + d(r - 1), d = +2 (train/gen_i_pc.py 'd', edge hold outside the contact signal).
(i-pc-g80): target h* < 100 -> 70 + (h* - 70)/3, so r = h*/h0 changes; the item is regenerated with the new r and the same
rounding key. Items whose target does not change are not written: they are the (i-pc) items (stored (i) input, (i-pc)
label). Storage: uint8 view frames + raw float64 label (train/u8_store.py), preprocessed at load (PhysNet, full-chunk
DiffNormalized, as the archived generator).
Route check (OPS2_IPC_VALIDATE=N): the first N unchanged items are regenerated through this same route with their own r
and compared with the stored (i) input SHA and the (i-pc) label SHA (cache/pools_i/PhysNet_i_d_all_manifest.csv).
Usage: gen_ipc_g80.py OUT_ROOT  -> OUT_ROOT/PhysNet/i-pc-g80/<rid>/..., OUT_ROOT/PhysNet_i-pc-g80_all_manifest.csv
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
from a1_common import *  # noqa: F401,F403
import u8_store  # noqa: E402
import gen_lab  # noqa: E402
from a3_i_lag import resample_video  # noqa: E402

MAIN_OUT = Path('<MACHINE_PATH>')  # read-only
D_FRAMES = 2
W = None


def init_worker():
    global W
    W = ViewWorker()
    atexit.register(W.close)


def job(args):
    out_root, rid, items, validate = args
    row = pure_index()[rid]
    files = frame_files(row)
    gt = contact(row, 'PURE', None)
    tau = int(tau_table()[rid]['tau_rec'])
    n, ng = len(files), len(gt)
    grid = np.arange(ng)
    physbox = W.view(read_pure(files, 0, 1), 'PhysNet')[1]
    folder = Path(out_root) / 'PhysNet' / 'i-pc-g80' / rid
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for r in items:
        tick = time.time()
        c, k, length = int(r['chunk_order']), int(r['augmentation_index']), int(r['chunk_length'])
        h0, t0 = float(r['reference_HR']), float(r['target_HR'])
        t = t0 if validate else gen_lab.target_for('pc_g80', t0, h0)
        scale = t / h0
        positions, plan = shifted_start(int(r['start_frame']), length, n, scale, tau)
        assert positions is not None
        sourcepos = positions[0]
        labelpos = sourcepos + D_FRAMES * (scale - 1)
        outside = max(0.0, float(-labelpos.min()), float(labelpos.max() - (ng - 1)))
        label = np.interp(np.clip(labelpos, 0, ng - 1), grid, gt)
        video = resample_video(files, sourcepos, np.random.default_rng([20261137, int(r['rounding_h'])]))
        view, _ = W.view(video, 'PhysNet', reuse_boxes=physbox)
        u8 = u8_store.to_u8(view)
        tag = 'check' if validate else 'k'
        vp = u8_store.save_view(folder / f'c{c}_{tag}{k}_view', u8)
        lp = u8_store.save_array(folder / f'c{c}_{tag}{k}_rawlabel', np.asarray(label, dtype=np.float64))
        x, y = u8_store.model_input('PhysNet', u8_store.read_array(vp), u8_store.read_array(lp), 0, length)
        rec = dict(r, mode='i-pc-g80', target_HR_original=t0, target_HR=t, target_changed=bool(t != t0), validate=validate,
                   scale_new=scale, source_start_new=float(plan['source_start']), label_shift_source_frames_new=D_FRAMES * (scale - 1),
                   edge_hold_samples=int(((labelpos < 0) | (labelpos > ng - 1)).sum()), edge_hold_max_frames=outside,
                   ops2_view_path=str(vp), ops2_rawlabel_path=str(lp), ops2_stats_path='',
                   ops2_view_bytes=vp.stat().st_size, ops2_rawlabel_bytes=lp.stat().st_size,
                   ops2_input_sha256=u8_store.npy_sha(x), ops2_label_sha256=u8_store.npy_sha(y),
                   ops2_u8_data_sha256=u8_store.data_sha(u8), view_is_integer=bool(np.array_equal(np.asarray(view), u8)),
                   ops2_CPU_seconds=time.time() - tick)
        if validate:
            rec['input_sha_equals_i'] = rec['ops2_input_sha256'] == r['input_sha256']
            rec['label_sha_equals_i_pc'] = rec['ops2_label_sha256'] == r['ipc_label_sha256']
            for p in (vp, lp):
                p.unlink()
        out.append(rec)
        del video, view, u8, x, y
    return out


def main():
    out_root = sys.argv[1]
    items = read_csv(MAIN_OUT / 'st9g_pool_PhysNet_i_manifest.csv')
    items = [r for r in items if r['model'] == 'PhysNet' and r['arm'] == 'i' and r['generation_excluded'] != 'True']
    ipc = {(r['chunk_id'], r['augmentation_index']): r['ops2_label_sha256'] for r in read_csv(WORK / 'cache/pools_i/PhysNet_i_d_all_manifest.csv')}
    validate_n = int(os.environ.get('OPS2_IPC_VALIDATE', '0'))
    if validate_n:
        sel = [dict(r, ipc_label_sha256=ipc[(r['chunk_id'], r['augmentation_index'])]) for r in items
               if gen_lab.target_for('pc_g80', float(r['target_HR']), float(r['reference_HR'])) == float(r['target_HR'])][:validate_n]
        validate = True
    else:
        sel = [r for r in items if gen_lab.target_for('pc_g80', float(r['target_HR']), float(r['reference_HR'])) != float(r['target_HR'])]
        validate = False
    by_rec = {}
    for r in sel:
        by_rec.setdefault(r['recording_id'], []).append(r)
    started, rows = time.time(), []
    jobs = [(out_root, rid, its, validate) for rid, its in sorted(by_rec.items(), key=lambda kv: -len(kv[1]))]
    with ProcessPoolExecutor(max_workers=int(os.environ.get('OPS2_WORKERS', '4')), mp_context=mp.get_context('spawn'),
                             initializer=init_worker) as ex:
        for i, o in enumerate(ex.map(job, jobs), 1):
            rows += o
            print(f'ipc-g80 {i}/{len(jobs)} recordings, {len(rows)} items, {time.time() - started:.0f}s', flush=True)
    name = 'PhysNet_i-pc-g80_route_check.csv' if validate else 'PhysNet_i-pc-g80_all_manifest.csv'
    write_rows(Path(out_root) / name, rows)
    if validate:
        print('route check', len(rows), 'input_sha_equals_i', sum(r['input_sha_equals_i'] for r in rows),
              'label_sha_equals_i_pc', sum(r['label_sha_equals_i_pc'] for r in rows))
    else:
        print('done', len(rows), 'changed', sum(r['target_changed'] for r in rows))


if __name__ == '__main__':
    main()
