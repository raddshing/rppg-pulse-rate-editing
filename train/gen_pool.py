"""(ii)/(ii-pc) pool items for PhysNet, RhythmFormer and matched FactorizePhys (brief 02, 2.1/2.4).

Per-item steps mirror the archived generators exactly:
  PhysNet  st9g/pool_worker.py (arm ii-h): rng [20261137,h]; PhysNet HC view; DiffNormalized input/label
  RF       st12c/rhythm_pool_worker.py:     rng [20261137,h]; RF HC 128 view; Standardized input/label
  FPm      st30g/pool_A_worker.py (arm ii): rng [20261202,h]; PhysNet HC 72 view; Raw input, Standardized label
Only the injected waveform differs in modes 'polarity_explicit' (ii-pc), 'polarity_inverted' (ii-inv) and
'lag_shift' (ii-lag) (editor_pc.edited inject argument).
Storage (brief 02 sections 6.1/6.6): kept uint8 frames of the editor output's model view, the
context statistics and the raw float64 label (train/u8_store.py); model preprocessing at load time. Each item is reloaded
and preprocessed here, and its tensor SHA is compared with the stored (ii) input SHA.
MODE may list several modes separated by commas (brief 02 6.19: notch, masks and calibration computed once per item;
each mode gets a fresh rounding generator with the item's key, so every mode equals its single-mode run).
Usage: gen_pool.py BACKBONE MODE[,MODE...] OUT_ROOT [ITEM_LIST.csv] ; writes OUT_ROOT/BACKBONE/MODE/<rid>/c{c}_k{k}_{view,rawlabel}.np[yz]
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
from editor_pc import *  # noqa: F401,F403
from infer_client import RhythmView  # noqa: E402
sys.path.insert(0, str(WORK / 'train'))
import u8_store  # noqa: E402

MANIFEST = {'PhysNet': 'outputs/st9g_pool_PhysNet_ii-h_manifest.csv', 'RhythmFormer': 'outputs/st12c_a_pool_manifest.csv',
            'FPm': 'outputs/st30g_FPm_ii_manifest.csv'}
SEED = {'PhysNet': 20261137, 'RhythmFormer': 20261137, 'FPm': 20261202}
W = R = None


def init_worker(backbone):
    global W, R
    W = ViewWorker()
    atexit.register(W.close)
    if backbone == 'RhythmFormer':
        R = RhythmView()
        atexit.register(R.close)


def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def job(args):
    backbone, modes, out_root, rid, items = args
    modes = modes.split(',')
    row = pure_index()[rid]
    files = frame_files(row)
    gt = contact(row, 'PURE', None)
    v = direction()
    tau = int(tau_table()[rid]['tau_rec'])
    native = read_pure(files, 0, len(files))
    phys_all, physboxes = W.view(native, 'PhysNet')
    rboxes = R.view(native[:1])[1] if backbone == 'RhythmFormer' else None
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
        for r in current:
            k, target = int(r['augmentation_index']), float(r['target_HR'])
            g0, gstar, _ = injection_wave(gt, cs, h0, target, tau, 'original')
            pulse, w, beta = calibration(physview, crop_native(notched, physboxes[0]), crop_native(face, physboxes[0]), g0, v)
            arm = 'ii' if backbone == 'FPm' else 'ii-h'
            for mode in modes:
              folder = folders[mode]
              rng = np.random.default_rng([SEED[backbone], int(r['rounding_h'])])
              _, _, inject = injection_wave(gt, cs, h0, target, tau, mode)
              output, diag = edited(clip, notched, face, roi, g0, gstar, v, h0, 'ED2W-S', pulse, w, r['chunk_id'], arm,
                                    rounding_rng=rng, inject=inject)
              raw_label = np.asarray(raw_target_label(gt, cs, h0, target))
              assert raw_label.dtype == np.float64 and raw_label.shape == (300,)
              if backbone == 'RhythmFormer':
                  view, _ = R.view(output, rboxes[0])
              else:
                  view, _ = W.view(output, 'PhysNet', reuse_boxes=physboxes)
              u8, stats = u8_store.pack(backbone, u8_store.to_u8(view), off, length)
              vp = u8_store.save_view(folder / f'c{c}_k{k}_view', u8)
              lp = u8_store.save_array(folder / f'c{c}_k{k}_rawlabel', raw_label)
              sp = None if stats is None else u8_store.save_array(folder / f'c{c}_k{k}_stats', stats)
              # reload exactly as the training loader does, then compare with the stored (ii) tensors
              x, y = u8_store.model_input(backbone, u8_store.read_array(vp), u8_store.read_array(lp), off, length,
                                          None if sp is None else u8_store.read_array(sp))
              xs, ys = u8_store.npy_sha(x), u8_store.npy_sha(y)
              out.append(dict(r, mode=mode, ops2_view_path=str(vp), ops2_rawlabel_path=str(lp),
                              ops2_stats_path='' if sp is None else str(sp),
                              ops2_view_bytes=vp.stat().st_size, ops2_rawlabel_bytes=lp.stat().st_size,
                              ops2_stats_bytes=0 if sp is None else sp.stat().st_size,
                              ops2_view_file_sha256=file_sha(vp), ops2_input_sha256=xs, ops2_label_sha256=ys,
                              ops2_input_data_sha256=u8_store.data_sha(x), ops2_label_data_sha256=u8_store.data_sha(y),
                              input_sha_equals_original=xs == r['input_sha256'], label_sha_equals_original=ys == r['label_sha256'],
                              ops2_injection_A=diag['injection_A'], ops2_beta=beta, ops2_tau_rec=tau, d=D_FRAMES if mode in SIGN else ('tau+T*/2' if mode == 'lag_shift' else ''),
                              s=SIGN.get(mode, -1) if mode != 'original' else '', ops2_CPU_seconds=time.time() - tick))
              del output, view, u8, x, y
        del face, roi, notched, physview
    return out


def main():
    backbone, mode, out_root = sys.argv[1], sys.argv[2], sys.argv[3]
    items = read_csv(VENDOR / MANIFEST[backbone])
    items = [r for r in items if str(r.get('generation_excluded', 'False')) != 'True']
    if len(sys.argv) > 4:
        keep = {(r['chunk_id'], r['augmentation_index']) for r in read_csv(sys.argv[4])}
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
            print(f'pool {backbone} {mode} {n}/{len(jobs)} recordings, {len(rows)} items, {time.time() - started:.0f}s', flush=True)
    tag = Path(sys.argv[4]).stem if len(sys.argv) > 4 else 'all'
    for m in mode.split(','):
        mr = [r for r in rows if r['mode'] == m]
        write_rows(Path(out_root) / f'{backbone}_{m}_{tag}_manifest.csv', mr)
        print('done', m, len(mr), 'input_sha_equals_original', sum(r['input_sha_equals_original'] for r in mr),
              'label_sha_equals_original', sum(r['label_sha_equals_original'] for r in mr))


if __name__ == '__main__':
    main()
