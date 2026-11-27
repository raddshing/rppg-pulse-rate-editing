"""7.09b E1: UBFC-rPPG editing pools (ii), (ii-pc), (ii-lag), from cache/ubfc709/plan/<backbone>_items.csv.

The per-item path of train/gen_pool.py (job), with UBFC inputs: frames and contact PPG as train/gen_ubfc.py (frame units,
fs = 30); face masks, notch, calibration in the PhysNet view, ED2W-S edit, stochastic rounding, label raw_target_label,
uint8 storage with context statistics (train/u8_store.pack). Corpus values (7.09b 가-1): direction v of UBFC (October
outputs/st31u_direction.csv, 59 UBFC windows); tau = the October UBFC delay-rule value of the recording; (ii) injects
+g*(t - tau), (ii-pc) -g*(t - 1) (s = -1, d = 1), (ii-lag) -g*(t - tau - T*/2). The amplitude A is computed from
g*(t - tau) in every mode, as in gen_pool. Rounding generator of item h: default_rng([20261159, j, h]), fresh per mode.
NOTCH (argument): 'source' = bands_for(h0) (f0, 2f0; computed once per chunk context, as the PURE pools) or 'target' =
bands_for(h0, h*) (f0, 2f0, f*, 2f*, per item, as the October stage-5 gate). A9 fixes which one the pools use.
Usage: gen_ubfc_ii.py BACKBONE MODES NOTCH OUT_ROOT [RECORDING ...]   (MODES: comma list of original,
polarity_ubfc,lag_shift; OPS2_WORKERS, OPS2_POOL_COMPRESS)
"""
import hashlib
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gen_ubfc as GU  # noqa: E402  (a1_common, UBFCReader, contact_ubfc, init_worker, ViewWorker, RhythmView, u8_store)
sys.path.insert(0, str(HERE.parent / 'audit'))
import editor_pc as E  # noqa: E402
np = GU.np
u8_store = GU.u8_store
D_UBFC, S_UBFC = 1, -1
import json  # 7.19
CFG = json.loads((Path(__file__).resolve().parents[1] / 'config' / 'paths.json').read_text())  # 7.19: machine paths, see config/paths.example.json
DIRECTION = str(Path(CFG['october_project']) / 'outputs' / 'st31u_direction.csv')
ARM = {'original': '(ii)', 'polarity_ubfc': '(ii-pc)', 'lag_shift': '(ii-lag)'}


def direction_ubfc():
    d = GU.read_csv(Path(DIRECTION))[0]
    return [float(d['v_' + c]) for c in 'RGB']


def inject_of(mode, gt, cs, h0, target, tau):
    if mode == 'original':
        return None
    if mode == 'polarity_ubfc':
        return S_UBFC * E.ppg_vectors(gt, cs, h0, target, D_UBFC)[1]
    assert mode == 'lag_shift'
    return E.lag_shift_wave(gt, cs, h0, target, tau)


def file_sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def job(args):
    backbone, modes, notch_kind, out_root, rid, items = args
    modes = modes.split(',')
    r0 = items[0]
    n = int(r0['frame_count'])
    native = GU.READER.read(r0['source_video_path'], 0, n)
    gt = GU.contact_ubfc(r0['ground_truth_path'], n)
    v = direction_ubfc()
    tau = int(r0['tau_rec'])
    W = GU.W if backbone != 'RhythmFormer' else None
    if backbone == 'RhythmFormer':
        phys = E.ViewWorker()
        phys_all, physboxes = phys.view(native, 'PhysNet')
        rboxes = GU.R.view(native[:1])[1]
    else:
        phys = None
        phys_all, physboxes = W.view(native, 'PhysNet')
    folders = {m: Path(out_root) / backbone / f'{m}_{notch_kind}' / rid for m in modes}
    for f in folders.values():
        f.mkdir(parents=True, exist_ok=True)
    out = []
    for c in sorted({int(r['chunk_order']) for r in items}):
        current = [r for r in items if int(r['chunk_order']) == c]
        ref = current[0]
        cs, off, length, h0 = int(ref['context_start']), int(ref['context_offset']), int(ref['chunk_length']), float(ref['reference_HR'])
        tick = time.time()
        clip = native[cs:cs + 300]
        face, roi, failure = E.face_masks(clip)
        notched_src = E.notch_all(clip, E.bands_for(h0)) if notch_kind == 'source' else None
        physview = phys_all[cs:cs + 300]
        for r in current:
            k, target = int(r['augmentation_index']), float(r['target_HR'])
            notched = notched_src if notch_kind == 'source' else E.notch_all(clip, E.bands_for(h0, target))
            g0, gstar = E.ppg_vectors(gt, cs, h0, target, tau)
            pulse, w, beta = E.calibration(physview, E.crop_native(notched, physboxes[0]), E.crop_native(face, physboxes[0]), g0, v)
            for mode in modes:
                rng = np.random.default_rng([GU.KEY, GU.J[backbone], int(r['rounding_h'])])
                output, diag = E.edited(clip, notched, face, roi, g0, gstar, v, h0, 'ED2W-S', pulse, w, r['chunk_id'], 'ii-h',
                                        rounding_rng=rng, inject=inject_of(mode, gt, cs, h0, target, tau))
                raw_label = np.asarray(E.raw_target_label(gt, cs, h0, target))
                assert raw_label.dtype == np.float64 and raw_label.shape == (300,)
                if backbone == 'RhythmFormer':
                    view, _ = GU.R.view(output, rboxes[0])
                else:
                    view, _ = W.view(output, 'PhysNet', reuse_boxes=physboxes)
                u8, stats = u8_store.pack(backbone, u8_store.to_u8(view), off, length)
                folder = folders[mode]
                vp = u8_store.save_view(folder / f'c{c}_k{k}_view', u8)
                lp = u8_store.save_array(folder / f'c{c}_k{k}_rawlabel', raw_label)
                sp = None if stats is None else u8_store.save_array(folder / f'c{c}_k{k}_stats', stats)
                x, y = u8_store.model_input(backbone, u8_store.read_array(vp), u8_store.read_array(lp), off, length,
                                            None if sp is None else u8_store.read_array(sp))
                out.append(dict(r, mode=mode, arm=ARM[mode], notch=notch_kind, ops2_view_path=str(vp), ops2_rawlabel_path=str(lp),
                                ops2_stats_path='' if sp is None else str(sp), ops2_view_bytes=vp.stat().st_size,
                                ops2_view_file_sha256=file_sha(vp), ops2_input_sha256=u8_store.npy_sha(x),
                                ops2_label_sha256=u8_store.npy_sha(y), injection_A=diag['injection_A'], beta=beta,
                                landmark_missing_fraction=failure, achieved_label_HR=GU.common_hr(raw_label, 'HAR'),
                                generation_excluded=False, ops2_CPU_seconds=time.time() - tick))
                del output, view, u8, x, y
        del face, roi, notched_src, physview
    if phys is not None:
        phys.close()
    return out


def main():
    backbone, modes, notch_kind, out_root = sys.argv[1], sys.argv[2], sys.argv[3], Path(sys.argv[4])
    only = set(sys.argv[5:])
    assert notch_kind in ('source', 'target') and all(m in ARM for m in modes.split(','))
    rows = GU.read_csv(GU.PLAN / f'{backbone}_items.csv')
    by_rec = {}
    for r in rows:
        if not only or r['recording_id'] in only:
            by_rec.setdefault(r['recording_id'], []).append(r)
    workers = int(os.environ.get('OPS2_WORKERS', '4'))
    jobs = [(backbone, modes, notch_kind, str(out_root), rid, its) for rid, its in sorted(by_rec.items(), key=lambda kv: -len(kv[1]))]
    started, out = time.time(), []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'), initializer=GU.init_worker,
                             initargs=(backbone,)) as ex:
        for i, o in enumerate(ex.map(job, jobs), 1):
            out += o
            print(f'ubfc ii {backbone} {notch_kind} {i}/{len(jobs)} recordings, {len(out)} rows, {time.time() - started:.0f}s', flush=True)
    tag = 'all' if not only else 'part'
    for m in modes.split(','):
        mr = [r for r in out if r['mode'] == m]
        GU.write_rows(out_root / f'{backbone}_{ARM[m].strip("()")}_{notch_kind}_{tag}_manifest.csv', mr)
        print('done', ARM[m], notch_kind, len(mr), flush=True)


if __name__ == '__main__':
    main()
