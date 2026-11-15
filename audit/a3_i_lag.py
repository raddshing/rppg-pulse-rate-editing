"""Brief 02 6.10: label lag of the (i) time-resampling items, 1a step-3 method (CPU only, no training).

For each of the 120 1a sample chunks/targets, the matching PhysNet (i) pool item is regenerated along the
archived pool path (6ebbc9cd src/st9g/pool_worker.py, arm 'i'); the lag/sign between the video pulsation and
the label p(rt) is measured on the generator's 300-frame achieved-HR context, next to the original context of the
same chunk. Positive controls follow 1a. Method fixed before computing: reports/1a_i_method.md.
"""
import atexit
import hashlib
import io
import sys
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a1_common import *  # noqa: F401,F403

MAIN_OUT = Path('<MACHINE_PATH>')  # read-only manifest
OUT_I = WORK / 'outputs' / '1a_i'
OUT_I.mkdir(parents=True, exist_ok=True)
CONTROLS = [(-10, 1), (-10, -1), (5, 1), (5, -1)]
W = None


def init_worker():
    global W
    W = ViewWorker()
    atexit.register(W.close)


def npy_sha(x):
    buf = io.BytesIO()
    np.save(buf, x)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def select_items():
    sample = read_csv(OUT / 'edit_sample.csv')
    manifest = {(r['chunk_id'], r['augmentation_index']): r for r in read_csv(MAIN_OUT / 'st9g_pool_PhysNet_i_manifest.csv')}
    items = []
    for s in sample:
        r = manifest[(s['chunk_id'], s['augmentation_index'])]
        assert float(r['target_HR']) == float(s['target_HR']) and float(r['reference_HR']) == float(s['reference_HR'])
        assert r['generation_excluded'] == 'False'
        items.append(dict(r, r_bin=s['r_bin'], bin_order=int(s['bin_order']), control=s['control'] == 'True'))
    return items


def resample_video(files, sourcepos, rng):
    lo, hi = int(np.floor(sourcepos.min())), int(np.ceil(sourcepos.max()))
    src = read_pure(files, lo, hi - lo + 1)
    out = np.empty((len(sourcepos),) + src.shape[1:], np.uint8)
    for i, p in enumerate(sourcepos - lo):
        left, right = int(np.floor(p)), int(np.ceil(p))
        fraction = p - left
        value = (1 - fraction) * src[left].astype(np.float64) + fraction * src[right]
        out[i] = np.floor(value + rng.random(value.shape)).astype(np.uint8)
    return out


def fp_view(files, frames, source_index):
    """FactorizePhys view with Y5F 30-frame boxes of the source frames (generator rule rint(position)//30)."""
    aligned = int(source_index.min()) // 30 * 30
    _, ob = W.view(read_pure(files, aligned, int(source_index.max()) + 1 - aligned), 'FactorizePhys')
    boxes = np.asarray(ob)[source_index // 30 - aligned // 30]
    view, _ = W.view(frames, 'FactorizePhys', reuse_boxes=boxes, reuse_boxes_per_frame=True)
    return view


def reflect(pos, n):
    pos = np.asarray(pos, float)
    pos = np.where(pos < 0, -pos, pos)
    return np.where(pos > n - 1, 2 * (n - 1) - pos, pos)


def job(item):
    t0 = time.time()
    pure = pure_index()
    rid = item['recording_id']
    row = pure[rid]
    files = frame_files(row)
    gt = contact(row, 'PURE', None)
    v = direction()
    original, target, tau = float(item['reference_HR']), float(item['target_HR']), int(item['tau_rec'])
    scale = target / original
    length, h, cs = int(item['chunk_length']), int(item['rounding_h']), int(item['context_start'])
    _, physbox = W.view(read_pure(files, 0, 1), 'PhysNet')

    # V3: exact 128-frame stored item (generator rule, n = number of frames)
    positions, plan = shifted_start(int(item['start_frame']), length, len(files), scale, tau)
    src128, lab128 = positions
    chunk = resample_video(files, src128, np.random.default_rng([20261137, h]))
    label128 = np.interp(lab128, np.arange(len(gt)), gt)
    pn128, _ = W.view(chunk, 'PhysNet', reuse_boxes=physbox)
    x = np.ascontiguousarray(diff_normalize_data(pn128)[0:length], dtype=np.float32)
    v3 = dict(V3_sha_match=npy_sha(x) == item['input_sha256'], V3_source_start_match=plan['source_start'] == float(item['source_start']),
              label_shift_source_frames=plan['label_shift_source_frames'])

    # 300-frame achieved-HR context (generator rule, n = len(gt))
    ctx, ctxplan = shifted_start(plan['source_start'] + (length - 300) * scale / 2, 300, len(gt), scale, tau)
    assert ctx is not None
    srcpos, labpos = ctx
    label = np.interp(labpos, np.arange(len(gt)), gt)
    achieved = common_hr(label, 'HAR')
    v4 = dict(V4_achieved_HR=achieved, V4_achieved_HR_manifest=float(item['achieved_context_HR']),
              V4_match=bool(np.isclose(achieved, float(item['achieved_context_HR']), rtol=0, atol=1e-9)),
              ctx_source_start=ctxplan['source_start'])
    out = resample_video(files, srcpos, np.random.default_rng([20261137, h]))
    q = editor_band(label)
    face, roi, failure = face_masks(out)
    pn, _ = W.view(out, 'PhysNet', reuse_boxes=physbox)
    fp = fp_view(files, out, np.rint(srcpos).astype(int))

    # same chunk's original 300-frame context
    clip = read_pure(files, cs)
    face_o, _, failure_o = face_masks(clip)
    pn_o, _ = W.view(clip, 'PhysNet', reuse_boxes=physbox)
    fp_o = fp_view(files, clip, cs + np.arange(300))
    p_nat = editor_band(gt[cs:cs + 300])

    base = dict(recording_id=rid, subject_id=item['subject_id'], chunk_id=item['chunk_id'],
                augmentation_index=item['augmentation_index'], context_start=cs, original_HR=original, target_HR=target,
                r=scale, r_bin=item['r_bin'], bin_order=item['bin_order'], tau_rec=tau,
                design_label_shift_output_frames=plan['label_shift_source_frames'] / scale,
                mask_failure_fraction_i=failure, mask_failure_fraction_original=failure_o, **v3, **v4)
    rows = []
    for view_name, yi, yo in [('PhysNet_view', view_signal(pn), view_signal(pn_o)),
                              ('FactorizePhys_view', view_signal(fp), view_signal(fp_o)),
                              ('face_mask_native', editor_band(green(weighted_rgb(out, face))),
                               editor_band(green(weighted_rgb(clip, face_o))))]:
        rows.append(dict(base, view=view_name, signal='i_item', **summarize(lag_curve(yi, q))))
        rows.append(dict(base, view=view_name, signal='natural_original', **summarize(lag_curve(yo, p_nat))))
    rows.append(dict(base, view='PhysNet_view', signal='i_item_stored128',
                     **summarize(lag_curve(view_signal(pn128), editor_band(label128)))))

    controls = []
    if item['control']:
        lead = labpos[0] + (np.arange(330) - 15) * scale
        g = normalize(np.interp(reflect(lead, len(gt)), np.arange(len(gt)), gt))
        reflected = int(((lead < 0) | (lead > len(gt) - 1)).sum())
        g0 = g[np.arange(300) + 15]
        pulse, w, beta = calibration(pn, crop_native(out, physbox[0]), crop_native(face, physbox[0]), g0, v)
        twin, _ = edited(out, out, face, roi, g0, g0, v, target, 'ED2W-S', 0., w, item['chunk_id'], 'i',
                         rounding_rng=np.random.default_rng([20261137, h]))
        sig_t = view_signal(W.view(twin, 'PhysNet', reuse_boxes=physbox)[0])
        for lag0, s in CONTROLS:
            goal = s * g[np.arange(300) + 15 - lag0]
            outc, diag = edited(out, out, face, roi, g0, goal, v, target, 'ED2W-S', pulse, w, item['chunk_id'], 'i',
                                rounding_rng=np.random.default_rng([20261137, h]))
            pn_c, _ = W.view(outc, 'PhysNet', reuse_boxes=physbox)
            res = summarize(lag_curve(view_signal(pn_c) - sig_t, q))
            controls.append(dict(base, known_lag=lag0, known_sign=s, injection_A=diag['injection_A'], beta=beta,
                                 reflected_label_samples=reflected, **res,
                                 lag_recovered=res['L_abs'] is not None and abs(res['L_abs'] - lag0) <= 1,
                                 sign_recovered=res['sign'] == s))
            del outc, pn_c
    for r in rows + controls:
        r['item_seconds'] = time.time() - t0
    return rows, controls


def main():
    items = select_items()
    workers = int(os.environ.get('OPS2_WORKERS', '2'))
    started = time.time()
    rows, controls = [], []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'), initializer=init_worker) as ex:
        for n, (r, c) in enumerate(ex.map(job, items), 1):
            rows += r
            controls += c
            if n % 6 == 0 or n == len(items):
                print(f'i-lag {n}/{len(items)} items, {time.time() - started:.0f}s', flush=True)
    write_rows(OUT_I / 'i_items.csv', rows)
    write_rows(OUT_I / 'i_positive_controls.csv', controls)
    print('done', len(rows), len(controls))


if __name__ == '__main__':
    main()
