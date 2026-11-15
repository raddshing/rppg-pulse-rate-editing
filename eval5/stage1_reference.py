"""Stage 1 of the Stage-5 evaluation: windows, reference HR, high-HR flag, populations, background flags and bootstrap
indices (ACCESS_OPS2_BRIEF_02 6.43 마, 6.45 나 4, 6.46 가 5, 6.48 라). No model, no prediction.

Seal guard: as stage 2 (eval5/stage2_predict.py seal_guard), before any data file is opened.

MMPD (the October MMPD-B path, st13b/prepare -> st11x.future.mmpd_b_references + st11x.pathways.mmpd_b_membership, commit
6ebbc9cd, paths changed): per recording of the input plan, the contact signal ('GT_ppg' of the provider .mat, 30 Hz) in five
300-sample windows; common HR (HAR, L0) and the reference estimators/classification of src/st4/signals.py, the defect map
d1-d4 of src/st3/c4.py, the high-frequency ratio T1b_high_freq (threshold 0.10676), and the automatic Q* (defect-free and
reference class 'agreement'; L0 also |h_L0 - h_HAR| <= one grid step). No PI-confirmation path. The only change to the
October function is that the fixed population check (435 recordings / 22 subjects) is an argument
(--expected-recordings/--expected-subjects): the rehearsal passes MMPD-A (220 / 11).
  Background flags (Amendment A2; tools/bg_spectrum.py and tools/b2_flag_robust.py flags(), unchanged rule): per recording
  the mean R, G, B outside the PhysNet HC face box of frame 0, Welch spectrum (300/150, nfft 512), unit power in 0.6-3.3 Hz;
  flagged = a local maximum with contrast >= 2 in bins 33-35 or 50-52; no background pixels -> 'no_background'.
MCD-rPPG (the 1b model-free pass, b1/mcd_meta.py = src/st5c mcd_packets/mcd_align at b81f6c9e6, unchanged): per recording
of the list (one per subject and state, front FullHDwebcam; db.csv rows of other subjects dropped before use), frame-count
class, raw-PPG clock groups, label-label alignment and the S5-3M2 10-s windows with reference estimates; GOP per recording
(packet VOP types; tools/mcd_gop.py). Populations: draft rule = recordings with a confirmed frame-to-PPG mapping and at
least one wide-band (HAR) primary window with reference class agreement and reference HR >= 100; windows high = agreement
and >= 100, low = < 100; M7 = subjects with both states confirmed, all primary windows (b1/analyze.py load_mcd).
Bootstrap indices: st11x.pathways.paired_bootstrap_indices (subjects in numeric order, default_rng(20261110), 10,000 x
subjects, then 10,000 x seeds), for 3 and 5 seeds, per population.
Contact signals for the recording-level benchmark score (brief 02 6.46 가 7, 6.54 다): labels/<recording>.npy (float64) on
the stage-2 prediction time axis and labels.csv (samples, finite samples, first/last finite index).
  MMPD: GT_ppg of the provider .mat (one sample per video frame), the whole recording.
  MCD-rPPG: recordings with a confirmed frame-to-PPG mapping; the synced contact PPG (ppg_sync column 0) at the meta frame
  times meta[k* : k* + len(sync)] (b1/mcd_windows.py), linearly interpolated to the stage-2 30 Hz input grid
  (uniform_frame_grid of [0, span], span from the meta file as in stage 2); grid points outside the mapped times are NaN.
  --labels-only writes only these files (and labels_hashes.txt) into an existing stage-1 output (MCD: its recordings.csv).
First-frame face detections (brief 02 6.67 다 4; video only): per recording, the first frame exactly as stage 2 prepares it
(MMPD round(video * 255) uint8 frame 0; MCD-rPPG first decoded frame, BGR -> RGB) and the Haar cascade call of the
rPPG-Toolbox face_detection ('HC': detectMultiScale on the uint8 frame, the toolbox's cascade file) with OpenCV on one thread
and no GPU visible (stage 2's CPU path);
the widths of all detections; recordings with two or more detections of the largest width (the tie that the order of
detections decides) and with two or more detections of any equal width are counted (first_frame_detections.csv, run.json).
Outputs (--out/<dataset>/): references.csv, recordings.csv, populations.csv, counts.csv, bg_flags.csv (MMPD), gop.csv (MCD),
bootstrap_<population>_<seeds>.npz, labels/, labels.csv, stage1_hashes.txt ('SHA-256  relative name'), run.json.
"""
import os  # noqa: E402
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
import argparse  # noqa: E402
import ast  # noqa: E402
import csv  # noqa: E402
import getpass  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
import multiprocessing as mp  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stage2_predict as S2  # noqa: E402  (seal guard, lists, hashes)

STEP = 30 / 512 * 60  # 3.515625 bpm
LOW, HIGH = (33, 34, 35), (50, 51, 52)
VENDOR = None
FIRST = [None]  # first-frame detection counts of this run (6.67 다 4)


def vendor_imports(vendor):
    global VENDOR
    VENDOR = Path(vendor).resolve()
    os.environ.setdefault('ST9_NAMESPACE', 'st9g')
    for p in (str(VENDOR), str(VENDOR / 'third_party' / 'rPPG-Toolbox')):
        if p not in sys.path:
            sys.path.insert(0, p)


# ------------------------------------------------------------------------------------------------------------- MMPD
def defect_map_function():
    """The exact src/st3/c4.py defect_map with its output hooks replaced (as st11x.future.mmpd_b_references)."""
    source = VENDOR / 'src/st3/c4.py'
    tree = ast.parse(source.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'defect_map')
    context = dict(np=np, pd=pd, write_rows=lambda *a, **k: None, update=lambda *a, **k: None)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), context)
    return context['defect_map']


def mmpd_references(recordings, load_contact, expected):
    """st11x.future.mmpd_b_references (commit 6ebbc9cd) with the population check as an argument."""
    from scipy.signal import periodogram, detrend
    from src.st11x.future import frozen_reference_functions, reference_rows
    from src.st11x.pathways import mmpd_b_membership
    if len(recordings) != expected[0] or len({r['subject_id'] for r in recordings}) != expected[1]:
        raise ValueError('Fixed population mismatch')
    helpers = frozen_reference_functions()
    defect_map = defect_map_function()
    out, segments = [], []
    for rec in recordings:
        raw = np.asarray(load_contact(rec))
        rows = []
        for j in range(5):
            window = raw[j * 300:j * 300 + 300]
            h = helpers['common_hr'](window, 'HAR')
            y = detrend(window.astype(np.float64), type='linear')
            f, p = periodogram(y, fs=30, window='hann', nfft=len(y), detrend=False, scaling='density')
            den = p[(f >= .6) & (f <= 15)].sum()
            ratio = float(p[(f > 3.5 * h / 60) & (f <= 15)].sum() / den) if den > 0 and np.isfinite(h) else np.nan
            rows.append(dict(window_id='MMPD|' + rec['recording_id'] + '|' + str(j), recording_id=rec['recording_id'],
                             subject_id=rec['subject_id'], window_index=j, start_seconds=10 * j, T1=-1, T1b=-1,
                             T1b_high_freq=int(ratio >= .10676) if np.isfinite(ratio) else -1, Q3_power_above_3p5_f0_ratio=ratio))
        frame = pd.DataFrame(rows)
        defects, parts = defect_map(frame, {rec['recording_id']: raw})
        segments.extend(parts)
        refs = reference_rows(raw, np.arange(len(raw)) / 30, rows)
        for r in refs:
            d = defects.loc[r['window_id']].to_dict()
            r.update(d)
            r.update(mmpd_b_membership(d['defect_any'], r.get('reference_class', 'unavailable'), r.get('h_FFT_HAR', np.nan), r.get('h_FFT_L0', np.nan)))
            r['PI_confirmation_path'] = False
            r['T1b_high_frequency_A_threshold'] = .10676
            out.append(r)
    return out, segments


def load_mmpd_contact(root, rid):
    from scipy.io import loadmat
    return loadmat(Path(root) / (rid + '.mat'), variable_names=['GT_ppg'])['GT_ppg'].T.reshape(-1)


def save_label(out, rid, values):
    """Contact signal of one recording on the stage-2 prediction time axis (labels/<recording>.npy, float64)."""
    d = out / 'labels'
    d.mkdir(exist_ok=True)
    np.save(d / (rid.replace('/', '__') + '.npy'), np.asarray(values, dtype=np.float64))


def mcd_label(row, k_star, span):
    """MCD-rPPG: synced contact PPG at the mapped meta frame times, interpolated to the stage-2 30 Hz grid (NaN outside)."""
    from src.st5c.mcd_align import timestamp_table
    from src.st11x.pathways import uniform_frame_grid
    _, stamps = timestamp_table(Path(row['meta']))
    meta = (stamps.astype(float) - float(stamps[0])) / 1e6
    sync = np.loadtxt(row['ppg_sync'], ndmin=2)[:, 0]
    mapped = meta[int(k_star):int(k_star) + len(sync)]
    grid = uniform_frame_grid(np.array([0.0, float(span)]))
    lab = np.interp(grid, mapped, sync)
    lab[(grid < mapped[0]) | (grid > mapped[-1])] = np.nan
    return lab


def mcd_labels(rows, recs, out):
    for r in recs.to_dict('records'):
        if bool(r['mapping_confirmed']):
            save_label(out, r['recording_id'], mcd_label(rows[r['recording_id']], r['k_star'], r['span_seconds']))
    label_table(out)


def label_table(out):
    rows = []
    for f in sorted((out / 'labels').glob('*.npy')):
        v = np.load(f)
        ok = np.flatnonzero(np.isfinite(v))
        rows.append(dict(recording_id=f.stem.replace('__', '/'), samples=len(v), finite_samples=len(ok),
                         first_finite=int(ok[0]) if len(ok) else '', last_finite=int(ok[-1]) if len(ok) else ''))
    S2.write_csv(out / 'labels.csv', rows)


def labels_only(args, out):
    """Contact signals only, into an existing stage-1 output (the same functions as the full run)."""
    if args.dataset == 'MMPD':
        plan = []
        with open(Path(args.plan_dir) / f'{args.plan_prefix}_FPm_input_plan.csv', newline='') as f:
            for r in csv.DictReader(f):
                if not plan or plan[-1] != r['recording_id']:
                    plan.append(r['recording_id'])
        keep = set(S2.read_subjects(args.subjects))
        assert all(r.split('/')[0] in keep for r in plan)
        for rid in plan:
            save_label(out, rid, load_mmpd_contact(args.data_root, rid))
        label_table(out)
    else:
        root = Path(args.data_root)
        keep = set(S2.read_subjects(args.subjects))
        recs = pd.read_csv(out / 'recordings.csv', dtype={'subject_id': str})
        assert all(s in keep for s in recs.subject_id)
        rows = {}
        with (root / 'db.csv').open(newline='') as f:
            for r in csv.DictReader(f):
                if r['patient_id'] not in keep:
                    continue  # dropped before use
                rid = f"{r['patient_id']}_{r['step']}"
                if r['view'] == 'front' and r['camera'] == 'FullHDwebcam':
                    rows[rid] = dict(meta=str(root / r['meta']), ppg_sync=str(root / r['ppg_sync']))
        mcd_labels(rows, recs, out)
    names = ['labels.csv'] + [f'labels/{p.name}' for p in sorted((out / 'labels').glob('*.npy'))]
    (out / 'labels_hashes.txt').write_text(''.join(f'{S2.file_sha(out / n)}  {args.dataset}/{n}\n' for n in names))
    print('labels', len(names) - 1, flush=True)


W = None


def bg_init(vendor):
    global W
    # A4 item 4(a) (brief 02 6.69 가): face detection on one CPU thread in every step. The October view worker
    # (src/st9e/view_worker.py, unchanged; it already hides the GPU) inherits this environment: OpenCV 4.5.2 reads
    # OPENCV_FOR_THREADS_NUM=1 as one thread (stage 2 imported here sets 16 for its own workers).
    os.environ['OPENCV_FOR_THREADS_NUM'] = '1'
    # 6.70 가: OpenCL off; the view worker loads eval5/face_guard/sitecustomize.py at start-up (cv2.ocl.setUseOpenCL(False), one
    # thread) and records its state in STAGE5_FACE_GUARD_LOG (set by run_mmpd)
    os.environ['OPENCV_OPENCL_DEVICE'] = 'disabled'
    os.environ['PYTHONPATH'] = os.pathsep.join([str(HERE / 'face_guard')] + [x for x in os.environ.get('PYTHONPATH', '').split(os.pathsep) if x])
    vendor_imports(vendor)
    import atexit
    from src.st9e.model_view import ViewWorker
    W = ViewWorker()
    atexit.register(W.close)


def bg_job(args):
    """tools/bg_spectrum.py job for one MMPD recording (background mean R, G, B outside the PhysNet HC box of frame 0)."""
    root, rid = args
    from scipy.io import loadmat
    from scipy.signal import welch, detrend
    x = np.round(np.asarray(loadmat(Path(root) / (rid + '.mat'), variable_names=['video'])['video']) * 255).astype(np.uint8)
    _, box = W.view(x[:1], 'PhysNet')
    bx, by, bw, bh = map(int, np.asarray(box).reshape(-1, 4)[0])
    y0, y1, x0, x1 = max(by, 0), min(by + bh, x.shape[1]), max(bx, 0), min(bx + bw, x.shape[2])
    total = x.reshape(len(x), -1, 3).sum(axis=1, dtype=np.float64)
    inner = x[:, y0:y1, x0:x1].reshape(len(x), -1, 3).sum(axis=1, dtype=np.float64)
    n_out = x.shape[1] * x.shape[2] - (y1 - y0) * (x1 - x0)
    rec = dict(recording_id=rid, frames=len(x), box=json.dumps([bx, by, bw, bh]), background_pixels=int(n_out))
    if n_out == 0:
        return rec, None
    bg = (total - inner) / n_out
    f, p = welch(detrend(bg.mean(axis=1)), fs=30, nperseg=300, noverlap=150, nfft=512)
    band = (f >= .6) & (f <= 3.3)
    pn = p[band] / p[band].sum()
    return rec, (np.round(f[band] * 60 / STEP).astype(int), pn)


def bg_flag(spec):
    """tools/b2_flag_robust.py flags() rule on the RGB-mean spectrum."""
    b, v = spec
    hit = set()
    for i in range(1, len(v) - 1):
        base = np.median(np.r_[v[max(0, i - 3):i], v[i + 1:i + 4]])
        if v[i] > v[i - 1] and v[i] > v[i + 1] and base > 0 and v[i] / base >= 2:
            hit.add(int(b[i]))
    return ('flagged_2Hz' if hit & set(LOW) else '') + ('flagged_3Hz' if hit & set(HIGH) else '')


def run_mmpd(args, out):
    plan = []
    with open(Path(args.plan_dir) / f'{args.plan_prefix}_FPm_input_plan.csv', newline='') as f:
        for r in csv.DictReader(f):
            if not plan or plan[-1]['recording_id'] != r['recording_id']:
                plan.append(dict(recording_id=r['recording_id'], subject_id=r['recording_id'].split('/')[0]))
    keep = set(S2.read_subjects(args.subjects))
    assert all(r['subject_id'] in keep for r in plan)
    lens = {}

    def load(rec):
        v = load_mmpd_contact(args.data_root, rec['recording_id'])
        lens[rec['recording_id']] = len(v)
        save_label(out, rec['recording_id'], v)
        return v
    audit, segments = mmpd_references(plan, load, (args.expected_recordings, args.expected_subjects))
    from src.st11x.pathways import BANDS
    pops = []
    for layer in BANDS:
        for r in audit:
            h = float(r.get('h_FFT_' + layer, np.nan))
            ok = bool(np.isfinite(h))
            pops.append(dict(window_id=r['window_id'], recording_id=r['recording_id'], subject_id=r['subject_id'],
                             window_index=r['window_index'], layer=layer, reference_HR=h, in_P=ok, in_Qstar=bool(r['Qstar_' + layer]),
                             high=bool(ok and h >= 100), reference_available=ok))
    started = time.time()
    bg_rows = []
    if args.no_bg:  # test only
        plan_bg = []
    else:
        plan_bg = plan
    log = out / 'face_workers_background.csv'  # 6.70 가: one line per view worker (pid, OpenCL in use, OpenCV threads, ...)
    log.write_text('worker_pid,opencl,threads,opencv,cuda_visible_devices,opencv_opencl_device\n')
    os.environ['STAGE5_FACE_GUARD_LOG'] = str(log)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn'), initializer=bg_init, initargs=(str(VENDOR),)) as ex:
        for n, (rec, spec) in enumerate(ex.map(bg_job, [(args.data_root, r['recording_id']) for r in plan_bg]), 1):
            rec['background_flag'] = 'no_background' if spec is None else ('flagged' if bg_flag(spec) else 'unflagged')
            rec['flag_kind'] = '' if spec is None else bg_flag(spec)
            bg_rows.append(rec)
            if n % 20 == 0 or n == len(plan):
                print(f'bg {n}/{len(plan)} {time.time() - started:.0f}s', flush=True)
    FIRST[0] = first_frames(args, out, [('MMPD', str(Path(args.data_root) / (r['recording_id'] + '.mat')), r['recording_id']) for r in plan])
    S2.write_csv(out / 'references.csv', audit)
    S2.write_csv(out / 'defect_segments.csv', segments or [dict(window_id='', type='', start_frame='', end_frame='')])
    S2.write_csv(out / 'populations.csv', pops)
    if bg_rows:
        S2.write_csv(out / 'bg_flags.csv', bg_rows)
    S2.write_csv(out / 'recordings.csv', [dict(r, contact_samples=lens[r['recording_id']]) for r in plan])
    counts, boots = [], {}
    for layer in BANDS:
        z = [r for r in pops if r['layer'] == layer]
        q = [r for r in z if r['in_Qstar']]
        counts.append(dict(population=f'Qstar_{layer}', windows=len(q), high=sum(r['high'] for r in q), low=sum(not r['high'] for r in q),
                           recordings=len({r['recording_id'] for r in q}), subjects=len({r['subject_id'] for r in q}),
                           all_windows=len(z), P=sum(r['in_P'] for r in z)))
    boots['P'] = {r['subject_id'] for r in pops if r['in_P']}
    return counts, boots


def first_frame_job(spec):
    """Brief 02 6.67 다 4: Haar detections on the first frame of one recording (video only)."""
    dataset, path, rid = spec
    import cv2
    if dataset == 'MMPD':
        from scipy.io import loadmat
        frame = np.round(np.asarray(loadmat(path, variable_names=['video'])['video'][0]) * 255).astype(np.uint8)
    else:
        cap = cv2.VideoCapture(path)
        ok, bgr = cap.read()
        cap.release()
        assert ok, rid
        frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    cv2.ocl.setUseOpenCL(False)  # 6.70 가
    n = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        state = dict(worker_pid=os.getpid(), opencl=bool(cv2.ocl.useOpenCL()), threads=cv2.getNumThreads())
        det = cv2.CascadeClassifier(str(VENDOR / 'third_party/rPPG-Toolbox/dataset/haarcascade_frontalface_default.xml'))
        zones = det.detectMultiScale(frame[:, :, :3].astype(np.uint8))
    finally:
        cv2.setNumThreads(n)
    widths = [int(z[2]) for z in zones] if len(zones) else []
    top = max(widths) if widths else 0
    return dict(recording_id=rid, detections=len(widths), widths=json.dumps(widths), largest_width=top,
                detections_at_largest_width=widths.count(top) if widths else 0,
                tie_at_largest_width=widths.count(top) >= 2 if widths else False,
                any_equal_width=len(widths) != len(set(widths)), **state)


def ff_init(vendor):
    """Worker setup of the first-frame detections: no GPU visible, as stage 2's workers (eval5/stage2_predict.init_worker).
    With a GPU visible, OpenCV runs the Haar cascade through OpenCL on the GPU (a GPU context, and possibly other
    detections than stage 2's CPU path); hidden, it uses the CPU path of stage 2."""
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['OPENCV_OPENCL_DEVICE'] = 'disabled'  # 6.70 가 (with cv2.ocl.setUseOpenCL(False) in the job)
    vendor_imports(vendor)


def first_frames(args, out, specs):
    started = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn'), initializer=ff_init,
                             initargs=(str(VENDOR),)) as ex:
        for n, r in enumerate(ex.map(first_frame_job, specs), 1):
            rows.append(r)
            if n % 50 == 0 or n == len(specs):
                print(f'first-frame detections {n}/{len(specs)} {time.time() - started:.0f}s', flush=True)
    S2.write_csv(out / 'first_frame_detections.csv', rows)
    return dict(recordings=len(rows), opencl_used=sum(r['opencl'] for r in rows), threads=sorted({r['threads'] for r in rows}),
                workers=len({r['worker_pid'] for r in rows}), no_detection=sum(r['detections'] == 0 for r in rows),
                several_detections=sum(r['detections'] >= 2 for r in rows),
                tie_at_largest_width=sum(r['tie_at_largest_width'] for r in rows),
                any_equal_width=sum(r['any_equal_width'] for r in rows))


# -------------------------------------------------------------------------------------------------------------- MCD
def run_mcd(args, out):
    import mcd_reference as mcd_meta  # b1/mcd_meta.job, copied unchanged
    root = Path(args.data_root)
    keep = set(S2.read_subjects(args.subjects))
    with open(args.recordings, newline='') as f:
        listed = [r['recording_id'] for r in csv.DictReader(f)]
    assert all(r.split('_')[0] in keep for r in listed)
    rows = {}
    with (root / 'db.csv').open(newline='') as f:
        for r in csv.DictReader(f):
            if r['patient_id'] not in keep:
                continue  # dropped before use
            rid = f"{r['patient_id']}_{r['step']}"
            if rid in listed and r['view'] == 'front' and r['camera'] == 'FullHDwebcam':
                rows[rid] = dict(recording_id=rid, subject_id=r['patient_id'], state=r['step'], video=str(root / r['video']),
                                 meta=str(root / r['meta']), ppg=str(root / r['ppg']), ppg_sync=str(root / r['ppg_sync']))  # absolute
    jobs = [rows[r] for r in listed]
    recs, wins = [], []
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn'), initializer=vendor_imports,
                             initargs=(str(VENDOR),)) as ex:
        for n, (r, w) in enumerate(ex.map(mcd_meta.job, jobs), 1):
            recs.append(r)
            wins += w
            if n % 25 == 0 or n == len(jobs):
                print(f'mcd-meta {n}/{len(jobs)} {time.time() - started:.0f}s', flush=True)
    FIRST[0] = first_frames(args, out, [('MCD', j['video'], j['recording_id']) for j in jobs])
    recs, w = pd.DataFrame(recs), pd.DataFrame(wins)
    recs.to_csv(out / 'recordings.csv', index=False)
    w.to_csv(out / 'references.csv', index=False)
    mcd_labels(rows, recs, out)
    # populations (b1/analyze.py load_mcd)
    confirmed = set(recs.recording_id[recs.mapping_confirmed.astype(bool)])
    prim = w[w.primary_included.astype(bool) & np.isfinite(w.h_FFT.astype(float))].copy()
    har = w[w.layer.eq('HAR') & w.primary_included.astype(bool) & w.agreement.astype(bool) & w.h_FFT.ge(100)]
    draft = set(har.recording_id) & confirmed
    d = prim[prim.recording_id.isin(draft)].copy()
    d['set'] = np.where(d.agreement.astype(bool) & (d.h_FFT >= 100), 'high', np.where(d.h_FFT < 100, 'low', ''))
    d = d[d.set.ne('')]
    d['population'] = 'draft_rule'
    st = recs[recs.mapping_confirmed.astype(bool)].groupby('subject_id').state.nunique()
    both = set(st[st == 2].index.astype(str))
    m7 = prim[prim.subject_id.astype(str).isin(both) & prim.recording_id.isin(confirmed)].copy()
    m7['set'] = np.where(m7.h_FFT >= 100, 'high', 'low')
    m7['population'] = 'M7_both_states'
    pops = pd.concat([d, m7], ignore_index=True)[['population', 'window_id', 'layer', 'recording_id', 'subject_id', 'state', 'h_FFT', 'agreement', 'set']]
    pops = pops.rename(columns={'h_FFT': 'reference_HR'})
    pops.to_csv(out / 'populations.csv', index=False)
    counts = []
    for (pop, layer), g in pops.groupby(['population', 'layer']):
        counts.append(dict(population=pop, layer=layer, windows=len(g), high=int((g.set == 'high').sum()), low=int((g.set == 'low').sum()),
                           recordings=g.recording_id.nunique(), subjects=g.subject_id.nunique()))
    boots = {'draft_rule': set(d.subject_id.astype(str)), 'M7_both_states': set(m7.subject_id.astype(str))}
    return counts, boots


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--dataset', required=True, choices=['MMPD', 'MCD'])
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--subjects', required=True)
    ap.add_argument('--recordings'), ap.add_argument('--plan-dir'), ap.add_argument('--plan-prefix', default='st30e_A')
    ap.add_argument('--expected-recordings', type=int, default=435), ap.add_argument('--expected-subjects', type=int, default=22)
    ap.add_argument('--manifest', nargs='+', required=True)
    ap.add_argument('--vendor', required=True), ap.add_argument('--out', required=True), ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--sealed-subjects', nargs='*')
    ap.add_argument('--no-bg', action='store_true', help='test only: skip the background flags')
    ap.add_argument('--labels-only', action='store_true', help='write only the contact signals into an existing stage-1 output')
    ap.add_argument('--unseal', action='store_true'), ap.add_argument('--amendment-commit'), ap.add_argument('--repo')
    args = ap.parse_args()
    vendor_imports(args.vendor)
    started = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    guard = S2.seal_guard(args, S2.read_subjects(args.subjects), S2.manifest_rows(args.manifest))  # before any data file
    out = Path(args.out) / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    if args.labels_only:
        labels_only(args, out)
        return
    counts, boots = run_mmpd(args, out) if args.dataset == 'MMPD' else run_mcd(args, out)
    if args.dataset == 'MMPD':
        label_table(out)
    from src.st11x.pathways import paired_bootstrap_indices
    for pop, subjects in boots.items():
        for seeds in (3, 5):
            ordered, si, se = paired_bootstrap_indices(subjects, seeds)
            np.savez(out / f'bootstrap_{pop}_{seeds}.npz', subjects=np.asarray(ordered), subject_indices=si, seed_indices=se)
    S2.write_csv(out / 'counts.csv', counts)
    names = sorted(p.name for p in out.iterdir() if p.suffix in ('.csv', '.npz')) + [f'labels/{p.name}' for p in sorted((out / 'labels').glob('*.npy'))]
    lines = [f'{S2.file_sha(out / n)}  {args.dataset}/{n}' for n in names]
    (out / 'stage1_hashes.txt').write_text(''.join(x + '\n' for x in lines))
    run = dict(stage=1, dataset=args.dataset, started=started, ended=time.strftime('%Y-%m-%dT%H:%M:%S%z'), operator=getpass.getuser(),
               guard=guard, subject_list_sha256=guard['subject_list_sha256'], counts=counts,
               expected=[args.expected_recordings, args.expected_subjects] if args.dataset == 'MMPD' else None,
               first_frame_detections=FIRST[0],
               hash_file_sha256=S2.file_sha(out / 'stage1_hashes.txt'))
    (out / 'run.json').write_text(json.dumps(run, indent=1, default=str))
    print(json.dumps(counts, indent=1, default=str))
    print("first-frame detections", json.dumps(FIRST[0]), flush=True)


if __name__ == '__main__':
    main()
