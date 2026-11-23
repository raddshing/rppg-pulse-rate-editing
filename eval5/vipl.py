"""Registered evaluation on VIPL-HR (amendment A7): stages (a) and (b).

The evaluation code of commit 19ecf5d is imported unchanged (eval5/stage2_predict.py: inference server, model list,
manifest and checkpoint checks, allocator-OOM guard, per-recording inference; vendor src/st11x and src/st4/src/st3:
30-Hz time axis, reference readout, reference estimators, defect map, Q* membership, bootstrap indices). This file only
adds what VIPL-HR needs (A7 items 1-6):
  - reading: the 14 original zips are read in memory; a video is passed to the decoder of the test run (OpenCV
    VideoCapture, FFmpeg) through a file in a memory file system (tmpfs), removed right after decoding; nothing is
    unpacked to disk.
  - sets: primary = source1 and source3 (frame times in time.txt), secondary = source2 (no frame times; 30 fps evenly).
  - preprocessing (as the MCD-rPPG test path, eval5/stage2_predict.mcd_decode): HC face box (larger box 1.5) found once in
    the first frame, every original frame cropped and resized (INTER_AREA; 72 PhysNet/FactorizePhys, 128 RhythmFormer),
    linear interpolation in time onto the 30-Hz grid from the frame times (no extrapolation), then DiffNormalized 128
    (PhysNet) / Raw 160 (FactorizePhys) / Standardized 160 (RhythmFormer). If the HC detector finds no face in the first
    frame, the box is the whole frame (the toolbox's own fallback box is not used; other data paths are unchanged).
  - windows (item 4): the first five non-overlapping 10-s windows (300 samples of the 30-Hz grid from the first frame)
    inside both the video and the BVP record; exclusions: video not decodable, time.txt not one line per frame, frame
    times not strictly increasing, no complete window, source2 duration difference > 2 s; flag: an interval > 200 ms
    between consecutive original frames inside the window (primary analysis excludes, sensitivity keeps).
  - reference (items 5, 6): BVP (wave.csv) placed on the 30-Hz axis from the first frame at rate fs (scipy resample_poly,
    st11x.pathways.reference_30hz), window readout common_hr (HAR, L0), reference estimators and classification
    (src/st4/signals.py), defect map d1-d4 (src/st3/c4.py, exactly as the MMPD-B stage-1 path), verified = Q*_HAR;
    gt_HR mean over the window (rows taken as one per second from the first frame). The BVP rate rule of item 6.
Real VIPL-HR data are opened only if the A7 entry is committed on the remote main branch of the repository (checked like
the seal check of the test run); test data must be in a folder marked SYNTHETIC_VIPL_TEST.
Usage:
  vipl.py a --zip-dir DIR --out OUT --vendor V (--repo R --remote URL --a7-commit C | synthetic folder) [--workers N] [--limit N]
  vipl.py b --zip-dir DIR --stage1 OUT_A --manifest M... --checkpoint-index I --set S --vendor V --out OUT_B --gpu G ...
"""
import os  # noqa: E402  (first: thread settings precede numpy / cv2 / torch imports; children inherit)
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENCV_FOR_THREADS_NUM='16',
                  OPENCV_OPENCL_DEVICE='disabled')
import argparse  # noqa: E402
import contextlib  # noqa: E402
import csv  # noqa: E402
import getpass  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import zipfile  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait  # noqa: E402
import multiprocessing as mp  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stage2_predict as S2  # noqa: E402  (19ecf5d, unchanged)

SOURCES = {'source1': 'primary', 'source3': 'primary', 'source2': 'secondary'}  # source4 (NIR) is not used
TMPFS = Path('/dev/shm')
A7_HEADING = 'Registered evaluation on VIPL-HR.'
RATIO_BAND = (0.98, 1.02)
NOMINAL_FS = 60.0
INF = None
INF_KEY = None


# ------------------------------------------------------------------------------------------------ guard and index
def data_guard(zdir, repo=None, a7_commit=None, remote=None):
    """Synthetic test folder, or the A7 entry committed on the remote main branch (as the test run's seal check)."""
    zdir = Path(zdir)
    if (zdir / 'SYNTHETIC_VIPL_TEST').exists() and not (zdir / 'TRANSFER_MANIFEST.sha256').exists():
        return dict(data='synthetic test zips')
    if not (repo and remote and a7_commit and re.fullmatch(r'[0-9a-f]{40}', a7_commit)):
        raise SystemExit('VIPL-HR: the A7 commit and the repository are required; nothing opened')
    git = lambda *a: subprocess.run(['git', '-C', str(repo), *a], capture_output=True, text=True)  # noqa: E731
    main = git('ls-remote', remote, 'refs/heads/main').stdout.split()
    head = git('rev-parse', 'HEAD').stdout.strip()
    if not main or main[0] != head:
        raise SystemExit('VIPL-HR: remote main is not the local main; nothing opened')
    if git('merge-base', '--is-ancestor', a7_commit, head).returncode != 0:
        raise SystemExit('VIPL-HR: the A7 commit is not on main; nothing opened')
    text = git('show', f'{a7_commit}:docs/PREREGISTRATION_STAGE5_AMENDMENTS.md').stdout
    if A7_HEADING not in text:
        raise SystemExit('VIPL-HR: the A7 entry is not in the amendments at that commit; nothing opened')
    return dict(data='VIPL-HR', a7_commit=a7_commit, remote_main=head)


def zip_files(zdir):
    zdir = Path(zdir)
    rest = sorted((zdir / 'VIPL_remainder').glob('*.zip'), key=lambda p: int(re.findall(r'\d+', p.name)[0]))
    return [zdir / 'VIPL-HR.zip'] + rest


def parse_member(name):
    parts = name.split('/')
    subj = next((p for p in parts if re.fullmatch(r'p\d+', p)), None)
    scene = next((p for p in parts if re.fullmatch(r'v\d(-2)?', p)), None)
    source = next((p for p in parts if re.fullmatch(r'source\d', p)), None)
    return subj, scene, source, parts[-1]


def index(zdir):
    """Recordings of source1/2/3 with their zip and member names (zip directories only)."""
    recs = {}
    for zp in zip_files(zdir):
        with zipfile.ZipFile(zp) as z:
            for i in z.infolist():
                subj, scene, source, base = parse_member(i.filename)
                if i.is_dir() or not (subj and scene and source in SOURCES):
                    continue
                if base not in ('video.avi', 'time.txt', 'wave.csv', 'gt_HR.csv'):
                    continue
                rid = f'{subj}_{scene}_{source}'
                r = recs.setdefault(rid, dict(recording_id=rid, subject_id=subj, scene=scene, source=source,
                                              set=SOURCES[source], zip=str(zp), members={}))
                assert r['zip'] == str(zp), f'{rid}: members in two zips'
                assert base not in r['members'], f'{rid}: two {base}'
                r['members'][base] = i.filename
    key = lambda r: (int(r['subject_id'][1:]), int(r['scene'][1]), r['scene'], r['source'])  # noqa: E731
    return sorted(recs.values(), key=key)


def read(rec, base):
    with zipfile.ZipFile(rec['zip']) as z:
        return z.read(rec['members'][base])


def column(raw, name):
    """Values of the single-column CSV (header line 'Wave' or 'HR')."""
    if raw.startswith(b'\xef\xbb\xbf'):
        raw = raw[3:]
    lines = [ln.strip() for ln in raw.decode('latin1').splitlines() if ln.strip()]
    assert lines and lines[0] == name, f'header {name} expected'
    return np.asarray([float(x) for x in lines[1:]], dtype=np.float64)


def frame_times(rec, n_frames):
    """Seconds from the first frame: time.txt (ms) for source1/3, evenly at 30 fps for source2. None, reason if unusable."""
    if rec['source'] == 'source2':
        return np.arange(n_frames, dtype=np.float64) / 30.0, ''
    if 'time.txt' not in rec['members']:
        return None, 'no time.txt'
    t = np.asarray([float(x) for x in read(rec, 'time.txt').decode('latin1').split()], dtype=np.float64)
    if len(t) != n_frames:
        return None, 'time.txt not one line per frame'
    t = (t - t[0]) / 1000.0
    if not (np.diff(t) > 0).all():
        return None, 'frame times not strictly increasing'
    return t, ''


# ------------------------------------------------------------------------------------------------ decoding
@contextlib.contextmanager
def video_file(rec):
    """The video as a file in tmpfs (memory), removed after use."""
    f = tempfile.NamedTemporaryFile(dir=TMPFS, prefix='ops2_vipl_', suffix='.avi', delete=False)
    try:
        f.write(read(rec, 'video.avi'))
        f.close()
        yield f.name
    finally:
        try:
            os.remove(f.name)
        except FileNotFoundError:
            pass


def face_box(rgb):
    """HC box of the test path (stage2_predict.hc_box: toolbox face_detection, larger box 1.5, one OpenCV thread, OpenCL
    off). If the same detector finds no face, the box is the whole frame (A7 item 3)."""
    import cv2
    cv2.ocl.setUseOpenCL(False)
    n = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        found = len(cv2.CascadeClassifier('./dataset/haarcascade_frontalface_default.xml').detectMultiScale(rgb[:, :, :3].astype(np.uint8)))
    finally:
        cv2.setNumThreads(n)
    if found == 0:
        return np.asarray([0, 0, rgb.shape[1], rgb.shape[0]], dtype=int), False
    return S2.hc_box(rgb), True


def decode(rec, streams=(), count_only=False):
    """Decode once. count_only: number of decoded frames. Otherwise the model-input chunks per stream (as mcd_decode)."""
    import cv2
    from dataset.data_loader.BaseLoader import BaseLoader
    from src.st11x.pathways import interpolate_face_input
    need72 = bool({'PhysNet', 'FPm'} & {s.split('@')[0] for s in streams})
    need128 = any(s.startswith('RhythmFormer') for s in streams)
    hc72, hc128, box, found, k = [], [], None, None, 0
    with video_file(rec) as path, contextlib.redirect_stdout(sys.stderr):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return dict(recording_id=rec['recording_id'], decoded_frames=0, error='video not opened')
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            k += 1
            if count_only:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if k == 1:
                box, found = face_box(rgb)
            a = rgb[max(box[1], 0):min(box[1] + box[3], rgb.shape[0]), max(box[0], 0):min(box[0] + box[2], rgb.shape[1])]
            if need72:
                hc72.append(cv2.resize(a, (72, 72), interpolation=cv2.INTER_AREA))
            if need128:
                hc128.append(cv2.resize(a, (128, 128), interpolation=cv2.INTER_AREA))
        cap.release()
    if count_only:
        return dict(recording_id=rec['recording_id'], decoded_frames=k)
    times, reason = frame_times(rec, k)
    assert times is not None, f"{rec['recording_id']}: {reason} (stage (a) excludes it)"
    out, grid = {}, None
    if need72:
        r72, grid = interpolate_face_input(np.asarray(hc72, dtype=np.float64), times)
        del hc72
        if 'PhysNet' in streams:
            out['PhysNet'] = S2.chunks_of(BaseLoader.diff_normalize_data(r72.copy()), 128)
        if 'FPm' in streams:
            out['FPm'] = S2.chunks_of(r72, 160)
        del r72
    if need128:
        r, grid = interpolate_face_input(np.asarray(hc128, dtype=np.float64), times)
        del hc128
        out['RhythmFormer'] = S2.chunks_of(BaseLoader.standardized_data(r), 160)
        del r
    return dict(recording_id=rec['recording_id'], decoded_frames=k, input_samples=len(grid), box=box.tolist(),
                face_found=found, chunks=out, **S2.FACE_STATE)


# ------------------------------------------------------------------------------------------------ stage (a)
def init_worker(vendor):
    S2.init_worker(vendor)  # vendor on the path, working directory = toolbox, GPU hidden, one torch thread


def gap_flags(t, n_windows):
    """Windows j that contain an interval > 200 ms between consecutive original frames (t: seconds from frame 0)."""
    big = np.where(np.diff(t) > 0.2)[0]
    return [bool(any(t[i] < 10 * j + 299 / 30 and t[i + 1] > 10 * j for i in big)) for j in range(n_windows)]


def a_job(rec):
    """Per recording, without any model: decoded frame count, frame times, BVP and gt_HR values."""
    d = decode(rec, count_only=True)
    row = dict(recording_id=rec['recording_id'], subject_id=rec['subject_id'], scene=rec['scene'], source=rec['source'],
               set=rec['set'], decoded_frames=d['decoded_frames'], exclusion='')
    if d['decoded_frames'] < 2:
        row['exclusion'] = 'video not decodable'
        return row, None
    t, reason = frame_times(rec, d['decoded_frames'])
    if t is None:
        row['exclusion'] = reason
        return row, None
    wave = column(read(rec, 'wave.csv'), 'Wave') if 'wave.csv' in rec['members'] else None
    hr = column(read(rec, 'gt_HR.csv'), 'HR') if 'gt_HR.csv' in rec['members'] else None
    if wave is None or len(wave) < 2:
        row['exclusion'] = 'no BVP record'
        return row, None
    row.update(span_seconds=float(t[-1]), bvp_samples=len(wave), gt_HR_rows=0 if hr is None else len(hr))
    return row, dict(t=t, wave=wave, hr=hr)


def windows_for(rec_row, data, fs, helpers, defect_map):
    """Windows of one recording at BVP rate fs: availability, 200-ms flags, readout, estimators, defects, gt_HR mean."""
    from src.st11x.pathways import uniform_frame_grid, complete_windows, reference_30hz, mmpd_b_membership, BANDS
    import pandas as pd
    t, wave, hr = data['t'], data['wave'], data['hr']
    grid = uniform_frame_grid(t)
    c30 = reference_30hz(wave, fs)
    excl = ''
    if rec_row['source'] == 'source2' and abs(rec_row['decoded_frames'] / 30 - len(wave) / fs) > 2:
        excl = 'source2 duration difference > 2 s'
    bounds = complete_windows(grid, [10.0 * j for j in range(5)])
    js = [j for j in range(5) if bounds[j]['available'] and bounds[j]['grid_start'] == 300 * j and 300 * (j + 1) <= len(c30)]
    js = [j for j in js if js[:js.index(j) + 1] == list(range(js.index(j) + 1))]  # consecutive from the first window
    if not excl and not js:
        excl = 'no complete window'
    if excl:
        return excl, []
    flags = gap_flags(t, 5) if rec_row['source'] != 'source2' else [False] * 5
    rid = rec_row['recording_id']
    from scipy.signal import detrend, periodogram
    fr = []
    for j in js:  # the window rows of the MMPD-B stage-1 path (src/st11x/future.mmpd_b_references), incl. T1b_high_freq
        w = c30[300 * j:300 * j + 300]
        h = helpers['common_hr'](w, 'HAR')
        y = detrend(w.astype(np.float64), type='linear')
        f, p = periodogram(y, fs=30, window='hann', nfft=len(y), detrend=False, scaling='density')
        den = p[(f >= .6) & (f <= 15)].sum()
        ratio = float(p[(f > 3.5 * h / 60) & (f <= 15)].sum() / den) if den > 0 and np.isfinite(h) else np.nan
        fr.append(dict(window_id=f'VIPL|{rid}|{j}', recording_id=rid, window_index=j, start_seconds=10 * j, T1=-1, T1b=-1,
                       T1b_high_freq=int(ratio >= .10676) if np.isfinite(ratio) else -1, Q3_power_above_3p5_f0_ratio=ratio))
    frame = pd.DataFrame(fr)
    defects, _ = defect_map(frame, {rid: c30})
    rows = []
    for j in js:
        w = c30[300 * j:300 * j + 300]
        h = {layer: helpers['common_hr'](w, layer) for layer in BANDS}
        diag = helpers['reference_estimators'](w, h['HAR'])
        cls = helpers['classify'](h['HAR'], diag['h_ACF'], diag['h_IBI'])
        dd = defects.loc[f'VIPL|{rid}|{j}'].to_dict()
        mem = mmpd_b_membership(dd['defect_any'], cls, h['HAR'], h['L0'])
        g = hr[10 * j:10 * j + 10] if hr is not None and len(hr) >= 10 * j + 10 else None
        rows.append(dict(window_id=f'VIPL|{rid}|{j}', recording_id=rid, subject_id=rec_row['subject_id'], scene=rec_row['scene'],
                         source=rec_row['source'], set=rec_row['set'], window_index=j, start_seconds=10 * j, grid_start=300 * j,
                         gap200=flags[j], h_FFT_HAR=h['HAR'], h_FFT_L0=h['L0'], reference_class=cls,
                         gt_HR_mean=float(np.mean(g)) if g is not None else np.nan,
                         **{k: v for k, v in diag.items()}, **{k: v for k, v in dd.items()}, verified=bool(mem['Qstar_HAR']),
                         Qstar_L0=bool(mem['Qstar_L0'])))
    return '', rows


def run_a(args):
    args.zip_dir, args.vendor = str(Path(args.zip_dir).resolve()), str(Path(args.vendor).resolve())
    guard = data_guard(args.zip_dir, args.repo, args.a7_commit, args.remote)
    out = Path(args.out).resolve() / 'VIPL'
    assert not out.exists(), 'stage (a) output exists'
    out.mkdir(parents=True)
    recs = index(args.zip_dir)
    if args.limit:
        recs = recs[:args.limit]
    started = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    rows, data = [], {}
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn'), initializer=init_worker,
                             initargs=(str(Path(args.vendor).resolve()),)) as ex:
        for n, (row, d) in enumerate(ex.map(a_job, recs), 1):
            rows.append(row)
            if d is not None:
                data[row['recording_id']] = d
            if n % 100 == 0:
                print(f'a {n}/{len(recs)}', flush=True)
    init_worker(str(Path(args.vendor).resolve()))
    from src.st11x.future import frozen_reference_functions
    import ast
    import pandas as pd
    helpers = frozen_reference_functions()
    src = Path(args.vendor).resolve() / 'src/st3/c4.py'  # the defect map exactly as the MMPD-B stage-1 path
    tree = ast.parse(src.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'defect_map')
    ctx = dict(np=np, pd=pd, write_rows=lambda *a, **k: None, update=lambda *a, **k: None)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(src), 'exec'), ctx)
    by = {r['recording_id']: r for r in rows}

    def at(rate_of):
        out_rows, excl = [], {}
        for rid, d in data.items():
            e, w = windows_for(by[rid], d, rate_of(by[rid]), helpers, ctx['defect_map'])
            excl[rid] = e
            out_rows += w
        v = [r for r in out_rows if r['verified'] and np.isfinite(r['gt_HR_mean']) and r['gt_HR_mean'] > 0]  # every verified window
        ratio = float(np.median([r['h_FFT_HAR'] / r['gt_HR_mean'] for r in v])) if v else np.nan
        return out_rows, excl, ratio, len(v)

    w60, e60, r60, n60 = at(lambda r: NOMINAL_FS)
    comparison = [dict(rate='60 Hz', median_ratio=r60, windows=n60)]
    choice, final, excl = '60 Hz', w60, e60
    if not (RATIO_BAND[0] <= r60 <= RATIO_BAND[1]):
        wr, er, rr, nr = at(lambda r: r['bvp_samples'] / r['gt_HR_rows'] if r.get('gt_HR_rows') else NOMINAL_FS)
        comparison.append(dict(rate='per record (BVP samples / gt_HR rows)', median_ratio=rr, windows=nr))
        if RATIO_BAND[0] <= rr <= RATIO_BAND[1]:
            choice, final, excl = 'per record', wr, er
        else:  # neither: gt_HR mean is the reference, the BVP (per-record rate) only verifies windows
            choice, final, excl = 'gt_HR mean (BVP verifies only, per-record rate)', wr, er
    for r in final:
        ref = r['gt_HR_mean'] if choice.startswith('gt_HR') else r['h_FFT_HAR']
        r.update(reference_HR=ref, high=bool(r['verified'] and np.isfinite(ref) and ref >= 100),
                 low=bool(r['verified'] and np.isfinite(ref) and ref < 100))
    for rid, e in excl.items():
        if e and not by[rid]['exclusion']:
            by[rid]['exclusion'] = e
    for r in rows:
        r['windows'] = sum(1 for w in final if w['recording_id'] == r['recording_id'])
    S2.write_csv(out / 'recordings.csv', rows)
    S2.write_csv(out / 'references.csv', final)
    (out / 'window_list.csv').write_text('recording_id,window_index,start_seconds\n' + ''.join(
        f"{w['recording_id']},{w['window_index']},{w['start_seconds']}\n" for w in final))
    (out / 'bvp_rate.json').write_text(json.dumps(dict(choice=choice, comparisons=comparison, band=RATIO_BAND), indent=1))
    from src.st11x.pathways import paired_bootstrap_indices
    for s in ('primary', 'secondary'):
        subs = sorted({w['subject_id'] for w in final if w['set'] == s})
        for k in (3, 6):
            if subs:
                order, a, b = paired_bootstrap_indices(subs, seeds=k)
                np.savez(out / f'bootstrap_{s}_{k}.npz', subjects=np.asarray(order), subject_draws=a, seed_draws=b)
    lines = [f'{S2.file_sha(p)}  VIPL/{p.name}' for p in sorted(out.iterdir()) if p.is_file()]
    (out / 'stage_a_hashes.txt').write_text(''.join(x + '\n' for x in lines))
    run = dict(stage='a', started=started, ended=time.strftime('%Y-%m-%dT%H:%M:%S%z'), operator=getpass.getuser(), guard=guard,
               recordings=len(rows), excluded=sum(bool(r['exclusion']) for r in rows), windows=len(final), bvp_rate=choice,
               limit=args.limit, python=sys.version.split()[0])
    (out / 'run.json').write_text(json.dumps(run, indent=1))
    print('done', json.dumps(run), flush=True)


# ------------------------------------------------------------------------------------------------ stage (b)
def b_job(args):
    """One recording: decode once, predict every model (random state reset per recording; as stage2_predict.mcd_job)."""
    global INF, INF_KEY
    rec, models, gpu, out, log = args
    t0 = time.time()
    streams = sorted({m['stream'] for m in models})
    d = decode(rec, streams)
    key = tuple(m['key'] for m in models)
    new_server = INF_KEY != key
    if new_server:
        if INF is not None:
            INF.close()
        INF = S2.Infer(models, gpu, log)
        INF_KEY = key
        assert not INF.failed, INF.failed
    before = {} if new_server else dict(INF.ooms)
    for s in streams:
        keys, p = INF.predict(s, d['chunks'][s], rng='reset')
        for i, k in enumerate(keys):
            dest = Path(out) / 'pred' / k
            dest.mkdir(parents=True, exist_ok=True)
            tmp = dest / (rec['recording_id'] + '.tmp.npy')
            np.save(tmp, p[i].reshape(-1).astype(np.float32))
            tmp.rename(dest / (rec['recording_id'] + '.npy'))
    row = dict(recording_id=rec['recording_id'], decoded_frames=d['decoded_frames'], input_samples=d['input_samples'],
               box=json.dumps(d['box']), face_found=d['face_found'], **{f'chunks_{s}': len(d['chunks'][s]) for s in streams},
               **{k: d[k] for k in ('face_worker_pid', 'face_opencl', 'face_threads')})
    ooms = {k: n - before.get(k, 0) for k, n in INF.ooms.items()}
    return row, time.time() - t0, ooms


def run_b(args):
    S2.VENDOR = Path(args.vendor).resolve()
    args.zip_dir = str(Path(args.zip_dir).resolve())
    guard = data_guard(args.zip_dir, args.repo, args.a7_commit, args.remote)
    keep = {r['recording_id'] for r in csv.DictReader(open(Path(args.stage1).resolve() / 'VIPL' / 'window_list.csv'))}  # only the window list of (a)
    recs = [r for r in index(args.zip_dir) if r['recording_id'] in keep]
    manifests = S2.manifest_rows(args.manifest)
    out = Path(args.out).resolve() / 'VIPL'
    out.mkdir(parents=True, exist_ok=True)
    log = str(out / 'server.log')
    started = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    args.dataset, args.blur, args.only_variants = 'VIPL', 0.0, None
    models = S2.models_from(args, manifests)
    used = [m for m in models if m['status'] == 'used']
    probe_ooms = {}
    if used:
        probe = S2.Infer(used, args.gpu, log)
        probe_ooms = dict(probe.ooms)
        failed = {f['key']: f['error'] for f in probe.failed}
        epochs = {x['key']: x['epoch'] for x in probe.loaded}
        probe.close()
        for m in models:
            if m['key'] in failed:
                m.update(status='defective', reason='load failure: ' + failed[m['key']])
            m['epoch'] = epochs.get(m['key'], '')
        used = [m for m in models if m['status'] == 'used']
    print('outputs', len(models), 'used', len(used), 'recordings', len(recs), flush=True)
    todo = [r for r in recs if not all((out / 'pred' / m['key'] / (r['recording_id'] + '.npy')).exists() for m in used)]
    rows, ooms = [], {}
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn'), initializer=init_worker,
                             initargs=(str(S2.VENDOR),)) as ex:
        inflight = {}
        while todo or inflight:
            while todo and len(inflight) < args.workers:
                r = todo.pop(0)
                inflight[ex.submit(b_job, (r, used, args.gpu, str(out), log))] = r
            fin, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
            for f in fin:
                inflight.pop(f)
                row, sec, oo = f.result()  # a failure stops the run
                rows.append(row)
                for k, n in oo.items():
                    ooms[k] = ooms.get(k, 0) + n
                if len(rows) % 50 == 0:
                    print(f'b {len(rows)}/{len(recs)}', flush=True)
    for m in models:
        m['allocator_ooms'] = (probe_ooms.get(m['key'], 0) + ooms.get(m['key'], 0)) if m['status'] == 'used' else ''
    S2.write_csv(out / 'recordings.csv', sorted(rows, key=lambda r: r['recording_id']))
    S2.write_csv(out / 'checkpoints.csv', [{k: v for k, v in m.items() if k not in ('checkpoint', 'run_id', 'md_inference')} for m in models])
    lines = [f"{S2.file_sha(p)}  VIPL/{p.relative_to(out).as_posix()}" for p in sorted((out / 'pred').rglob('*.npy'))]
    lines += [f"{S2.file_sha(out / n)}  VIPL/{n}" for n in ('recordings.csv', 'checkpoints.csv')]
    (out / 'stage2_hashes.txt').write_text(''.join(x + '\n' for x in lines))
    run = dict(stage='b', dataset='VIPL', set=args.set, started=started, ended=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
               operator=getpass.getuser(), guard=guard, manifests=[Path(p).name for p in args.manifest],
               manifest_sha256={Path(p).name: S2.file_sha(p) for p in args.manifest}, outputs=len(models), used=len(used),
               recordings=len(rows), prediction_files=len(lines) - 2, hash_file_sha256=S2.file_sha(out / 'stage2_hashes.txt'),
               allocator_ooms_total=sum(int(m['allocator_ooms'] or 0) for m in models), python=sys.version.split()[0])
    (out / 'run.json').write_text(json.dumps(run, indent=1))
    print('done', json.dumps({k: run[k] for k in ('set', 'used', 'recordings', 'prediction_files')}), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('stage', choices=['a', 'b'])
    ap.add_argument('--zip-dir', required=True), ap.add_argument('--out', required=True), ap.add_argument('--vendor', required=True)
    ap.add_argument('--repo'), ap.add_argument('--a7-commit'), ap.add_argument('--remote')
    ap.add_argument('--workers', type=int, default=3), ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--stage1'), ap.add_argument('--manifest', nargs='+'), ap.add_argument('--checkpoint-index')
    ap.add_argument('--set', choices=['confirmatory', 'exploratory']), ap.add_argument('--gpu', default='0')
    ap.add_argument('--seeds'), ap.add_argument('--arms'), ap.add_argument('--backbones')
    ap.add_argument('--fsam-off', action='store_true'), ap.add_argument('--batch1', action='store_true')
    args = ap.parse_args()
    run_a(args) if args.stage == 'a' else run_b(args)


if __name__ == '__main__':
    main()
