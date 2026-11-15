"""Stage 2 of the Stage-5 evaluation: predicted waveforms of every checkpoint (ACCESS_OPS2_BRIEF_02 6.43 마, 6.45 나 4,
6.46 가, 6.48 마).

This process receives no reference path and opens no reference signal (no PPG, no reference HR, no window list): it reads
the checkpoint manifests, the checkpoint index, the subject list, the input plan (MMPD) or the recording list and the
MCD-rPPG index table (db.csv; rows of other subjects are dropped before use), the videos and the frame-time (meta) files.

Seal guard: the subject list and the input plan are checked before any other file is opened. A list or plan of a sealed test
set (MMPD-B, MCD-rPPG test half) stops the run unless --unseal is given with --amendment-commit equal to the last commit
of docs/PREREGISTRATION_STAGE5_AMENDMENTS.md in --repo.

Checkpoints: every checkpoint row of the given manifests (kind = checkpoint) is matched to a file of the checkpoint index;
the file's SHA-256 is compared with the manifest before loading. A missing file, a hash mismatch or a load failure marks
that checkpoint defective (reason recorded; it is not used). Variants (output names):
  PhysNet, RhythmFormer: 'std'; RhythmFormer on MCD-rPPG with --blur: also 'blur15' (Gaussian, sigma pixels, on the
  128 x 128 input before the 30 Hz interpolation and standardization);
  matched FactorizePhys: 'fsam-on' (primary), with --fsam-off also 'fsam-off', with --batch1 also 'fsam-on-batch1'.

Model input and batching (Amendment A3 item 1):
  MMPD: the October input plan in its full order (one global chunk order per view), batches of four that may cross
  recording boundaries, each model's random state carried across batches from seed 100. Frames = round(video * 255) as
  uint8, HC 1.5 face box on frame 0, INTER_AREA 72 (PhysNet, FactorizePhys) / 128 (RhythmFormer), native 30 Hz,
  DiffNormalized (PhysNet) / Standardized (RhythmFormer) / Raw (FactorizePhys), full chunks 128/160/160.
  MCD-rPPG: per recording, all frames decoded, HC 1.5 box on frame 0, INTER_AREA, endpoint-linear frame times over the
  meta span, 30 Hz interpolation, the same transforms, chunks in temporal order, batches of four (the last batch of a
  recording one to three chunks), random state reset per recording.
Outputs (--out, relative names only in the records):
  <dataset>/pred/<key>/<recording>.npy   float32, the concatenated chunk outputs (np.save; byte-reproducible)
  <dataset>/recordings.csv               per recording: input samples, chunks per view, decoded frames, span, face box, and
                                         the face-detection worker's pid, OpenCL in use and OpenCV threads (6.70 가)
  <dataset>/checkpoints.csv              per output key: backbone, arm, seed, variant, set, manifest and file SHA-256,
                                         allocator_ooms (brief 02 6.67 다 3: CUDA allocator OOMs counted during this
                                         checkpoint's loads and forward passes, all workers; any OOM stops stage 2),
                                         epoch, status (used / defective) and reason
  <dataset>/stage2_hashes.txt            'SHA-256  relative name' of every prediction file and of the two CSVs
  <dataset>/run.json                     arguments (relative names), start/end time, operator, versions
Usage: see --help.
"""
import os  # noqa: E402  (first: thread settings precede numpy / cv2 / torch imports; children inherit)
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENCV_FOR_THREADS_NUM='16',
                  OPENCV_OPENCL_DEVICE='disabled')  # 6.70 가: no OpenCL in any worker (with cv2.ocl.setUseOpenCL(False))
import argparse  # noqa: E402
import contextlib  # noqa: E402
import csv  # noqa: E402
import getpass  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait  # noqa: E402
import multiprocessing as mp  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
VIEW = {'FactorizePhys-matched': 'FPm', 'PhysNet': 'PhysNet', 'RhythmFormer': 'RhythmFormer'}
SIZE = {'PhysNet': 72, 'FPm': 72, 'RhythmFormer': 128}
LENGTH = {'PhysNet': 128, 'FPm': 160, 'RhythmFormer': 160}
SEALED_LIST_ROWS = ('mcd_test_subjects',)  # manifest rows that identify a sealed subject list by SHA-256
LOADER = None
VENDOR = None
INF = None
INF_KEY = None


# ----------------------------------------------------------------------------------------------- lists, guard, hashes
def file_sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def read_subjects(path):
    ids = [x.strip() for x in Path(path).read_text().splitlines() if x.strip()]
    if ids and ids[0] == 'subject_id':
        ids = ids[1:]
    assert len(ids) == len(set(ids)), 'duplicate subject IDs'
    return ids


def list_sha(ids):
    """Manifest rule (Section 8): IDs sorted as strings, one per line, UTF-8, LF line ends including the last line."""
    return hashlib.sha256(''.join(i + '\n' for i in sorted(ids)).encode()).hexdigest()


def manifest_rows(paths):
    rows = []
    for p in paths:
        with open(p, newline='') as f:
            rows += [dict(r, manifest=Path(p).name) for r in csv.DictReader(f)]
    return rows


def seal_guard(args, subjects, manifests):
    """Stops before any data file is opened if the subject list or plan is a sealed test list (brief 02 6.43 마)."""
    sealed = []
    sha = list_sha(subjects)
    for r in manifests:
        if r['kind'] in SEALED_LIST_ROWS and r['sha256'] == sha:
            sealed.append(f"subject list = manifest row {r['kind']}")
    if args.sealed_subjects:  # optional frozen split file(s): one sealed subject ID per line
        sealed_ids = set()
        for p in args.sealed_subjects:
            sealed_ids |= set(read_subjects(p))
        n = len(set(subjects) & sealed_ids)
        if n:
            sealed.append(f'{n} subjects of a sealed list')
    if args.dataset == 'MMPD':
        for v in ('FPm', 'PhysNet', 'RhythmFormer'):
            with open(Path(args.plan_dir) / f'{args.plan_prefix}_{v}_input_plan.csv', newline='') as f:
                ds = {r['dataset'] for r in csv.DictReader(f)}
            if ds != {'A'}:
                sealed.append(f'{v} input plan datasets {sorted(ds)}')
    if not sealed:
        return dict(sealed=False, subject_list_sha256=sha)
    if not args.unseal:
        sys.exit('SEALED: ' + '; '.join(sealed) + ' -- stopped before opening any data file')
    if not (args.amendment_commit and args.repo):
        sys.exit('SEALED: --unseal needs --amendment-commit and --repo')
    last = subprocess.run(['git', '-C', args.repo, 'log', '-1', '--format=%H', '--', 'docs/PREREGISTRATION_STAGE5_AMENDMENTS.md'],
                          capture_output=True, text=True, check=True).stdout.strip()
    if last != args.amendment_commit:
        sys.exit(f'SEALED: last amendment commit {last} != --amendment-commit {args.amendment_commit}')
    return dict(sealed=True, unsealed_with=last, reasons=sealed, subject_list_sha256=sha)


def models_from(args, manifests):
    """Output keys: one per (checkpoint, variant); defective checkpoints are listed with their reason."""
    index = {}
    with open(args.checkpoint_index, newline='') as f:
        for r in csv.DictReader(f):
            index[(r['backbone'], r['arm'], int(r['seed']))] = r
    seeds = {int(s) for s in args.seeds.split(',')} if args.seeds else None
    arms = set(args.arms.split(',')) if args.arms else None
    out = []
    for r in manifests:
        if r['kind'] != 'checkpoint':
            continue
        b, arm, seed = r['backbone'], r['arm'], int(r['seed'])
        if (seeds and seed not in seeds) or (arms and arm not in arms) or (args.backbones and b not in args.backbones.split(',')):
            continue
        view = VIEW[b]
        variants = ['std']
        if view == 'FPm':
            variants = ['fsam-on'] + (['fsam-off'] if args.fsam_off else []) + (['fsam-on-batch1'] if args.batch1 else [])
        if view == 'RhythmFormer' and args.dataset == 'MCD' and args.blur:
            variants += [f'blur{int(round(args.blur * 10))}']
        if args.only_variants:
            variants = [v for v in variants if v in args.only_variants.split(',')]
        ix = index.get((b, arm, seed))
        status, reason, sha = 'used', '', ''
        if ix is None or not Path(ix['path']).is_file():
            status, reason = 'defective', 'no checkpoint file'
        else:
            sha = file_sha(ix['path'])
            if sha != r['sha256']:
                status, reason = 'defective', 'manifest hash mismatch'
        for v in variants:
            key = f"{view}__{arm.strip('()')}__s{seed}__{v}"
            out.append(dict(key=key, backbone=b, arm=arm, seed=seed, variant=v, set=args.set, manifest=r['manifest'],
                            manifest_sha256=r['sha256'], file_sha256=sha, status=status, reason=reason, view=view,
                            stream=view + ('@' + v if v.startswith('blur') else ''), batch=1 if v.endswith('batch1') else 4,
                            md_inference=False if v == 'fsam-off' else None, checkpoint=ix['path'] if ix else None,
                            run_id=(ix.get('run_id') or None) if ix else None))
    return out


# ------------------------------------------------------------------------------------------------- inference client
class Infer:
    def __init__(self, models, gpu, log):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONDONTWRITEBYTECODE='1', STAGE5_VENDOR=str(VENDOR))
        self.log = open(log, 'ab')
        self.p = subprocess.Popen([sys.executable, '-B', '-s', str(HERE / 'server.py')], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=self.log, env=env)
        load = [dict(key=m['key'], view=m['view'], checkpoint=m['checkpoint'], run_id=m['run_id'], stream=m['stream'],
                     batch=m['batch'], **({'md_inference': m['md_inference']} if m['md_inference'] is not None else {}))
                for m in models]
        res = self._send(dict(load=load))
        self.loaded, self.failed = res['loaded'], res['failed']
        self.max_reserved_mb = 0
        self.ooms = {}
        self._add_ooms(res)
        self._oom_check(res)

    def _add_ooms(self, meta):
        for k, n in meta.get('ooms_by_key', {}).items():
            self.ooms[k] = self.ooms.get(k, 0) + int(n)

    def _oom_check(self, meta):
        # cuDNN catches a workspace OOM and silently runs another plan (different rounding, kept for later calls); any
        # allocator OOM in the server therefore stops stage 2 (repeat it from the start with more free GPU memory)
        if meta.get('num_ooms', 0):
            by = {k: n for k, n in meta.get('ooms_by_key', {}).items() if n}
            raise RuntimeError(f"GPU memory shortage in the inference server (allocator OOM count {meta['num_ooms']}; by checkpoint {by}): "
                               'outputs may depend on free memory; stage 2 stops (repeat with more free GPU memory)')

    def _send(self, header, payload=None):
        self.p.stdin.write((json.dumps(header) + '\n').encode())
        if payload is not None:
            self.p.stdin.write(payload)
        self.p.stdin.flush()
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError('inference server ended')
        return json.loads(line)

    def predict(self, stream, x, rng):
        x = np.ascontiguousarray(x, dtype=np.float32)
        meta = self._send(dict(predict=stream, shape=list(x.shape), rng=rng), x.tobytes())
        n = int(np.prod(meta['shape'])) * 4
        raw = self.p.stdout.read(n)
        assert len(raw) == n
        self._add_ooms(meta)
        self._oom_check(meta)
        self.max_reserved_mb = max(self.max_reserved_mb, meta.get('reserved_mb', 0))
        return meta['keys'], np.frombuffer(raw, np.float32).reshape(meta['shape'])

    def close(self):
        if self.p.poll() is None:
            self.p.stdin.write(b'{"close": true}\n')
            self.p.stdin.flush()
            self.p.stdin.close()
            self.p.wait(timeout=300)
        self.log.close()


# -------------------------------------------------------------------------------------------------- CPU input preparation
def hc_box(frame):
    """HC face box (rPPG-Toolbox face_detection, larger box 1.5) with OpenCV restricted to 1 thread during the detection
    (11-14, brief 02 6.54 나 4): with several threads the Haar cascade may return its detections in a different order, and
    when two detections share the largest width the toolbox keeps the first (MCD-dev 1298_after: 1 of 100 repeats with 16
    threads picked the other face). One thread fixes the order; the detections themselves do not change.
    6.70 가: OpenCL is switched off (cv2.ocl.setUseOpenCL(False); OPENCV_OPENCL_DEVICE=disabled) and the state during the
    detection is kept in FACE_STATE (worker pid, OpenCL in use, OpenCV threads), written per recording to recordings.csv."""
    import cv2
    cv2.ocl.setUseOpenCL(False)
    n = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        FACE_STATE.update(face_worker_pid=os.getpid(), face_opencl=bool(cv2.ocl.useOpenCL()), face_threads=cv2.getNumThreads())
        return np.asarray(LOADER.face_detection(frame, 'HC', True, 1.5), dtype=int)
    finally:
        cv2.setNumThreads(n)


FACE_STATE = {}


def init_worker(vendor):
    global LOADER, VENDOR
    VENDOR = Path(vendor)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['OPENCV_OPENCL_DEVICE'] = 'disabled'
    os.environ.setdefault('ST9_NAMESPACE', 'st9g')
    sys.path.insert(0, str(VENDOR / 'third_party' / 'rPPG-Toolbox'))
    sys.path.insert(0, str(VENDOR))
    os.chdir(VENDOR / 'third_party' / 'rPPG-Toolbox')
    import torch
    torch.set_num_threads(1)
    with contextlib.redirect_stdout(sys.stderr):
        from dataset.data_loader.BaseLoader import BaseLoader
        LOADER = object.__new__(BaseLoader)


def chunks_of(x, length):
    n = len(x) // length
    return np.ascontiguousarray([x[i * length:(i + 1) * length] for i in range(n)], dtype=np.float32)


def mmpd_prepare(args):
    rid, root, views = args
    import cv2
    from scipy.io import loadmat
    from dataset.data_loader.BaseLoader import BaseLoader
    t0 = time.time()
    mat = loadmat(Path(root) / (rid + '.mat'), variable_names=['video'])
    assert set(mat) - {'__header__', '__version__', '__globals__'} == {'video'}
    frames = np.round(np.asarray(mat.pop('video')) * 255).astype(np.uint8)
    del mat
    with contextlib.redirect_stdout(sys.stderr):
        box = hc_box(frames[0])
    resized = {s: [] for s in {SIZE[v] for v in views}}
    for frame in frames:
        a = frame[max(box[1], 0):min(box[1] + box[3], frame.shape[0]), max(box[0], 0):min(box[0] + box[2], frame.shape[1])]
        for s in resized:
            resized[s].append(cv2.resize(a, (s, s), interpolation=cv2.INTER_AREA))
    native = len(frames)
    del frames
    out = {}
    for v in views:
        x = np.asarray(resized[SIZE[v]], dtype=np.float64)
        if v == 'PhysNet':
            x = BaseLoader.diff_normalize_data(x)
        elif v == 'RhythmFormer':
            x = BaseLoader.standardized_data(x)
        out[v] = chunks_of(x, LENGTH[v])
    return dict(recording_id=rid, native_frames=native, input_samples=native, box=box.tolist(), chunks=out, seconds=time.time() - t0,
                **FACE_STATE)


def mcd_decode(rec, streams):
    """Decoded, cropped and resized frames of one MCD-rPPG recording -> model-input chunks per stream."""
    import cv2
    from dataset.data_loader.BaseLoader import BaseLoader
    from src.st11x.pathways import interpolate_face_input
    cap = cv2.VideoCapture(str(rec['video_path']))
    assert cap.isOpened()
    need72 = bool({'PhysNet', 'FPm'} & {s.split('@')[0] for s in streams})
    need128 = any(s.startswith('RhythmFormer') for s in streams)
    hc72, hc128, box, k = [], [], None, 0
    with contextlib.redirect_stdout(sys.stderr):
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if k == 0:
                box = hc_box(rgb)
            a = rgb[max(box[1], 0):min(box[1] + box[3], rgb.shape[0]), max(box[0], 0):min(box[0] + box[2], rgb.shape[1])]
            if need72:
                hc72.append(cv2.resize(a, (72, 72), interpolation=cv2.INTER_AREA))
            if need128:
                hc128.append(cv2.resize(a, (128, 128), interpolation=cv2.INTER_AREA))
            k += 1
    cap.release()
    n = k
    assert n > 1
    times = np.arange(n, dtype=np.float64) * float(rec['span_seconds']) / (n - 1)
    out, grid = {}, None
    if need72:
        r72, grid = interpolate_face_input(np.asarray(hc72, dtype=np.float64), times)
        r72 = r72.astype(np.float64, copy=False)
        del hc72
        if 'PhysNet' in streams:
            out['PhysNet'] = chunks_of(BaseLoader.diff_normalize_data(r72.copy()), 128)
        if 'FPm' in streams:
            out['FPm'] = chunks_of(r72, 160)
        del r72
    if need128:
        x128 = np.asarray(hc128, dtype=np.float64)
        del hc128
        for s in sorted(st for st in streams if st.startswith('RhythmFormer')):
            x = x128
            if '@blur' in s:
                sigma = int(s.split('@blur')[1]) / 10
                x = np.empty_like(x128)
                for i in range(len(x128)):
                    x[i] = cv2.GaussianBlur(x128[i], (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)
            r, grid = interpolate_face_input(x, times)
            out[s] = chunks_of(BaseLoader.standardized_data(r.astype(np.float64, copy=False)), 160)
            del r
    return dict(recording_id=rec['recording_id'], decoded_frames=n, span_seconds=float(rec['span_seconds']),
                input_samples=len(grid), box=box.tolist(), chunks=out, **FACE_STATE)


def mcd_job(args):
    """One MCD-rPPG recording: decode once, predict every model (random state reset per recording)."""
    global INF, INF_KEY
    rec, models, gpu, out, log = args
    t0 = time.time()
    streams = sorted({m['stream'] for m in models})
    d = mcd_decode(rec, streams)
    key = tuple(m['key'] for m in models)
    new_server = INF_KEY != key
    if new_server:
        if INF is not None:
            INF.close()
        INF = Infer(models, gpu, log)
        INF_KEY = key
        assert not INF.failed, INF.failed  # load failures are found once in the main process before any recording
    before = {} if new_server else dict(INF.ooms)  # a new server's load OOMs count with its first recording
    for s in streams:
        keys, p = INF.predict(s, d['chunks'][s], rng='reset')
        for i, k in enumerate(keys):
            dest = Path(out) / 'pred' / k
            dest.mkdir(parents=True, exist_ok=True)
            tmp = dest / (rec['recording_id'] + '.tmp.npy')
            np.save(tmp, p[i].reshape(-1).astype(np.float32))
            tmp.rename(dest / (rec['recording_id'] + '.npy'))
    row = dict(recording_id=rec['recording_id'], decoded_frames=d['decoded_frames'], span_seconds=d['span_seconds'],
               input_samples=d['input_samples'], box=json.dumps(d['box']),
               **{f'chunks_{s}': len(d['chunks'][s]) for s in streams},
               **{k: d[k] for k in ('face_worker_pid', 'face_opencl', 'face_threads')})
    ooms = {k: n - before.get(k, 0) for k, n in INF.ooms.items()}
    return row, time.time() - t0, ooms


# ---------------------------------------------------------------------------------------------------------- datasets
def mcd_recordings(args, subjects):
    """Recording list (one per subject and state, designated camera) with the meta-file span; db.csv rows of other subjects
    are dropped before any use."""
    sys.path.insert(0, str(VENDOR))
    os.environ.setdefault('ST9_NAMESPACE', 'st9g')
    from src.st5c.mcd_packets import timestamp_table
    keep = set(subjects)
    with open(args.recordings, newline='') as f:
        listed = [r['recording_id'] for r in csv.DictReader(f)]
    assert all(r.split('_')[0] in keep for r in listed), 'recording list has subjects outside the subject list'
    root = Path(args.data_root)
    db = {}
    with (root / 'db.csv').open(newline='') as f:
        for r in csv.DictReader(f):
            if r['patient_id'] not in keep:
                continue  # dropped before use
            if r['view'] == 'front' and r['camera'] == 'FullHDwebcam':
                rid = f"{r['patient_id']}_{r['step']}"
                assert rid not in db, rid
                db[rid] = r
    rows = []
    for rid in listed:
        r = db[rid]
        _, stamps = timestamp_table(root / r['meta'])
        rows.append(dict(recording_id=rid, subject_id=r['patient_id'], video_path=str(root / r['video']),
                         span_seconds=(int(stamps[-1]) - int(stamps[0])) / 1e6))
    return rows


def run_mcd(args, models, out, log):
    rows = mcd_recordings(args, read_subjects(args.subjects))
    if args.limit:
        rows = rows[:args.limit]
    rows = [r for r in rows if not all((out / 'pred' / m['key'] / (r['recording_id'] + '.npy')).exists() for m in models)]
    print('MCD recordings to predict', len(rows), flush=True)
    recs, ooms = [], {}
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn'), initializer=init_worker,
                             initargs=(str(VENDOR),)) as ex:
        todo = list(rows)
        inflight = {}
        while todo or inflight:
            while todo and len(inflight) < args.workers:
                r = todo.pop(0)
                inflight[ex.submit(mcd_job, (r, models, args.gpu, str(out), log))] = r
            fin, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
            for f in fin:
                r = inflight.pop(f)
                row, sec, oo = f.result()  # a failure stops the run (stage 2 is repeated from the start, not patched)
                recs.append(row)
                for k, n in oo.items():
                    ooms[k] = ooms.get(k, 0) + n
                print(f"mcd {len(recs)}/{len(rows)} {r['recording_id']} {sec:.0f}s total {time.time() - started:.0f}s", flush=True)
    return recs, ooms


def run_mmpd(args, models, out, log):
    views = sorted({m['view'] for m in models})
    plan = {}
    for v in views:
        with open(Path(args.plan_dir) / f'{args.plan_prefix}_{v}_input_plan.csv', newline='') as f:
            plan[v] = [(r['recording_id'], int(r['chunk_index'])) for r in csv.DictReader(f)]
    order = []
    for rid, _ in plan[views[0]]:
        if not order or order[-1] != rid:
            order.append(rid)
    keep = set(read_subjects(args.subjects))
    assert all(rid.split('/')[0] in keep for rid in order), 'plan has recordings outside the subject list'
    if args.limit:
        order = order[:args.limit]
        plan = {v: [x for x in plan[v] if x[0] in set(order)] for v in plan}
    for v in views:
        o = []
        for rid, _ in plan[v]:
            if not o or o[-1] != rid:
                o.append(rid)
        assert o == order, 'views differ in recording order'
    plan_j = {v: {} for v in views}
    for v in views:
        for rid, j in plan[v]:
            plan_j[v].setdefault(rid, []).append(j)
    assert not any((out / 'pred' / m['key']).exists() for m in models), 'outputs exist; the global order needs a fresh run'
    inf = Infer(models, args.gpu, log)
    assert not inf.failed, inf.failed
    buffers = {v: [] for v in views}
    parts = {(m['key'], r): {} for m in models for r in order}
    remaining, recs = {}, {}
    view_of = {m['key']: m['view'] for m in models}

    def write(rid, k, p):
        dest = out / 'pred' / k
        dest.mkdir(parents=True, exist_ok=True)
        name = rid.replace('/', '__')
        tmp = dest / (name + '.tmp.npy')
        np.save(tmp, np.concatenate([p[j].reshape(-1) for j in sorted(p)]).astype(np.float32))
        tmp.rename(dest / (name + '.npy'))

    def flush(v, final=False):
        buf = buffers[v]
        n = len(buf) if final else (len(buf) // 4) * 4
        if n == 0:
            return
        batch, buffers[v] = buf[:n], buf[n:]
        keys, p = inf.predict(v, np.stack([a for _, _, a in batch]), rng='carry')
        for i, (rid, j, _) in enumerate(batch):
            for ki, key in enumerate(keys):
                parts[(key, rid)][j] = p[ki, i]
        for rid in {rid for rid, _, _ in batch}:
            if all(len(parts[(k, rid)]) == remaining[(view_of[k], rid)] for k in keys):
                for k in keys:
                    write(rid, k, parts.pop((k, rid)))

    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn'), initializer=init_worker,
                             initargs=(str(VENDOR),)) as ex:
        futures = [ex.submit(mmpd_prepare, (r, args.data_root, views)) for r in order[:args.workers + 2]]
        nxt = len(futures)
        for i, rid in enumerate(order):
            res = futures[i].result()
            futures[i] = None
            if nxt < len(order):
                futures.append(ex.submit(mmpd_prepare, (order[nxt], args.data_root, views)))
                nxt += 1
            assert res['recording_id'] == rid
            recs[rid] = dict(recording_id=rid, native_frames=res['native_frames'], input_samples=res['input_samples'],
                             box=json.dumps(res['box']), **{f'chunks_{v}': len(res['chunks'][v]) for v in views},
                             **{k: res[k] for k in ('face_worker_pid', 'face_opencl', 'face_threads')})
            for v in views:
                arr = res['chunks'][v]
                assert sorted(plan_j[v][rid]) == list(range(len(arr))), (rid, v)
                remaining[(v, rid)] = len(arr)
                buffers[v].extend((rid, j, arr[j]) for j in plan_j[v][rid])
                flush(v)
            print(f"mmpd {i + 1}/{len(order)} {rid} prepare {res['seconds']:.0f}s total {time.time() - started:.0f}s", flush=True)
    for v in views:
        flush(v, final=True)
    ooms = dict(inf.ooms)
    inf.close()
    assert not parts, 'unwritten predictions remain'
    return [recs[r] for r in order], ooms


# -------------------------------------------------------------------------------------------------------------- main
def write_csv(path, rows):
    cols = []
    for r in rows:
        cols += [c for c in r if c not in cols]
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def main():
    global VENDOR
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--dataset', required=True, choices=['MMPD', 'MCD'])
    ap.add_argument('--data-root', required=True, help='MMPD: folder of <recording>.mat; MCD: dataset root with db.csv')
    ap.add_argument('--subjects', required=True, help='subject list (one ID per line)')
    ap.add_argument('--recordings', help='MCD: recording list CSV (recording_id)')
    ap.add_argument('--plan-dir', help='MMPD: folder of <prefix>_<view>_input_plan.csv')
    ap.add_argument('--plan-prefix', default='st30e_A')
    ap.add_argument('--manifest', nargs='+', required=True, help='checkpoint manifests (CSV: kind,backbone,arm,seed,n,sha256)')
    ap.add_argument('--checkpoint-index', required=True, help='CSV: backbone,arm,seed,path[,run_id]')
    ap.add_argument('--set', required=True, choices=['confirmatory', 'exploratory'])
    ap.add_argument('--seeds'), ap.add_argument('--arms'), ap.add_argument('--backbones')
    ap.add_argument('--fsam-off', action='store_true'), ap.add_argument('--batch1', action='store_true')
    ap.add_argument('--blur', type=float, default=0.0, help='MCD: RhythmFormer blur sigma (pixels), e.g. 1.5')
    ap.add_argument('--only-variants', help='comma list of variants to run (e.g. fsam-off)')
    ap.add_argument('--vendor', required=True, help='archived source tree (src/, configs/, third_party/rPPG-Toolbox)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--gpu', default='0'), ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--sealed-subjects', nargs='*', help='optional sealed subject lists (one ID per line)')
    ap.add_argument('--unseal', action='store_true'), ap.add_argument('--amendment-commit'), ap.add_argument('--repo')
    ap.add_argument('--limit', type=int, default=0, help='test only: first N recordings (recorded in run.json)')
    args = ap.parse_args()
    VENDOR = Path(args.vendor).resolve()
    manifests = manifest_rows(args.manifest)
    subjects = read_subjects(args.subjects)
    guard = seal_guard(args, subjects, manifests)  # before any data file is opened
    out = Path(args.out) / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    log = str(out / 'server.log')
    started = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    models = models_from(args, manifests)
    used = [m for m in models if m['status'] == 'used']
    probe_ooms = {}
    if used:  # load check in the main process: load failures become defects before any recording is opened
        probe = Infer(used, args.gpu, log)
        probe_ooms = dict(probe.ooms)
        failed = {f['key']: f['error'] for f in probe.failed}
        epochs = {x['key']: x['epoch'] for x in probe.loaded}
        probe.close()
        for m in models:
            if m['key'] in failed:
                m.update(status='defective', reason='load failure: ' + failed[m['key']])
            m['epoch'] = epochs.get(m['key'], '')
        used = [m for m in models if m['status'] == 'used']
    print('outputs', len(models), 'used', len(used), 'defective', len(models) - len(used), flush=True)
    recs, ooms = run_mmpd(args, used, out, log) if args.dataset == 'MMPD' else run_mcd(args, used, out, log)
    for m in models:  # brief 02 6.67 다 3: allocator OOMs per checkpoint (load check, loads and forward passes of all workers)
        m['allocator_ooms'] = (probe_ooms.get(m['key'], 0) + ooms.get(m['key'], 0)) if m['status'] == 'used' else ''
    write_csv(out / 'recordings.csv', sorted(recs, key=lambda r: r['recording_id']))
    write_csv(out / 'checkpoints.csv', [{k: v for k, v in m.items() if k not in ('checkpoint', 'run_id', 'md_inference')} for m in models])
    lines = []
    for p in sorted((out / 'pred').rglob('*.npy')):
        lines.append(f"{file_sha(p)}  {args.dataset}/{p.relative_to(out).as_posix()}")
    for n in ('recordings.csv', 'checkpoints.csv'):
        lines.append(f"{file_sha(out / n)}  {args.dataset}/{n}")
    (out / 'stage2_hashes.txt').write_text(''.join(x + '\n' for x in lines))
    run = dict(stage=2, dataset=args.dataset, set=args.set, started=started, ended=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
               operator=getpass.getuser(), guard=guard, gpu_memory_guard='stop on any CUDA allocator OOM in the inference server (num_ooms > 0)', manifests=[Path(p).name for p in args.manifest],
               manifest_sha256={Path(p).name: file_sha(p) for p in args.manifest}, subject_list_sha256=guard['subject_list_sha256'],
               variants=dict(fsam_off=args.fsam_off, batch1=args.batch1, blur=args.blur, only=args.only_variants),
               filters=dict(seeds=args.seeds, arms=args.arms, backbones=args.backbones, limit=args.limit), outputs=len(models), used=len(used),
               recordings=len(recs), prediction_files=len(lines) - 2, hash_file_sha256=file_sha(out / 'stage2_hashes.txt'),
               allocator_ooms_total=sum(int(m['allocator_ooms'] or 0) for m in models),
               python=sys.version.split()[0])
    (out / 'run.json').write_text(json.dumps(run, indent=1))
    print('done', json.dumps({k: run[k] for k in ('dataset', 'set', 'used', 'recordings', 'prediction_files')}), flush=True)


if __name__ == '__main__':
    main()
