"""Brief 02 6.97 나: cross-corpus natural audit (face green vs contact PPG lag), CPU only, no model.

The PURE 1a audit code is used unchanged (audit/a1_natural.py job, audit/a1_common.py lag_curve/summarize, archived
st9e face_masks / weighted_rgb / green / editor_band / common_hr): per non-overlapping complete 300-frame context from
frame 0, the face-oval-mask weighted mean RGB -> G / mean(G) - 1 -> first-order 0.6-3.3 Hz zero-phase filter (30 Hz
assumed), the contact PPG with the same filter, the Pearson correlation for lags -30..+30 (lag > 0: video later than PPG),
and the context HR of the contact PPG (common_hr 'HAR'). Only data reading and paths change:
- PURE: as 1a (PNG frames, label JSON waveform interpolated to the frame count); fps from the /Image timestamps.
- UBFC-rPPG DATASET_2 (42): vid.avi decoded in order (BGR -> RGB); ground_truth.txt line 1 (PPG), interpolated to the
  decoded frame count as the PURE loader does (identity when the lengths agree); fps from line 3 (seconds).
- MMPD-A (220, the draft's A reference list only; no B file is opened): 'video' -> rint(255 x) uint8 as the archived
  frame_window; 'GT_ppg' per frame; fps 30 (nominal, no timestamps).
- MCD-rPPG development half, draft-rule recordings (89): db.csv rows of non-dev subjects dropped before use; video decoded
  in order; ppg_sync column 0, row j paired with frame k* + j (k* of the 1b alignment); fps from the frame timestamps.
A context whose face masks fail (archived checks) is recorded and left out. Outputs stay on the server.
Usage: audit_cross.py CORPUS [--workers N] [--limit N] [--max-contexts N] [--tag TAG]
"""
import os
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
import multiprocessing as mp  # noqa: E402
from pathlib import Path  # noqa: E402

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK / 'audit'))
from a1_common import *  # noqa: E402,F401,F403  (vendor st9e helpers, lag_curve, summarize, LAGS, pure_index, tau_table)

CFG = json.loads((Path(__file__).resolve().parents[1] / 'config' / 'paths.json').read_text())  # 7.08: machine paths, see config/paths.example.json
UBFC = Path(CFG['ubfc2_raw'])  # read-only
MMPD = Path(CFG['mmpd_raw'])  # read-only, A recordings only
MCD = Path(CFG['mcd_raw'])  # read-only, dev rows only
OUTX = WORK / 'outputs' / 'audit_cross'


def init_worker():
    import cv2
    cv2.setNumThreads(1)


# ------------------------------------------------------------------ recording lists
def ids(corpus):
    if corpus == 'PURE':
        r = sorted(tau_table())
        assert len(r) == 59
        return r
    if corpus == 'UBFC':
        r = sorted((p.name for p in UBFC.iterdir() if p.is_dir() and p.name.startswith('subject')), key=lambda s: int(s[7:]))
        assert len(r) == 42
        return r
    if corpus == 'MMPD':
        r = sorted({x['recording_id'] for x in read_csv(VENDOR / 'outputs/st30a_new_A_reference.csv')})
        assert len(r) == 220
        return r
    if corpus == 'MCD':
        import pandas as pd
        recs = pd.read_csv(WORK / 'outputs/1b/mcd_dev_recordings.csv', dtype={'subject_id': str})
        w = pd.read_csv(WORK / 'outputs/1b/mcd_dev_windows.csv', dtype={'subject_id': str})
        conf = set(recs.recording_id[recs.mapping_confirmed.astype(bool)])
        har = w[w.layer.eq('HAR') & w.primary_included & w.agreement & w.h_FFT.ge(100)]
        r = sorted(set(har.recording_id) & conf)  # b1/analyze.py load_mcd draft_rule recordings
        assert len(r) == 89
        return r
    raise ValueError(corpus)


def mcd_rows(wanted):
    import pandas as pd
    recs = pd.read_csv(WORK / 'outputs/1b/mcd_dev_recordings.csv', dtype={'subject_id': str}).set_index('recording_id')
    dev = set(recs.subject_id)
    out = {}
    with (MCD / 'db.csv').open(newline='') as f:
        for r in csv.DictReader(f):
            if r['patient_id'] not in dev:
                continue  # non-dev row dropped before any use
            rid = f"{r['patient_id']}_{r['step']}"
            if rid in wanted and r['view'] == 'front' and r['camera'] == 'FullHDwebcam':
                out[rid] = dict(video=r['video'], meta=r['meta'], ppg_sync=r['ppg_sync'], k_star=int(recs.loc[rid, 'k_star']))
    return out


# ------------------------------------------------------------------ readers: (n_frames, ppg per frame, fps, context reader)
# Video files are decoded by cv2.VideoCapture.read() of the repro environment (OpenCV 4.5.2, as the study's UBFC and MCD
# readers) in a child process (tools/decode_frames_repro.py); the main environment's OpenCV 5.0 crashes on UBFC vid.avi.
REPRO = Path(CFG['python_repro'])  # read-only, run only
DECODER = WORK / 'tools' / 'decode_frames_repro.py'
CHILD_ENV = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')


def video_contexts(path):
    import subprocess
    p = subprocess.Popen([str(REPRO), '-B', '-s', str(DECODER), str(path), 'stream'], stdout=subprocess.PIPE,
                         env=CHILD_ENV)
    try:
        head = p.stdout.read(8)
        if len(head) < 8:
            return
        h, w = (int(x) for x in np.frombuffer(head, dtype=np.uint32))
        size = 300 * h * w * 3
        while True:
            buf = p.stdout.read(size)
            if len(buf) < size:
                return
            yield np.frombuffer(buf, dtype=np.uint8).reshape(300, h, w, 3).copy()
    finally:
        p.stdout.close()
        p.kill()
        p.wait()


def decoded_count(path):
    import subprocess
    r = subprocess.run([str(REPRO), '-B', '-s', str(DECODER), str(path), 'count'], capture_output=True, text=True,
                       env=CHILD_ENV, check=True)
    return int(r.stdout.strip().splitlines()[-1])


def source(corpus, rid, extra=None):
    if corpus == 'PURE':
        row = pure_index()[rid]
        files = frame_files(row)
        gt = contact(row, 'PURE', None)
        labels = json.loads(Path(row['label_path']).read_text())
        t = np.asarray([v['Timestamp'] for v in labels['/Image']], float)
        fps = (len(t) - 1) / ((t[-1] - t[0]) / 1e9)
        return len(files), gt, fps, (read_pure(files, j * 300) for j in range(len(files) // 300)), dict(fps_source='image timestamps')
    if corpus == 'UBFC':
        d = UBFC / rid
        lines = [l for l in (d / 'ground_truth.txt').read_text().split('\n') if l.strip()]
        raw = np.asarray([float(x) for x in lines[0].split()])
        tt = np.asarray([float(x) for x in lines[2].split()])
        n = decoded_count(d / 'vid.avi')
        gt = np.interp(np.linspace(1, len(raw), n), np.linspace(1, len(raw), len(raw)), raw)
        fps = (len(tt) - 1) / (tt[-1] - tt[0])
        return n, gt, fps, video_contexts(d / 'vid.avi'), dict(fps_source='ground_truth time line', ppg_samples=len(raw))
    if corpus == 'MMPD':
        from scipy.io import loadmat
        p = MMPD / (rid + '.mat')
        video = loadmat(p, variable_names=['video'])['video']
        gt = loadmat(p, variable_names=['GT_ppg'])['GT_ppg'].T.reshape(-1).astype(np.float64)
        check('MMPD frame and label counts', len(video) == len(gt) and video.ndim == 4 and video.shape[-1] == 3)

        def ctx():
            for j in range(len(video) // 300):
                x = video[j * 300:(j + 1) * 300]
                check('Native MMPD window normalized range', np.isfinite(x).all() and x.min() >= 0 and x.max() <= 1)
                yield np.rint(255 * x).astype(np.uint8)
        return len(video), gt, 30.0, ctx(), dict(fps_source='nominal 30')
    if corpus == 'MCD':
        from src.st5c.mcd_align import timestamp_table
        e = extra
        sync = np.loadtxt(MCD / e['ppg_sync'], ndmin=2)[:, 0]
        _, stamps = timestamp_table(MCD / e['meta'])
        k = e['k_star']
        nd = decoded_count(MCD / e['video'])
        n = min(nd, k + len(sync))
        gt = np.full(n, np.nan)
        gt[k:n] = sync[:n - k]
        meta = (stamps.astype(float) - float(stamps[0])) / 1e6
        fps = (n - 1) / (meta[n - 1] - meta[0])
        info = dict(fps_source='frame timestamps', decoded_frames=nd, sync_rows=len(sync), k_star=k,
                    max_frame_gap_s=float(np.max(np.diff(meta[:n]))))

        def ctx():
            # frames before k* have no PPG: skip them in whole contexts from frame k*
            check('k* = 0 for the draft-rule recordings', k == 0)
            yield from video_contexts(MCD / e['video'])
        return n, gt, fps, ctx(), info
    raise ValueError(corpus)


def job(args):
    corpus, rid, extra, max_contexts = args
    t0 = time.time()
    rows, curves = [], []
    try:
        n, gt, fps, contexts, info = source(corpus, rid, extra)
        rec = dict(corpus=corpus, recording_id=rid, n_frames=n, fps=float(fps), **info)
        for j, clip in enumerate(contexts):
            if j >= n // 300 or (max_contexts and j >= max_contexts):
                break
            start = j * 300
            check('Complete context', clip.shape[0] == 300)
            p_raw = gt[start:start + 300]
            base = dict(corpus=corpus, recording_id=rid, context_index=j, start_frame=start)
            if not np.isfinite(p_raw).all():
                rows.append(dict(base, status='ppg_unavailable'))
                continue
            try:
                face, _, failure = face_masks(clip)
            except Exception as ex:  # archived checks (no landmark in the window, empty mask)
                rows.append(dict(base, status='mask_failure', error=str(ex)[:120], HR=common_hr(p_raw, 'HAR')))
                continue
            y = editor_band(green(weighted_rgb(clip, face)))
            p = editor_band(p_raw)
            c = lag_curve(y, p)
            curves.append(c)
            rows.append(dict(base, status='ok', HR=common_hr(p_raw, 'HAR'), mask_failure_fraction=failure, **summarize(c)))
            del clip, face
        rec.update(status='ok', contexts_ok=sum(r['status'] == 'ok' for r in rows), contexts_total=len(rows),
                   seconds=time.time() - t0)
    except Exception:
        rec = dict(corpus=corpus, recording_id=rid, status='error', error=traceback.format_exc()[-400:], seconds=time.time() - t0)
    return rec, rows, curves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('corpus', choices=['PURE', 'UBFC', 'MMPD', 'MCD'])
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--max-contexts', type=int, default=0)
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    OUTX.mkdir(parents=True, exist_ok=True)
    rids = ids(a.corpus)
    if a.limit:
        rids = rids[:a.limit]
    extra = mcd_rows(set(rids)) if a.corpus == 'MCD' else {}
    if a.corpus == 'MCD':
        assert set(extra) == set(rids)
    jobs = [(a.corpus, r, extra.get(r), a.max_contexts) for r in rids]
    started = time.time()
    recs, rows, curves = [], [], []
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context('spawn'), initializer=init_worker) as ex:
        for k, (rec, rr, cc) in enumerate(ex.map(job, jobs), 1):
            recs.append(rec)
            rows += rr
            curves += cc
            print(f"{a.corpus} {k}/{len(jobs)} {rec['recording_id']} {rec['status']} "
                  f"{rec.get('contexts_ok')}/{rec.get('contexts_total')} {time.time() - started:.0f}s", flush=True)
    stem = OUTX / f'{a.corpus}{a.tag}'
    write_rows(Path(f'{stem}_recordings_meta.csv'), recs)
    write_rows(Path(f'{stem}_contexts.csv'), rows)
    ok = [r for r in rows if r['status'] == 'ok']
    np.savez(f'{stem}_curves.npz', lags=LAGS, curves=np.asarray(curves),
             keys=np.asarray([f"{r['recording_id']}|{r['context_index']}" for r in ok]))
    print('done', a.corpus, len(recs), 'recordings', len(ok), 'contexts ok', len(rows) - len(ok), 'not ok',
          sum(r['status'] != 'ok' for r in recs), 'recording errors', f'{time.time() - started:.0f}s', flush=True)


if __name__ == '__main__':
    main()
