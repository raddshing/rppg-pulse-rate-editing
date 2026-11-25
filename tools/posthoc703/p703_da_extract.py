"""Brief 02 7.03 다 (part 1): per-frame face-mask weighted RGB of the recordings of the cross-corpus audit (6.97 나,
Table VI), CPU only, no model. Post hoc, existing outputs and training/development data only.

The audit tool tools/audit_cross.py is imported unchanged (recording lists, readers, decoder, archived st9e face masks).
Per recording and per non-overlapping complete 300-frame context from frame 0, exactly as the audit job: face masks of the
context (archived face_masks), the face-oval-mask weighted mean RGB per frame (weighted_rgb), the contact PPG per frame of
the audit reader. This tool keeps, in addition to the audit's per-context lag curve (recomputed here as a check against
the stored audit curves), the per-frame weighted RGB and PPG of every context, so that the recording-level cross spectrum
can be computed without reading the video again. Nothing else is changed. Outputs stay on the server (cache).
Usage: p703_da_extract.py CORPUS [--workers N] [--limit N]
"""
import os
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
import multiprocessing as mp  # noqa: E402
from pathlib import Path  # noqa: E402

WORK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORK / 'tools'))
import audit_cross as AC  # noqa: E402  (imports audit/a1_common: vendor st9e helpers, lag_curve, summarize)
import numpy as np  # noqa: E402

OUTD = WORK / 'cache' / 'posthoc703' / 'da_traces'


def job(args):
    """tools/audit_cross.py job, with the per-frame weighted RGB and PPG of each context kept."""
    corpus, rid, extra = args
    t0 = time.time()
    rows, curves, keys, rgb, ppg, starts = [], [], [], [], [], []
    try:
        n, gt, fps, contexts, info = AC.source(corpus, rid, extra)
        rec = dict(corpus=corpus, recording_id=rid, n_frames=n, fps=float(fps), **info)
        for j, clip in enumerate(contexts):
            if j >= n // 300:
                break
            start = j * 300
            AC.check('Complete context', clip.shape[0] == 300)
            p_raw = gt[start:start + 300]
            base = dict(corpus=corpus, recording_id=rid, context_index=j, start_frame=start)
            if not np.isfinite(p_raw).all():
                rows.append(dict(base, status='ppg_unavailable'))
                continue
            try:
                face, _, failure = AC.face_masks(clip)
            except Exception as ex:  # archived checks (no landmark in the window, empty mask)
                rows.append(dict(base, status='mask_failure', error=str(ex)[:120]))
                continue
            w = AC.weighted_rgb(clip, face)
            y = AC.editor_band(AC.green(w))
            p = AC.editor_band(p_raw)
            curves.append(AC.lag_curve(y, p))
            keys.append(f'{rid}|{j}')
            rgb.append(w)
            ppg.append(np.asarray(p_raw, np.float64))
            starts.append(start)
            rows.append(dict(base, status='ok', mask_failure_fraction=failure))
            del clip, face
        rec.update(status='ok', contexts_ok=sum(r['status'] == 'ok' for r in rows), contexts_total=len(rows),
                   seconds=time.time() - t0)
        out = OUTD / corpus
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / (rid.replace('/', '__') + '.npz'), rgb=np.asarray(rgb), ppg=np.asarray(ppg), starts=np.asarray(starts),
                 curves=np.asarray(curves), keys=np.asarray(keys), fps=float(fps), n_frames=int(n))
    except Exception:
        rec = dict(corpus=corpus, recording_id=rid, status='error', error=traceback.format_exc()[-400:], seconds=time.time() - t0)
    return rec, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('corpus', choices=['PURE', 'UBFC', 'MMPD', 'MCD'])
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    rids = AC.ids(a.corpus)
    if a.limit:
        rids = rids[:a.limit]
    extra = AC.mcd_rows(set(rids)) if a.corpus == 'MCD' else {}
    if a.corpus == 'MCD':
        assert set(extra) == set(rids)
    jobs = [(a.corpus, r, extra.get(r)) for r in rids]
    started = time.time()
    recs, rows = [], []
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context('spawn'), initializer=AC.init_worker) as ex:
        for k, (rec, rr) in enumerate(ex.map(job, jobs), 1):
            recs.append(rec)
            rows += rr
            print(f"{a.corpus} {k}/{len(jobs)} {rec['status']} {rec.get('contexts_ok')}/{rec.get('contexts_total')} "
                  f"{time.time() - started:.0f}s", flush=True)
    OUTD.mkdir(parents=True, exist_ok=True)
    tag = f'_limit{a.limit}' if a.limit else ''
    (OUTD / f'{a.corpus}{tag}_run.json').write_text(json.dumps(dict(
        corpus=a.corpus, recordings=len(recs), errors=sum(r['status'] != 'ok' for r in recs),
        contexts_ok=sum(r['status'] == 'ok' for r in rows), contexts_not_ok=sum(r['status'] != 'ok' for r in rows),
        seconds=time.time() - started, ended=time.strftime('%Y-%m-%dT%H:%M:%S%z'), recordings_meta=recs), indent=1, default=str))
    print('done', a.corpus, len(recs), 'recordings', sum(r['status'] == 'ok' for r in rows), 'contexts ok',
          sum(r['status'] != 'ok' for r in rows), 'not ok', sum(r['status'] != 'ok' for r in recs), 'recording errors',
          f'{time.time() - started:.0f}s', flush=True)


if __name__ == '__main__':
    main()
