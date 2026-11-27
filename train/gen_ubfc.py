"""7.09b E1: UBFC-rPPG originals and time-resampling pools (family i), from cache/ubfc709/plan manifests.

Frames: tools/ubfc_video_reader.py (OpenCV 4.5.2, as the October UBFC reader); contact PPG: ground_truth.txt line 1, one
value per frame; all quantities in frame units with fs = 30 (as the October UBFC work and rPPG-Toolbox).

originals: per recording, the model view of the whole recording (PhysNet: ViewWorker 'PhysNet', 72; RhythmFormer: the
  RhythmFormer view worker, 128; FactorizePhys: the PhysNet 72 view, raw) and the archived whole-recording normalization
  (PhysNet diff_normalize_data/label; RhythmFormer standardized_data/label; FactorizePhys raw input, standardized_label),
  as the October UBFC originals (src/st30g/ubfc_worker.py, arm original) and the October PURE FactorizePhys originals
  (src/st30g/fpm_originals.py). Stored per recording: the uint8 view (delta-zlib), the scalar statistics of the
  normalization and the raw contact PPG; a chunk tensor is rebuilt at load time by original_tensor(). Every chunk tensor
  rebuilt this way is compared with the slice of the whole-recording arrays (must be equal) and, for PhysNet and
  RhythmFormer, its SHA-256 with the October UBFC original manifest.
family_i: per item, the archived (i) temporal resampling (shifted_start, linear interpolation of uint8 frames with stochastic
  rounding, generator default_rng([20261159, j, h])) and three labels: (i) shift tau_rec (r - 1), (i-pc) shift d (r - 1)
  with d = 1 (UBFC audit, 7.09b), (i-simple) no shift; edge hold as train/gen_i_pc.py. Stored as in the second pool
  (uint8 view frames of the chunk, raw float64 labels); model inputs at load time by train/u8_store.model_input.
Usage: gen_ubfc.py BACKBONE originals|family_i OUT_ROOT [RECORDING ...]   (OPS2_WORKERS, OPS2_POOL_COMPRESS)
"""
import atexit
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK / 'audit'))
sys.path.insert(0, str(WORK / 'train'))
from a1_common import *  # noqa: F401,F403  (ViewWorker, shifted_start, common_hr, write_rows, read_csv, np)
from infer_client import RhythmView  # noqa: E402
import u8_store  # noqa: E402
from src.st9g.preprocess import diff_normalize_data, diff_normalize_label, standardized_data, standardized_label  # noqa: E402

PLAN = WORK / 'cache' / 'ubfc709' / 'plan'
CFG = json.loads((Path(__file__).resolve().parents[1] / 'config' / 'paths.json').read_text())  # 7.19: machine paths, see config/paths.example.json
REPRO = Path(CFG['python_repro'])
KEY = 20261159
J = {'PhysNet': 1, 'RhythmFormer': 1, 'FPm': 2}
D_UBFC = 1
EDGE_HOLD_STOP_FRAMES = 3.0
W = R = READER = None


class UBFCReader:
    def __init__(self):
        self.log = (WORK / 'logs' / 'ubfc_video_reader.log').open('ab')
        self.p = subprocess.Popen([str(REPRO), '-B', '-s', str(WORK / 'tools' / 'ubfc_video_reader.py')], cwd=WORK,
                                  env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=self.log)

    def read(self, path, start, count):
        self.p.stdin.write((json.dumps(dict(path=path, start=start, count=count)) + '\n').encode())
        self.p.stdin.flush()
        meta = json.loads(self.p.stdout.readline())
        assert meta['opencv_version'] == '4.5.2'
        n = int(np.prod(meta['shape']))
        raw = self.p.stdout.read(n)
        assert len(raw) == n
        return np.frombuffer(raw, np.uint8).reshape(meta['shape'])

    def close(self):
        if self.p.poll() is None:
            self.p.stdin.write(b'{"close":true}\n')
            self.p.stdin.flush()
            self.p.stdin.close()
            self.p.wait(timeout=60)
        self.log.close()


def init_worker(backbone):
    global W, R, READER
    READER = UBFCReader()
    atexit.register(READER.close)
    if backbone == 'RhythmFormer':
        R = RhythmView()
        atexit.register(R.close)
    else:
        W = ViewWorker()
        atexit.register(W.close)


def contact_ubfc(path, frames):
    with open(path) as f:
        gt = np.asarray([float(x) for x in f.readline().split()])
    assert len(gt) == frames, 'one contact value per frame'
    return gt


def std_repro(d):
    """float32 np.std of a float32 array computed by numpy 1.22 (repro interpreter), as the October UBFC originals.

    numpy 2.4.6 (main) and 1.22 (repro) can differ by one float32 ulp on whole-recording arrays (subject1: 0.0075678094 vs
    0.00756781; the October SHA-256 of all 12 chunks match only the second)."""
    tmp = WORK / 'cache' / 'ubfc709' / 'tmp'
    tmp.mkdir(parents=True, exist_ok=True)
    path = tmp / f'diff_{os.getpid()}.npy'
    np.save(path, np.ascontiguousarray(d, dtype=np.float32))
    try:
        out = subprocess.run([str(REPRO), '-B', '-s', '-c', 'import sys, numpy as np; d = np.load(sys.argv[1]); '
                              'print(int(np.asarray(np.std(d), dtype=np.float32).view(np.uint32)))', str(path)],
                             capture_output=True, text=True, check=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
    finally:
        path.unlink()
    return np.asarray([int(out.stdout.strip())], dtype=np.uint32).view(np.float32)


def stats_of(backbone, view_u8):
    v = view_u8.astype(np.float64)
    if backbone == 'PhysNet':
        return std_repro(u8_store._diff_frames(v))
    if backbone == 'RhythmFormer':
        mean = np.mean(v)
        return np.asarray([mean, np.std(v - mean)], dtype=np.float64)
    return None


def original_tensor(backbone, view_u8, stats, gt, start, length):
    """(x, y) float32 of one original chunk from the whole-recording uint8 view, statistics and contact PPG."""
    n = len(view_u8)
    if backbone == 'PhysNet':
        v = view_u8[start:min(start + length + 1, n)].astype(np.float64)
        x = u8_store._diff_frames(v) / np.float32(stats[0])
        if start + length == n:
            x = np.append(x, np.zeros((1,) + x.shape[1:], dtype=np.float32), axis=0)
        x[np.isnan(x)] = 0
        y = diff_normalize_label(gt)[start:start + length]
    elif backbone == 'RhythmFormer':
        x = (view_u8[start:start + length].astype(np.float64) - np.float64(stats[0])) / np.float64(stats[1])
        x[np.isnan(x)] = 0
        y = standardized_label(gt)[start:start + length]
    else:
        x = view_u8[start:start + length].astype(np.float64)
        y = standardized_label(gt)[start:start + length]
    assert len(x) == length and len(y) == length
    return np.ascontiguousarray(x, dtype=np.float32), np.ascontiguousarray(y, dtype=np.float32)


def job_originals(args):
    backbone, out_root, rid, items = args
    r0 = items[0]
    n = int(r0['frame_count'])
    native = READER.read(r0['source_video_path'], 0, n)
    gt = contact_ubfc(r0['ground_truth_path'], n)
    tick = time.time()
    if backbone == 'RhythmFormer':
        view, boxes = R.view(native)
    else:
        view, boxes = W.view(native, 'PhysNet')
    del native
    u8 = u8_store.to_u8(view)
    if backbone == 'PhysNet':  # diff_normalize_data with the numpy-1.22 std (std_repro); label as archived
        d = u8_store._diff_frames(view)
        whole_x = np.append(d / std_repro(d)[0], np.zeros((1,) + d.shape[1:], dtype=np.float32), axis=0)
        whole_x[np.isnan(whole_x)] = 0
        whole_y = diff_normalize_label(gt)
        del d
    elif backbone == 'RhythmFormer':
        whole_x, whole_y = standardized_data(view), standardized_label(gt)
    else:
        whole_x, whole_y = view, standardized_label(gt)
    del view
    stats = stats_of(backbone, u8)
    folder = Path(out_root) / backbone / 'original' / rid
    folder.mkdir(parents=True, exist_ok=True)
    vp = u8_store.save_view(folder / 'recording_view', u8)
    gp = u8_store.save_array(folder / 'contact_ppg', gt)
    sp = None if stats is None else u8_store.save_array(folder / 'stats', stats)
    reread = u8_store.read_array(vp)
    assert np.array_equal(reread, u8)
    out = []
    for r in items:
        s, length = int(r['start_frame']), int(r['chunk_length'])
        x, y = original_tensor(backbone, reread, stats, gt, s, length)
        xs = np.ascontiguousarray(whole_x[s:s + length], dtype=np.float32)
        ys = np.ascontiguousarray(whole_y[s:s + length], dtype=np.float32)
        out.append(dict(r, ubfc_view_path=str(vp), ubfc_contact_path=str(gp), ubfc_stats_path='' if sp is None else str(sp),
                        view_frames=len(u8), ops2_input_sha256=u8_store.npy_sha(x), ops2_label_sha256=u8_store.npy_sha(y),
                        load_equals_whole_recording=bool(np.array_equal(x, xs) and np.array_equal(y, ys)),
                        input_sha_equals_october=(u8_store.npy_sha(x) == r['october_input_sha256']) if r['october_input_sha256'] else '',
                        label_sha_equals_october=(u8_store.npy_sha(y) == r['october_label_sha256']) if r['october_label_sha256'] else '',
                        view_box=json.dumps(np.asarray(boxes).tolist()[:1]), ops2_CPU_seconds=time.time() - tick))
    return out


def job_family_i(args):
    backbone, out_root, rid, items = args
    r0 = items[0]
    n = int(r0['frame_count'])
    native = READER.read(r0['source_video_path'], 0, n)
    gt = contact_ubfc(r0['ground_truth_path'], n)
    grid = np.arange(n)
    tau = int(r0['tau_rec'])
    if backbone == 'RhythmFormer':
        box = R.view(native[:1])[1][0]
    else:
        physbox = W.view(native[:1], 'PhysNet')[1]
    folder = Path(out_root) / backbone / 'i' / rid
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for r in items:
        tick = time.time()
        c, k, length = int(r['chunk_order']), int(r['augmentation_index']), int(r['chunk_length'])
        scale = float(r['target_HR']) / float(r['reference_HR'])
        positions, plan = shifted_start(int(r['start_frame']), length, n, scale, tau)
        if positions is None:
            out.append(dict(r, mode='', generation_excluded=True, exclusion_reason='source_or_label_interval_exceeds_recording'))
            continue
        sourcepos = positions[0]
        lo, hi = int(np.floor(sourcepos.min())), int(np.ceil(sourcepos.max()))
        src = native[lo:hi + 1]
        rng = (np.random.default_rng([20261202, int(r['rounding_h'])]) if r.get('october_check') == 'True'
               else np.random.default_rng([KEY, J[backbone], int(r['rounding_h'])]))
        output = np.empty((length,) + src.shape[1:], np.uint8)
        for j, p in enumerate(sourcepos - lo):
            left, right = int(np.floor(p)), int(np.ceil(p))
            frac = p - left
            value = (1 - frac) * src[left].astype(np.float64) + frac * src[right]
            output[j] = np.floor(value + rng.random(value.shape)).astype(np.uint8)
        view, _ = R.view(output, box) if backbone == 'RhythmFormer' else W.view(output, 'PhysNet', reuse_boxes=physbox)
        vp = u8_store.save_view(folder / f'c{c}_k{k}_view', u8_store.to_u8(view))
        for mode, d in (('original', float(tau)), ('d', float(D_UBFC)), ('zero', 0.0)):
            shift = d * (scale - 1)
            labelpos = sourcepos + shift
            outside = max(0.0, float(-labelpos.min()), float(labelpos.max() - (n - 1)))
            if outside > EDGE_HOLD_STOP_FRAMES:
                raise RuntimeError(f"{r['chunk_id']} k{k} {mode}: label positions {outside:.3f} frames outside")
            label = np.interp(np.clip(labelpos, 0, n - 1), grid, gt)
            lp = u8_store.save_array(folder / f'c{c}_k{k}_rawlabel_{mode}', np.asarray(label, dtype=np.float64))
            x, y = u8_store.model_input(backbone, u8_store.read_array(vp), u8_store.read_array(lp), 0, length)
            ctx, _ = shifted_start(plan['source_start'] + (length - 300) * scale / 2, 300, n, scale, tau)
            achieved = common_hr(np.interp(np.clip(ctx[0] + shift, 0, n - 1), grid, gt), 'HAR') if ctx is not None else np.nan
            out.append(dict(r, mode=mode, arm={'original': '(i)', 'd': '(i-pc)', 'zero': '(i-simple)'}[mode],
                            generation_excluded=False, exclusion_reason='', scale=scale, source_start=plan['source_start'],
                            label_shift_source_frames=shift, edge_hold_samples=int(((labelpos < 0) | (labelpos > n - 1)).sum()),
                            edge_hold_max_frames=outside, achieved_context_HR=achieved, ops2_view_path=str(vp),
                            ops2_rawlabel_path=str(lp), ops2_stats_path='', ops2_view_offset=0,
                            ops2_view_bytes=vp.stat().st_size, ops2_rawlabel_bytes=lp.stat().st_size,
                            ops2_input_sha256=u8_store.npy_sha(x), ops2_label_sha256=u8_store.npy_sha(y),
                            ops2_CPU_seconds=time.time() - tick))
        del src, output, view
    return out


def main():
    backbone, kind, out_root = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    only = set(sys.argv[4:])
    assert backbone in J and kind in ('originals', 'family_i', 'check_i_october')
    if kind == 'check_i_october':  # October UBFC (i) items (their targets and rounding [20261202, h]) for a SHA check
        index = {r['recording_id']: r for r in read_csv(Path(CFG['october_project']) / 'data' / 'index' / 'ubfc_index_st1.csv')}
        rows = [dict(r, frame_count=index[r['recording_id']]['frame_count'], october_check='True')
                for r in read_csv(Path(CFG['october_project']) / 'outputs' / f'st30g_UBFC_{backbone}_i_manifest.csv')]
    else:
        rows = read_csv(PLAN / f'{backbone}_{"originals" if kind == "originals" else "items"}.csv')
    by_rec = {}
    for r in rows:
        if not only or r['recording_id'] in only:
            by_rec.setdefault(r['recording_id'], []).append(r)
    workers = int(os.environ.get('OPS2_WORKERS', '4'))
    fn = job_originals if kind == 'originals' else job_family_i
    if kind == 'check_i_october':
        out_root = out_root / 'october_check'
    jobs = [(backbone, str(out_root), rid, its) for rid, its in sorted(by_rec.items(), key=lambda kv: -len(kv[1]))]
    started, out = time.time(), []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'), initializer=init_worker,
                             initargs=(backbone,)) as ex:
        for i, o in enumerate(ex.map(fn, jobs), 1):
            out += o
            print(f'ubfc {backbone} {kind} {i}/{len(jobs)} recordings, {len(out)} rows, {time.time() - started:.0f}s', flush=True)
    tag = 'all' if not only else 'part'
    if kind == 'check_i_october':
        mr = [r for r in out if r.get('mode') == 'original']
        write_rows(out_root / f'{backbone}_i_october_check_manifest.csv', mr)
        print('done check (i) items', len(mr), 'input==October', sum(r['ops2_input_sha256'] == r['input_sha256'] for r in mr),
              'label==October', sum(r['ops2_label_sha256'] == r['label_sha256'] for r in mr), flush=True)
        return
    if kind == 'originals':
        write_rows(out_root / f'{backbone}_original_{tag}_manifest.csv', out)
        print('done originals', len(out), 'load==whole', sum(r['load_equals_whole_recording'] for r in out),
              'input==October', sum(r['input_sha_equals_october'] is True for r in out),
              'label==October', sum(r['label_sha_equals_october'] is True for r in out), flush=True)
    else:
        for mode, arm in (('original', 'i'), ('d', 'i-pc'), ('zero', 'i-simple')):
            mr = [r for r in out if r.get('mode') == mode]
            excl = [r for r in out if r.get('generation_excluded') is True]
            write_rows(out_root / f'{backbone}_{arm}_{tag}_manifest.csv', mr + excl)
            print(f'done {backbone} ({arm}) items {len(mr)} excluded {len(excl)}', flush=True)


if __name__ == '__main__':
    main()
