"""7.09a (training-data preparation for E1): the 360-window editor validity check of PURE (audit/a2_validity.py), run on
UBFC-rPPG. CPU only; no test data.

Same path as audit/a2_validity.py (archived E3R-b, st9e/brange.py) with these UBFC inputs, all read-only from the October
project and identical to those of the October UBFC gate G-U (src/st31u/gate.py, 11-28 Oct):
  cases      outputs/st30u_gate_samples.csv, rows not excluded (360: 4 target intervals x 90 windows)
  direction  outputs/st31u_direction.csv (v from 59 UBFC windows)
  index      data/index/ubfc_index_st1.csv (video and label paths, frame counts)
  contact    ground_truth.txt line 1, one value per frame (October ubfc_contact: length equal to the frame count)
  frames     tools/ubfc_video_reader.py in the frozen decoding environment (OpenCV 4.5.2), as the October UBFCReader
  tau        tau_rec of the sample rows (the October UBFC tau)
Views: PhysNet and FactorizePhys (as the PURE check) and RhythmFormer (as the October gate). Modes: (ii) and (ii-pc) of
audit/editor_pc.py (s = -1, d = 2 frames, as for PURE); the HR readout does not depend on the delay of the injection.
Writes outputs/2u/validity360_ubfc_readouts.csv, _baselines.csv, _summary.csv (summary as audit/a2_validity_arms_summary.py).
OPS2_UBFC_NOTCH=target (7.09b 가-4): the notch also covers the target, bands_for(original HAR, target), as in the October
stage-5 gate (src/st5/editor_gate.py); computed per case, and the calibration uses that notched clip. Output folder 2u_target.
Default 'source': bands_for(original HAR) (f0 and 2f0), as the pools and the PURE check.
"""
import atexit
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from editor_pc import *  # noqa: F401,F403
from infer_client import RhythmView  # noqa: E402
import pandas as pd  # noqa: E402
from src.st9e.brange import view_signals, readout_rows, baseline_rows  # noqa: E402
from src.st9e.masks import face_from_points  # noqa: E402

CFG = json.loads((Path(__file__).resolve().parents[1] / 'config' / 'paths.json').read_text())  # 7.19: machine paths, see config/paths.example.json
OCT = Path(CFG['october_project'])
REPRO = Path(CFG['python_repro'])
NOTCH = os.environ.get('OPS2_UBFC_NOTCH', 'source')
assert NOTCH in ('source', 'target')
OUT = WORK / 'outputs' / ('2u' if NOTCH == 'source' else '2u_target')
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


def init_worker():
    global W, R, READER
    W = ViewWorker()
    atexit.register(W.close)
    R = RhythmView()
    atexit.register(R.close)
    READER = UBFCReader()
    atexit.register(READER.close)


def cases():
    c = pd.read_csv(OCT / 'outputs/st30u_gate_samples.csv', dtype={'excluded': str})
    c = c[c.excluded.eq('False')].copy()
    assert len(c) == 360
    c['task'] = ''
    return c


def job(rid):
    import mediapipe as mp_
    from mediapipe.tasks.python import vision
    group = cases()
    group = group[group.recording_id.eq(rid)]
    index = {r['recording_id']: r for r in read_csv(OCT / 'data/index/ubfc_index_st1.csv')}
    row = index[rid]
    roidef = json.loads((VENDOR / 'outputs/st3_roi_indices.json').read_text())
    d = read_csv(OCT / 'outputs/st31u_direction.csv')[0]
    direction_rgb = [float(d['v_' + c]) for c in 'RGB']
    with open(row['label_path']) as f:
        gt = np.asarray([float(x) for x in f.readline().split()])
    assert len(gt) == int(row['frame_count'])
    wanted = set(group.window_index.astype(int))
    native = READER.read(row['path'], 0, (max(wanted) + 1) * 300)
    first = native[0].copy()
    rboxes = R.view(native[:1])[1]
    previous = None
    frames, points, oldmasks = [], [], []
    result, baselines, diags = [], [], []
    with vision.FaceLandmarker.create_from_options(tracking_options()) as tracker:
        for i, rgb in enumerate(native):
            value = tracker.detect_for_video(mp_.Image(image_format=mp_.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)), round(i * 1000 / 30))
            p = np.full((478, 2), np.nan, np.float32)
            if value.face_landmarks:
                h, w = rgb.shape[:2]
                observed = np.asarray([[v.x * w, v.y * h] for v in value.face_landmarks[0]], dtype=np.float64)
                p[:] = observed
                previous = skin_mask(rgb.shape, observed, roidef)
            j = i // 300
            if j not in wanted:
                continue
            assert previous is not None and previous.any(), 'Every editor window has valid skin landmarks'
            frames.append(rgb)
            points.append(p)
            oldmasks.append(previous)
            if (i + 1) % 300:
                continue
            clip = np.asarray(frames)
            roi = np.asarray(oldmasks)
            face, failure = face_from_points(np.asarray(points), clip.shape[1:])
            wid = f'UBFC|{rid}|{j}'
            start = j * 300
            original = {layer: common_hr(gt[start:start + 300], layer) for layer in BANDS}
            notched = notch_all(clip, bands_for(original['HAR'])) if NOTCH == 'source' else None
            before, boxes = {}, {}
            for view in ['PhysNet', 'FactorizePhys', 'RhythmFormer']:
                if view == 'RhythmFormer':
                    resized, box = R.view(clip, rboxes[0])
                else:
                    resized, box = W.view(clip, view, first if view == 'PhysNet' else None)
                before[view] = view_signals(resized)
                boxes[view] = box
                if view == 'PhysNet':
                    physview, physbox = resized, box[0]
                baselines.extend(dict(r, dataset='UBFC') for r in baseline_rows(wid, '', view, before[view], original))
            for case in group[group.window_index.eq(j)].sort_values('range_low').to_dict('records'):
                assert abs(float(case['h0']) - original['HAR']) < 1e-9
                tau = int(case['tau_rec'])
                if NOTCH == 'target':
                    notched = notch_all(clip, bands_for(original['HAR'], float(case['target_HR'])))
                g0, gstar = ppg_vectors(gt, start, original['HAR'], case['target_HR'], tau)
                pulse, w, beta = calibration(physview, crop_native(notched, physbox), crop_native(face, physbox), g0, direction_rgb)
                target_ref = {layer: common_hr(gstar, layer) for layer in BANDS}
                seedcond = f'tgt-{case["range_low"]}-{case["range_high"]}'
                for mode in ['ii', 'ii-pc']:
                    inject = None if mode == 'ii' else S_SIGN * ppg_vectors(gt, start, original['HAR'], case['target_HR'], D_FRAMES)[1]
                    output, diag = edited(clip, notched, face, roi, g0, gstar, direction_rgb, original['HAR'], 'ED2W-S', pulse, w,
                                          wid, seedcond, inject=inject)
                    for view in ['PhysNet', 'FactorizePhys', 'RhythmFormer']:
                        if view == 'RhythmFormer':
                            resized, _ = R.view(output, rboxes[0])
                        else:
                            resized, _ = W.view(output, view, reuse_boxes=boxes[view])
                        for r in readout_rows(case, view, before[view], view_signals(resized), original, target_ref, tau,
                                              'tau_rec_original_boxes'):
                            result.append(dict(r, dataset='UBFC', population='October G-U sample 360', mode=mode, notch=NOTCH,
                                               injection_A=diag['injection_A']))
                        del resized
                    diags.append(dict(window_id=wid, range_low=int(case['range_low']), nominal_target_HR=float(case['target_HR']),
                                      mode=mode, tau_rec=tau, beta=beta, landmark_missing_fraction=failure, notch=NOTCH, **diag))
                    del output
            frames, points, oldmasks = [], [], []
            del clip, roi, face, notched, physview
    return result, baselines, diags


def summary(r, b):
    b = b.drop_duplicates(['window_id', 'model_view', 'reader', 'layer'])
    base = b.groupby(['model_view', 'reader', 'layer']).baseline_hit.mean()
    rows = []
    for (mode, view, reader, layer, lo, hi), g in r.groupby(['mode', 'model_view', 'reader', 'layer', 'range_low', 'range_high']):
        rate = float(base.loc[(view, reader, layer)])
        t, o = float(g.target_hit.mean()), float(g.original_hit_after.mean())
        rows.append(dict(mode=mode, model_view=view, reader=reader, layer=layer, range=f'{lo}-{hi}', windows=len(g),
                         target_hit_rate=t, original_hit_after_rate=o, unedited_hit_rate=rate,
                         B1_target_ge_unedited_minus_012=t >= rate - .12, B2_original_le_half_unedited=o <= rate / 2))
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rids = sorted(cases().recording_id.unique())
    workers = int(os.environ.get('OPS2_WORKERS', '4'))
    started = time.time()
    result, baselines, diags = [], [], []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'), initializer=init_worker) as ex:
        for n, (r, b, d) in enumerate(ex.map(job, rids), 1):
            result += r
            baselines += b
            diags += d
            print(f'validity ubfc {n}/{len(rids)} recordings, {time.time() - started:.0f}s', flush=True)
    write_rows(OUT / 'validity360_ubfc_readouts.csv', result)
    write_rows(OUT / 'validity360_ubfc_baselines.csv', baselines)
    write_rows(OUT / 'validity360_ubfc_diagnostics.csv', diags)
    s = summary(pd.DataFrame(result), pd.DataFrame(baselines))
    s.to_csv(OUT / 'validity360_ubfc_summary.csv', index=False)
    print('done', len(result), len(baselines), f'{time.time() - started:.0f}s')


if __name__ == '__main__':
    main()
