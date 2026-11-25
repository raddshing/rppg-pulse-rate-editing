"""Brief 02 7.02 다: HR of the labels actually presented in training (training data only, no model).

Items: per backbone, the original chunks and the pool items of every trained arm, built as the trainer builds them
(train/train_pc.py setup + manifest merge; the (i) and (ii) arms use the archived pools of their October runs).
Label as presented to the network: the file the trainer loader reads (train/u8_store.py U8Tensors): the item's stored
preprocessed label (label_path, ops2_label_path) or, for uint8 pools, the raw float64 context label (ops2_rawlabel_path)
preprocessed at load (PhysNet DiffNormalized, RhythmFormer and matched FactorizePhys Standardized, chunk slice), float32.
Where the trainer's label file is no longer on the server (uint8 pools kept on the encrypted Drive copy):
  1. stored copy: a stored label file of another pool whose SHA-256 equals the item's recorded label SHA-256 (the generator
     records the SHA of the preprocessed label, ops2_label_sha256); the file is hashed again before use;
  2. recomputed: the label alone is recomputed from the manifest and the PURE contact PPG with the generator's own label
     function (no video): raw_target_label(gt, context_start, reference_HR, target_HR) for the (ii)-family and label arms
     (train/gen_pool.py, train/gen_lab.py), the shifted label positions for (i-pc-g80) (train/gen_ipc_g80.py); then the same
     preprocessing; its SHA-256 must equal the recorded one. Items whose SHA does not match are left out and counted.
Readout (manuscript IV-B, wide band): PhysNet DiffNormalized labels are cumulated (diff flag), then the readout preprocessing of
the test (src/eval/grid.py _prepare: detrend lambda 100, first-order Butterworth 0.6-3.3 Hz, filtfilt) and the highest peak of a
512-point periodogram within 0.6-3.3 Hz (30 Hz; bin 3.515625 bpm). The chunk (128 or 160 samples) is zero-padded to 512 points.
Variant 'next_pow2': the periodogram length of the code's fft_hr (next power of two of the chunk length: 128 -> 128, 160 -> 256),
for reference.
Nominal: the chunk's reference HR (originals) or the item's target HR (pool items; label arms: the changed target).
Report: bin proportions (<70, 70-80, 80-90, 90-100, 100-110, 110-120, 120-140, >=140 bpm) of the readout and of the nominal HR;
readout - nominal: median, fraction > 1 bin (|d| > 3.515625 + 1e-6), fraction > 3 bins. Parts: original chunks, pool items, and
the expected mixture of a batch (2 originals + 2 pool items: originals drawn by epoch permutation, pool items uniformly,
src/st9s/streams.py SourceStreams.batch), i.e. weight 1/2 on each part.
Writes outputs/posthoc702/da/{da_distribution,da_sources}.csv.
"""
import hashlib
import os
import sys
from collections import Counter

from p702_common import GRID, SRC_POOL, TOL, VENDOR, WORK, np, pd, stamp, time, write

sys.path.insert(0, str(WORK / 'audit'))
sys.path.insert(0, str(WORK / 'train'))
sys.path.insert(0, str(VENDOR))
import train_pc as T  # noqa: E402  (setup functions only; chdir to the vendor copy at import)
import u8_store  # noqa: E402
from a1_common import contact, frame_files, pure_index, raw_target_label, shifted_start, tau_table  # noqa: E402
from scipy.signal import periodogram  # noqa: E402
from src.eval import grid as G, har  # noqa: E402
from src.st9g.preprocess import diff_normalize_label, standardized_label  # noqa: E402

ARMS = {'PhysNet': ['i', 'i-pc', 'i-simple', 'ii', 'ii-pc', 'ii-lag', 'ii-pc-g', 'ii-pc-g80', 'ii-pc-g2', 'i-pc-g80', 'sham-pc', 'roi-pc'],
        'RhythmFormer': ['i', 'i-pc', 'i-simple', 'ii', 'ii-pc', 'ii-lag'],
        'FPm': ['i', 'i-pc', 'i-simple', 'ii', 'ii-pc', 'ii-lag', 'ii-pc-g', 'ii-pc-g80', 'ii-pc-g2', 'sham-pc']}
NAME = {'PhysNet': 'PhysNet', 'RhythmFormer': 'RhythmFormer', 'FPm': 'FactorizePhys (matched)'}
BANDS = [('<70', -np.inf, 70), ('70-80', 70, 80), ('80-90', 80, 90), ('90-100', 90, 100), ('100-110', 100, 110),
         ('110-120', 110, 120), ('120-140', 120, 140), ('>=140', 140, np.inf)]
BAND = (.6, 3.3)
RUNS = {}
for f in sorted((WORK / 'cache' / 'train').glob('*_s20261131/result.json')):
    bb, rest = f.parent.name.split('_', 1)
    import json
    RUNS[(bb, rest.rsplit('_s', 1)[0])] = json.loads(f.read_text())['pool_manifest']


def file_sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


def items(b, arm):
    kind = 'i' if arm == 'i' or arm.startswith('i-') else 'ii'
    _, original, pool, *_ = T.setup(b, kind)
    if arm in ('i', 'ii'):
        return original, pool, {}
    new = {(r['chunk_id'], r['augmentation_index']): r for r in T.rows(RUNS[(b, arm)])}
    keys = ('ops2_view_path', 'ops2_rawlabel_path', 'ops2_stats_path', 'ops2_label_path', 'ops2_view_offset')
    pool = [dict(r, **{k: new[(r['chunk_id'], r['augmentation_index'])].get(k, '') for k in keys})
            if (r['chunk_id'], r['augmentation_index']) in new else r for r in pool]
    pool = [dict(r, target_HR=new[(r['chunk_id'], r['augmentation_index'])]['target_HR'])
            if new.get((r['chunk_id'], r['augmentation_index']), {}).get('target_HR_original') not in (None, '') else r for r in pool]
    return original, pool, new


def trainer_source(r):
    if r.get('ops2_view_path'):
        return 'ops2_rawlabel_path', r['ops2_rawlabel_path']
    if r.get('ops2_label_path'):
        return 'ops2_label_path', r['ops2_label_path']
    return 'label_path', r['label_path']


def recorded_sha(r, new):
    key = (r['chunk_id'], r['augmentation_index'])
    if r.get('ops2_view_path') or r.get('ops2_label_path'):
        return new[key].get('ops2_label_sha256', '')
    return r.get('label_sha256', '')


def preprocess(b, raw, off, length):
    """train/u8_store.py model_input, label part."""
    y = diff_normalize_label(raw) if b == 'PhysNet' else standardized_label(raw)
    lo = 0 if len(raw) == length else off
    return np.ascontiguousarray(y[lo:lo + length], dtype=np.float32)


GT, NFR, TAU = {}, {}, None


def recompute(b, arm, r):
    global TAU
    rid = r['recording_id']
    if rid not in GT:
        row = pure_index()[rid]
        GT[rid] = contact(row, 'PURE', None)
        NFR[rid] = len(frame_files(row))
    gt = GT[rid]
    length, h0, t = int(r['chunk_length']), float(r['reference_HR']), float(r['target_HR'])
    if b == 'PhysNet' and arm == 'i-pc-g80':  # train/gen_ipc_g80.py
        if TAU is None:
            TAU = tau_table()
        scale = t / h0
        positions, plan = shifted_start(int(r['start_frame']), length, NFR[rid], scale, int(TAU[rid]['tau_rec']))
        labelpos = positions[0] + 2 * (scale - 1)
        ng = len(gt)
        raw = np.interp(np.clip(labelpos, 0, ng - 1), np.arange(ng), gt).astype(np.float64)
        return preprocess(b, raw, 0, length)
    raw = np.asarray(raw_target_label(gt, int(r['context_start']), h0, t), dtype=np.float64)  # train/gen_pool.py, gen_lab.py
    return preprocess(b, raw, int(r['context_offset']), length)


def readout(y, diff):
    x = G._prepare(y, diff, BAND, 'default', 30)
    f, p = periodogram(x, fs=30, nfft=512, detrend=False)
    m = (f >= BAND[0]) & (f <= BAND[1])
    return float(f[m][np.argmax(p[m])] * 60), float(har.fft_hr(x, 30, BAND[0], BAND[1]))


def describe(hr, nominal, w=None):
    hr, nominal = np.asarray(hr, float), np.asarray(nominal, float)
    w = np.ones(len(hr)) / len(hr) if w is None else np.asarray(w, float)
    out = {}
    for name, lo, hi in BANDS:
        out[f'share_{name}'] = float(w[(hr >= lo) & (hr < hi)].sum())
    d = hr - nominal
    o = np.argsort(d)
    cw = np.cumsum(w[o])
    out['median_difference'] = float(d[o][np.searchsorted(cw, 0.5)]) if len(set(np.round(w, 15))) > 1 else float(np.median(d))
    out['share_gt_1bin'] = float(w[np.abs(d) > GRID + TOL].sum())
    out['share_gt_3bins'] = float(w[np.abs(d) > 3 * GRID + TOL].sum())
    return out


def shares(hr, w=None):
    return {k: v for k, v in describe(hr, hr, w).items() if k[len('share_'):] in [n for n, _, _ in BANDS]}


def main():
    t0 = time.time()
    # index: recorded label SHA -> stored label file (trainer files that exist on the server, every arm and the originals)
    plans, index = {}, {}
    for b, arms in ARMS.items():
        for arm in arms:
            original, pool, new = items(b, arm)
            plans[(b, arm)] = (original, pool, new)
            for r in pool:
                col, p = trainer_source(r)
                sh = recorded_sha(r, new)
                if sh and os.path.exists(p):
                    index.setdefault(sh, p)
            for r in original:
                if os.path.exists(r['label_path']):
                    index.setdefault(r['label_sha256'], r['label_path'])
    print('index', len(index), round(time.time() - t0), flush=True)
    dist, sources = [], []
    originals = {}
    for (b, arm), (original, pool, new) in plans.items():
        diff = b == 'PhysNet'
        parts = {}
        if b not in originals:  # the same original chunks for every arm of a backbone
            res, src = [], Counter()
            for r in original:
                y = np.load(r['label_path'], allow_pickle=False)
                ok = file_sha(r['label_path']) == r['label_sha256']
                src[('stored (trainer file)', ok)] += 1
                h512, hp2 = readout(np.ascontiguousarray(y, dtype=np.float32), diff)
                res.append((h512, hp2, float(r['reference_HR'])))
            originals[b] = (np.array(res), src, [r['chunk_id'] for r in original])
            for (how, ok), n in sorted(src.items()):
                sources.append(dict(backbone=NAME[b], arm='(original chunks)', part='original chunks', label_source=how,
                                    sha_matches_record=ok, items=n, source=SRC_POOL))
        assert originals[b][2] == [r['chunk_id'] for r in original], 'original chunks differ between arms'
        parts['original chunks'] = originals[b][:2]
        res, src = [], Counter()
        for r in pool:
            col, p = trainer_source(r)
            sh = recorded_sha(r, new)
            if os.path.exists(p):
                if col == 'ops2_rawlabel_path':
                    raw = u8_store.read_array(p)
                    off = r.get('ops2_view_offset')
                    off = int(float(off)) if off not in (None, '') else int(r['context_offset'])
                    y = preprocess(b, raw, off, int(r['chunk_length']))
                    ok = u8_store.npy_sha(y) == sh
                else:
                    y = np.load(p, allow_pickle=False)
                    ok = file_sha(p) == sh
                how = 'stored (trainer file)'
            elif sh in index:
                y = np.load(index[sh], allow_pickle=False)
                ok = file_sha(index[sh]) == sh
                how = 'stored (identical SHA, other pool)'
            else:
                y = recompute(b, arm, r)
                ok = u8_store.npy_sha(y) == sh
                how = 'recomputed (manifest + contact PPG)'
            src[(how, ok)] += 1
            if not ok:
                continue
            h512, hp2 = readout(np.ascontiguousarray(y, dtype=np.float32), diff)
            res.append((h512, hp2, float(r['target_HR'])))
        parts['pool items'] = (np.array(res), src)
        for part, (a, src) in [('pool items', parts['pool items'])]:
            for (how, ok), n in sorted(src.items()):
                sources.append(dict(backbone=NAME[b], arm=f'({arm})', part=part, label_source=how, sha_matches_record=ok, items=n,
                                    source=SRC_POOL))
        o, p = parts['original chunks'][0], parts['pool items'][0]
        for vi, variant in enumerate(('512-point', 'next_pow2')):
            for part, a in (('original chunks', o), ('pool items', p)):
                dist.append(dict(backbone=NAME[b], arm=f'({arm})', part=part, readout=variant, quantity='readout', items=len(a),
                                 **describe(a[:, vi], a[:, 2]), source=SRC_POOL))
            w = np.r_[np.full(len(o), .5 / len(o)), np.full(len(p), .5 / len(p))]
            m = np.r_[o, p]
            dist.append(dict(backbone=NAME[b], arm=f'({arm})', part='expected batch mixture (2 originals + 2 pool items)', readout=variant,
                             quantity='readout', items=len(m), **describe(m[:, vi], m[:, 2], w), source=SRC_POOL))
        for part, a in (('original chunks', o), ('pool items', p)):
            dist.append(dict(backbone=NAME[b], arm=f'({arm})', part=part, readout='', quantity='nominal', items=len(a),
                             **shares(a[:, 2]),
                             source=SRC_POOL))
        dist.append(dict(backbone=NAME[b], arm=f'({arm})', part='expected batch mixture (2 originals + 2 pool items)', readout='',
                         quantity='nominal', items=len(o) + len(p),
                         **shares(np.r_[o[:, 2], p[:, 2]], w), source=SRC_POOL))
        print(b, arm, dict(parts['pool items'][1]), round(time.time() - t0), flush=True)
    dist, sources = pd.DataFrame(dist), pd.DataFrame(sources)
    write(dist, 'da_distribution.csv', 'da')
    write(sources, 'da_sources.csv', 'da')
    stamp('da', seconds=time.time() - t0, readout='HAR band 0.6-3.3 Hz, 512-point periodogram (variant next_pow2)',
          sha_mismatch_items=int(sources[~sources.sha_matches_record]['items'].sum()) if len(sources) else 0)


if __name__ == '__main__':
    main()
