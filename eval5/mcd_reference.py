"""Stage-5 evaluation, MCD-rPPG reference side: the 1b model-free pass per recording (frame-count class, raw-PPG clock groups,
label-label alignment and the S5-3M2 10-s windows with reference estimates), the function job() copied unchanged from the OPS2
development module b1/mcd_meta.py, which applies src/st5c/mcd_packets.py and mcd_align.py of commit b81f6c9e6 unchanged.
The caller puts the archived source tree on sys.path and passes rows with absolute file paths (video, meta, ppg, ppg_sync).
"""
from pathlib import Path

import numpy as np  # noqa: F401  (used by job)
from src.st5c.mcd_align import *  # noqa: F401,F403  (offset_alignment, aligned_windows, timestamp_table, packet_metadata)
from src.st5c.mcd_packets import clock_groups  # noqa: F401

MCD = Path('/')  # rows carry absolute paths; MCD / path = path


def job(row):
    rid = row['recording_id']
    base = dict(subject_id=row['subject_id'], state=row['state'], recording_id=rid)
    index, timestamps = timestamp_table(MCD / row['meta'])
    raw, rawstamps = timestamp_table(MCD / row['ppg'])
    sync_full = np.loadtxt(MCD / row['ppg_sync'], ndmin=2)
    sync = sync_full[:, 0]
    parsed = packet_metadata(MCD / row['video'])
    nf, ns, nm = parsed['positive_packet_count'], len(sync_full), len(index)
    category = 'F0' if nf == ns == nm else 'F1' if nf == ns and nf != nm else 'F2' if nf == nm and nf != ns else 'other'
    g = clock_groups(rawstamps, timestamps[0])
    meta = (timestamps.astype(float) - float(timestamps[0])) / 1e6
    meta_invalid = int(np.sum(timestamps == np.iinfo(np.int64).min))
    rawtimes = float(g['clock_a_seconds']) + float(g['clock_b_seconds_per_sample']) * np.arange(len(raw)) if np.isfinite(g.get('clock_a_seconds', np.nan)) else None
    if bool(g['raw_clock_irregular']) or not np.isfinite(raw).all():
        result = dict(alignment_status='raw_clock_irregular')
    elif meta_invalid > 0 or not np.all(np.diff(meta) > 0) or len(meta) < len(sync) or not np.isfinite(sync).all():
        result = dict(alignment_status='unverified')
    else:
        result, _ = offset_alignment(meta, sync, rawtimes, raw)
    confirmed = result['alignment_status'] == 'verified' and category in ['F0', 'F1', 'F2'] and not bool(parsed['frame_count_ambiguous'])
    rec = dict(base, N_frames=nf, N_sync=ns, N_meta=nm, frame_count_class=category, alignment_status=result['alignment_status'],
               mapping_confirmed=confirmed, k_star=result.get('k_star'),
               first_meta_microseconds=int(timestamps[0]), last_meta_microseconds=int(timestamps[-1]),
               span_seconds=(int(timestamps[-1]) - int(timestamps[0])) / 1e6, meta_rows=nm,
               zero_size_packet_count=parsed.get('zero_size_packet_count'))
    windows = []
    if confirmed:
        k = int(result['k_star'])
        mapped = meta[k:k + len(sync)]
        local, edge = aligned_windows(base, mapped, sync, rawtimes, raw)
        rec['n_endpoint_excluded_windows'] = edge
        windows = local
    return rec, windows
