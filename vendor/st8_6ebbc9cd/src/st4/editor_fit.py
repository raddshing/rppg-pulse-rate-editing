"""S4-1b fixed color regression and dataset ED2 directions before editing."""
from src.st4.base import *
from src.st4.editor_inputs import *
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


def editor_band(values):
    # ED1 changes only the inherited band, preserving its first-order filter.
    b, a = butter(1, [.6 / 15, 3.3 / 15], btype='bandpass')
    return filtfilt(b, a, values, axis=0)


def normalize(values):
    y = editor_band(values)
    check('Finite nonzero contact waveform for editor', np.isfinite(y).all() and y.std() > 0)
    return (y - y.mean()) / y.std()


def fit(clip, masks, gt, start):
    means = np.asarray([frame[m].mean(axis=0) for frame, m in zip(clip, masks)])
    normalized = means / means.mean(axis=0) - 1
    colors = editor_band(normalized)
    g = normalize(gt[reflected_indices(np.arange(start - 15, start + 315), len(gt))])
    fits = []
    for lag in range(-15, 16):
        wave = g[np.arange(300) + 15 - lag]
        wave = wave - wave.mean()
        centered = colors - colors.mean(axis=0)
        coef = (wave[:, None] * centered).sum(axis=0) / (wave**2).sum()
        residual = centered - wave[:, None] * coef
        total = (centered**2).sum(axis=0)
        r2 = np.divide(total - (residual**2).sum(axis=0), total, out=np.zeros(3), where=total > 0)
        fits.append((r2.sum(), lag, coef, r2))
    _, lag, coef, r2 = max(fits, key=lambda x: x[0])
    return dict(tau_frames=int(lag), a_R=float(coef[0]), a_G=float(coef[1]), a_B=float(coef[2]),
                R2_R=float(r2[0]), R2_G=float(r2[1]), R2_B=float(r2[2])), means


def main():
    cases, mm, pure, allowed = target_cases()
    write_rows('outputs/st4_editor_cases.csv', cases)
    out = rows('outputs/st4_editor_fit.csv') if (ROOT / 'outputs/st4_editor_fit.csv').exists() else []
    done = {r['window_id'] for r in out}
    started = time.monotonic()
    newly_done = 0
    for dataset, rid in sorted({(r['dataset'], r['recording_id']) for r in cases}):
        group = [r for r in cases if (r['dataset'], r['recording_id']) == (dataset, rid) and r['window_id'] not in done]
        if not group:
            continue
        row = (mm if dataset == 'MMPD' else pure)[rid]
        source = load_recording(row, dataset, allowed)
        gt = contact(row, dataset, allowed)
        for case in group:
            start = int(case['window_index']) * 300
            whole = frame_window(source, dataset, start)
            box = hc_box(whole, dataset)
            check('Every prescribed gate window has the fixed face box', box is not None)
            left, top, right, bottom = box
            clip = np.ascontiguousarray(whole[:, top:bottom, left:right])
            masks, failure = interpolate_masks(clip)
            calibration, means = fit(clip, masks, gt, start)
            dest = geometry_file(case['window_id'])
            check('No uncommitted calibration cache overwritten', not dest.exists())
            save_geometry(dest, masks, box, clip)
            out.append(dict(**case, **calibration, detection_failure_ratio=failure, box_left=left,
                            box_top=top, box_right=right, box_bottom=bottom,
                            original_HAR=common_hr(gt[start:start + 300], 'HAR'),
                            original_L0=common_hr(gt[start:start + 300], 'L0'),
                            geometry_path=str(dest), geometry_sha256=sha(dest),
                            starting_uint8_sha256=hashlib.sha256(clip.tobytes()).hexdigest()))
            write_rows('outputs/st4_editor_fit.csv', out)
            done.add(case['window_id'])
            newly_done += 1
            progress('editor_fit', len(done), len(cases), started)
            elapsed = time.monotonic() - started
            if newly_done >= 6:
                projected_remaining = elapsed / newly_done * (len(cases) - len(done))
                # Remaining editor and dependent tasks retain the conservative allowance.
                check('Calibration projection fits remaining CPU bundle', budget_check() + projected_remaining + 20 * 3600 < 129600)
            del whole, clip, masks
        del source
    table = pd.DataFrame(out)
    directions = []
    for dataset, n in [('MMPD', 162), ('PURE', 60)]:
        q = table[table.dataset.eq(dataset)]
        check('Exact dataset direction-calibration population', len(q) == n)
        coef = q[['a_R', 'a_G', 'a_B']].to_numpy(float).copy()
        coef *= np.where(coef[:, 1] < 0, -1, 1)[:, None]
        norms = np.linalg.norm(coef, axis=1)
        check('All regression vectors finite and nonzero', np.isfinite(coef).all() and np.all(norms > 0))
        weights = np.maximum(q.R2_G.to_numpy(float), 0)
        direction = ((coef / norms[:, None]) * weights[:, None]).sum(axis=0)
        check('Finite nonzero weighted direction and positive green component', np.linalg.norm(direction) > 0 and direction[1] > 0)
        direction /= np.linalg.norm(direction)
        directions.append(dict(dataset=dataset, n_windows=n, v_R=float(direction[0]), v_G=float(direction[1]),
                               v_B=float(direction[2]), R2_G_weight_sum=float(weights.sum())))
    write_rows('outputs/st4_editor_directions.csv', directions)
    return dict(calibration_complete=True, ED2_direction_value_commit_required=True)


if __name__ == '__main__':
    finish('editor_fit', main)
