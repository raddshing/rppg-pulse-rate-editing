"""ST1 harmonized rPPG evaluation, derived from the pinned official definitions.

Reference: rPPG-Toolbox b7500b848f84ad7f86e277b4612563b69f4f88f9,
evaluation/post_process.py. This module contains no legacy project code.

The banded smoothness-prior solve and FFT circular correlation are algebraically
equivalent to the official dense inverse and lag loop. Reference parity is
tested separately in the official ``repro`` environment. No sealed file may
enter the file-reading API; the converter can evaluate arrays in memory.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Union

import numpy as np
from scipy.linalg import solveh_banded
from scipy.signal import butter, filtfilt, periodogram

ROOT = Path(__file__).resolve().parents[2]
FS = 30.0
WINDOW_FRAMES = 300
LOW_HZ = 0.6
HIGH_HZ = 3.3
DETREND_LAMBDA = 100.0
METRIC_KEYS = (
    "hr_pred_bpm", "hr_label_bpm", "snr_db", "macc", "abs_error_bpm",
    "squared_error_bpm2", "ape_percent", "tail_gt_10_bpm",
)
NPZ_KEYS = (
    "pred", "label", "chunk_starts", "chunk_len", "fs", "diff_flag",
    "model", "train_source", "seed", "checkpoint_sha256", "recording_id",
    "subject_id",
)


class SealedPathError(PermissionError):
    """An analysis entry point attempted to access MMPD-B sealed output."""


def assert_analysis_path(path: Union[str, Path]) -> Path:
    """Reject lexical and symlink aliases of the authoritative sealed subtree."""
    paths = json.loads((ROOT / "paths.yaml").read_text())
    sealed = Path(paths["sealed_mmpd_b"])
    candidate = Path(path).expanduser()
    lexical = Path(os.path.abspath(str(candidate)))
    resolved = candidate.resolve()
    roots = (Path(os.path.abspath(str(sealed))), sealed.resolve())
    for target in (lexical, resolved):
        # Also reject a literal sealed-path spelling before such a tree exists.
        parts = target.parts
        literal_seal = any(parts[i:i + 2] == ("sealed", "mmpd_B")
                           for i in range(len(parts) - 1))
        if literal_seal or any(target == root or root in target.parents for root in roots):
            raise SealedPathError("Analysis cannot read sealed MMPD-B output: " + str(path))
    return resolved


def next_power_of_two(length: int) -> int:
    return 1 if length == 0 else 2 ** (int(length) - 1).bit_length()


def detrend(signal: np.ndarray, lambda_value: float = DETREND_LAMBDA) -> np.ndarray:
    """Return (I - (I + lambda**2 D.T D)^-1) y using a pentadiagonal solve."""
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 1 or len(values) < 3:
        raise ValueError("Detrending requires a one-dimensional signal of length >= 3.")
    n = len(values)
    scale = float(lambda_value) ** 2
    diagonal = np.ones(n, dtype=np.float64)
    diagonal[:-2] += scale
    diagonal[1:-1] += 4 * scale
    diagonal[2:] += scale
    first = np.zeros(n - 1, dtype=np.float64)
    first[:-1] -= 2 * scale
    first[1:] -= 2 * scale
    banded = np.zeros((3, n), dtype=np.float64)
    banded[0] = diagonal
    banded[1, :-1] = first
    banded[2, :-2] = scale
    trend = solveh_banded(banded, values, lower=True, check_finite=True)
    return values - trend


def prepare_signal(signal: np.ndarray, diff_flag: bool, fs: float = FS) -> np.ndarray:
    """Apply native-dtype cumulative sum, detrend(100), then butter(1)/filtfilt."""
    values = np.asarray(signal)
    if values.ndim != 1:
        raise ValueError("A signal must be one-dimensional.")
    # np.cumsum(float32) must remain float32, as in the reference implementation.
    if diff_flag:
        values = np.cumsum(values)
    values = detrend(values)
    b, a = butter(1, [LOW_HZ / fs * 2, HIGH_HZ / fs * 2], btype="bandpass")
    return filtfilt(b, a, np.asarray(values, dtype=np.float64))


def fft_hr(signal: np.ndarray, fs: float = FS,
           low_hz: float = LOW_HZ, high_hz: float = HIGH_HZ) -> float:
    """Official periodogram HR, including its first-bin tie/zero-power rule."""
    values = np.asarray(signal)
    frequencies, power = periodogram(values, fs=fs,
                                    nfft=next_power_of_two(len(values)), detrend=False)
    mask = (frequencies >= low_hz) & (frequencies <= high_hz)
    if not np.any(mask):
        raise ValueError("No periodogram bin lies in the HR band.")
    return float(frequencies[mask][np.argmax(power[mask])] * 60)


def snr(signal: np.ndarray, hr_label_bpm: float, fs: float = FS,
        low_hz: float = LOW_HZ, high_hz: float = HIGH_HZ) -> float:
    """Exact official harmonic-power masks, including numerator outside the band.

    Harmonics are summed separately, not clipped to the remainder's band. The
    native zero-remainder result is 0 dB; a zero numerator with positive
    remainder produces -inf. These conventions are intentionally retained.
    """
    values = np.asarray(signal)
    frequencies, power = periodogram(values, fs=fs,
                                    nfft=next_power_of_two(len(values)), detrend=False)
    first_hz = float(hr_label_bpm) / 60
    h1 = (frequencies >= first_hz - 0.1) & (frequencies <= first_hz + 0.1)
    h2 = (frequencies >= 2 * first_hz - 0.1) & (frequencies <= 2 * first_hz + 0.1)
    remainder = ((frequencies >= low_hz) & (frequencies <= high_hz) & ~h1 & ~h2)
    remainder_power = float(np.sum(power[remainder]))
    if remainder_power == 0:
        return 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(10 * np.log10((np.sum(power[h1]) + np.sum(power[h2])) /
                                   remainder_power))


def macc(prediction: np.ndarray, label: np.ndarray) -> float:
    """Maximum absolute circular Pearson correlation over lags 0,...,N-2."""
    pred = np.asarray(prediction, dtype=np.float64).reshape(-1)
    truth = np.asarray(label, dtype=np.float64).reshape(-1)
    n = min(len(pred), len(truth))
    if n < 2:
        return float("nan")
    pred = pred[:n] - np.mean(pred[:n])
    truth = truth[:n] - np.mean(truth[:n])
    denominator = np.sqrt(np.dot(pred, pred) * np.dot(truth, truth))
    if not np.isfinite(denominator) or denominator == 0:
        return float("nan")
    circular = np.fft.irfft(np.fft.rfft(pred) * np.conj(np.fft.rfft(truth)), n=n)
    # np.corrcoef clips its real result to [-1,1]; retain that behavior.
    correlations = np.clip(circular[:-1] / denominator, -1.0, 1.0)
    return float(np.max(np.abs(correlations)))


def _undefined(status: str, flags: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    result = {key: float("nan") for key in METRIC_KEYS}
    result.update(status=status, flags=list(flags or []))
    return result


def evaluate_signal(prediction: np.ndarray, label: np.ndarray,
                    diff_flag: bool = False, fs: float = FS) -> Dict[str, Any]:
    """Evaluate one contiguous unit. Keep undefined units and their flags."""
    pred = np.asarray(prediction)
    truth = np.asarray(label)
    if pred.ndim != 1 or truth.ndim != 1 or len(pred) != len(truth):
        raise ValueError("Prediction and label must be equal-length one-dimensional arrays.")
    if not np.isfinite(fs) or fs <= 2 * HIGH_HZ:
        raise ValueError("Sampling rate must exceed twice the upper passband frequency.")
    flags = []
    if not np.all(np.isfinite(pred)):
        flags.append("nonfinite_prediction")
    if not np.all(np.isfinite(truth)):
        flags.append("nonfinite_label")
    if flags:
        return _undefined("nonfinite_signal", flags)
    if len(pred) <= 9:
        return _undefined("too_short", ["filtfilt_requires_more_than_9_samples"])
    if np.all(pred == pred[0]):
        flags.append("constant_prediction")
    if np.all(truth == truth[0]):
        flags.append("constant_label")
    pred = prepare_signal(pred, diff_flag, fs)
    truth = prepare_signal(truth, diff_flag, fs)
    hr_pred = fft_hr(pred, fs)
    hr_label = fft_hr(truth, fs)
    error = hr_pred - hr_label
    return {
        "hr_pred_bpm": hr_pred, "hr_label_bpm": hr_label,
        "snr_db": snr(pred, hr_label, fs), "macc": macc(pred, truth),
        "abs_error_bpm": abs(error), "squared_error_bpm2": error ** 2,
        "ape_percent": abs(error / hr_label) * 100,
        "tail_gt_10_bpm": float(abs(error) > 10),
        "status": "ok", "flags": flags,
    }


def _integer_vector(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or (array.size and array.dtype.kind not in "iu"):
        raise ValueError(name + " must be a one-dimensional integer array.")
    return array.astype(np.int64, copy=False)


def evaluate_arrays(pred: np.ndarray, label: np.ndarray, chunk_starts: Sequence[int],
                    chunk_len: Any, fs: float, diff_flag: bool,
                    recording_n_frames: int,
                    metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Evaluate packed chunks on their original recording time axis.

    ``chunk_len`` may be a positive integer scalar or one positive length per
    chunk. Chunks must be ordered and nonoverlapping; their lengths must sum to
    the packed signal length. The raw index frame count is mandatory so missing
    tail windows cannot disappear from the denominator.

    Recording metrics evaluate a contiguous prefix from frame 0. Leading or
    interior gaps make the recording metric undefined. All complete anchored
    300-frame windows still receive a row; uncovered windows have no metrics.
    A final raw fragment shorter than 300 frames is counted separately.
    """
    prediction = np.asarray(pred)
    truth = np.asarray(label)
    if prediction.ndim != 1 or truth.ndim != 1 or len(prediction) != len(truth):
        raise ValueError("Packed prediction and label must have matching one-dimensional shapes.")
    if prediction.dtype.kind != "f" or truth.dtype.kind != "f":
        raise ValueError("Prediction and label must contain floating-point signals.")
    if not np.isfinite(fs) or float(fs) != FS:
        raise ValueError("ST1 recording and 300-frame windows require fs=30.")
    if isinstance(recording_n_frames, (bool, np.bool_)) or int(recording_n_frames) != recording_n_frames or recording_n_frames < 0:
        raise ValueError("recording_n_frames must be a nonnegative integer from the raw index.")
    n_frames = int(recording_n_frames)
    if not isinstance(diff_flag, (bool, np.bool_, int, np.integer)) or diff_flag not in (False, True):
        raise ValueError("diff_flag must be a boolean (or integer 0/1).")
    starts = _integer_vector(chunk_starts, "chunk_starts")
    lengths = np.asarray(chunk_len)
    if lengths.ndim == 0:
        if lengths.dtype.kind not in "iu" or int(lengths) <= 0:
            raise ValueError("chunk_len must be positive.")
        lengths = np.full(len(starts), int(lengths), dtype=np.int64)
    else:
        lengths = _integer_vector(lengths, "chunk_len")
    if len(starts) != len(lengths) or np.any(starts < 0) or np.any(lengths <= 0):
        raise ValueError("Chunk starts/lengths are inconsistent.")
    if int(np.sum(lengths)) != len(prediction):
        raise ValueError("Packed signal length differs from the sum of chunk lengths.")
    ends = starts + lengths
    if np.any(ends < starts) or np.any(ends > n_frames):
        raise ValueError("Chunk interval falls outside the raw recording.")
    if len(starts) > 1 and np.any(starts[1:] < ends[:-1]):
        raise ValueError("Chunks must be ordered and nonoverlapping.")

    covered = np.zeros(n_frames, dtype=bool)
    timeline_pred = np.full(n_frames, np.nan, dtype=prediction.dtype)
    timeline_label = np.full(n_frames, np.nan, dtype=truth.dtype)
    offset = 0
    for start, end, length in zip(starts, ends, lengths):
        covered[start:end] = True
        timeline_pred[start:end] = prediction[offset:offset + length]
        timeline_label[start:end] = truth[offset:offset + length]
        offset += int(length)

    identity = dict(metadata or {})
    identity.update(fs=FS, diff_flag=bool(diff_flag), recording_n_frames=n_frames)

    def row(unit: str, start: int, end: int, metrics: Dict[str, Any],
            window_index: Optional[int] = None) -> Dict[str, Any]:
        result = dict(identity)
        result.update(metrics)
        result.update(unit=unit, window_index=window_index, start_frame=start,
                      end_frame=end, n_frames=end - start,
                      covered_frames=int(np.count_nonzero(covered[start:end])))
        return result

    contiguous_prefix = (len(starts) > 0 and starts[0] == 0 and
                         (len(starts) == 1 or np.all(starts[1:] == ends[:-1])))
    if contiguous_prefix:
        prefix_end = int(ends[-1])
        recording_metrics = evaluate_signal(timeline_pred[:prefix_end],
                                            timeline_label[:prefix_end], bool(diff_flag), fs)
        if prefix_end < n_frames:
            recording_metrics["flags"].append("uncovered_recording_tail")
        recording = row("recording", 0, prefix_end, recording_metrics)
    else:
        recording = row("recording", 0, n_frames,
                        _undefined("not_evaluated_noncontiguous_coverage",
                                   ["no_prediction_coverage" if not len(starts) else
                                    "leading_or_interior_prediction_gap"]))
    windows = []
    for index, start in enumerate(range(0, n_frames - WINDOW_FRAMES + 1, WINDOW_FRAMES)):
        end = start + WINDOW_FRAMES
        if np.all(covered[start:end]):
            metrics = evaluate_signal(timeline_pred[start:end], timeline_label[start:end],
                                      bool(diff_flag), fs)
        else:
            metrics = _undefined("not_covered", ["prediction_does_not_cover_full_window"])
        windows.append(row("10s", start, end, metrics, index))
    coverage = {
        "recording_n_frames": n_frames, "predicted_frames": len(prediction),
        "uncovered_frames": n_frames - len(prediction),
        "first_predicted_frame": int(starts[0]) if len(starts) else None,
        "last_predicted_frame_exclusive": int(ends[-1]) if len(ends) else None,
        "contiguous_prefix": bool(contiguous_prefix),
        "expected_full_windows": n_frames // WINDOW_FRAMES,
        "covered_full_windows": sum(w["status"] != "not_covered" for w in windows),
        "uncovered_full_windows": sum(w["status"] == "not_covered" for w in windows),
        "evaluated_full_windows": sum(w["status"] == "ok" for w in windows),
        "invalid_covered_windows": sum(w["status"] not in ("ok", "not_covered") for w in windows),
        "partial_tail_frames_not_a_full_window": n_frames % WINDOW_FRAMES,
    }
    return {"recording": recording, "windows": windows, "coverage": coverage}


def _scalar(value: np.ndarray, name: str) -> Any:
    if np.asarray(value).ndim != 0:
        raise ValueError(name + " must be an NPZ scalar.")
    result = np.asarray(value).item()
    return result.decode("utf-8") if isinstance(result, bytes) else result


def load_prediction(path: Union[str, Path]) -> Dict[str, Any]:
    """Read one new ST1 NPZ without pickle, after the sealed-path guard."""
    checked = assert_analysis_path(path)
    with np.load(str(checked), allow_pickle=False) as source:
        missing = set(NPZ_KEYS) - set(source.files)
        if missing:
            raise ValueError("Missing ST1 prediction fields: " + ", ".join(sorted(missing)))
        result = {key: source[key] for key in NPZ_KEYS}
    for key in NPZ_KEYS[4:]:
        result[key] = _scalar(result[key], key)
    result["source_path"] = str(checked)
    return result


def evaluate_npz(path: Union[str, Path], recording_n_frames: int) -> Dict[str, Any]:
    """File convenience API; the index-derived frame count remains required."""
    source = load_prediction(path)
    metadata = {key: source[key] for key in NPZ_KEYS[6:]}
    metadata["source_path"] = source["source_path"]
    return evaluate_arrays(source["pred"], source["label"], source["chunk_starts"],
                           source["chunk_len"], source["fs"], source["diff_flag"],
                           recording_n_frames, metadata)


def _unit_weights(rows: Sequence[Mapping[str, Any]], weighting: str) -> np.ndarray:
    recordings = Counter(str(row["recording_id"]) for row in rows)
    owners = {}
    for row in rows:
        recording, subject = str(row["recording_id"]), str(row["subject_id"])
        if recording in owners and owners[recording] != subject:
            raise ValueError("A recording_id maps to multiple subjects.")
        owners[recording] = subject
    subject_recordings = Counter(owners.values())
    if weighting == "recording":
        raw = [1.0 / recordings[str(row["recording_id"])] for row in rows]
    else:
        raw = [1.0 / (subject_recordings[str(row["subject_id"])] *
                      recordings[str(row["recording_id"])]) for row in rows]
    weights = np.asarray(raw, dtype=np.float64)
    return weights / np.sum(weights)


def aggregate(units: Iterable[Mapping[str, Any]], weighting: str = "recording") -> Dict[str, Any]:
    """Aggregate one model/protocol/unit type, with explicit missingness counts.

    Recording mode: equal total weight per recording, equal weight per valid
    unit within a recording. Subject mode: equal total weight per subject,
    equal recording weight within subject, equal unit weight within recording.
    RMSE is the root of weighted squared error, not a mean of group RMSEs.
    Pearson is computed from pooled weighted moments, never averaged group
    correlations. Undefined values are retained in counts and excluded only
    from that metric's denominator; +/-inf SNR values retain native arithmetic.
    """
    if weighting not in ("recording", "subject"):
        raise ValueError("weighting must be recording or subject.")
    rows = list(units)
    if len({row.get("unit", "recording") for row in rows}) > 1:
        raise ValueError("Aggregate recording and 10s units separately.")
    for row in rows:
        if not row.get("recording_id") or not row.get("subject_id"):
            raise ValueError("Aggregation requires recording_id and subject_id.")
    result = {
        "weighting": weighting, "n_units_total": len(rows),
        "n_units_evaluated": sum(row.get("status", "ok") == "ok" for row in rows),
        "n_flagged_units": sum(bool(row.get("flags")) for row in rows),
        "n_recordings": len({row["recording_id"] for row in rows}),
        "n_subjects": len({row["subject_id"] for row in rows}),
        "MAE": float("nan"), "RMSE": float("nan"), "MAPE": float("nan"),
        "Pearson_r": float("nan"), "SNR": float("nan"), "MACC": float("nan"),
        "tail_gt_10_ratio": float("nan"),
    }
    paired = [row for row in rows if row.get("status", "ok") == "ok" and
              np.isfinite(row.get("hr_pred_bpm", np.nan)) and
              np.isfinite(row.get("hr_label_bpm", np.nan))]
    result["n_hr_pairs"] = len(paired)
    result["n_units_without_hr_pair"] = len(rows) - len(paired)
    result["n_recordings_with_hr_pair"] = len({row["recording_id"] for row in paired})
    result["n_subjects_with_hr_pair"] = len({row["subject_id"] for row in paired})
    if paired:
        weights = _unit_weights(paired, weighting)
        pred = np.asarray([row["hr_pred_bpm"] for row in paired], dtype=np.float64)
        truth = np.asarray([row["hr_label_bpm"] for row in paired], dtype=np.float64)
        error = pred - truth
        result["MAE"] = float(np.dot(weights, np.abs(error)))
        result["RMSE"] = float(np.sqrt(np.dot(weights, error ** 2)))
        with np.errstate(divide="ignore", invalid="ignore"):
            result["MAPE"] = float(np.dot(weights, np.abs(error / truth)) * 100)
        result["tail_gt_10_ratio"] = float(np.dot(weights, np.abs(error) > 10))
        pred_centered = pred - np.dot(weights, pred)
        truth_centered = truth - np.dot(weights, truth)
        denominator = np.sqrt(np.dot(weights, pred_centered ** 2) *
                              np.dot(weights, truth_centered ** 2))
        if len(paired) >= 2 and denominator > 0:
            result["Pearson_r"] = float(np.clip(np.dot(weights, pred_centered * truth_centered) /
                                                denominator, -1, 1))
    for key, output in (("snr_db", "SNR"), ("macc", "MACC")):
        available = [row for row in rows if row.get("status", "ok") == "ok" and
                     not np.isnan(row.get(key, np.nan))]
        result["n_" + output] = len(available)
        if available:
            with np.errstate(invalid="ignore"):
                result[output] = float(np.dot(_unit_weights(available, weighting),
                                             [row[key] for row in available]))
    return result
