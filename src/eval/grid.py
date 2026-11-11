"""ST1 WP3's fixed 108-setting evaluation grid and descriptive summaries.

Real execution is an operational WP3 gate: call ``run_grid`` only after release
of the requested WP2 protocol set. Importing this module performs no file I/O.
Classical methods retain UNSUP_<target> comparison groups and train_source=none.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, get_window, welch
from scipy.stats import kendalltau, rankdata

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.eval import har
from src.eval.build_metrics_tables import FIXTURE_ID, dataset_name, infer_protocol, read_csv, run_artifact
from src.mmpd_contract import approved_metadata, public_analysis_registry

FACTOR_LEVELS = {
    "F1": ("periodogram", "welch", "peak"),
    "F2": ("recording", "30s", "10s"),
    "F3": ("0.6-3.3", "0.75-2.5", "0.7-3.0"),
    "F4": ("default", "no_detrend"),
    "F5": ("same_rule", "peak"),
}
BANDS = {"0.6-3.3": (0.6, 3.3), "0.75-2.5": (0.75, 2.5), "0.7-3.0": (0.7, 3.0)}
AUTHOR_SOURCES = [
    "https://github.com/zizheng-guo/RhythmMamba/blob/1533ad21cb4ae0d31e9dbb9afb0b1eefbed54aed/evaluation/post_process.py#L10-L12",
    "https://github.com/zizheng-guo/RhythmFormer/blob/7c9cea2284cecc50e267abefce711a0eb6d72b6e/evaluation/post_process.py#L10-L12",
]
SCIPY_SOURCE = "https://github.com/scipy/scipy/blob/v1.17.1/scipy/signal/_spectral_py.py"


@dataclass(frozen=True)
class Setting:
    F1: str
    F2: str
    F3: str
    F4: str
    F5: str

    @property
    def grid_id(self):
        return "|".join(getattr(self, key) for key in FACTOR_LEVELS)

    def fields(self):
        return {key: getattr(self, key) for key in FACTOR_LEVELS}


SETTINGS = tuple(Setting(*values) for values in itertools.product(*FACTOR_LEVELS.values()))
BY_ID = {setting.grid_id: setting for setting in SETTINGS}
BASELINE = SETTINGS[0]
BASELINE_10S = Setting("periodogram", "10s", "0.6-3.3", "default", "same_rule")


def welch_parameters(length, fs=30.0):
    """Freeze author-call defaults, including SciPy's observed int(nfft)."""
    nperseg = min(int(length) - 1, 256)
    if nperseg < 1:
        raise ValueError("Welch needs at least two samples")
    return dict(fs=fs, window=get_window("hann", nperseg, fftbins=True),
                nperseg=nperseg, noverlap=nperseg // 2, nfft=int(1e5 / fs),
                detrend="constant", return_onesided=True, scaling="density", average="mean")


def welch_hr(signal, fs=30.0, low_hz=0.6, high_hz=3.3):
    frequencies, power = welch(signal, **welch_parameters(len(signal), fs))
    mask = (frequencies > low_hz) & (frequencies < high_hz)  # Author strict bounds.
    if not np.any(mask):
        return float("nan")
    return float(frequencies[mask][np.argmax(power[mask])] * 60)


def peak_hr(signal, fs=30.0):
    """Official find_peaks HR, with its undefined empty-interval case explicit."""
    peaks, _ = find_peaks(signal)
    if len(peaks) < 2:
        return float("nan")
    return float(60 / (np.mean(np.diff(peaks)) / fs))


def _prepare(signal, diff_flag, band, detrend_mode, fs):
    values = np.asarray(signal)
    if diff_flag:
        values = np.cumsum(values)  # Preserve official float32 accumulation.
    if detrend_mode == "default":
        values = har.detrend(values, 100)
    else:
        values = np.asarray(values, dtype=np.float64)
    b, a = butter(1, [band[0] / fs * 2, band[1] / fs * 2], btype="bandpass")
    return filtfilt(b, a, np.asarray(values, dtype=np.float64))


def _estimate(signal, method, band, fs):
    if method == "periodogram":
        return har.fft_hr(signal, fs, band[0], band[1])
    if method == "welch":
        return welch_hr(signal, fs, band[0], band[1])
    return peak_hr(signal, fs)


def evaluate_recording_grid(pred, label, chunk_starts, chunk_len, fs, diff_flag,
                            recording_n_frames, metadata=None):
    """Return per-unit rows for all settings, preserving original time anchors.

    Recording units follow HAR's available contiguous-prefix rule. Fixed windows
    require full coverage; neither gaps nor partial terminal windows are joined.
    The existing HAR path validates the packed representation and supplies the
    baseline's coverage semantics. Its SNR/MACC are not emitted in this grid.
    """
    baseline = har.evaluate_arrays(pred, label, chunk_starts, chunk_len, fs, diff_flag,
                                   recording_n_frames, metadata)
    pred, label = np.asarray(pred), np.asarray(label)
    starts = np.asarray(chunk_starts, dtype=np.int64)
    lengths = np.asarray(chunk_len)
    if lengths.ndim == 0:
        lengths = np.full(len(starts), int(lengths), dtype=np.int64)
    n_frames = int(recording_n_frames)
    timeline_pred = np.full(n_frames, np.nan, dtype=pred.dtype)
    timeline_label = np.full(n_frames, np.nan, dtype=label.dtype)
    covered = np.zeros(n_frames, dtype=bool)
    offset = 0
    for start, length in zip(starts, lengths):
        end = int(start + length)
        timeline_pred[start:end] = pred[offset:offset + length]
        timeline_label[start:end] = label[offset:offset + length]
        covered[start:end] = True
        offset += int(length)
    units = {"recording": [baseline["recording"]], "10s": baseline["windows"], "30s": []}
    for number, start in enumerate(range(0, n_frames - 900 + 1, 900)):
        end = start + 900
        row = dict(metadata or {}, unit="30s", window_index=number, start_frame=start,
                   end_frame=end, n_frames=900, covered_frames=int(np.count_nonzero(covered[start:end])),
                   fs=float(fs), diff_flag=bool(diff_flag), recording_n_frames=n_frames, flags=[])
        if not np.all(covered[start:end]):
            row.update(status="not_covered", flags=["prediction_does_not_cover_full_window"])
        elif not np.all(np.isfinite(timeline_pred[start:end])) or not np.all(np.isfinite(timeline_label[start:end])):
            row.update(status="nonfinite_signal", flags=["nonfinite_prediction_or_label"])
        else:
            row["status"] = "ok"
            for signal, name in ((timeline_pred[start:end], "prediction"), (timeline_label[start:end], "label")):
                if np.all(signal == signal[0]):
                    row["flags"].append("constant_" + name)
        units["30s"].append(row)
    prepared, estimates, result = {}, {}, {}
    for setting in SETTINGS:
        output = []
        band = BANDS[setting.F3]
        for template in units[setting.F2]:
            row = {key: value for key, value in template.items() if key not in har.METRIC_KEYS}
            row["flags"] = list(template.get("flags", []))
            row.update(hr_pred_bpm=float("nan"), hr_label_bpm=float("nan"))
            if row["status"] == "ok":
                start, end = row["start_frame"], row["end_frame"]
                cache_key = (setting.F2, start, end, setting.F3, setting.F4)
                if cache_key not in prepared:
                    prepared[cache_key] = tuple(_prepare(signal[start:end], diff_flag, band, setting.F4, fs)
                                                for signal in (timeline_pred, timeline_label))
                truth_method = setting.F1 if setting.F5 == "same_rule" else "peak"
                for method in (setting.F1, truth_method):
                    key = cache_key + (method,)
                    if key not in estimates:
                        estimates[key] = tuple(_estimate(signal, method, band, fs)
                                               for signal in prepared[cache_key])
                row["hr_pred_bpm"] = estimates[cache_key + (setting.F1,)][0]
                row["hr_label_bpm"] = estimates[cache_key + (truth_method,)][1]
                if not np.isfinite(row["hr_pred_bpm"]) or not np.isfinite(row["hr_label_bpm"]):
                    row["status"] = "undefined_hr"
                    row["flags"].append("estimator_has_no_finite_hr")
            output.append(row)
        result[setting.grid_id] = output
    return {"units": result, "coverage": baseline["coverage"],
            "cache_counts": {"prepared_signal_pairs": len(prepared), "estimated_hr_pairs": len(estimates)}}


def aggregate_setting(units, setting, identity, recording_count):
    metrics = har.aggregate(units, weighting="recording")
    row = dict(identity, grid_id=setting.grid_id, **setting.fields(), weighting="recording")
    row.update({key: metrics[key] for key in ("MAE", "RMSE", "MAPE", "Pearson_r",
                                             "n_recordings", "n_subjects", "n_recordings_with_hr_pair",
                                             "n_subjects_with_hr_pair", "n_flagged_units")})
    row.update(r=metrics["Pearson_r"], n_total=metrics["n_units_total"], n_valid=metrics["n_hr_pairs"],
               n_invalid=metrics["n_units_without_hr_pair"], n_recordings_in_run=recording_count,
               n_uncovered=sum(unit["status"] in ("not_covered", "not_evaluated_noncontiguous_coverage") for unit in units))
    row["n_undefined"] = row["n_invalid"] - row["n_uncovered"]
    row["status"] = "no_full_units" if not row["n_total"] else "partial" if row["n_invalid"] else "ok"
    return row


def _number(value):
    return float("nan") if value is None or value == "" else float(value)


def _group_key(row):
    return tuple(str(row.get(key, "")) for key in
                 ("kind", "model", "train_source", "comparison_protocol", "population"))


def competitor_scores(rows):
    """Pool RT seeds only; other kinds remain separate registered-run competitors."""
    grouped = defaultdict(list)
    for row in rows:
        key = _group_key(row)
        suffix = "all_available_seeds" if row["kind"] == "RT" else str(row["run_id"])
        grouped[(key, suffix, row["grid_id"])].append(row)
    scores = []
    for (key, suffix, grid_id), selected in sorted(grouped.items()):
        seeds = [str(row["seed"]) for row in selected]
        if len(seeds) != len(set(seeds)):
            raise ValueError("Multiple successful runs share one competitor/seed/setting; select attempts explicitly")
        valid = [row for row in selected if np.isfinite(_number(row["MAE"]))]
        first = selected[0]
        score = {name: first.get(name, "") for name in
                 ("kind", "model", "train_source", "protocol", "comparison_protocol", "population")}
        score.update(summary_type="competitor_setting", competitor_id="|".join(key + (suffix,)),
                     grid_id=grid_id, **BY_ID[grid_id].fields(),
                     seed_aggregation="arithmetic_mean_of_available_seed_MAE" if first["kind"] == "RT" else "single_registered_run",
                     seed_count=len(valid), registered_seed_count=len(selected),
                     planned_seed_count=3 if first["kind"] == "RT" else 1,
                     available_seeds=json.dumps(sorted(str(row["seed"]) for row in valid)),
                     registered_seeds=json.dumps(sorted(seeds)),
                     run_ids=json.dumps(sorted(str(row["run_id"]) for row in selected)),
                     mean_seed_MAE=float(np.mean([_number(row["MAE"]) for row in valid])) if valid else float("nan"))
        scores.append(score)
    return scores


def summarize_grid(rows):
    """Ranges, one-factor-at-default deltas, and Kendall tau-b ranking changes."""
    by_run = defaultdict(list)
    for row in rows:
        by_run[row["run_id"]].append(row)
    summary = []
    for run_id, selected in sorted(by_run.items()):
        settings = {row["grid_id"]: row for row in selected}
        if len(settings) != len(selected) or set(settings) != set(BY_ID):
            raise ValueError("Every run must have exactly the fixed 108 settings: " + run_id)
        base = settings[BASELINE.grid_id]
        identity = {key: base.get(key, "") for key in
                    ("run_id", "kind", "model", "train_source", "protocol", "comparison_protocol", "population", "seed")}
        finite = [row for row in selected if np.isfinite(_number(row["MAE"]))]
        low = min(_number(row["MAE"]) for row in finite) if finite else float("nan")
        high = max(_number(row["MAE"]) for row in finite) if finite else float("nan")
        summary.append(dict(identity, summary_type="run_range", settings_total=108, settings_valid=len(finite),
                            minimum_MAE=low, maximum_MAE=high,
                            minimum_settings=json.dumps([BY_ID[row["grid_id"]].fields() for row in finite if _number(row["MAE"]) == low]),
                            maximum_settings=json.dumps([BY_ID[row["grid_id"]].fields() for row in finite if _number(row["MAE"]) == high])))
        for setting in SETTINGS:
            changed = [key for key in FACTOR_LEVELS if getattr(setting, key) != getattr(BASELINE, key)]
            if len(changed) != 1:
                continue
            row = settings[setting.grid_id]
            summary.append(dict(identity, summary_type="one_factor_at_default", factor=changed[0],
                                level=getattr(setting, changed[0]), grid_id=setting.grid_id,
                                baseline_grid_id=BASELINE.grid_id, baseline_MAE=_number(base["MAE"]),
                                setting_MAE=_number(row["MAE"]), MAE_delta=_number(row["MAE"]) - _number(base["MAE"]),
                                baseline_valid_units=base["n_valid"], setting_valid_units=row["n_valid"]))
    scores = competitor_scores(rows)
    summary.extend(scores)
    comparison_groups = defaultdict(dict)
    for score in scores:
        group = (score["comparison_protocol"], score["population"])
        comparison_groups[group].setdefault(score["grid_id"], {})[score["competitor_id"]] = score
    for (protocol, population), group in sorted(comparison_groups.items()):
        baseline = group.get(BASELINE.grid_id, {})
        for setting in SETTINGS:
            current = group.get(setting.grid_id, {})
            common = sorted(key for key in set(baseline) & set(current)
                            if np.isfinite(baseline[key]["mean_seed_MAE"]) and np.isfinite(current[key]["mean_seed_MAE"]))
            before = rankdata([baseline[key]["mean_seed_MAE"] for key in common], method="average")
            after = rankdata([current[key]["mean_seed_MAE"] for key in common], method="average")
            tau = float(kendalltau(before, after, variant="b")[0]) if len(common) >= 2 else float("nan")
            seed_counts = {key: {"baseline": baseline[key]["seed_count"], "setting": current[key]["seed_count"],
                                 "planned": baseline[key]["planned_seed_count"],
                                 "baseline_registered": baseline[key]["registered_seed_count"],
                                 "setting_registered": current[key]["registered_seed_count"]}
                           for key in common}
            dropped = sorted((set(baseline) | set(current)) - set(common))
            partial_seeds = any(value["baseline"] < value["planned"] or value["setting"] < value["planned"]
                                for value in seed_counts.values())
            summary.append(dict(summary_type="rank_kendall_tau", comparison_protocol=protocol,
                                population=population, grid_id=setting.grid_id, **setting.fields(),
                                baseline_grid_id=BASELINE.grid_id, kendall_tau_b=tau,
                                competitors_common=len(common), competitors_excluded=len(dropped),
                                excluded_competitors=json.dumps(dropped), seed_counts=json.dumps(seed_counts, sort_keys=True),
                                competitor_ranks=json.dumps({key: {"baseline": float(before[i]), "setting": float(after[i])}
                                                             for i, key in enumerate(common)}, sort_keys=True),
                                status="undefined_rank_correlation" if not np.isfinite(tau) else
                                "partial_cohort" if dropped else "partial_seed_coverage" if partial_seeds else "ok"))
    return summary


def eligible_run(row):
    if str(row.get("exit_code", "")) != "0" or not str(row.get("run_id", "")).startswith("st1_"):
        return False
    if FIXTURE_ID.search(str(row["run_id"])) or FIXTURE_ID.search(str(row.get("model", ""))):
        return False
    if str(row.get('target', '')) not in ('PURE', 'UBFC', 'UBFC-rPPG', 'MMPD'):
        return False
    # Generated-input QC and NAT-off are distinct protocol checks. Only the
    # NAT-all public A prediction corpus supplies MMPD grid/WP7 observations.
    if str(row['run_id']).startswith('st1_rg_qc_') or '_off_' in str(row['run_id']):
        return False
    return ((row.get('target')=='MMPD' and row.get('WP') in ('MCPU','MGPU') and
             (row.get('kind') in ('RW','RT','RG') or str(row.get('model','')).upper() in ('CHROM','POS'))) or
            (row.get("WP"), row.get("kind")) in (("WP2", "RW"), ("WP2", "RG"), ("WP4", "RT"), ("WP5", "RG")) or
            (row.get("WP") == "WP2" and str(row.get("model", "")).upper() in ("CHROM", "POS")))


def run_grid(run_ids=None, root=None, progress=None):
    """Read only registered, successful, converted public outputs after release."""
    root = Path(root) if root is not None else ROOT
    registry = public_analysis_registry(root)
    runs = [row for row in registry if eligible_run(row) and (run_ids is None or row["run_id"] in run_ids)]
    if len({row["run_id"] for row in runs}) != len(runs):
        raise ValueError("Duplicate successful registered run IDs")
    if run_ids is not None and {row["run_id"] for row in runs} != set(run_ids):
        raise ValueError("Requested run IDs must all be registered successful eligible runs")
    rows, diagnostics = [], []
    processed = 0
    progress_total = sum(int(row.get('expected_recordings') or 59) for row in runs)
    progress_started = __import__('time').monotonic()
    for run in runs:
        target = dataset_name(run["target"])
        if target not in ("PURE", "UBFC", "MMPD"):
            raise ValueError("Unsupported target dataset")
        a_subjects = None
        if target == "MMPD":
            split_path = har.assert_analysis_path(root / "data/splits/mmpd_split_st1.json")
            if not split_path.is_file() or split_path.stat().st_mode & 0o222:
                raise PermissionError("MMPD grid requires the PI-approved read-only split")
            _, split, index, _ = approved_metadata(root=root)
            a_subjects = set(split["A"])
            if (len(a_subjects) != 11 or len(set(split["B"])) != 22 or a_subjects & set(split["B"]) or
                    len(split["A"]) != 11 or len(split["B"]) != 22):
                raise ValueError("Invalid approved split")
        if target != "MMPD":
            index = {item["recording_id"]: item for item in read_csv(root / "data/index" / (target.lower() + "_index_st1.csv"))}
        manifest_path = run_artifact(root / "outputs/st1_predictions" / run["run_id"] / "predictions.csv",
                                     root, run["run_id"], "outputs/st1_predictions")
        complete_path = run_artifact(manifest_path.parent / "har_aggregates.json", root, run["run_id"], "outputs/st1_predictions")
        if not manifest_path.is_file() or not complete_path.is_file():
            diagnostics.append(dict(run_id=run["run_id"], status="conversion_not_complete"))
            continue
        manifest = read_csv(manifest_path)
        if not manifest or len({item["recording_id"] for item in manifest}) != len(manifest):
            raise ValueError("Public prediction list is empty or has duplicate recordings")
        protocol = infer_protocol(run)
        identity = {key: run.get(key, "") for key in ("run_id", "kind", "model", "train_source", "seed")}
        identity.update(source_WP=run["WP"], analysis_WP="WP3", target=target, protocol=protocol,
                        comparison_protocol=protocol, population="MMPD-A" if target == "MMPD" else target)
        expected_count = sum(a_subjects is None or item["subject_id"] in a_subjects for item in index.values())
        identity.update(expected_recordings=expected_count,
                        recording_scope_status="complete" if len(manifest) == expected_count else "partial")
        by_setting = {setting.grid_id: [] for setting in SETTINGS}
        for item in sorted(manifest, key=lambda row: row["recording_id"]):
            if item["recording_id"] not in index or item["subject_id"] != index[item["recording_id"]]["subject_id"]:
                raise ValueError("Prediction list disagrees with the fresh index")
            # Check cohort before reading a waveform, even if a B file were
            # incorrectly listed in a public manifest outside the sealed tree.
            if a_subjects is not None and item["subject_id"] not in a_subjects:
                raise har.SealedPathError("MMPD grid cannot read a B-subject waveform")
            path = run_artifact(Path(item["path"]), root, run["run_id"], "outputs/st1_predictions")
            source = har.load_prediction(path)
            for key in ("recording_id", "subject_id"):
                if str(source[key]) != item[key]:
                    raise ValueError("NPZ identity disagrees with its public prediction list")
            if (source["model"] != run["model"] or dataset_name(source["train_source"]) != dataset_name(run["train_source"]) or
                    str(source["seed"]) != str(run["seed"])):
                raise ValueError("NPZ model/source/seed disagrees with its registered run")
            if run.get("checkpoint_sha256") and source["checkpoint_sha256"] != run["checkpoint_sha256"]:
                raise ValueError("NPZ checkpoint disagrees with its registered run")
            metadata = dict(identity, recording_id=item["recording_id"], subject_id=item["subject_id"])
            evaluated = evaluate_recording_grid(source["pred"], source["label"], source["chunk_starts"],
                                                 source["chunk_len"], source["fs"], source["diff_flag"],
                                               int(index[item["recording_id"]]["frame_count"]), metadata)
            processed += 1
            if progress is not None:
                elapsed = __import__('time').monotonic() - progress_started
                progress(dict(stage='grid_recordings', done=processed, total=progress_total,
                              elapsed_seconds=elapsed, remaining_seconds=elapsed / processed * (progress_total - processed)))
            for grid_id, units in evaluated["units"].items():
                by_setting[grid_id].extend(units)
        rows.extend(aggregate_setting(by_setting[setting.grid_id], setting, identity, len(manifest)) for setting in SETTINGS)
    return rows, summarize_grid(rows), diagnostics


def write_outputs(rows, summary, output_dir=None):
    directory = har.assert_analysis_path(output_dir if output_dir is not None else ROOT / "outputs")
    directory.mkdir(parents=True, exist_ok=True)
    for name, items in (("st1_eval_grid.csv", rows), ("st1_eval_grid_summary.csv", summary)):
        path = har.assert_analysis_path(directory / name)
        fields = list(dict.fromkeys(["run_id", "kind", "model", "train_source", "protocol", "comparison_protocol", "population", "seed"] +
                                   [key for row in items for key in row]))
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writer.writerows(items)


def verify_har_baselines(rows, root=None, tolerance=1e-6):
    """Compare saved grid baselines with registered public converter aggregates.

    Both-NaN values are explicitly undefined, not numerical zeros or finite
    agreement. Count fields must match exactly. No prediction/unit files open.
    """
    root = Path(root) if root is not None else ROOT
    registry_rows = public_analysis_registry(root)
    registry = {row["run_id"]: row for row in registry_rows}
    if len(registry) != len(registry_rows):
        raise ValueError("Duplicate registered run IDs during HAR baseline verification")
    by_run = defaultdict(list)
    for row in rows:
        by_run[row["run_id"]].append(row)
    checks, failures = [], []
    metric_fields = {"MAE": "MAE", "RMSE": "RMSE", "MAPE": "MAPE", "r": "Pearson_r"}
    count_fields = {"n_total": "n_units_total", "n_valid": "n_hr_pairs", "n_invalid": "n_units_without_hr_pair",
                    "n_recordings": "n_recordings", "n_subjects": "n_subjects",
                    "n_recordings_with_hr_pair": "n_recordings_with_hr_pair",
                    "n_subjects_with_hr_pair": "n_subjects_with_hr_pair", "n_flagged_units": "n_flagged_units"}

    def recorded(value):
        return None if np.isnan(value) else float(value) if np.isfinite(value) else str(value)

    for run_id, run_rows in sorted(by_run.items()):
        run = registry.get(run_id)
        if run is None or not eligible_run(run):
            raise ValueError("HAR comparison requires a successful registered eligible run: " + run_id)
        target = dataset_name(run["target"])
        population = "MMPD-A" if target == "MMPD" else target
        if target == "MMPD":
            split = har.assert_analysis_path(root / "data/splits/mmpd_split_st1.json")
            if not split.is_file() or split.stat().st_mode & 0o222:
                raise PermissionError("MMPD HAR comparison requires the PI-approved read-only split")
            approved_metadata(root=root)
        aggregate_path = run_artifact(root / "outputs/st1_predictions" / run_id / "har_aggregates.json",
                                      root, run_id, "outputs/st1_predictions")
        aggregates = json.loads(aggregate_path.read_text())
        for setting in (BASELINE, BASELINE_10S):
            selected = [row for row in run_rows if row.get("grid_id") == setting.grid_id]
            references = [row for row in aggregates if row.get("population") == population and
                          row.get("unit") == setting.F2 and row.get("weighting") == "recording"]
            if len(selected) != 1 or len(references) != 1:
                failures.append(dict(run_id=run_id, unit=setting.F2, issue="missing_or_duplicate_baseline",
                                     grid_rows=len(selected), HAR_rows=len(references)))
                continue
            actual, reference = selected[0], references[0]
            identity = {"model": run["model"], "kind": run["kind"], "seed": str(run["seed"]),
                        "train_source": dataset_name(run["train_source"]), "protocol": infer_protocol(run),
                        "population": population, "weighting": "recording", **setting.fields()}
            for key, expected in identity.items():
                value = dataset_name(actual.get(key, "")) if key == "train_source" else str(actual.get(key, ""))
                if value != expected:
                    raise ValueError("HAR comparison grid/registry identity mismatch: " + run_id + ": " + key)
            check = dict(run_id=run_id, unit=setting.F2, population=population, weighting="recording",
                         aggregate_path=str(aggregate_path), metrics={}, counts={})
            for grid_key, har_key in metric_fields.items():
                a, b = _number(actual.get(grid_key)), _number(reference.get(har_key))
                present = grid_key in actual and har_key in reference
                both_undefined = present and np.isnan(a) and np.isnan(b)
                difference = abs(a - b) if np.isfinite(a) and np.isfinite(b) else None
                passed = present and (both_undefined or (difference is not None and difference <= tolerance))
                status = "both_undefined" if both_undefined else "matched_finite" if passed else "mismatch"
                check["metrics"][grid_key] = dict(grid=recorded(a), HAR=recorded(b),
                                                 absolute_difference=difference, status=status)
                if not passed:
                    failures.append(dict(run_id=run_id, unit=setting.F2, field=grid_key, issue="metric_mismatch"))
            # The public CSV emits both names; neither alias may silently drift.
            if "Pearson_r" in actual:
                a, b = _number(actual["r"]), _number(actual["Pearson_r"])
                if not ((np.isnan(a) and np.isnan(b)) or (np.isfinite(a) and np.isfinite(b) and a == b)):
                    failures.append(dict(run_id=run_id, unit=setting.F2, field="Pearson_r", issue="r_alias_mismatch"))
            for grid_key, har_key in count_fields.items():
                a, b = _number(actual.get(grid_key)), _number(reference.get(har_key))
                passed = bool(np.isfinite(a) and np.isfinite(b) and a >= 0 and a.is_integer() and a == b)
                check["counts"][grid_key] = dict(grid=recorded(a), HAR=recorded(b), exact_match=passed)
                if not passed:
                    failures.append(dict(run_id=run_id, unit=setting.F2, field=grid_key, issue="count_mismatch"))
            checks.append(check)
    return dict(status="failed" if failures else "passed" if by_run else "not_run_no_grid_rows",
                absolute_tolerance=tolerance, run_count=len(by_run), unit_checks=len(checks),
                metrics=list(metric_fields), weighting="recording", checks=checks, failures=failures,
                signal_files_read=0, scope="Saved grid recording/10s baselines versus each run's public converter HAR aggregates.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", action="append", help="Optional explicit set; output tables contain only that selected set")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    grid_rows, summaries, skipped = run_grid(args.run_id)
    write_outputs(grid_rows, summaries, args.output_dir)
    baseline_check = verify_har_baselines(read_csv(args.output_dir / "st1_eval_grid.csv"))
    baseline_check["grid_source_path"] = str(har.assert_analysis_path(args.output_dir / "st1_eval_grid.csv"))
    evidence_path = har.assert_analysis_path(ROOT / "reports/wp3/har_baseline_validation.json")
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(baseline_check, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"grid_rows": len(grid_rows), "summary_rows": len(summaries), "diagnostics": skipped,
                      "HAR_baseline_status": baseline_check["status"], "HAR_baseline_evidence": str(evidence_path)}))
    if baseline_check["status"] != "passed":
        raise SystemExit("HAR baseline verification did not pass; inspect " + str(evidence_path))
