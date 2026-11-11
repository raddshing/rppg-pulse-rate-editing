"""Copy NAT log values and HAR aggregates into source-backed ST1 metric tables.

NAT numbers, including their printed standard errors and NaN spellings, remain
verbatim strings. This module never evaluates signals, opens prediction files,
fills unreported native metrics, or claims completion of a study's run plan.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.eval.har import assert_analysis_path

METRICS = ("MAE", "RMSE", "MAPE", "Pearson_r", "SNR", "MACC", "tail_gt_10_ratio")
NUMBER = r"[-+]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|nan|inf(?:inity)?)"
NATIVE_METRIC = re.compile(
    r"^(FFT|PEAK)\s+(MAE|RMSE|MAPE|Pearson|SNR)\s+\((FFT|Peak) Label\):\s*(" +
    NUMBER + r")\s*\+/-\s*(" + NUMBER + r")\s*(?:\(dB\))?\s*$", re.I)
NATIVE_MACC = re.compile(r"^MACC\s+\(avg\):\s*(" + NUMBER + r")\s*\+/-\s*(" + NUMBER + r")\s*$", re.I)
METHOD_START = re.compile(r"^===Unsupervised Method\s*\(\s*([^\s)]+)\s*\)\s*Predicting\s*===$")
METHOD_USED = re.compile(r"^Used Unsupervised Method:\s*(\S+)\s*$")
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
FIXTURE_ID = re.compile(r"(?:^|[_-])(smoke|test|fixture|synthetic)(?:[_-]|$)", re.I)


def parse_native_log(text, default_method=""):
    """Extract separate official metric blocks without recomputing any value."""
    blocks, current = [], None
    method = default_method.upper()
    for line_number, raw in enumerate(text.split("\n"), 1):
        line = ANSI.sub("", raw).rsplit("\r", 1)[-1].strip()
        marker = METHOD_START.fullmatch(line) or METHOD_USED.fullmatch(line)
        if marker:
            if current and current["values"]:
                blocks.append(current)
            current = None
            method = marker.group(1).upper()
            continue
        matched = NATIVE_METRIC.fullmatch(line)
        macc = NATIVE_MACC.fullmatch(line)
        if not matched and not macc:
            continue
        if current is None:
            current = dict(method=method, values={}, standard_errors={}, line_numbers={},
                           raw_lines={}, native_hr_method="unreported")
        if matched:
            prefix, metric, _, value, standard_error = matched.groups()
            metric = "Pearson_r" if metric.lower() == "pearson" else metric.upper()
            # The official peak-HR branch still prints an FFT label for SNR.
            # Preserve that raw line, but do not mistake it for another HR mode.
            if metric != "SNR":
                hr_method = "FFT" if prefix.upper() == "FFT" else "Peak"
                prior = current["native_hr_method"]
                if prior not in ("unreported", hr_method):
                    raise ValueError("Mixed native HR modes in one metric block")
                current["native_hr_method"] = hr_method
        else:
            metric, (value, standard_error) = "MACC", macc.groups()
        if metric in current["values"]:
            raise ValueError("Repeated native metric without a new method block: " + metric)
        current["values"][metric] = value
        current["standard_errors"][metric] = standard_error
        current["line_numbers"][metric] = line_number
        current["raw_lines"][metric] = line
    if current and current["values"]:
        blocks.append(current)
    return blocks


def read_csv(path):
    checked = assert_analysis_path(path)
    with checked.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path):
    return json.loads(assert_analysis_path(path).read_text())


def eligible_run(row):
    run_id = str(row.get("run_id", ""))
    if not re.fullmatch(r"st1_[A-Za-z0-9_]+", run_id) or FIXTURE_ID.search(run_id):
        return False
    if FIXTURE_ID.search(str(row.get("model", ""))):
        return False
    if str(row.get("exit_code", "")).strip() != "0" or row.get("WP") not in ("WP2", "WP4"):
        return False
    return row.get("kind") in ("RW", "RT") or str(row.get("model", "")).upper() in ("CHROM", "POS")


def dataset_name(value):
    text = str(value).strip()
    aliases = {"PURE": "PURE", "UBFC": "UBFC", "UBFC-RPPG": "UBFC", "MMPD": "MMPD",
               "NONE": "none", "": "none", "NA": "none", "N/A": "none"}
    return aliases.get(text.upper(), text)


def infer_protocol(row):
    source, target = dataset_name(row.get("train_source", "")), dataset_name(row.get("target", ""))
    if source == "none":
        inferred = "UNSUP_" + target
    else:
        inferred = {("PURE", "UBFC"): "PU", ("UBFC", "PURE"): "UP",
                    ("PURE", "MMPD"): "PM", ("UBFC", "MMPD"): "UM",
                    ("MAUBFC", "PURE"): "UP", ("MAUBFC", "MMPD"): "UM"}.get((source, target), "")
    explicit = row.get("protocol", "")
    if explicit and inferred and explicit != inferred:
        raise ValueError("Registry protocol disagrees with source/target for " + row["run_id"])
    return explicit or inferred or "unreported"


def run_artifact(path, root, run_id, area):
    """A run may only use artifacts whose resolved path still names that run."""
    checked = assert_analysis_path(path)
    directory = assert_analysis_path(root / area)
    try:
        relative = checked.relative_to(directory)
    except ValueError:
        raise ValueError("Run artifact is outside its expected project area: " + str(path))
    if len(relative.parts) < 2 or relative.parts[0] != run_id:
        raise ValueError("Run artifact belongs to a different run: " + str(path))
    return checked


def verify_har_identity(path, row, protocol):
    units = read_csv(path)
    if not units:
        return False
    expected = {"run_id": row["run_id"], "model": row["model"], "seed": str(row.get("seed", "")),
                "train_source": dataset_name(row.get("train_source", "")), "protocol": protocol,
                "checkpoint_sha256": row.get("checkpoint_sha256", "")}
    for unit in units:
        for key, value in expected.items():
            actual = unit.get(key, "")
            if key == "train_source":
                actual = dataset_name(actual)
            if actual != value:
                raise ValueError("HAR identity mismatch for " + row["run_id"] + ": " + key)
    return True


def _base_row(run):
    return {key: run.get(key, "") for key in
            ("run_id", "WP", "kind", "model", "train_source", "target", "seed",
             "config_path", "config_sha256", "checkpoint_sha256", "toolbox_commit")}


def _missing_row(base, layer, population, status):
    row = dict(base, layer=layer, population=population,
               unit="native_config" if layer == "NAT" else "unreported",
               weighting="official" if layer == "NAT" else "unreported", source_status=status,
               scope_status="partial", unreported_metrics=json.dumps(list(METRICS)))
    return row


def build_rows(registry_path=None, root=None, run_ids=None):
    root = Path(root) if root is not None else ROOT
    registry_path = Path(registry_path) if registry_path is not None else root / "runs/st1_runs.csv"
    registry = read_csv(registry_path)
    allowed = None if run_ids is None else set(run_ids)
    # Instruction-02 MMPD native logs are sealed from creation. Its dedicated
    # table writer consumes public native tokens/subset aggregates instead.
    # Keep these historical PURE/UBFC tables from attempting any sealed read.
    runs = [row for row in registry if (allowed is None or row['run_id'] in allowed) and eligible_run(row)
            and dataset_name(row.get('target', '')) in ('PURE', 'UBFC')]
    ids = [row["run_id"] for row in runs]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate eligible run IDs in the registry")
    rows, diagnostics, index_cache = [], [], {}
    for run in runs:
        run_id = run["run_id"]
        target = dataset_name(run.get("target", ""))
        protocol = infer_protocol(run)
        base = dict(_base_row(run), protocol=protocol)
        population = "MMPD full" if target == "MMPD" else target
        split = None
        if target == "MMPD":
            split_path = assert_analysis_path(root / "data/splits/mmpd_split_st1.json")
            if not split_path.is_file() or split_path.stat().st_mode & 0o222:
                diagnostics.append(dict(run_id=run_id, issue="mmpd_split_not_approved_no_artifacts_read"))
                continue
            split = read_json(split_path)
        if target not in index_cache:
            index_path = assert_analysis_path(root / "data/index" / (target.lower() + "_index_st1.csv"))
            index_cache[target] = read_csv(index_path) if index_path.is_file() else None
        index = index_cache[target]
        expected_recordings = len(index) if index is not None else None

        log_value = run.get("native_log", "")
        if not log_value:
            note_match = re.search(r"(?:^|;)\s*native_log=([^;]+)", run.get("notes", ""))
            log_value = note_match.group(1).strip() if note_match else str(root / "runs" / run_id / "native.log")
        log = run_artifact(Path(log_value), root, run_id, "runs")
        expected_method = run["model"].upper() if run["model"].upper() in ("CHROM", "POS") else ""
        if log.is_file():
            blocks = parse_native_log(assert_analysis_path(log).read_text(), expected_method)
            matching = [block for block in blocks if block["method"] == expected_method]
            if len(matching) > 1:
                raise ValueError("Multiple matching native blocks for " + run_id)
            other_methods = sorted({block["method"] for block in blocks if block["method"] != expected_method})
            if other_methods:
                diagnostics.append(dict(run_id=run_id, issue="other_native_method_blocks_not_assigned",
                                        methods=other_methods))
            if matching:
                block = matching[0]
                native = dict(base, layer="NAT", population=population, unit="native_config",
                              weighting="official", source_status="reported", scope_status="native_scope_unreported",
                              source_path=str(log), native_hr_method=block["native_hr_method"],
                              expected_recordings=expected_recordings)
                native.update(block["values"])
                native.update({key + "_SE": value for key, value in block["standard_errors"].items()})
                native["native_line_numbers"] = json.dumps(block["line_numbers"], sort_keys=True)
                native["native_raw_lines"] = json.dumps(block["raw_lines"], sort_keys=True)
                native["unreported_metrics"] = json.dumps([key for key in METRICS if key not in block["values"]])
                rows.append(native)
            else:
                rows.append(_missing_row(base, "NAT", population, "native_metrics_unreported"))
        else:
            rows.append(_missing_row(base, "NAT", population, "missing_native_log"))

        aggregates_path = run_artifact(root / "outputs/st1_predictions" / run_id / "har_aggregates.json",
                                       root, run_id, "outputs/st1_predictions")
        units_path = run_artifact(aggregates_path.parent / "har_units.csv", root, run_id,
                                 "outputs/st1_predictions")
        if not aggregates_path.is_file() or not units_path.is_file():
            rows.append(_missing_row(base, "HAR", population, "missing_har_aggregates_or_identity"))
            continue
        if not verify_har_identity(units_path, run, protocol):
            rows.append(_missing_row(base, "HAR", population, "empty_har_identity"))
            continue
        aggregates = read_json(aggregates_path)
        if not isinstance(aggregates, list) or not aggregates:
            raise ValueError("HAR aggregate file must be a nonempty list for " + run_id)
        seen = set()
        for metrics in aggregates:
            current_population = metrics.get("population")
            allowed_populations = {"MMPD full", "MMPD-A"} if target == "MMPD" else {target}
            if current_population not in allowed_populations:
                raise ValueError("Unexpected HAR population for " + run_id)
            coordinate = (current_population, metrics.get("unit"), metrics.get("weighting"))
            if coordinate in seen or coordinate[1] not in ("recording", "10s") or coordinate[2] not in ("recording", "subject"):
                raise ValueError("Invalid or duplicate HAR aggregate coordinate for " + run_id)
            seen.add(coordinate)
            expected = expected_recordings
            if current_population == "MMPD-A" and index is not None:
                expected = sum(item["subject_id"] in set(split["A"]) for item in index)
            partial = (expected is None or metrics.get("n_recordings") != expected or
                       metrics.get("n_units_without_hr_pair", 0) > 0)
            result = dict(base, layer="HAR", population=current_population, unit=coordinate[1],
                          weighting=coordinate[2], source_status="reported", source_path=str(aggregates_path),
                          scope_status="partial" if partial else "complete_available_units",
                          expected_recordings=expected)
            result.update({key: value for key, value in metrics.items() if key in METRICS or key.startswith("n_")})
            result["unreported_metrics"] = json.dumps([key for key in METRICS if key not in metrics])
            rows.append(result)
    summary = {
        "status": "partial_registry_snapshot", "study_completion_claimed": False,
        "registry_path": str(assert_analysis_path(registry_path)), "registry_rows": len(registry),
        "eligible_successful_runs": len(runs), "excluded_registry_rows": len(registry) - len(runs),
        "table_rows": len(rows), "diagnostics": diagnostics,
        "scope": "Registered successful PURE/UBFC WP2 RW/classical and WP4 RT runs only. Instruction-02 MMPD/RG tables have a separate public-artifact writer; absent requested runs remain unreported.",
        "NAT": "Exact printed mean and standard-error tokens; unreported metrics are never recomputed.",
        "HAR": "Copied from converter aggregates after checking public unit identities; no signal files read.",
    }
    return rows, summary


def write_tables(rows, summary, output_dir=None, summary_path=None):
    output_dir = Path(output_dir) if output_dir is not None else ROOT / "outputs"
    output_dir = assert_analysis_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_fields = ["run_id", "WP", "kind", "model", "train_source", "target", "protocol", "seed",
                   "layer", "population", "unit", "weighting", "source_status", "scope_status"]
    metric_fields = [item for metric in METRICS for item in (metric, metric + "_SE")]
    outputs = []
    for name, wp in (("st1_rw_metrics.csv", "WP2"), ("st1_rt_metrics.csv", "WP4")):
        selected = [row for row in rows if row["WP"] == wp]
        fields = list(dict.fromkeys(base_fields + metric_fields +
                                   [key for row in selected for key in row]))
        path = assert_analysis_path(output_dir / name)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(selected)
        outputs.append(str(path))
    summary = dict(summary, output_tables=outputs)
    summary_path = Path(summary_path) if summary_path is not None else ROOT / "reports/wp2/metrics_table_status.json"
    summary_path = assert_analysis_path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=ROOT / "runs/st1_runs.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    table_rows, status = build_rows(args.registry)
    print(json.dumps(write_tables(table_rows, status, args.output_dir), indent=2))
