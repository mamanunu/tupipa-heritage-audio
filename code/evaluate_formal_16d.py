#!/usr/bin/env python3
"""Evaluate the formal 16-dimensional Tupipa acoustic representation.

The script reads the per-file feature table, fits playing-intensity centers and
fixed scales using session 1 only, and evaluates session 2 as a held-out query
set. It writes parameters, per-query retrieval outcomes, and a results summary.

Usage:
    python evaluate_formal_16d.py FEATURE_CSV OUTPUT_DIRECTORY
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np


INTENSITIES = ("P", "MF", "F")
FEATURE_GROUPS = {
    "energy": (
        "log_peak_rms", "log_rms_onset", "log_rms_body", "log_rms_decay",
    ),
    "temporal_envelope": ("log_body_to_onset", "log_decay_to_body"),
    "f0_relative_spectral": (
        "onset_centroid_over_f0", "body_centroid_over_f0", "decay_centroid_over_f0",
    ),
    "harmonic": (
        "logit_onset_harmonic_ratio", "logit_body_harmonic_ratio",
        "logit_decay_harmonic_ratio", "body_relative_h2_db", "body_relative_h3_db",
        "body_relative_h4_db", "body_relative_h5_db",
    ),
}
SINGLE_BASELINES = {
    "Peak RMS": "log_peak_rms",
    "Body spectral centroid": "body_centroid_over_f0",
    "20 dB decay time": "decay_20db_s",
    "Body harmonic ratio": "logit_body_harmonic_ratio",
}


def read_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, object] = dict(raw)
            for key, value in raw.items():
                if key not in {"filename", "intensity", "file"}:
                    row[key] = float(value)
            row["session"] = int(row["session"])
            row["position"] = int(row["position"])
            row["take"] = int(row["take"])
            rows.append(row)
    return rows


def log_value(value: float) -> float:
    return math.log(max(value, 1e-12))


def logit(value: float) -> float:
    clipped = min(max(value, 1e-6), 1.0 - 1e-6)
    return math.log(clipped / (1.0 - clipped))


def feature_value(row: dict[str, object], name: str) -> float:
    value = lambda key: float(row[key])
    if name == "log_peak_rms":
        return log_value(value("peak_rms"))
    if name == "log_rms_onset":
        return log_value(value("rms_onset"))
    if name == "log_rms_body":
        return log_value(value("rms_body"))
    if name == "log_rms_decay":
        return log_value(value("rms_decay"))
    if name == "log_body_to_onset":
        return log_value(value("rms_body") / max(value("rms_onset"), 1e-12))
    if name == "log_decay_to_body":
        return log_value(value("rms_decay") / max(value("rms_body"), 1e-12))
    if name == "onset_centroid_over_f0":
        return value("onset_centroid_hz") / max(value("f0_hz"), 1e-12)
    if name == "body_centroid_over_f0":
        return value("body_centroid_hz") / max(value("f0_hz"), 1e-12)
    if name == "decay_centroid_over_f0":
        return value("decay_centroid_hz") / max(value("f0_hz"), 1e-12)
    if name == "logit_onset_harmonic_ratio":
        return logit(value("onset_harmonic_ratio"))
    if name == "logit_body_harmonic_ratio":
        return logit(value("body_harmonic_ratio"))
    if name == "logit_decay_harmonic_ratio":
        return logit(value("decay_harmonic_ratio"))
    if name in {
        "body_relative_h2_db", "body_relative_h3_db", "body_relative_h4_db",
        "body_relative_h5_db", "decay_20db_s",
    }:
        return value(name)
    raise KeyError(name)


def feature_matrix(rows: list[dict[str, object]], names: tuple[str, ...]) -> np.ndarray:
    return np.asarray(
        [[feature_value(row, name) for name in names] for row in rows], dtype=np.float64
    )


def robust_scale(reference: np.ndarray) -> np.ndarray:
    """Return the bounded scale specified for the reported analysis."""
    median = np.median(reference, axis=0)
    mad_scale = np.median(np.abs(reference - median), axis=0) * 1.4826
    standard_deviation = np.std(reference, axis=0, ddof=1)
    iqr = np.percentile(reference, 75, axis=0) - np.percentile(reference, 25, axis=0)
    return np.maximum.reduce(
        (mad_scale, 0.25 * standard_deviation, 0.10 * iqr, np.full_like(median, 1e-8))
    )


def transform_multi(
    rows: list[dict[str, object]], centered_groups: frozenset[str]
) -> tuple[np.ndarray, dict[str, object]]:
    sessions = np.asarray([int(row["session"]) for row in rows])
    intensities = np.asarray([str(row["intensity"]) for row in rows])
    reference = sessions == 1
    parts: list[np.ndarray] = []
    details: dict[str, object] = {}
    for group, names in FEATURE_GROUPS.items():
        raw = feature_matrix(rows, names)
        transformed = raw.copy()
        centers: dict[str, list[float]] | None = None
        if group in centered_groups:
            centers = {}
            for intensity in INTENSITIES:
                center = np.median(raw[reference & (intensities == intensity)], axis=0)
                transformed[intensities == intensity] -= center
                centers[intensity] = [float(item) for item in center]
        location = np.median(transformed[reference], axis=0)
        scale = robust_scale(raw[reference])
        standardized = (transformed - location) / scale
        parts.append(standardized / math.sqrt(len(names)))
        details[group] = {
            "features": list(names),
            "condition_centers_session1": centers,
            "reference_location": [float(item) for item in location],
            "reference_scale": [float(item) for item in scale],
            "group_weight_divisor": math.sqrt(len(names)),
        }
    return np.hstack(parts), details


def transform_single(rows: list[dict[str, object]], name: str) -> np.ndarray:
    values = feature_matrix(rows, (name,))
    reference = np.asarray([int(row["session"]) == 1 for row in rows])
    return (values - np.median(values[reference], axis=0)) / robust_scale(values[reference])


def distance_matrix(representation: np.ndarray) -> np.ndarray:
    differences = representation[:, None, :] - representation[None, :, :]
    return np.sqrt(np.sum(differences * differences, axis=2))


def retrieval_queries(
    rows: list[dict[str, object]], distances: np.ndarray, cross_intensity: bool
) -> list[dict[str, object]]:
    sessions = np.asarray([int(row["session"]) for row in rows])
    positions = np.asarray([int(row["position"]) for row in rows])
    intensities = np.asarray([str(row["intensity"]) for row in rows])
    references = np.where(sessions == 1)[0]
    queries: list[dict[str, object]] = []
    for query in np.where(sessions == 2)[0]:
        candidates = references
        if cross_intensity:
            candidates = candidates[intensities[candidates] != intensities[query]]
        nearest = candidates[int(np.argmin(distances[query, candidates]))]
        queries.append({
            "query_filename": str(rows[query]["filename"]),
            "query_position": int(positions[query]),
            "query_intensity": str(intensities[query]),
            "eligible_reference_count": int(len(candidates)),
            "nearest_reference_filename": str(rows[nearest]["filename"]),
            "nearest_reference_position": int(positions[nearest]),
            "reference_intensity": str(intensities[nearest]),
            "nearest_distance": float(distances[query, nearest]),
            "correct": int(positions[nearest] == positions[query]),
        })
    return queries


def evaluate(rows: list[dict[str, object]], representation: np.ndarray) -> dict[str, object]:
    distances = distance_matrix(representation)
    positions = np.asarray([int(row["position"]) for row in rows])
    intensities = np.asarray([str(row["intensity"]) for row in rows])
    within, between = [], []
    for left, right in combinations(range(len(rows)), 2):
        if positions[left] == positions[right] and intensities[left] != intensities[right]:
            within.append(float(distances[left, right]))
        if positions[left] != positions[right] and intensities[left] == intensities[right]:
            between.append(float(distances[left, right]))
    same_label = retrieval_queries(rows, distances, cross_intensity=False)
    direct = retrieval_queries(rows, distances, cross_intensity=True)
    within_mean = float(np.mean(within))
    between_mean = float(np.mean(between))
    return {
        "n_samples": len(rows),
        "n_within_position_cross_intensity_pairs": len(within),
        "n_matched_intensity_between_position_pairs": len(between),
        "within_position_cross_intensity_mean_distance": within_mean,
        "matched_intensity_between_position_mean_distance": between_mean,
        "between_within_ratio": between_mean / within_mean,
        "direct_cross_intensity_queries": direct,
        "direct_cross_intensity_top1_correct": int(sum(q["correct"] for q in direct)),
        "direct_cross_intensity_top1_total": len(direct),
        "direct_cross_intensity_top1_accuracy": float(sum(q["correct"] for q in direct) / len(direct)),
        "same_intensity_allowed_queries": same_label,
        "same_intensity_allowed_top1_correct": int(sum(q["correct"] for q in same_label)),
        "same_intensity_allowed_top1_total": len(same_label),
        "same_intensity_allowed_top1_accuracy": float(sum(q["correct"] for q in same_label) / len(same_label)),
    }


def exact_mcnemar(baseline: dict[str, object], comparison: dict[str, object]) -> dict[str, object]:
    base = {str(q["query_filename"]): bool(q["correct"]) for q in baseline["direct_cross_intensity_queries"]}
    tested = {str(q["query_filename"]): bool(q["correct"]) for q in comparison["direct_cross_intensity_queries"]}
    gains = sum(not base[name] and tested[name] for name in base)
    losses = sum(base[name] and not tested[name] for name in base)
    discordant = gains + losses
    tail = sum(math.comb(discordant, k) for k in range(min(gains, losses) + 1))
    return {
        "uncentered_incorrect_centered_correct": gains,
        "uncentered_correct_centered_incorrect": losses,
        "discordant_queries": discordant,
        "exact_two_sided_p": min(1.0, 2.0 * tail / (2**discordant)) if discordant else 1.0,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the formal Tupipa 16D method.")
    parser.add_argument("feature_csv", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    all_rows = read_rows(args.feature_csv)

    f0_values = np.asarray([float(row["f0_hz"]) for row in all_rows])
    body_rms_values = np.asarray([float(row["rms_body"]) for row in all_rows])
    median_f0 = float(np.median(f0_values))
    median_body_rms = float(np.median(body_rms_values))
    excluded = [
        row for row in all_rows
        if abs(float(row["f0_hz"]) - median_f0) > 1.0
        and float(row["rms_body"]) < 0.10 * median_body_rms
    ]
    rows = [row for row in all_rows if row not in excluded]

    uncentered_rep, _ = transform_multi(rows, frozenset())
    energy_rep, _ = transform_multi(rows, frozenset({"energy"}))
    formal_rep, formal_details = transform_multi(rows, frozenset(FEATURE_GROUPS))
    results = {
        "Peak RMS": evaluate(rows, transform_single(rows, SINGLE_BASELINES["Peak RMS"])),
        "Body spectral centroid": evaluate(rows, transform_single(rows, SINGLE_BASELINES["Body spectral centroid"])),
        "20 dB decay time": evaluate(rows, transform_single(rows, SINGLE_BASELINES["20 dB decay time"])),
        "Body harmonic ratio": evaluate(rows, transform_single(rows, SINGLE_BASELINES["Body harmonic ratio"])),
        "Uncentered 16-feature": evaluate(rows, uncentered_rep),
        "Energy-only centering": evaluate(rows, energy_rep),
        "All-group 16-feature centering": evaluate(rows, formal_rep),
    }
    formal, uncentered = results["All-group 16-feature centering"], results["Uncentered 16-feature"]
    parameters = {
        "method": "playing-intensity-normalized 16-dimensional multi-feature acoustic representation",
        "feature_groups": formal_details,
        "quality_control": {
            "rule": "absolute F0 deviation > 1.0 Hz and body RMS < 10% of corpus median body RMS",
            "corpus_median_f0_hz": median_f0,
            "corpus_median_body_rms": median_body_rms,
            "excluded_filenames": [str(row["filename"]) for row in excluded],
        },
        "evaluation_design": {
            "reference_session": 1,
            "query_session": 2,
            "direct_cross_intensity_retrieval": "Session-1 references with the query intensity label are excluded.",
            "same_intensity_allowed_retrieval": "All session-1 references are available.",
        },
        "uncentered_vs_formal_direct_cross_intensity_mcnemar": exact_mcnemar(uncentered, formal),
    }
    (args.output_directory / "formal_16d_parameters.json").write_text(
        json.dumps(parameters, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(args.output_directory / "formal_16d_direct_cross_intensity_retrieval.csv", formal["direct_cross_intensity_queries"])
    write_csv(args.output_directory / "formal_16d_same_intensity_allowed_retrieval.csv", formal["same_intensity_allowed_queries"])
    summary_rows = []
    for name, result in results.items():
        summary_rows.append({
            "representation": name,
            "n_samples": result["n_samples"],
            "n_within_position_cross_intensity_pairs": result["n_within_position_cross_intensity_pairs"],
            "n_matched_intensity_between_position_pairs": result["n_matched_intensity_between_position_pairs"],
            "within_position_cross_intensity_mean_distance": result["within_position_cross_intensity_mean_distance"],
            "matched_intensity_between_position_mean_distance": result["matched_intensity_between_position_mean_distance"],
            "between_within_ratio": result["between_within_ratio"],
            "direct_cross_intensity_top1_correct": result["direct_cross_intensity_top1_correct"],
            "direct_cross_intensity_top1_total": result["direct_cross_intensity_top1_total"],
            "direct_cross_intensity_top1_accuracy": result["direct_cross_intensity_top1_accuracy"],
            "same_intensity_allowed_top1_correct": result["same_intensity_allowed_top1_correct"],
            "same_intensity_allowed_top1_total": result["same_intensity_allowed_top1_total"],
            "same_intensity_allowed_top1_accuracy": result["same_intensity_allowed_top1_accuracy"],
        })
    write_csv(args.output_directory / "formal_16d_results_summary.csv", summary_rows)
    print(
        "Formal 16D direct cross-intensity retrieval: "
        f"{formal['direct_cross_intensity_top1_correct']}/{formal['direct_cross_intensity_top1_total']} "
        f"({formal['direct_cross_intensity_top1_accuracy']:.3f})"
    )


if __name__ == "__main__":
    main()
