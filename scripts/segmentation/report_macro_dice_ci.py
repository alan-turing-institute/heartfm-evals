#!/usr/bin/env python3
"""Report mean macro Dice and confidence intervals from segmentation CSV files.

This script scans segmentation result CSV files, computes the average
``macro_dice`` for each file, and estimates a confidence interval using
bootstrap resampling.

The averaging can be performed either:
- at the sample level, using every non-empty ``macro_dice`` row directly
- at the patient level, first averaging rows within each ``pid``

By default it reads ``*_per_sample.csv`` files from a dataset directory and
writes ``macro_dice_ci_summary.csv`` alongside them.

Usage:
    python scripts/segmentation/report_macro_dice_ci.py
    python scripts/segmentation/report_macro_dice_ci.py --dataset-dir results/segmentation/acdc
    python scripts/segmentation/report_macro_dice_ci.py --pattern '*.csv' --output acdc_macro_dice_ci.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

BACKBONE_DISPLAY = {
    "cinema": "CineMA",
    "dinov3": "Dino",
    "sam": "SAM",
    "sam2": "SAM2",
}

DECODER_DISPLAY = {
    "linear_probe": "Linear",
    "conv_decoder": "Conv",
    "unetr": "UNETR",
}


def infer_aggregation_unit(columns: list[str]) -> str:
    """Infer the row aggregation unit from CSV columns."""
    column_set = set(columns)
    if "n_slices" in column_set:
        return "sample"
    if "z_idx" in column_set:
        return "slice"
    if "stack_idx" in column_set:
        return "stack"
    return "unknown"


def percentile_bootstrap_ci(
    values: np.ndarray,
    confidence_level: float = 0.95,
    n_bootstrap: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    """Estimate a percentile bootstrap CI for the sample mean."""
    if values.size == 0:
        raise ValueError("Cannot compute confidence interval for an empty sample.")

    if values.size == 1:
        only_value = float(values[0])
        return only_value, only_value

    rng = np.random.default_rng(seed)
    sample_idx = rng.integers(0, values.size, size=(n_bootstrap, values.size))
    bootstrap_means = values[sample_idx].mean(axis=1)

    alpha = 1.0 - confidence_level
    lower = float(np.quantile(bootstrap_means, alpha / 2))
    upper = float(np.quantile(bootstrap_means, 1 - alpha / 2))
    return lower, upper


def load_json_metadata(csv_path: Path) -> dict[str, str]:
    """Load sidecar JSON metadata for a CSV result file when available."""
    stem = csv_path.stem
    suffixes = (
        "_per_sample",
        "_per_slice",
        "_per_stack",
    )
    base_stem = stem
    for suffix in suffixes:
        if stem.endswith(suffix):
            base_stem = stem.removesuffix(suffix)
            break

    json_path = csv_path.with_name(f"{base_stem}.json")
    if not json_path.exists():
        return {}

    data = json.loads(json_path.read_text())
    config = data.get("config", {})
    backbone_raw = config.get("backbone", "")
    decoder_raw = config.get("decoder", "")
    return {
        "backbone": BACKBONE_DISPLAY.get(backbone_raw, backbone_raw),
        "model_name": str(config.get("model_name", "")),
        "decoder": DECODER_DISPLAY.get(decoder_raw, decoder_raw),
    }


def collect_macro_dice_by_level(
    csv_path: Path,
    summary_level: str,
) -> tuple[np.ndarray, list[str]]:
    """Collect macro Dice values at the requested summary level."""
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "macro_dice" not in reader.fieldnames:
            raise ValueError(f"{csv_path} does not contain a 'macro_dice' column.")

        if summary_level == "sample":
            values = [
                float(row["macro_dice"])
                for row in reader
                if row.get("macro_dice") not in ("", None)
            ]
            return np.asarray(values, dtype=float), reader.fieldnames

        if "pid" not in reader.fieldnames:
            raise ValueError(
                f"{csv_path} does not contain a 'pid' column required for patient-level summaries."
            )

        patient_scores: dict[str, list[float]] = {}
        for row in reader:
            raw_value = row.get("macro_dice", "")
            if raw_value in ("", None):
                continue
            patient_id = row.get("pid", "")
            if not patient_id:
                continue
            patient_scores.setdefault(patient_id, []).append(float(raw_value))

    values = np.asarray(
        [float(np.mean(scores)) for scores in patient_scores.values()],
        dtype=float,
    )
    return values, reader.fieldnames


def summarize_csv(
    csv_path: Path,
    confidence_level: float,
    n_bootstrap: int,
    seed: int,
    summary_level: str,
) -> dict[str, str | int | float]:
    """Summarize macro Dice statistics for a single CSV file."""
    macro_dice_values, fieldnames = collect_macro_dice_by_level(csv_path, summary_level)
    if macro_dice_values.size == 0:
        raise ValueError(f"{csv_path} has no non-empty macro_dice values.")

    mean_macro_dice = float(macro_dice_values.mean())
    ci_lower, ci_upper = percentile_bootstrap_ci(
        macro_dice_values,
        confidence_level=confidence_level,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    metadata = load_json_metadata(csv_path)

    return {
        "result_file": csv_path.name,
        "backbone": metadata.get("backbone", ""),
        "model_name": metadata.get("model_name", ""),
        "decoder": metadata.get("decoder", ""),
        "row_unit": infer_aggregation_unit(fieldnames),
        "summary_level": summary_level,
        "n": int(macro_dice_values.size),
        "mean_macro_dice": mean_macro_dice,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "confidence_level": confidence_level,
    }


def format_summary_value(value: str | int | float) -> str:
    """Format values for CSV output."""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def build_summary(
    dataset_dir: Path,
    pattern: str,
    confidence_level: float,
    n_bootstrap: int,
    seed: int,
    summary_level: str,
) -> list[dict[str, str | int | float]]:
    """Build summary rows for matching CSV files in a dataset directory."""
    rows = [
        summarize_csv(path, confidence_level, n_bootstrap, seed, summary_level)
        for path in sorted(dataset_dir.glob(pattern))
    ]

    rows.sort(
        key=lambda row: (
            str(row["backbone"]),
            str(row["model_name"]),
            str(row["decoder"]),
            str(row["result_file"]),
        )
    )
    return rows


def write_summary(
    rows: list[dict[str, str | int | float]],
    output_path: Path,
) -> None:
    """Write summary rows to CSV."""
    fieldnames = [
        "result_file",
        "backbone",
        "model_name",
        "decoder",
        "row_unit",
        "summary_level",
        "n",
        "mean_macro_dice",
        "ci_lower",
        "ci_upper",
        "confidence_level",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_summary_value(row[key]) for key in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report mean macro Dice and confidence intervals from segmentation CSV results."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("results/segmentation/acdc"),
        help="Directory containing segmentation result CSV files.",
    )
    parser.add_argument(
        "--pattern",
        default="*_per_sample.csv",
        help="Glob pattern for selecting result CSV files.",
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
        help="Confidence level for the interval estimate.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=10_000,
        help="Number of bootstrap resamples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for bootstrap resampling.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output CSV path. Defaults to <dataset-dir>/macro_dice_ci_summary.csv.",
    )
    parser.add_argument(
        "--summary-level",
        choices=("sample", "patient"),
        default="sample",
        help=(
            "Compute the summary across rows ('sample') or across patient IDs "
            "after averaging rows within each patient ('patient')."
        ),
    )
    args = parser.parse_args()

    if not args.dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {args.dataset_dir}")

    rows = build_summary(
        dataset_dir=args.dataset_dir,
        pattern=args.pattern,
        confidence_level=args.confidence_level,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        summary_level=args.summary_level,
    )
    if not rows:
        raise FileNotFoundError(
            f"No CSV files matched pattern {args.pattern!r} in {args.dataset_dir}"
        )

    output_path = args.output or args.dataset_dir / "macro_dice_ci_summary.csv"
    write_summary(rows, output_path)

    print(f"Wrote {len(rows)} summaries to {output_path}")
    for row in rows:
        model_label = row["model_name"] or row["result_file"]
        print(
            f"{model_label} ({row['summary_level']}, row unit: {row['row_unit']}): "
            f"{row['mean_macro_dice']:.4f} "
            f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]"
        )


if __name__ == "__main__":
    main()
