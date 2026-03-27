#!/usr/bin/env python3
"""Build the summary table in results/classification/summary.md.

Scans all .md result files (excluding summary.md itself) in the results
directory, extracts metrics, and rebuilds the summary table.

Usage:
    python scripts/classification/build_summary.py
    python scripts/classification/build_summary.py --results-dir results/classification
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

BACKBONE_DISPLAY = {
    "cinema": "CineMA",
    "dinov3": "Dino",
    "sam": "SAM",
}

EVAL_MODE_DISPLAY = {
    "logreg": "logreg",
    "ftfrozen": "ft-frozen",
    "ftfull": "ft-full",
}


def parse_result_file(path: Path) -> dict | None:
    """Extract metrics from a single result .md file."""
    text = path.read_text()

    def find(pattern: str) -> str | None:
        m = re.search(pattern, text)
        return m.group(1) if m else None

    backbone_raw = find(r"Backbone:\s*(\w+)")
    if backbone_raw is None:
        return None

    embed_dim = find(r"embed_dim:\s*(\d+)")

    # Determine eval mode from filename stem (e.g. cinema_logreg, dinov3_ftfrozen)
    stem = path.stem
    for tag, display in EVAL_MODE_DISPLAY.items():
        if stem.endswith(f"_{tag}"):
            eval_mode = display
            break
    else:
        # Fallback: parse from text
        if "Logistic Regression" in text:
            eval_mode = "logreg"
        elif "frozen backbone" in text.lower():
            eval_mode = "ft-frozen"
        else:
            eval_mode = "ft-full"

    # Split into 5-way and binary sections
    binary_section = ""
    if "Binary Disease Detection:" in text:
        binary_section = text[text.index("Binary Disease Detection:"):]

    def find_in(section: str, pattern: str) -> str | None:
        m = re.search(pattern, section)
        return m.group(1) if m else None

    pooling = find(r"Pooling:\s*(cls|gap)")
    if pooling is None:
        # Old format: "Pooling: ED-mean + ES-mean → ..." didn't record cls/gap
        pooling = "cls"

    metrics = {
        "backbone": BACKBONE_DISPLAY.get(backbone_raw, backbone_raw),
        "embed_dim": embed_dim or "?",
        "eval_mode": eval_mode,
        "pooling": pooling or "cls",
        "five_acc": find(r"Test Accuracy:\s*([\d.]+)"),
        "five_f1": find(r"Test Macro F1:\s*([\d.]+)"),
        "five_auc": find(r"Macro ROC AUC:\s*([\d.]+)"),
        "bin_acc": find_in(binary_section, r"Accuracy:\s*([\d.]+)"),
        "bin_f1": find_in(binary_section, r"F1 Score:\s*([\d.]+)"),
        "bin_sens": find_in(binary_section, r"Sensitivity:\s*([\d.]+)"),
        "bin_spec": find_in(binary_section, r"Specificity:\s*([\d.]+)"),
        "bin_auc": find_in(binary_section, r"ROC AUC:\s*([\d.]+)"),
    }

    return metrics


def build_summary(results_dir: Path) -> str:
    """Build the full summary.md content."""
    md_files = sorted(
        p for p in results_dir.glob("*.md") if p.name != "summary.md"
    )

    rows = []
    for f in md_files:
        m = parse_result_file(f)
        if m:
            rows.append(m)

    # Sort: backbone name, then eval mode
    mode_order = {"logreg": 0, "ft-frozen": 1, "ft-full": 2}
    rows.sort(key=lambda r: (r["backbone"], mode_order.get(r["eval_mode"], 9)))

    header = "| Backbone | Embed Dim | Eval Mode | Pooling | 5-way Acc | 5-way F1 | 5-way AUC | Binary Acc | Binary F1 | Binary Sens | Binary Spec | Binary AUC |"
    sep = "| -------- | --------- | --------- | ------- | --------- | -------- | --------- | ---------- | --------- | ----------- | ----------- | ---------- |"

    table_rows = []
    for r in rows:
        def v(key, width):
            val = r.get(key) or "—"
            return f"{val:<{width}s}"

        table_rows.append(
            f"| {v('backbone', 8)} | {v('embed_dim', 9)} | {v('eval_mode', 9)} "
            f"| {v('pooling', 7)} "
            f"| {v('five_acc', 9)} | {v('five_f1', 8)} | {v('five_auc', 9)} "
            f"| {v('bin_acc', 10)} | {v('bin_f1', 9)} | {v('bin_sens', 11)} "
            f"| {v('bin_spec', 11)} | {v('bin_auc', 10)} |"
        )

    lines = [
        "# ACDC Pathology Classification — Summary",
        "",
        "5-way patient-level classification (100 train / 50 test) + binary disease detection (NOR vs disease).",
        "Per-run details (per-class metrics, confusion matrices, plots) are in the individual result files.",
        "",
        header,
        sep,
        *table_rows,
    ]
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description="Build classification summary table")
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/classification"),
    )
    args = p.parse_args()

    summary = build_summary(args.results_dir)
    out_path = args.results_dir / "summary.md"
    out_path.write_text(summary)
    print(summary)
    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
