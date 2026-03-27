#!/usr/bin/env python3
"""Build the summary table in results/classification/summary.md.

Scans all .json result files in the results directory, extracts metrics,
and rebuilds the summary table.

Usage:
    python scripts/classification/build_summary.py
    python scripts/classification/build_summary.py --results-dir results/classification
"""

from __future__ import annotations

import argparse
import json
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
    """Extract metrics from a single result .json file."""
    data = json.loads(path.read_text())

    cfg = data.get("config", {})
    backbone_raw = cfg.get("backbone")
    if backbone_raw is None:
        return None

    eval_mode_raw = cfg.get("eval_mode")
    if eval_mode_raw == "logreg":
        eval_mode = "logreg"
    elif cfg.get("freeze_backbone", True):
        eval_mode = "ft-frozen"
    else:
        eval_mode = "ft-full"

    five = data.get("five_way", {})
    roc = data.get("roc_auc", {})
    binary = data.get("binary", {})

    def fmt(val) -> str | None:
        return f"{val:.4f}" if val is not None else None

    return {
        "backbone": BACKBONE_DISPLAY.get(backbone_raw, backbone_raw),
        "embed_dim": str(cfg.get("embed_dim", "?")),
        "eval_mode": eval_mode,
        "pooling": cfg.get("pooling", "cls"),
        "five_acc": fmt(five.get("accuracy")),
        "five_f1": fmt(five.get("macro_f1")),
        "five_auc": fmt(roc.get("macro_auc")),
        "bin_acc": fmt(binary.get("accuracy")),
        "bin_f1": fmt(binary.get("f1")),
        "bin_sens": fmt(binary.get("sensitivity")),
        "bin_spec": fmt(binary.get("specificity")),
        "bin_auc": fmt(binary.get("roc_auc")),
    }


def build_summary(results_dir: Path) -> str:
    """Build the full summary.md content."""
    json_files = sorted(results_dir.glob("*.json"))

    rows = []
    for f in json_files:
        m = parse_result_file(f)
        if m:
            rows.append(m)

    # Sort: backbone name, then eval mode, then pooling
    mode_order = {"logreg": 0, "ft-frozen": 1, "ft-full": 2}
    rows.sort(key=lambda r: (r["backbone"], mode_order.get(r["eval_mode"], 9), r["pooling"]))

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
