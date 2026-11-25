"""Build the summary table in results/segmentation/<dataset>/summary.csv.

Scans all .json result files in each dataset subdirectory, extracts metrics,
and rebuilds the summary table: one row per run, so a config trained under
several ``--seed-training`` values has one row per seed. Two runs of the same
config with the same seeds are rejected rather than double-counted.

Results written before issue #86 have no ``seed_training``: their training was
unseeded, so that column is left blank. Their ``seed`` did set the ACDC split, so
it is reported as the data-split seed.

Usage:
    python scripts/segmentation/build_summary.py
    python scripts/segmentation/build_summary.py --results-dir results/segmentation
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

BACKBONE_DISPLAY = {
    "cinema": "CineMA",
    "dinov3": "Dino",
    "sam": "SAM",
}

DECODER_DISPLAY = {
    "linear_probe": "Linear",
    "conv_decoder": "Conv",
    "unetr": "UNETR",
}

DECODER_ORDER = {
    "Linear": 0,
    "Conv": 1,
    "UNETR": 2,
}


def parse_result_file(path: Path) -> dict | None:
    """Extract metrics from a single segmentation result .json file."""
    data = json.loads(path.read_text())

    cfg = data.get("config", {})
    backbone_raw = cfg.get("backbone")
    if backbone_raw is None:
        return None

    decoder_raw = cfg.get("decoder", "")
    probe_layers = cfg.get("probe_layers", [])
    per_class = data.get("per_class_dice", {})

    def fmt(val) -> str | None:
        return f"{val:.4f}" if val is not None else None

    def fmt_seed(val) -> str:
        return "" if val is None else str(val)

    row: dict = {
        "backbone": BACKBONE_DISPLAY.get(backbone_raw, backbone_raw),
        "model_name": cfg.get("model_name", ""),
        "embed_dim": str(cfg.get("embed_dim", "?")),
        "decoder": DECODER_DISPLAY.get(decoder_raw, decoder_raw),
        "probe_layers": ",".join(str(layer) for layer in probe_layers),
        "trainable_params": str(cfg.get("trainable_params", "")),
        # Legacy results recorded only ``seed``, which set the split, not training.
        "seed_data_split": fmt_seed(cfg.get("seed_data_split", cfg.get("seed"))),
        "seed_training": fmt_seed(cfg.get("seed_training")),
    }

    # Per-class dice scores (dynamic keys)
    for cls_name, val in per_class.items():
        row[f"dice_{cls_name}"] = fmt(val)

    row["macro_dice"] = fmt(data.get("macro_dice"))
    row["best_val_dice"] = fmt(data.get("best_val_dice"))
    row["best_epoch"] = str(data.get("best_epoch", ""))

    return row


# Fixed columns (before and after per-class dice)
COLUMNS_PREFIX = [
    ("backbone", "Backbone"),
    ("model_name", "Model"),
    ("embed_dim", "Embed Dim"),
    ("decoder", "Decoder"),
    ("probe_layers", "Probe Layers"),
    ("trainable_params", "Trainable Params"),
    ("seed_data_split", "Seed Data Split"),
    ("seed_training", "Seed Training"),
]

# Fields that identify one run; two result files sharing all of them are duplicates.
RUN_KEY = ("model_name", "decoder", "probe_layers", "seed_data_split", "seed_training")

COLUMNS_SUFFIX = [
    ("macro_dice", "Macro Dice"),
    ("best_val_dice", "Best Val Dice"),
    ("best_epoch", "Best Epoch"),
]


def build_summary(dataset_dir: Path) -> str:
    """Build summary.csv content for a single dataset directory."""
    json_files = sorted(dataset_dir.glob("*.json"))

    rows: list[dict] = []
    class_names: set[str] = set()
    seen: dict[tuple[str, ...], str] = {}
    for f in json_files:
        m = parse_result_file(f)
        if m:
            key = tuple(m[k] for k in RUN_KEY)
            if key in seen:
                raise ValueError(
                    f"Duplicate result for {'/'.join(key)} in {dataset_dir}: "
                    f"{seen[key]} and {f.name}. Delete one before rebuilding."
                )
            seen[key] = f.name
            rows.append(m)
            class_names.update(
                k.removeprefix("dice_") for k in m if k.startswith("dice_")
            )

    rows.sort(
        key=lambda r: (
            r["backbone"],
            DECODER_ORDER.get(r["decoder"], 9),
            r["model_name"],
            r["seed_data_split"],
            # numeric seed order; blank (unseeded legacy) first
            int(r["seed_training"]) if r["seed_training"] else -1,
        )
    )

    # Build dynamic per-class dice columns in a stable order
    sorted_classes = sorted(class_names)
    dice_columns = [(f"dice_{c}", f"{c} Dice") for c in sorted_classes]
    all_columns = COLUMNS_PREFIX + dice_columns + COLUMNS_SUFFIX

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([display for _, display in all_columns])
    for r in rows:
        writer.writerow([r.get(key) or "" for key, _ in all_columns])
    return buf.getvalue()


def main():
    p = argparse.ArgumentParser(description="Build segmentation summary table")
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/segmentation"),
    )
    args = p.parse_args()

    for subdir in sorted(args.results_dir.iterdir()):
        if not subdir.is_dir() or not list(subdir.glob("*.json")):
            continue
        summary = build_summary(subdir)
        out_path = subdir / "summary.csv"
        out_path.write_text(summary)
        print(f"=== {subdir.name} ===")
        print(summary)
        print(f"Written to {out_path}\n")


if __name__ == "__main__":
    main()
