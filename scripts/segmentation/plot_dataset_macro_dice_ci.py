"""Per-dataset grouped bar plots of macro Dice with bootstrap CI error bars.

Reads results/segmentation/<dataset>/macro_dice_ci_summary.csv (produced by
``report_macro_dice_ci.py``) and writes, for each dataset:
    macro_dice_ci_by_backbone.png
    macro_dice_ci_by_decoder.png

Plot style matches scripts/segmentation/aggregate_summary.py, except bars use
``mean_macro_dice`` from the CI summary and error bars span [ci_lower, ci_upper].

Usage:
    python scripts/segmentation/plot_dataset_macro_dice_ci.py
    python scripts/segmentation/plot_dataset_macro_dice_ci.py --datasets acdc
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DATASETS = ["acdc", "mnm", "mnm2"]
DECODER_ORDER = ["Linear", "Conv", "UNETR"]
BACKBONE_ORDER = ["CineMA", "Dino", "SAM"]

DATASET_DISPLAY = {
    "acdc": "ACDC",
    "mnm": "M&Ms",
    "mnm2": "M&Ms-2",
}

DECODER_COLORS = {
    "Linear": "#4C72B0",
    "Conv": "#DD8452",
    "UNETR": "#55A467",
}

BACKBONE_COLORS = {
    "CineMA": "#4C72B0",
    "Dino": "#DD8452",
    "SAM": "#937860",
}


def load_ci_summary(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.rename(
        columns={
            "backbone": "Backbone",
            "model_name": "Model",
            "decoder": "Decoder",
        }
    )
    missing = df[["Backbone", "Decoder", "Model"]].isna().any(axis=1) | (
        df[["Backbone", "Decoder", "Model"]].eq("").any(axis=1)
    )
    if missing.any():
        dropped = df.loc[missing, "result_file"].tolist()
        print(
            f"  skipping {missing.sum()} row(s) without backbone/decoder metadata: "
            f"{dropped}"
        )
        df = df.loc[~missing].reset_index(drop=True)

    df["_backbone_order"] = df["Backbone"].map(
        {b: i for i, b in enumerate(BACKBONE_ORDER)}
    )
    df["_decoder_order"] = df["Decoder"].map(
        {d: i for i, d in enumerate(DECODER_ORDER)}
    )
    df = df.sort_values(["_backbone_order", "_decoder_order", "Model"]).drop(
        columns=["_backbone_order", "_decoder_order"]
    )
    return df.reset_index(drop=True)


def _grouped_bar_plot_with_ci(
    df: pd.DataFrame,
    group_col: str,
    color_col: str,
    group_order: list[str],
    color_map: dict[str, str],
    title: str,
    y_label: str,
    out_path: Path,
) -> None:
    """Grouped bars colored by `color_col`, with asymmetric CI error bars."""
    present = list(df[group_col].unique())
    groups = [g for g in group_order if g in present] + [
        g for g in present if g not in group_order
    ]

    fallback_palette = ["#8172B2", "#937860", "#DA8BC3", "#8C8C8C", "#CCB974"]
    fallback_idx = 0
    color_map = dict(color_map)
    for key in df[color_col].unique():
        if key not in color_map:
            color_map[key] = fallback_palette[fallback_idx % len(fallback_palette)]
            fallback_idx += 1

    fig, ax = plt.subplots(figsize=(6, 6))

    bar_width = 0.8
    gap_between_groups = 1.5
    cursor = 0.0
    group_centers: list[float] = []
    seen: set[str] = set()
    other_col = "Decoder" if group_col == "Backbone" else "Backbone"

    for group in groups:
        sub = df[df[group_col] == group].reset_index(drop=True)
        n = len(sub)
        positions = [cursor + i * bar_width for i in range(n)]
        group_centers.append(sum(positions) / n)

        for pos, (_, row) in zip(positions, sub.iterrows(), strict=True):
            color_key = row[color_col]
            label = color_key if color_key not in seen else None
            seen.add(color_key)
            mean = row["mean_macro_dice"]
            err_low = mean - row["ci_lower"]
            err_high = row["ci_upper"] - mean
            ax.bar(
                pos,
                mean,
                width=bar_width,
                color=color_map.get(color_key, "#888"),
                edgecolor="black",
                linewidth=0.5,
                label=label,
                yerr=[[err_low], [err_high]],
                capsize=3,
                error_kw={"elinewidth": 1.0, "ecolor": "black"},
            )
            ax.text(
                pos,
                row["ci_upper"] + 0.005,
                f"{row['Model']}\n{row[other_col]}",
                ha="center",
                va="bottom",
                fontsize=10,
                rotation=90,
            )

        cursor += n * bar_width + gap_between_groups

    ax.set_xticks(group_centers)
    ax.set_xticklabels(groups, fontsize=12, fontweight="bold")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_ylim(0.6, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(title=color_col, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_dataset(df: pd.DataFrame, dataset: str, out_dir: Path) -> None:
    display = DATASET_DISPLAY.get(dataset, dataset)
    confidence = (
        df["confidence_level"].iloc[0] if "confidence_level" in df.columns else 0.95
    )
    pct = int(round(confidence * 100))
    y_label = f"Mean Macro Dice ({pct}% bootstrap CI)"

    _grouped_bar_plot_with_ci(
        df,
        group_col="Backbone",
        color_col="Decoder",
        group_order=BACKBONE_ORDER,
        color_map=DECODER_COLORS,
        title=f"Segmentation Macro Dice — {display} — grouped by Backbone",
        y_label=y_label,
        out_path=out_dir / "macro_dice_ci_by_backbone.png",
    )

    _grouped_bar_plot_with_ci(
        df,
        group_col="Decoder",
        color_col="Backbone",
        group_order=DECODER_ORDER,
        color_map=BACKBONE_COLORS,
        title=f"Segmentation Macro Dice — {display} — grouped by Decoder",
        y_label=y_label,
        out_path=out_dir / "macro_dice_ci_by_decoder.png",
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/segmentation"),
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=DATASETS,
        help="Dataset subdirectories to process.",
    )
    args = p.parse_args()

    for dataset in args.datasets:
        ci_path = args.results_dir / dataset / "macro_dice_ci_summary.csv"
        if not ci_path.exists():
            print(f"[{dataset}] skipping — {ci_path} not found")
            continue

        print(f"[{dataset}] reading {ci_path}")
        df = load_ci_summary(ci_path)
        if df.empty:
            print(f"[{dataset}] no usable rows after filtering — skipping")
            continue

        plot_dataset(df, dataset, args.results_dir / dataset)
        print(
            f"[{dataset}] wrote macro_dice_ci_by_backbone.png and "
            f"macro_dice_ci_by_decoder.png"
        )


if __name__ == "__main__":
    main()
