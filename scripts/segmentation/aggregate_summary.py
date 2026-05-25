"""Aggregate per-dataset segmentation summaries into one cross-dataset summary.

Reads results/segmentation/{acdc,mnm,mnm2}/summary.csv, averages Macro Dice
for each (Backbone, Model, Decoder) combination across the three datasets,
writes results/segmentation/summary_aggregated.csv, and saves a grouped bar
plot to results/segmentation/macro_dice_by_backbone.png.

Usage:
    python scripts/segmentation/aggregate_summary.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DATASETS = ["acdc", "mnm", "mnm2"]
DECODER_ORDER = ["Linear", "Conv", "UNETR"]
BACKBONE_ORDER = ["CineMA", "Dino", "SAM2"]


def load_dataset(results_dir: Path, dataset: str) -> pd.DataFrame:
    df = pd.read_csv(results_dir / dataset / "summary.csv")
    df["Dataset"] = dataset
    return df[["Backbone", "Model", "Decoder", "Macro Dice", "Dataset"]]


def aggregate(results_dir: Path) -> pd.DataFrame:
    frames = [load_dataset(results_dir, d) for d in DATASETS]
    long_df = pd.concat(frames, ignore_index=True)

    wide = long_df.pivot_table(
        index=["Backbone", "Model", "Decoder"],
        columns="Dataset",
        values="Macro Dice",
    ).reset_index()

    wide["Mean Macro Dice"] = wide[DATASETS].mean(axis=1)

    wide["_backbone_order"] = wide["Backbone"].map(
        {b: i for i, b in enumerate(BACKBONE_ORDER)}
    )
    wide["_decoder_order"] = wide["Decoder"].map(
        {d: i for i, d in enumerate(DECODER_ORDER)}
    )
    wide = wide.sort_values(
        ["_backbone_order", "_decoder_order", "Model"]
    ).drop(columns=["_backbone_order", "_decoder_order"])

    column_order = (
        ["Backbone", "Model", "Decoder"] + DATASETS + ["Mean Macro Dice"]
    )
    return wide[column_order].reset_index(drop=True)


DECODER_COLORS = {
    "Linear": "#4C72B0",
    "Conv": "#DD8452",
    "UNETR": "#55A467",
}

BACKBONE_COLORS = {
    "CineMA": "#4C72B0",
    "Dino": "#DD8452",
    "SAM2": "#55A467",
}


def _grouped_bar_plot(
    agg: pd.DataFrame,
    group_col: str,
    color_col: str,
    group_order: list[str],
    color_map: dict[str, str],
    title: str,
    out_path: Path,
) -> None:
    """Render a grouped bar plot with one group per `group_col` value.

    Bars within a group are colored by `color_col` and labeled with Model + the
    other categorical (Decoder or Backbone).
    """
    groups = [g for g in group_order if g in agg[group_col].unique()]

    fig, ax = plt.subplots(figsize=(14, 6))

    bar_width = 0.8
    gap_between_groups = 1.5
    cursor = 0.0
    group_centers: list[float] = []
    seen: set[str] = set()
    other_col = "Decoder" if group_col == "Backbone" else "Backbone"

    for group in groups:
        sub = agg[agg[group_col] == group].reset_index(drop=True)
        n = len(sub)
        positions = [cursor + i * bar_width for i in range(n)]
        group_centers.append(sum(positions) / n)

        for pos, (_, row) in zip(positions, sub.iterrows()):
            color_key = row[color_col]
            label = color_key if color_key not in seen else None
            seen.add(color_key)
            ax.bar(
                pos,
                row["Mean Macro Dice"],
                width=bar_width,
                color=color_map.get(color_key, "#888"),
                edgecolor="black",
                linewidth=0.5,
                label=label,
            )
            ax.text(
                pos,
                row["Mean Macro Dice"] + 0.005,
                f"{row['Model']}\n{row[other_col]}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )

        cursor += n * bar_width + gap_between_groups

    ax.set_xticks(group_centers)
    ax.set_xticklabels(groups, fontsize=12, fontweight="bold")
    ax.set_ylabel("Mean Macro Dice (avg over ACDC, M&Ms, M&Ms-2)")
    ax.set_title(title)
    ax.set_ylim(0.6, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(title=color_col, loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_by_backbone(agg: pd.DataFrame, out_path: Path) -> None:
    _grouped_bar_plot(
        agg,
        group_col="Backbone",
        color_col="Decoder",
        group_order=BACKBONE_ORDER,
        color_map=DECODER_COLORS,
        title="Segmentation Macro Dice — grouped by Backbone",
        out_path=out_path,
    )


def plot_by_decoder(agg: pd.DataFrame, out_path: Path) -> None:
    sorted_agg = agg.copy()
    sorted_agg["_decoder_order"] = sorted_agg["Decoder"].map(
        {d: i for i, d in enumerate(DECODER_ORDER)}
    )
    sorted_agg["_backbone_order"] = sorted_agg["Backbone"].map(
        {b: i for i, b in enumerate(BACKBONE_ORDER)}
    )
    sorted_agg = sorted_agg.sort_values(
        ["_decoder_order", "_backbone_order", "Model"]
    ).drop(columns=["_decoder_order", "_backbone_order"])

    _grouped_bar_plot(
        sorted_agg,
        group_col="Decoder",
        color_col="Backbone",
        group_order=DECODER_ORDER,
        color_map=BACKBONE_COLORS,
        title="Segmentation Macro Dice — grouped by Decoder",
        out_path=out_path,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/segmentation"),
    )
    args = p.parse_args()

    agg = aggregate(args.results_dir)

    csv_path = args.results_dir / "summary_aggregated.csv"
    agg.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"Wrote {csv_path}")
    print(agg.to_string(index=False))

    by_backbone_path = args.results_dir / "macro_dice_by_backbone.png"
    plot_by_backbone(agg, by_backbone_path)
    print(f"Wrote {by_backbone_path}")

    by_decoder_path = args.results_dir / "macro_dice_by_decoder.png"
    plot_by_decoder(agg, by_decoder_path)
    print(f"Wrote {by_decoder_path}")


if __name__ == "__main__":
    main()
