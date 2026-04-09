#!/usr/bin/env python3
"""Pairwise McNemar's test between classifiers.

For each dataset, loads per-sample predictions from all JSON result files,
computes pairwise McNemar's tests (exact binomial version), and outputs:
  1. A CSV with the pairwise p-values
  2. A heatmap of -log10(p-values) highlighting significant pairs

Usage:
    python scripts/classification/mcnemar_test.py
    python scripts/classification/mcnemar_test.py --results-dir results/classification
    python scripts/classification/mcnemar_test.py --dataset acdc
    python scripts/classification/mcnemar_test.py --task binary
    python scripts/classification/mcnemar_test.py --alpha 0.01
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import binomtest


def classifier_label(config: dict, eval_mode: str) -> str:
    """Build a short human-readable label for a classifier."""
    backbone = config.get("backbone", "?")
    model = config.get("model_name", "?")
    pooling = config.get("pooling", "cls")
    # Shorten model name: drop backbone prefix if redundant
    short_model = model.replace("cinema_", "").replace("dinov3_", "").replace("sam_", "")
    return f"{backbone}_{short_model}_{eval_mode}_{pooling}"


def load_classifiers(dataset_dir: Path, task: str) -> dict[str, dict]:
    """Load predictions and true labels from all JSON files in a dataset dir.

    Returns dict mapping classifier_label -> {
        "predictions": np.array,
        "true_labels": np.array,
    }
    """
    classifiers = {}
    for f in sorted(dataset_dir.glob("*.json")):
        data = json.loads(f.read_text())
        cfg = data.get("config", {})

        eval_mode_raw = cfg.get("eval_mode")
        if eval_mode_raw == "logreg":
            eval_mode = "logreg"
        elif cfg.get("freeze_backbone", True):
            eval_mode = "ft-frozen"
        else:
            eval_mode = "ft-full"

        label = classifier_label(cfg, eval_mode)

        if task == "five_way":
            section = data.get("five_way", {})
            preds = section.get("predictions")
            true_labels = section.get("true_labels")
        elif task == "binary":
            section = data.get("binary", {})
            true_labels = section.get("binary_labels")
            # Binary predictions from thresholding probs at 0.5
            probs = section.get("binary_probs")
            if probs is not None and true_labels is not None:
                preds = [int(p >= 0.5) for p in probs]
            else:
                preds = None
        else:
            raise ValueError(f"Unknown task: {task}")

        if preds is not None and true_labels is not None:
            classifiers[label] = {
                "predictions": np.array(preds),
                "true_labels": np.array(true_labels),
            }

    return classifiers


def mcnemar_pairwise(
    classifiers: dict[str, dict],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute pairwise McNemar's test (exact) between all classifiers.

    Returns (p_value_matrix, contingency_info) as DataFrames.
    contingency_info has string entries like "b=5,c=12" for the discordant cells.
    """
    names = list(classifiers.keys())
    n = len(names)
    pvals = pd.DataFrame(np.ones((n, n)), index=names, columns=names)
    info = pd.DataFrame("", index=names, columns=names)

    for i in range(n):
        ci = classifiers[names[i]]
        correct_i = ci["predictions"] == ci["true_labels"]
        for j in range(i + 1, n):
            cj = classifiers[names[j]]
            correct_j = cj["predictions"] == cj["true_labels"]

            # Discordant pairs
            b = int(np.sum(correct_i & ~correct_j))  # only i correct
            c = int(np.sum(~correct_i & correct_j))  # only j correct

            if b + c == 0:
                p = 1.0
            else:
                # Exact McNemar: two-sided binomial test on b vs c
                result = binomtest(b, b + c, 0.5, alternative="two-sided")
                p = result.pvalue

            pvals.iloc[i, j] = p
            pvals.iloc[j, i] = p
            info.iloc[i, j] = f"b={b},c={c}"
            info.iloc[j, i] = f"b={c},c={b}"

    return pvals, info


def plot_heatmap(
    pvals: pd.DataFrame,
    classifiers: dict[str, dict],
    dataset: str,
    task: str,
    alpha: float,
    output_path: Path,
) -> None:
    """Plot a heatmap of -log10(p-values), sorted by accuracy descending."""
    # Compute macro F1 for each classifier
    from sklearn.metrics import f1_score

    f1s = {
        name: float(f1_score(c["true_labels"], c["predictions"], average="macro"))
        for name, c in classifiers.items()
    }

    # Sort by F1 descending
    sorted_names = sorted(pvals.index, key=lambda n: f1s[n], reverse=True)

    # Build display labels: "name (F1=0.64)"
    display_labels = [f"{n}  (F1={f1s[n]:.2f})" for n in sorted_names]

    # Reorder the matrix
    pvals_sorted = pvals.loc[sorted_names, sorted_names]
    log_pvals = -np.log10(pvals_sorted.values.astype(float))
    np.fill_diagonal(log_pvals, np.nan)

    n = len(pvals)
    fig, ax = plt.subplots(figsize=(max(12, n * 0.55), max(10, n * 0.5)))

    mask = np.eye(n, dtype=bool)
    sns.heatmap(
        log_pvals,
        xticklabels=display_labels,
        yticklabels=display_labels,
        mask=mask,
        cmap="YlOrRd",
        vmin=0,
        vmax=max(3, np.nanmax(log_pvals) * 1.1),
        annot=True,
        fmt=".1f",
        annot_kws={"fontsize": 6},
        ax=ax,
        cbar_kws={"label": "$-\\log_{10}(p)$"},
    )

    # Mark significance threshold
    threshold = -np.log10(alpha)
    ax.set_title(
        f"McNemar's test: {dataset} ({task}) — sorted by macro F1 (best top-left)\n"
        f"Values are $-\\log_{{10}}(p)$;  significant if > {threshold:.2f}  (α={alpha})",
        fontsize=11,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=7)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Heatmap saved to {output_path}")


def summarise_significant(
    pvals: pd.DataFrame, info: pd.DataFrame, classifiers: dict, alpha: float
) -> pd.DataFrame:
    """Extract significant pairs with accuracy context."""
    rows = []
    names = list(pvals.index)
    accs = {
        name: float(np.mean(c["predictions"] == c["true_labels"]))
        for name, c in classifiers.items()
    }
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            p = pvals.iloc[i, j]
            if p < alpha:
                rows.append({
                    "classifier_a": names[i],
                    "classifier_b": names[j],
                    "acc_a": f"{accs[names[i]]:.4f}",
                    "acc_b": f"{accs[names[j]]:.4f}",
                    "discordant": info.iloc[i, j],
                    "p_value": f"{p:.6f}",
                })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description="Pairwise McNemar's test")
    p.add_argument("--results-dir", type=Path, default=Path("results/classification"))
    p.add_argument("--dataset", type=str, default=None, help="Run for a single dataset")
    p.add_argument(
        "--task",
        choices=["five_way", "binary"],
        default="five_way",
        help="Which classification task to compare",
    )
    p.add_argument("--alpha", type=float, default=0.05, help="Significance threshold")
    args = p.parse_args()

    if args.dataset:
        dataset_dirs = [args.results_dir / args.dataset]
    else:
        dataset_dirs = sorted(
            d for d in args.results_dir.iterdir()
            if d.is_dir() and list(d.glob("*.json"))
        )

    for dataset_dir in dataset_dirs:
        dataset = dataset_dir.name
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset} | Task: {args.task} | α = {args.alpha}")
        print(f"{'='*60}")

        classifiers = load_classifiers(dataset_dir, args.task)
        if len(classifiers) < 2:
            print(f"  Skipping: only {len(classifiers)} classifier(s) found")
            continue

        print(f"  Loaded {len(classifiers)} classifiers, "
              f"{len(next(iter(classifiers.values()))['true_labels'])} samples")

        pvals, info = mcnemar_pairwise(classifiers)

        # Save p-value matrix
        csv_path = dataset_dir / f"mcnemar_{args.task}.csv"
        pvals.to_csv(csv_path)
        print(f"  P-value matrix saved to {csv_path}")

        # Plot heatmap
        fig_path = dataset_dir / f"mcnemar_{args.task}.png"
        plot_heatmap(pvals, classifiers, dataset, args.task, args.alpha, fig_path)

        # Report significant pairs
        sig = summarise_significant(pvals, info, classifiers, args.alpha)
        if len(sig) > 0:
            print(f"\n  Significant pairs (p < {args.alpha}):")
            print(sig.to_string(index=False))
        else:
            print(f"\n  No significant pairs found at α = {args.alpha}")

        # Summary stats
        n_pairs = len(classifiers) * (len(classifiers) - 1) // 2
        n_sig = len(sig)
        print(f"\n  {n_sig}/{n_pairs} pairs significant at α = {args.alpha}")


if __name__ == "__main__":
    main()
