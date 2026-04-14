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
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score

from _common import (
    align_and_validate_classifiers,
    load_classifiers,
    plot_pvalue_heatmap,
)


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

            b = int(np.sum(correct_i & ~correct_j))
            c = int(np.sum(~correct_i & correct_j))

            if b + c == 0:
                p = 1.0
            else:
                result = binomtest(b, b + c, 0.5, alternative="two-sided")
                p = result.pvalue

            pvals.iloc[i, j] = p
            pvals.iloc[j, i] = p
            info.iloc[i, j] = f"b={b},c={c}"
            info.iloc[j, i] = f"b={c},c={b}"

    return pvals, info


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

        classifiers = align_and_validate_classifiers(classifiers)

        print(f"  Loaded {len(classifiers)} classifiers, "
              f"{len(next(iter(classifiers.values()))['true_labels'])} samples")

        pvals, info = mcnemar_pairwise(classifiers)

        csv_path = dataset_dir / f"mcnemar_{args.task}.csv"
        pvals.to_csv(csv_path)
        print(f"  P-value matrix saved to {csv_path}")

        f1s = {
            name: float(f1_score(c["true_labels"], c["predictions"], average="macro"))
            for name, c in classifiers.items()
        }
        fig_path = dataset_dir / f"mcnemar_{args.task}.png"
        plot_pvalue_heatmap(
            pvals,
            scores=f1s,
            score_name="F1",
            title=f"McNemar's test: {dataset} ({args.task}) — sorted by macro F1 (best top-left)",
            alpha=args.alpha,
            output_path=fig_path,
        )

        sig = summarise_significant(pvals, info, classifiers, args.alpha)
        if len(sig) > 0:
            print(f"\n  Significant pairs (p < {args.alpha}):")
            print(sig.to_string(index=False))
        else:
            print(f"\n  No significant pairs found at α = {args.alpha}")

        n_pairs = len(classifiers) * (len(classifiers) - 1) // 2
        n_sig = len(sig)
        print(f"\n  {n_sig}/{n_pairs} pairs significant at α = {args.alpha}")


if __name__ == "__main__":
    main()
