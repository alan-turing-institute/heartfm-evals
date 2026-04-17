#!/usr/bin/env python3
"""Paired bootstrap test between classifiers.

For each dataset, repeatedly resamples the test set with replacement
(sampling sample indices *once per iteration* and scoring every classifier
on the same resampled subset -> paired metric vectors). For every pair of
classifiers we then compute:
  - a two-sided paired-bootstrap p-value on the metric difference
  - a 95% percentile CI on the metric difference

Outputs per dataset (next to the JSON result files):
  1. ``bootstrap_<task>_<metric>.csv`` — pairwise p-value matrix
  2. ``bootstrap_<task>_<metric>_ci.csv`` — pairwise mean diff + 95% CI
  3. ``bootstrap_<task>_<metric>.png`` — heatmap of -log10(p-values),
     sorted by the chosen metric (best top-left)

Usage:
    python scripts/classification/bootstrap_test.py
    python scripts/classification/bootstrap_test.py --dataset acdc
    python scripts/classification/bootstrap_test.py --task binary
    python scripts/classification/bootstrap_test.py --n-bootstrap 2000 --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from _common import (
    align_and_validate_classifiers,
    load_classifiers,
    plot_pvalue_heatmap,
)


METRIC_FNS = {
    "macro_f1": lambda y, p: f1_score(y, p, average="macro"),
    "accuracy": accuracy_score,
    "balanced_accuracy": balanced_accuracy_score,
}


def paired_bootstrap(
    classifiers: dict[str, dict],
    metric: str,
    n_bootstrap: int,
    seed: int,
) -> np.ndarray:
    """Return an (n_classifiers, n_bootstrap) matrix of metric values.

    Each column corresponds to a single resample: the same sample indices
    are used for every classifier, so comparisons across classifiers are
    paired within each bootstrap iteration.
    """
    metric_fn = METRIC_FNS[metric]
    names = list(classifiers.keys())
    n_samples = len(next(iter(classifiers.values()))["true_labels"])

    rng = np.random.default_rng(seed)
    scores = np.zeros((len(names), n_bootstrap), dtype=float)

    for b in range(n_bootstrap):
        idx = rng.integers(0, n_samples, size=n_samples)
        for i, name in enumerate(names):
            c = classifiers[name]
            y_true = c["true_labels"][idx]
            y_pred = c["predictions"][idx]
            scores[i, b] = float(metric_fn(y_true, y_pred))

    return scores


def pairwise_bootstrap_stats(
    names: list[str],
    boot_scores: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute pairwise p-values and 95% CIs on the metric difference.

    Two-sided paired bootstrap p-value:
        p = 2 * min(mean(diff <= 0), mean(diff >= 0))

    The CI is the percentile CI on (score_i - score_j) across iterations.
    """
    n = len(names)
    pvals = pd.DataFrame(np.ones((n, n)), index=names, columns=names)
    ci_info = pd.DataFrame("", index=names, columns=names)

    for i in range(n):
        for j in range(i + 1, n):
            diff = boot_scores[i] - boot_scores[j]
            mean_diff = float(np.mean(diff))
            lo = float(np.quantile(diff, 0.025))
            hi = float(np.quantile(diff, 0.975))

            p_left = float(np.mean(diff <= 0))
            p_right = float(np.mean(diff >= 0))
            p = min(1.0, 2.0 * min(p_left, p_right))
            # Floor at 1/B: the bootstrap cannot resolve smaller p, and p=0
            # would give -log10(p)=inf and collapse the heatmap colormap.
            p = max(p, 1.0 / len(diff))

            pvals.iloc[i, j] = p
            pvals.iloc[j, i] = p
            ci_info.iloc[i, j] = f"mean={mean_diff:+.4f},ci95=[{lo:+.4f},{hi:+.4f}]"
            ci_info.iloc[j, i] = f"mean={-mean_diff:+.4f},ci95=[{-hi:+.4f},{-lo:+.4f}]"

    return pvals, ci_info


def summarise_significant(
    pvals: pd.DataFrame,
    ci_info: pd.DataFrame,
    observed_scores: dict[str, float],
    alpha: float,
) -> pd.DataFrame:
    """Extract significant pairs with observed-metric context."""
    rows = []
    names = list(pvals.index)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            p = pvals.iloc[i, j]
            if p < alpha:
                rows.append({
                    "classifier_a": names[i],
                    "classifier_b": names[j],
                    "score_a": f"{observed_scores[names[i]]:.4f}",
                    "score_b": f"{observed_scores[names[j]]:.4f}",
                    "diff_ci": ci_info.iloc[i, j],
                    "p_value": f"{p:.6f}",
                })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description="Paired bootstrap test between classifiers")
    p.add_argument("--results-dir", type=Path, default=Path("results/classification"))
    p.add_argument("--dataset", type=str, default=None, help="Run for a single dataset")
    p.add_argument(
        "--task",
        choices=["five_way", "binary"],
        default="five_way",
        help="Which classification task to compare",
    )
    p.add_argument(
        "--metric",
        choices=list(METRIC_FNS.keys()),
        default="macro_f1",
        help="Metric used for the paired comparison",
    )
    p.add_argument("--n-bootstrap", type=int, default=1000, help="Number of bootstrap iterations")
    p.add_argument("--seed", type=int, default=0, help="RNG seed")
    p.add_argument("--alpha", type=float, default=0.05, help="Significance threshold")
    args = p.parse_args()

    metric_fn = METRIC_FNS[args.metric]

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
        print(
            f"Dataset: {dataset} | Task: {args.task} | Metric: {args.metric} "
            f"| B={args.n_bootstrap} | α = {args.alpha}"
        )
        print(f"{'='*60}")

        classifiers = load_classifiers(dataset_dir, args.task)
        if len(classifiers) < 2:
            print(f"  Skipping: only {len(classifiers)} classifier(s) found")
            continue

        classifiers = align_and_validate_classifiers(classifiers)
        names = list(classifiers.keys())

        print(f"  Loaded {len(classifiers)} classifiers, "
              f"{len(next(iter(classifiers.values()))['true_labels'])} samples")

        observed_scores = {
            name: float(metric_fn(c["true_labels"], c["predictions"]))
            for name, c in classifiers.items()
        }

        boot_scores = paired_bootstrap(classifiers, args.metric, args.n_bootstrap, args.seed)
        pvals, ci_info = pairwise_bootstrap_stats(names, boot_scores)

        stem = f"bootstrap_{args.task}_{args.metric}"
        csv_path = dataset_dir / f"{stem}.csv"
        pvals.to_csv(csv_path)
        print(f"  P-value matrix saved to {csv_path}")

        ci_path = dataset_dir / f"{stem}_ci.csv"
        ci_info.to_csv(ci_path)
        print(f"  CI matrix saved to {ci_path}")

        fig_path = dataset_dir / f"{stem}.png"
        plot_pvalue_heatmap(
            pvals,
            scores=observed_scores,
            score_name=args.metric,
            title=(
                f"Paired bootstrap: {dataset} ({args.task}, {args.metric}, "
                f"B={args.n_bootstrap}) — sorted by {args.metric} (best top-left)"
            ),
            alpha=args.alpha,
            output_path=fig_path,
        )

        sig = summarise_significant(pvals, ci_info, observed_scores, args.alpha)
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
