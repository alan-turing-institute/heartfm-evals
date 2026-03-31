#!/usr/bin/env python3
"""Bootstrap confidence intervals for classification results.

Resamples test patients (with replacement) from the saved per-patient
predictions in each JSON result file and recomputes metrics on each
bootstrap sample to produce 95% confidence intervals.

Also runs pairwise bootstrap hypothesis tests across all model pairs
to determine whether differences in the chosen metric are significant.
The default comparison metric is 5-way AUC (ROC AUC, macro one-vs-rest).

Usage:
    python scripts/classification/bootstrap_ci.py
    python scripts/classification/bootstrap_ci.py --n-boot 10000
    python scripts/classification/bootstrap_ci.py --output results/classification/bootstrap_ci.csv
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


EVAL_MODE_DISPLAY = {
    ("logreg", True): "logreg",
    ("logreg", False): "logreg",
    ("finetune", True): "ft-frozen",
    ("finetune", False): "ft-full",
}


def _eval_mode_label(cfg: dict) -> str:
    return EVAL_MODE_DISPLAY.get(
        (cfg["eval_mode"], cfg.get("freeze_backbone", True)),
        cfg["eval_mode"],
    )


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float]:
    """Compute 5-way and binary metrics for a single (possibly resampled) set."""
    n_classes = y_prob.shape[1]

    # 5-way
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # 5-way AUC (macro, one-vs-rest) — needs all classes present
    try:
        auc5 = roc_auc_score(
            y_true, y_prob, multi_class="ovr", average="macro",
            labels=list(range(n_classes)),
        )
    except ValueError:
        auc5 = np.nan

    # Binary: normal (class 0) vs disease (classes 1-4)
    bin_true = (y_true != 0).astype(int)
    bin_prob = 1.0 - y_prob[:, 0]
    bin_pred = (bin_prob >= 0.5).astype(int)

    bin_acc = accuracy_score(bin_true, bin_pred)
    bin_f1 = f1_score(bin_true, bin_pred, zero_division=0)

    # Binary sensitivity / specificity
    pos = bin_true == 1
    neg = bin_true == 0
    bin_sens = float(bin_pred[pos].mean()) if pos.any() else np.nan
    bin_spec = float(1.0 - bin_pred[neg].mean()) if neg.any() else np.nan

    try:
        bin_auc = roc_auc_score(bin_true, bin_prob)
    except ValueError:
        bin_auc = np.nan

    return {
        "5way_acc": acc,
        "5way_f1": f1,
        "5way_auc": auc5,
        "bin_acc": bin_acc,
        "bin_f1": bin_f1,
        "bin_sens": bin_sens,
        "bin_spec": bin_spec,
        "bin_auc": bin_auc,
    }


def load_results(results_dir: Path) -> list[dict]:
    """Load per-patient predictions from all JSON result files."""
    records = []
    for path in sorted(results_dir.glob("*.json")):
        with open(path) as f:
            data = json.load(f)
        cfg = data["config"]
        fw = data["five_way"]

        records.append({
            "file": path.name,
            "backbone": cfg["backbone"],
            "model_name": cfg["model_name"],
            "eval_mode": _eval_mode_label(cfg),
            "pooling": cfg["pooling"],
            "embed_dim": cfg["embed_dim"],
            "y_true": np.array(fw["true_labels"]),
            "y_pred": np.array(fw["predictions"]),
            "y_prob": np.array(fw["probabilities"]),
            "patient_ids": fw["patient_ids"],
        })
    return records


def bootstrap_ci(
    rec: dict,
    n_boot: int = 5000,
    ci: float = 0.95,
    rng: np.random.Generator | None = None,
) -> dict[str, dict[str, float]]:
    """Compute bootstrap CIs for all metrics of a single model.

    Returns {metric_name: {"point": ..., "lo": ..., "hi": ..., "std": ...}}.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    y_true = rec["y_true"]
    y_pred = rec["y_pred"]
    y_prob = rec["y_prob"]
    n = len(y_true)

    # Point estimates
    point = _compute_metrics(y_true, y_pred, y_prob)

    # Bootstrap
    boot_metrics = {k: [] for k in point}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        m = _compute_metrics(y_true[idx], y_pred[idx], y_prob[idx])
        for k, v in m.items():
            boot_metrics[k].append(v)

    alpha = (1.0 - ci) / 2.0
    result = {}
    for k in point:
        vals = np.array(boot_metrics[k])
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            result[k] = {"point": point[k], "lo": np.nan, "hi": np.nan, "std": np.nan}
        else:
            result[k] = {
                "point": point[k],
                "lo": float(np.percentile(vals, 100 * alpha)),
                "hi": float(np.percentile(vals, 100 * (1 - alpha))),
                "std": float(np.std(vals)),
            }
    return result


def pairwise_bootstrap_test(
    records: list[dict],
    metric: str = "5way_acc",
    n_boot: int = 5000,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Bootstrap hypothesis test for all pairs: is the difference in metric significant?

    For each pair (A, B), resamples patients and computes metric_A - metric_B.
    The p-value is the proportion of bootstrap samples where the sign of the
    difference is opposite to the observed difference (two-sided).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = len(records[0]["y_true"])

    # Pre-generate shared bootstrap indices so all models use the same resamples
    boot_indices = rng.integers(0, n, size=(n_boot, n))

    # Compute bootstrap distribution of the metric for each model
    boot_vals = {}
    for rec in records:
        y_true = rec["y_true"]
        y_pred = rec["y_pred"]
        y_prob = rec["y_prob"]

        vals = []
        for i in range(n_boot):
            idx = boot_indices[i]
            m = _compute_metrics(y_true[idx], y_pred[idx], y_prob[idx])
            vals.append(m[metric])
        boot_vals[rec["file"]] = np.array(vals)

    rows = []
    for a, b in combinations(records, 2):
        # Point estimates
        point_a = _compute_metrics(a["y_true"], a["y_pred"], a["y_prob"])[metric]
        point_b = _compute_metrics(b["y_true"], b["y_pred"], b["y_prob"])[metric]
        observed_diff = point_a - point_b

        # Bootstrap difference
        diff = boot_vals[a["file"]] - boot_vals[b["file"]]
        diff = diff[~np.isnan(diff)]

        if len(diff) == 0 or observed_diff == 0:
            p_val = 1.0
        else:
            # Two-sided: proportion of times the sign flips
            if observed_diff > 0:
                p_val = float(np.mean(diff <= 0)) * 2
            else:
                p_val = float(np.mean(diff >= 0)) * 2
            p_val = min(p_val, 1.0)

        ci_lo = float(np.percentile(diff, 2.5))
        ci_hi = float(np.percentile(diff, 97.5))

        rows.append({
            "metric": metric,
            "model_a": a["model_name"],
            "mode_a": a["eval_mode"],
            "pooling_a": a["pooling"],
            "model_b": b["model_name"],
            "mode_b": b["eval_mode"],
            "pooling_b": b["pooling"],
            f"{metric}_a": point_a,
            f"{metric}_b": point_b,
            "diff": observed_diff,
            "diff_ci_lo": ci_lo,
            "diff_ci_hi": ci_hi,
            "p_value": p_val,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("p_value")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap confidence intervals for classification results"
    )
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results/classification"),
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output CSV path (default: results/classification/bootstrap_ci.csv)",
    )
    parser.add_argument(
        "--n-boot", type=int, default=5000,
        help="Number of bootstrap resamples (default: 5000)",
    )
    parser.add_argument(
        "--pairwise-metric", type=str, default="5way_auc",
        choices=["5way_acc", "5way_f1", "5way_auc", "bin_acc", "bin_auc"],
        help="Metric for pairwise comparisons (default: 5way_auc)",
    )
    parser.add_argument(
        "--skip-pairwise", action="store_true",
        help="Skip pairwise comparisons (faster)",
    )
    args = parser.parse_args()

    output_path = args.output or args.results_dir / "bootstrap_ci.csv"
    rng = np.random.default_rng(42)

    records = load_results(args.results_dir)
    print(f"Loaded {len(records)} result files from {args.results_dir}")

    # ── Per-model bootstrap CIs ──────────────────────────────────────────────
    metrics = ["5way_acc", "5way_f1", "5way_auc", "bin_acc", "bin_f1",
               "bin_sens", "bin_spec", "bin_auc"]

    ci_rows = []
    for rec in records:
        ci = bootstrap_ci(rec, n_boot=args.n_boot, rng=rng)
        row = {
            "backbone": rec["backbone"],
            "model": rec["model_name"],
            "eval_mode": rec["eval_mode"],
            "pooling": rec["pooling"],
            "embed_dim": rec["embed_dim"],
        }
        for m in metrics:
            row[f"{m}"] = ci[m]["point"]
            row[f"{m}_lo"] = ci[m]["lo"]
            row[f"{m}_hi"] = ci[m]["hi"]
        ci_rows.append(row)

    ci_df = pd.DataFrame(ci_rows)
    ci_df.to_csv(output_path, index=False, float_format="%.4f")
    print(f"\nSaved per-model CIs to {output_path}")

    # ── Print summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("BOOTSTRAP 95% CONFIDENCE INTERVALS")
    print("=" * 100)
    for _, row in ci_df.iterrows():
        label = f"{row['backbone']:6s} | {row['model']:25s} | {row['eval_mode']:9s} | {row['pooling']:3s}"
        auc5 = f"5way_auc: {row['5way_auc']:.2f} [{row['5way_auc_lo']:.2f}, {row['5way_auc_hi']:.2f}]"
        f1 = f"5way_f1: {row['5way_f1']:.2f} [{row['5way_f1_lo']:.2f}, {row['5way_f1_hi']:.2f}]"
        acc = f"5way_acc: {row['5way_acc']:.2f} [{row['5way_acc_lo']:.2f}, {row['5way_acc_hi']:.2f}]"
        bauc = f"bin_auc: {row['bin_auc']:.2f} [{row['bin_auc_lo']:.2f}, {row['bin_auc_hi']:.2f}]"
        bsens = f"bin_sens: {row['bin_sens']:.2f} [{row['bin_sens_lo']:.2f}, {row['bin_sens_hi']:.2f}]"
        bspec = f"bin_spec: {row['bin_spec']:.2f} [{row['bin_spec_lo']:.2f}, {row['bin_spec_hi']:.2f}]"
        print(f"  {label}")
        print(f"    5-way: {auc5} | {f1} | {acc}")
        print(f"    Binary: {bauc} | {bsens} | {bspec}")

    # ── Pairwise bootstrap tests ─────────────────────────────────────────────
    if not args.skip_pairwise:
        print(f"\nRunning pairwise bootstrap tests on {args.pairwise_metric}...")
        pw_df = pairwise_bootstrap_test(
            records,
            metric=args.pairwise_metric,
            n_boot=args.n_boot,
            rng=rng,
        )
        pw_path = output_path.with_name(output_path.stem + "_pairwise.csv")
        pw_df.to_csv(pw_path, index=False, float_format="%.4f")
        print(f"Saved pairwise comparisons to {pw_path}")

        sig = pw_df[pw_df["p_value"] < 0.05]
        print(f"\n{len(sig)} / {len(pw_df)} pairs significant at p<0.05")

        if not sig.empty:
            print("\nSignificant pairs:")
            for _, r in sig.head(20).iterrows():
                a_label = f"{r['model_a']}_{r['mode_a']}_{r['pooling_a']}"
                b_label = f"{r['model_b']}_{r['mode_b']}_{r['pooling_b']}"
                print(
                    f"  {a_label} vs {b_label}: "
                    f"diff={r['diff']:+.2f} [{r['diff_ci_lo']:+.2f}, {r['diff_ci_hi']:+.2f}] "
                    f"p={r['p_value']:.4f}"
                )

        # Show top non-significant comparisons people might care about
        ns_cross = pw_df[
            (pw_df["p_value"] >= 0.05)
            & (pw_df["model_a"].str.split("_").str[0] != pw_df["model_b"].str.split("_").str[0])
        ].head(10)
        if not ns_cross.empty:
            print("\nNotable non-significant cross-backbone pairs:")
            for _, r in ns_cross.iterrows():
                a_label = f"{r['model_a']}_{r['mode_a']}_{r['pooling_a']}"
                b_label = f"{r['model_b']}_{r['mode_b']}_{r['pooling_b']}"
                print(
                    f"  {a_label} vs {b_label}: "
                    f"diff={r['diff']:+.2f} [{r['diff_ci_lo']:+.2f}, {r['diff_ci_hi']:+.2f}] "
                    f"p={r['p_value']:.4f}"
                )


if __name__ == "__main__":
    main()
