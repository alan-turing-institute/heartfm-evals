"""Shared utilities for classifier comparison tests (McNemar, bootstrap).

Handles loading predictions from per-classifier JSON files, aligning them
sample-by-sample via patient_ids, and plotting pairwise p-value heatmaps.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def classifier_label(config: dict, eval_mode: str) -> str:
    """Build a short human-readable label for a classifier."""
    backbone = config.get("backbone", "?")
    model = config.get("model_name", "?")
    pooling = config.get("pooling", "cls")
    short_model = model.replace("cinema_", "").replace("dinov3_", "").replace("sam_", "")
    return f"{backbone}_{short_model}_{eval_mode}_{pooling}"


def load_classifiers(dataset_dir: Path, task: str) -> dict[str, dict]:
    """Load predictions and true labels from all JSON files in a dataset dir.

    Returns dict mapping classifier_label -> {
        "predictions": np.array,
        "true_labels": np.array,
        "patient_ids": np.array | None,
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
            patient_ids = section.get("patient_ids")
        elif task == "binary":
            section = data.get("binary", {})
            true_labels = section.get("binary_labels")
            probs = section.get("binary_probs")
            if probs is not None and true_labels is not None:
                preds = [int(p >= 0.5) for p in probs]
            else:
                preds = None

            patient_ids = section.get("patient_ids")
            if patient_ids is None:
                patient_ids = data.get("five_way", {}).get("patient_ids")
        else:
            raise ValueError(f"Unknown task: {task}")

        if preds is not None and true_labels is not None:
            if len(preds) != len(true_labels):
                raise ValueError(
                    f"Length mismatch in {f.name}: "
                    f"len(predictions)={len(preds)} vs len(true_labels)={len(true_labels)}"
                )
            if patient_ids is not None and len(patient_ids) != len(true_labels):
                raise ValueError(
                    f"Length mismatch in {f.name}: "
                    f"len(patient_ids)={len(patient_ids)} vs len(true_labels)={len(true_labels)}"
                )

            classifiers[label] = {
                "predictions": np.array(preds),
                "true_labels": np.array(true_labels),
                "patient_ids": np.array(patient_ids) if patient_ids is not None else None,
            }

    return classifiers


def align_and_validate_classifiers(classifiers: dict[str, dict]) -> dict[str, dict]:
    """Ensure all classifiers are aligned sample-by-sample and labels are consistent.

    If patient_ids are available for all classifiers, they are used as the canonical
    alignment key and each classifier is reordered to match the reference order.
    If IDs are unavailable for all classifiers, falls back to strict checks on length
    and exact true-label sequence equality.
    """
    if not classifiers:
        return classifiers

    names = list(classifiers.keys())
    reference_name = names[0]
    reference = classifiers[reference_name]
    reference_ids = reference["patient_ids"]
    use_ids = reference_ids is not None

    if use_ids:
        ref_ids_list = reference_ids.tolist()
        if len(set(ref_ids_list)) != len(ref_ids_list):
            raise ValueError(
                f"Duplicate patient_ids found in reference classifier {reference_name}"
            )

    for name in names[1:]:
        current = classifiers[name]

        if len(current["predictions"]) != len(reference["predictions"]):
            raise ValueError(
                f"Sample count mismatch between {reference_name} and {name}: "
                f"{len(reference['predictions'])} vs {len(current['predictions'])}"
            )

        current_ids = current["patient_ids"]
        if use_ids and current_ids is None:
            raise ValueError(
                f"Missing patient_ids for {name}. Cannot verify sample-level alignment."
            )
        if (not use_ids) and (current_ids is not None):
            raise ValueError(
                "Mixed availability of patient_ids across classifiers. "
                "Provide patient_ids for all classifiers to validate alignment."
            )

        if use_ids:
            curr_ids_list = current_ids.tolist()
            if len(set(curr_ids_list)) != len(curr_ids_list):
                raise ValueError(f"Duplicate patient_ids found in classifier {name}")

            if set(curr_ids_list) != set(ref_ids_list):
                missing = sorted(set(ref_ids_list) - set(curr_ids_list))
                extra = sorted(set(curr_ids_list) - set(ref_ids_list))
                raise ValueError(
                    f"patient_ids mismatch between {reference_name} and {name}. "
                    f"missing={missing[:5]}{'...' if len(missing) > 5 else ''}, "
                    f"extra={extra[:5]}{'...' if len(extra) > 5 else ''}"
                )

            curr_id_to_idx = {pid: idx for idx, pid in enumerate(curr_ids_list)}
            reorder_idx = np.array([curr_id_to_idx[pid] for pid in ref_ids_list], dtype=int)
            current["predictions"] = current["predictions"][reorder_idx]
            current["true_labels"] = current["true_labels"][reorder_idx]
            current["patient_ids"] = current["patient_ids"][reorder_idx]

            if not np.array_equal(current["true_labels"], reference["true_labels"]):
                raise ValueError(
                    f"true_labels mismatch after patient_id alignment between "
                    f"{reference_name} and {name}."
                )
        else:
            if not np.array_equal(current["true_labels"], reference["true_labels"]):
                raise ValueError(
                    f"true_labels mismatch between {reference_name} and {name}. "
                    "Without patient_ids, sample alignment cannot be guaranteed."
                )

    return classifiers


def plot_pvalue_heatmap(
    pvals: pd.DataFrame,
    scores: dict[str, float],
    score_name: str,
    title: str,
    alpha: float,
    output_path: Path,
) -> None:
    """Plot a heatmap of -log10(p-values), sorted by ``scores`` descending.

    ``scores`` maps classifier label -> scalar used for sorting and display
    (e.g. macro-F1). ``title`` is the full plot title (caller decides
    dataset/task/test-type wording). ``alpha`` is shown in the title as the
    significance threshold.
    """
    sorted_names = sorted(pvals.index, key=lambda n: scores[n], reverse=True)
    display_labels = [f"{n}  ({score_name}={scores[n]:.2f})" for n in sorted_names]

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

    threshold = -np.log10(alpha)
    full_title = (
        f"{title}\n"
        f"Values are $-\\log_{{10}}(p)$;  significant if > {threshold:.2f}  (α={alpha})"
    )
    ax.set_title(full_title, fontsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=7)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Heatmap saved to {output_path}")
