#!/usr/bin/env python3
"""CLI script for pathology classification evaluation on ACDC, MnM, or MnM2.

Runs the full pipeline (load data, load backbone, train, evaluate) and saves
results as both a Markdown summary and a JSON file containing all data needed
to regenerate plots.

Usage examples:
    python scripts/classification/run_classification.py --dataset acdc --backbone cinema --eval-mode logreg
    python scripts/classification/run_classification.py --dataset mnm --backbone dinov3 --eval-mode finetune
    python scripts/classification/run_classification.py --dataset mnm2 --backbone sam --eval-mode logreg --pooling gap
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from cinema.segmentation.dataset import EndDiastoleEndSystoleDataset
from monai.transforms import ScaleIntensityd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

from heartfm_evals.backbones import load_backbone as _load_backbone
from heartfm_evals.classification_probe import (
    build_patient_features,
    cache_cinema_cls_features,
    cache_cls_features,
    cache_sam_cls_features,
    evaluate_binary_detection,
    evaluate_classification,
    get_pathology_classes,
    get_pathology_map,
    load_cached_cls_features,
    sweep_C_and_train,
    validate_split_pathology_labels,
)
from heartfm_evals.device import detect_device as _detect_device
from heartfm_evals.finetune_classification import (
    ClassificationHeadPredictor,
    finetune_sweep_and_train,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pathology classification evaluation")
    p.add_argument("--dataset", default="acdc", choices=["acdc", "mnm", "mnm2"])
    p.add_argument("--backbone", required=True, choices=["cinema", "dinov3", "sam"])
    p.add_argument("--eval-mode", required=True, choices=["logreg", "finetune"])
    p.add_argument("--pooling", default="cls", choices=["cls", "gap"])
    p.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Override data dir (default: data/heartfm/processed/{dataset})",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output dir (default: results/classification/{dataset})",
    )
    p.add_argument(
        "--device", default=None, help="Override device (default: auto-detect)"
    )
    # Model paths
    p.add_argument("--dinov3-repo-dir", default="models/dinov3/")
    p.add_argument("--dinov3-model-name", default="dinov3_vits16")
    p.add_argument("--dinov3-weights-path", default=None)
    p.add_argument("--sam-model-id", default="facebook/sam-vit-base")
    p.add_argument("--hf-cache-dir", type=Path, default=Path("model_weights/hf"))
    p.add_argument(
        "--cls-cache-dir",
        type=Path,
        default=None,
        help="Feature cache dir (default: auto)",
    )
    p.add_argument(
        "--max-patients",
        type=int,
        default=None,
        help="Limit number of patients (for debugging)",
    )
    p.add_argument(
        "--no-auto-download", action="store_true", help="Disable HF auto-download"
    )
    args = p.parse_args()

    # Set defaults that depend on --dataset
    if args.data_dir is None:
        args.data_dir = Path(f"data/heartfm/processed/{args.dataset}")
    if args.output_dir is None:
        args.output_dir = Path(f"results/classification/{args.dataset}")

    return args


def detect_device(override: str | None) -> torch.device:
    return _detect_device(override)


def derive_model_name(args) -> str:
    if args.backbone == "cinema":
        return "cinema_pretrained"
    if args.backbone == "sam":
        return args.sam_model_id.split("/")[-1].replace("-", "_")
    return args.dinov3_model_name


def eval_mode_tag(eval_mode: str, freeze_backbone: bool) -> str:
    """Return a short tag string for the evaluation mode."""
    if eval_mode == "logreg":
        return "logreg"
    return "ftfrozen" if freeze_backbone else "ftfull"


def build_results_dict(
    args,
    model_name,
    embed_dim,
    best_hyperparam,
    sweep_results,
    test_metrics,
    test_labels_eval,
    test_pids_eval,
    macro_auc,
    per_class_auc,
    binary_metrics,
    pathology_classes,
) -> dict:
    """Build a JSON-serialisable results dictionary."""

    def to_list(x):
        if isinstance(x, (np.ndarray, torch.Tensor)):
            return x.tolist()
        return x

    return {
        "config": {
            "dataset": args.dataset,
            "backbone": args.backbone,
            "model_name": model_name,
            "eval_mode": args.eval_mode,
            "pooling": args.pooling,
            "embed_dim": embed_dim,
            "best_hyperparam": best_hyperparam,
            "hyperparam_name": "C" if args.eval_mode == "logreg" else "lr",
            "num_classes": len(pathology_classes),
            "class_names": list(pathology_classes.keys()),
        },
        "sweep_results": [
            {
                "param_value": r.get("C", r.get("lr")),
                "mean_cv_acc": r["mean_cv_acc"],
                "std_cv_acc": r["std_cv_acc"],
            }
            for r in sweep_results
        ],
        "five_way": {
            "accuracy": test_metrics["accuracy"],
            "macro_f1": test_metrics["macro_f1"],
            "macro_sensitivity": test_metrics["macro_sensitivity"],
            "macro_specificity": test_metrics["macro_specificity"],
            "per_class_sensitivity": test_metrics["per_class_sensitivity"],
            "per_class_specificity": test_metrics["per_class_specificity"],
            "confusion_matrix": to_list(test_metrics["confusion_matrix"]),
            "classification_report": test_metrics["classification_report"],
            "probabilities": to_list(test_metrics["probabilities"]),
            "predictions": to_list(test_metrics["predictions"]),
            "true_labels": to_list(test_labels_eval),
            "patient_ids": to_list(test_pids_eval),
        },
        "roc_auc": {
            "macro_auc": float(macro_auc),
            "per_class_auc": {
                cls_name: float(auc_val)
                for cls_name, auc_val in zip(
                    pathology_classes.keys(), per_class_auc, strict=False
                )
            },
        },
        "binary": {
            "accuracy": binary_metrics["accuracy"],
            "f1": binary_metrics["f1"],
            "sensitivity": binary_metrics["sensitivity"],
            "specificity": binary_metrics["specificity"],
            "roc_auc": binary_metrics["roc_auc"],
            "binary_labels": to_list(binary_metrics["binary_labels"]),
            "binary_probs": to_list(binary_metrics["binary_probs"]),
        },
    }


def main():
    args = parse_args()
    model_name = derive_model_name(args)
    tag = eval_mode_tag(args.eval_mode, True)
    base_name = f"{model_name}_{tag}_{args.pooling}"
    if args.max_patients:
        base_name += "_smoke"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{base_name}.json"
    if json_path.exists():
        print(f"Skipping: {json_path} already exists.")
        return

    device = detect_device(args.device)
    pathology_classes = get_pathology_classes(args.dataset)
    num_classes = len(pathology_classes)
    nor_idx = pathology_classes["NOR"]

    print(
        f"Dataset: {args.dataset} ({num_classes} classes: {list(pathology_classes.keys())})"
    )
    print(f"Device: {device}")
    print(f"Backbone: {args.backbone} ({model_name})")
    print(f"Eval mode: {args.eval_mode}")
    print(f"Pooling: {args.pooling}")

    # ── Load metadata and datasets ──
    train_meta_df = pd.read_csv(
        args.data_dir / "train_metadata.csv", dtype={"pid": str}
    )
    test_meta_df = pd.read_csv(args.data_dir / "test_metadata.csv", dtype={"pid": str})

    # Auto-detect val split
    val_meta_path = args.data_dir / "val_metadata.csv"
    has_val_split = val_meta_path.exists()
    val_meta_df = (
        pd.read_csv(val_meta_path, dtype={"pid": str}) if has_val_split else None
    )

    if args.max_patients:
        train_meta_df = train_meta_df.head(args.max_patients)
        test_meta_df = test_meta_df.head(args.max_patients)
        if val_meta_df is not None:
            val_meta_df = val_meta_df.head(args.max_patients)

    print(f"Train: {len(train_meta_df)} patients, Test: {len(test_meta_df)} patients")
    if has_val_split:
        print(f"Val: {len(val_meta_df)} patients (using dedicated val split)")
    else:
        print("No val split found, using K-fold CV")

    transform = ScaleIntensityd(keys="sax_image", factor=1 / 255, channel_wise=False)
    train_cinema = EndDiastoleEndSystoleDataset(
        data_dir=args.data_dir / "train",
        meta_df=train_meta_df,
        views="sax",
        transform=transform,
    )
    test_cinema = EndDiastoleEndSystoleDataset(
        data_dir=args.data_dir / "test",
        meta_df=test_meta_df,
        views="sax",
        transform=transform,
    )
    val_cinema = None
    if has_val_split:
        val_cinema = EndDiastoleEndSystoleDataset(
            data_dir=args.data_dir / "val",
            meta_df=val_meta_df,
            views="sax",
            transform=transform,
        )

    # ── Load backbone ──
    backbone_kwargs: dict = {}
    if args.backbone == "dinov3":
        backbone_kwargs["dinov3_model_name"] = args.dinov3_model_name
        backbone_kwargs["dinov3_repo_dir"] = args.dinov3_repo_dir
        if args.dinov3_weights_path:
            backbone_kwargs["dinov3_weights_path"] = args.dinov3_weights_path
    elif args.backbone == "sam":
        backbone_kwargs["sam_model_id"] = args.sam_model_id
        backbone_kwargs["hf_cache_dir"] = str(args.hf_cache_dir)
        backbone_kwargs["auto_download"] = not args.no_auto_download
    elif args.backbone == "cinema":
        backbone_kwargs["hf_cache_dir"] = str(args.hf_cache_dir)
        backbone_kwargs["auto_download"] = not args.no_auto_download

    backbone, info = _load_backbone(args.backbone, device, **backbone_kwargs)
    embed_dim = info["embed_dim"]
    sam_image_processor = info.get("sam_image_processor")
    print(f"Loaded backbone: embed_dim={embed_dim}")

    # ── Pathology maps ──
    train_pathology_map = get_pathology_map(train_meta_df)
    test_pathology_map = get_pathology_map(test_meta_df)
    val_pathology_map = (
        get_pathology_map(val_meta_df) if val_meta_df is not None else None
    )
    validate_split_pathology_labels(
        train_pathology_map,
        pathology_classes=pathology_classes,
        val_pathology_map=val_pathology_map,
        test_pathology_map=test_pathology_map,
    )

    # ── Train ──
    print("Training...")

    # ── Feature caching (shared by logreg and finetune) ──
    cls_cache_dir = args.cls_cache_dir or Path(
        f"classification_feature_cache/{args.dataset}/{model_name}/{args.pooling}"
    )

    if args.backbone == "cinema":
        cache_fn = lambda m, ds, cd, dev: cache_cinema_cls_features(
            m, ds, cd, device=dev, pooling=args.pooling
        )
    elif args.backbone == "sam":
        cache_fn = lambda m, ds, cd, dev: cache_sam_cls_features(
            m, sam_image_processor, ds, cd, device=dev
        )
    else:
        cache_fn = lambda m, ds, cd, dev: cache_cls_features(
            m, ds, cd, device=dev, pooling=args.pooling
        )

    print("Caching training features...")
    train_manifest = cache_fn(backbone, train_cinema, cls_cache_dir / "train", device)
    print("Caching test features...")
    test_manifest = cache_fn(backbone, test_cinema, cls_cache_dir / "test", device)

    train_cls = load_cached_cls_features(train_manifest)
    test_cls = load_cached_cls_features(test_manifest)

    train_features, train_labels, train_pids = build_patient_features(
        train_cls,
        train_pathology_map,
        pathology_classes=pathology_classes,
    )
    test_features, test_labels, test_pids = build_patient_features(
        test_cls,
        test_pathology_map,
        pathology_classes=pathology_classes,
    )
    print(f"Feature shape: {train_features.shape}")

    # Cache val features if val split exists
    val_features, val_labels = None, None
    if has_val_split:
        print("Caching val features...")
        val_manifest = cache_fn(backbone, val_cinema, cls_cache_dir / "val", device)
        val_cls = load_cached_cls_features(val_manifest)
        val_features, val_labels, _ = build_patient_features(
            val_cls,
            val_pathology_map,
            pathology_classes=pathology_classes,
        )

    if args.eval_mode == "logreg":
        best_C, final_model, sweep_results = sweep_C_and_train(
            train_features,
            train_labels,
            n_folds=10,
            val_features=val_features,
            val_labels=val_labels,
        )
        best_hyperparam = best_C
        print(f"Best C = {best_C:.4g}")
    else:
        best_lr, ft_head, sweep_results, ft_scaler = finetune_sweep_and_train(
            train_features,
            train_labels,
            device=device,
            n_folds=10,
            num_classes=num_classes,
            val_features=val_features,
            val_labels=val_labels,
        )
        final_model = ClassificationHeadPredictor(ft_head, ft_scaler, device)
        best_hyperparam = best_lr
        print(f"Best LR = {best_lr:.4g}")

    # ── Evaluate 5-way ──
    print("Evaluating...")
    test_metrics = evaluate_classification(
        final_model,
        test_features,
        test_labels,
        pathology_classes=pathology_classes,
    )
    test_pids_eval = test_pids
    test_labels_eval = test_labels

    print(
        f"5-way Accuracy: {test_metrics['accuracy']:.4f}, Macro F1: {test_metrics['macro_f1']:.4f}"
    )

    # ── ROC AUC ──
    y_true_bin = label_binarize(
        test_labels_eval.numpy(), classes=list(range(num_classes))
    )
    y_prob = test_metrics["probabilities"]
    macro_auc = roc_auc_score(y_true_bin, y_prob, multi_class="ovr", average="macro")
    per_class_auc = roc_auc_score(y_true_bin, y_prob, multi_class="ovr", average=None)
    print(f"Macro ROC AUC: {macro_auc:.4f}")

    # ── Binary detection ──
    binary_metrics = evaluate_binary_detection(
        test_metrics["probabilities"],
        test_labels_eval,
        nor_idx=nor_idx,
    )
    print(
        f"Binary Acc: {binary_metrics['accuracy']:.4f}, F1: {binary_metrics['f1']:.4f}"
    )

    # ── Save results ──
    results = build_results_dict(
        args,
        model_name,
        embed_dim,
        best_hyperparam,
        sweep_results,
        test_metrics,
        test_labels_eval,
        test_pids_eval,
        macro_auc,
        per_class_auc,
        binary_metrics,
        pathology_classes,
    )
    json_path.write_text(json.dumps(results, indent=2))
    print(f"Saved results: {json_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
