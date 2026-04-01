#!/usr/bin/env python3
"""CLI script for ACDC pathology classification evaluation.

Runs the full pipeline (load data, load backbone, train, evaluate) and saves
results as both a Markdown summary and a JSON file containing all data needed
to regenerate plots.

Usage examples:
    python scripts/run_acdc_classification.py --backbone cinema --eval-mode logreg
    python scripts/run_acdc_classification.py --backbone dinov3 --eval-mode finetune --no-freeze-backbone
    python scripts/run_acdc_classification.py --backbone sam --eval-mode logreg --pooling gap
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from cinema.segmentation.dataset import EndDiastoleEndSystoleDataset
from monai.transforms import ScaleIntensityd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

from heartfm_evals.classification_probe import (
    NUM_PATHOLOGIES,
    PATHOLOGY_CLASSES,
    PATHOLOGY_NAMES,
    build_patient_features,
    cache_cinema_cls_features,
    cache_cls_features,
    cache_sam_cls_features,
    evaluate_binary_detection,
    evaluate_classification,
    get_pathology_map,
    load_cached_cls_features,
    sweep_C_and_train,
)
from heartfm_evals.finetune_classification import (
    evaluate_finetune_classification,
    finetune_sweep_and_train,
)

# DINOv3 embed dims by model name
DINOV3_EMBED_DIMS = {
    "dinov3_vits16": 384,
    "dinov3_vitb16": 768,
    "dinov3_vitl16": 1024,
    "dinov3_vit7b16": 4096,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ACDC pathology classification evaluation")
    p.add_argument("--backbone", required=True, choices=["cinema", "dinov3", "sam"])
    p.add_argument("--eval-mode", required=True, choices=["logreg", "finetune"])
    p.add_argument(
        "--freeze-backbone",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Freeze backbone in finetune mode (default: True). Use --no-freeze-backbone to unfreeze.",
    )
    p.add_argument("--pooling", default="cls", choices=["cls", "gap"])
    p.add_argument("--data-dir", type=Path, default=Path("data/heartfm/processed/acdc"))
    p.add_argument("--output-dir", type=Path, default=Path("results/classification"))
    p.add_argument("--device", default=None, help="Override device (default: auto-detect)")
    # Model paths
    p.add_argument("--dinov3-repo-dir", default="models/dinov3/")
    p.add_argument("--dinov3-model-name", default="dinov3_vits16")
    p.add_argument("--dinov3-weights-path", default=None)
    p.add_argument("--sam-model-id", default="facebook/sam-vit-base")
    p.add_argument("--hf-cache-dir", type=Path, default=Path("model_weights/hf"))
    p.add_argument("--cls-cache-dir", type=Path, default=None, help="Feature cache dir (default: auto)")
    p.add_argument("--max-patients", type=int, default=None, help="Limit number of patients (for debugging)")
    p.add_argument("--no-auto-download", action="store_true", help="Disable HF auto-download")
    return p.parse_args()


def detect_device(override: str | None) -> torch.device:
    if override:
        return torch.device(override)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def derive_model_name(args) -> str:
    if args.backbone == "cinema":
        return "cinema_pretrained"
    if args.backbone == "sam":
        return args.sam_model_id.split("/")[-1].replace("-", "_")
    return args.dinov3_model_name


def eval_mode_tag(eval_mode: str, freeze_backbone: bool) -> str:
    if eval_mode == "logreg":
        return "logreg"
    return "ftfrozen" if freeze_backbone else "ftfull"


def load_backbone(args, device):
    """Load backbone model and return (backbone, embed_dim, sam_image_processor_or_None)."""
    sam_image_processor = None
    auto_download = not args.no_auto_download

    if args.backbone == "cinema":
        from cinema import CineMA

        args.hf_cache_dir.mkdir(parents=True, exist_ok=True)
        backbone = CineMA.from_pretrained(
            cache_dir=str(args.hf_cache_dir),
            local_files_only=not auto_download,
        )
        embed_dim = backbone.enc_down_dict["sax"].patch_embed.proj.out_features

    elif args.backbone == "sam":
        from transformers import SamImageProcessor, SamModel

        args.hf_cache_dir.mkdir(parents=True, exist_ok=True)
        sam_image_processor = SamImageProcessor.from_pretrained(
            args.sam_model_id,
            cache_dir=str(args.hf_cache_dir),
            local_files_only=not auto_download,
        )
        backbone = SamModel.from_pretrained(
            args.sam_model_id,
            cache_dir=str(args.hf_cache_dir),
            local_files_only=not auto_download,
        )
        embed_dim = backbone.config.vision_config.hidden_size

    else:  # dinov3
        if args.dinov3_weights_path:
            weights_path = args.dinov3_weights_path
        else:
            # Match exact name or name with hash suffix (e.g. dinov3_vitl16-8aa4cbdd.pth)
            candidates = glob.glob(f"model_weights/{args.dinov3_model_name}*.pth")
            weights_path = candidates[0] if candidates else None
        if weights_path is None or not Path(weights_path).exists():
            print(f"Weights not found for {args.dinov3_model_name}, downloading from default source...")
            weights_path = None
        backbone = torch.hub.load(
            args.dinov3_repo_dir,
            args.dinov3_model_name,
            source="local",
            weights=weights_path,
        )
        embed_dim = DINOV3_EMBED_DIMS[args.dinov3_model_name]

    backbone.eval().to(device)
    return backbone, embed_dim, sam_image_processor



def build_results_dict(
    args, model_name, embed_dim, best_hyperparam, sweep_results,
    test_metrics, test_labels_eval, test_pids_eval,
    macro_auc, per_class_auc, binary_metrics,
) -> dict:
    """Build a JSON-serialisable results dictionary."""

    def to_list(x):
        if isinstance(x, (np.ndarray, torch.Tensor)):
            return x.tolist()
        return x

    return {
        "config": {
            "backbone": args.backbone,
            "model_name": model_name,
            "eval_mode": args.eval_mode,
            "freeze_backbone": args.freeze_backbone,
            "pooling": args.pooling,
            "embed_dim": embed_dim,
            "best_hyperparam": best_hyperparam,
            "hyperparam_name": "C" if args.eval_mode == "logreg" else "lr",
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
                for cls_name, auc_val in zip(PATHOLOGY_CLASSES.keys(), per_class_auc)
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
    tag = eval_mode_tag(args.eval_mode, args.freeze_backbone)
    base_name = f"{model_name}_{tag}_{args.pooling}"
    if args.max_patients:
        base_name += "_smoke"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{base_name}.json"
    if json_path.exists():
        print(f"Skipping: {json_path} already exists.")
        return

    device = detect_device(args.device)
    print(f"Device: {device}")
    print(f"Backbone: {args.backbone} ({model_name})")
    mode_desc = args.eval_mode if args.eval_mode == "logreg" else f"finetune (freeze={args.freeze_backbone})"
    print(f"Eval mode: {mode_desc}")
    print(f"Pooling: {args.pooling}")

    # ── Load metadata and datasets ──
    train_meta_df = pd.read_csv(args.data_dir / "train_metadata.csv")
    test_meta_df = pd.read_csv(args.data_dir / "test_metadata.csv")
    if args.max_patients:
        train_meta_df = train_meta_df.head(args.max_patients)
        test_meta_df = test_meta_df.head(args.max_patients)
    print(f"Train: {len(train_meta_df)} patients, Test: {len(test_meta_df)} patients")

    transform = ScaleIntensityd(keys="sax_image", factor=1 / 255, channel_wise=False)
    train_cinema = EndDiastoleEndSystoleDataset(
        data_dir=args.data_dir / "train", meta_df=train_meta_df, views="sax", transform=transform,
    )
    test_cinema = EndDiastoleEndSystoleDataset(
        data_dir=args.data_dir / "test", meta_df=test_meta_df, views="sax", transform=transform,
    )

    # ── Load backbone ──
    backbone, embed_dim, sam_image_processor = load_backbone(args, device)
    print(f"Loaded backbone: embed_dim={embed_dim}")

    # Freeze if needed
    if args.eval_mode == "logreg" or args.freeze_backbone:
        for p in backbone.parameters():
            p.requires_grad = False

    # ── Pathology maps ──
    train_pathology_map = get_pathology_map(train_meta_df)
    test_pathology_map = get_pathology_map(test_meta_df)

    # ── Feature caching (logreg only) ──
    if args.eval_mode == "logreg":
        cls_cache_dir = args.cls_cache_dir or Path(f"cls_feature_cache/{model_name}/{args.pooling}")

        if args.backbone == "cinema":
            cache_fn = lambda m, ds, cd, dev: cache_cinema_cls_features(m, ds, cd, device=dev, pooling=args.pooling)
        elif args.backbone == "sam":
            cache_fn = lambda m, ds, cd, dev: cache_sam_cls_features(m, sam_image_processor, ds, cd, device=dev)
        else:
            cache_fn = lambda m, ds, cd, dev: cache_cls_features(m, ds, cd, device=dev, pooling=args.pooling)

        print("Caching training features...")
        train_manifest = cache_fn(backbone, train_cinema, cls_cache_dir / "train", device)
        print("Caching test features...")
        test_manifest = cache_fn(backbone, test_cinema, cls_cache_dir / "test", device)

        train_cls = load_cached_cls_features(train_manifest)
        test_cls = load_cached_cls_features(test_manifest)

        train_features, train_labels, train_pids = build_patient_features(train_cls, train_pathology_map)
        test_features, test_labels, test_pids = build_patient_features(test_cls, test_pathology_map)
        print(f"Feature shape: {train_features.shape}")

    # ── Train ──
    print("Training...")
    if args.eval_mode == "logreg":
        best_C, final_model, sweep_results = sweep_C_and_train(
            train_features, train_labels, n_folds=10,
        )
        best_hyperparam = best_C
        print(f"Best C = {best_C:.4g}")
    else:
        image_proc = sam_image_processor if args.backbone == "sam" else None
        best_lr, backbone, ft_head, sweep_results, ft_scaler = finetune_sweep_and_train(
            backbone=backbone,
            cinema_dataset=train_cinema,
            pathology_map=train_pathology_map,
            embed_dim=embed_dim,
            device=device,
            backbone_type=args.backbone,
            freeze_backbone=args.freeze_backbone,
            image_processor=image_proc,
            n_folds=10,
            pooling=args.pooling,
        )
        best_hyperparam = best_lr
        print(f"Best LR = {best_lr:.4g}")

    # ── Evaluate 5-way ──
    print("Evaluating...")
    if args.eval_mode == "logreg":
        test_metrics = evaluate_classification(final_model, test_features, test_labels)
        test_pids_eval = test_pids
        test_labels_eval = test_labels
    else:
        image_proc = sam_image_processor if args.backbone == "sam" else None
        test_metrics = evaluate_finetune_classification(
            backbone, ft_head, test_cinema, test_pathology_map,
            device, args.backbone, image_processor=image_proc,
            scaler=ft_scaler, pooling=args.pooling,
        )
        test_pids_eval = test_metrics["pids"]
        test_labels_eval = torch.tensor(test_metrics["labels"], dtype=torch.long)

    print(f"5-way Accuracy: {test_metrics['accuracy']:.4f}, Macro F1: {test_metrics['macro_f1']:.4f}")

    # ── ROC AUC ──
    y_true_bin = label_binarize(test_labels_eval.numpy(), classes=list(range(NUM_PATHOLOGIES)))
    y_prob = test_metrics["probabilities"]
    macro_auc = roc_auc_score(y_true_bin, y_prob, multi_class="ovr", average="macro")
    per_class_auc = roc_auc_score(y_true_bin, y_prob, multi_class="ovr", average=None)
    print(f"Macro ROC AUC: {macro_auc:.4f}")

    # ── Binary detection ──
    binary_metrics = evaluate_binary_detection(test_metrics["probabilities"], test_labels_eval)
    print(f"Binary Acc: {binary_metrics['accuracy']:.4f}, F1: {binary_metrics['f1']:.4f}")

    # ── Save results ──
    results = build_results_dict(
        args, model_name, embed_dim, best_hyperparam, sweep_results,
        test_metrics, test_labels_eval, test_pids_eval,
        macro_auc, per_class_auc, binary_metrics,
    )
    json_path.write_text(json.dumps(results, indent=2))
    print(f"Saved results: {json_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
