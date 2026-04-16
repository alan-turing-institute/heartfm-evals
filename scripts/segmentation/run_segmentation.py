#!/usr/bin/env python3
"""Unified segmentation evaluation script.

Supports all backbone × decoder × dataset combinations:
    --backbone {dinov3,cinema,sam2}
    --decoder  {linear_probe,conv_decoder,unetr}
    --dataset  {acdc,mnm,mnm2}

Usage examples:
    python scripts/segmentation/run_segmentation.py --backbone dinov3 --decoder linear_probe --dataset acdc
    python scripts/segmentation/run_segmentation.py --backbone cinema --decoder unetr --dataset mnm
    python scripts/segmentation/run_segmentation.py --backbone sam2 --decoder conv_decoder --dataset mnm2 --sam2-model-id facebook/sam2.1-hiera-tiny
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from heartfm_evals.backbones import load_backbone
from heartfm_evals.caching import (
    CachedCinemaVolumeDataset,
    CachedFeatureDataset,
    CachedVolumeDataset,
    cache_cinema_2d_features,
    cache_cinema_volume_features,
    cache_dino_volume_features,
    cache_features,
    cache_sam2_2d_features,
    cache_sam_volume_features,
)
from heartfm_evals.constants import CLASS_NAMES, NUM_CLASSES
from heartfm_evals.data import load_segmentation_datasets
from heartfm_evals.decoders import get_decoder
from heartfm_evals.device import detect_device
from heartfm_evals.losses import CombinedLoss, MaskedVolumeLoss, WeightedCombinedLoss
from heartfm_evals.reproducibility import set_seed
from heartfm_evals.training import (
    evaluate,
    evaluate_per_sample,
    evaluate_vol,
    evaluate_vol_per_slice,
    evaluate_vol_per_sample,
    train_segmentation,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified segmentation evaluation")
    p.add_argument("--backbone", required=True, choices=["dinov3", "cinema", "sam2"])
    p.add_argument(
        "--decoder", required=True, choices=["linear_probe", "conv_decoder", "unetr"]
    )
    p.add_argument("--dataset", default="acdc", choices=["acdc", "mnm", "mnm2"])
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--cache-dir", type=Path, default=None)

    # Model selection
    p.add_argument("--dinov3-model-name", default="dinov3_vits16")
    p.add_argument("--dinov3-repo-dir", default="models/dinov3/")
    p.add_argument("--dinov3-weights-path", default=None)
    p.add_argument("--sam2-model-id", default="facebook/sam2.1-hiera-base-plus")
    p.add_argument("--hf-cache-dir", type=Path, default=Path("model_weights/hf"))

    # Layer selection
    p.add_argument(
        "--use-layers",
        type=int,
        nargs="+",
        default=None,
        help="Layer indices for probe (default: last layer for linear_probe, all for others)",
    )

    # Training
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--n-epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument(
        "--no-auto-download", action="store_true", help="Disable HF auto-download"
    )

    args = p.parse_args()

    # Defaults
    if args.data_dir is None:
        args.data_dir = Path(f"data/heartfm/processed/{args.dataset}")
    if args.output_dir is None:
        args.output_dir = Path(f"results/segmentation/{args.dataset}")

    return args


def derive_model_name(args: argparse.Namespace) -> str:
    if args.backbone == "cinema":
        return "cinema_pretrained"
    if args.backbone == "sam2":
        return args.sam2_model_id.split("/")[-1].replace("-", "_").replace(".", "_")
    return args.dinov3_model_name


def derive_cache_dir(args: argparse.Namespace, model_name: str) -> Path:
    if args.cache_dir is not None:
        return args.cache_dir
    suffix = "unetr3d" if args.decoder == "unetr" else ""
    name = f"{model_name}{'_' + suffix if suffix else ''}"
    return Path(f"feature_cache/{args.dataset}/{name}")


def compute_class_weights(
    train_manifest: list[dict], device: torch.device
) -> torch.Tensor:
    """Compute inverse-frequency class weights from cached training labels."""
    class_counts = torch.zeros(NUM_CLASSES, dtype=torch.long)
    for entry in train_manifest:
        data = torch.load(entry["path"], weights_only=True)
        label = data["label"]
        # For volume caches, exclude z-padded slices (label 0 = BG)
        if "n_slices" in data:
            ns = data["n_slices"]
            label = label[..., :ns]
        class_counts += torch.bincount(label.long().reshape(-1), minlength=NUM_CLASSES)

    class_weights = class_counts.sum().float() / (
        NUM_CLASSES * class_counts.clamp_min(1).float()
    )
    class_weights[0] = class_weights[0] * 0.5  # downweight background
    class_weights = class_weights / class_weights.mean()
    return class_weights.to(device)


def metric_columns(metric_level: str) -> list[str]:
    """Return output columns for a metrics CSV."""
    base_columns = ["pid", "frame", "is_ed"]
    if metric_level == "volume":
        base_columns.append("n_slices")
    elif metric_level == "slice":
        base_columns.append("z_idx")
    else:
        raise ValueError(f"Unknown metric level: {metric_level}")
    dice_columns = [f"dice_{CLASS_NAMES.get(c, f'C{c}')}" for c in range(NUM_CLASSES)]
    return [*base_columns, *dice_columns, "macro_dice"]


def save_metrics_csv(
    rows: list[dict],
    output_path: Path,
    metric_level: str,
) -> int:
    """Write metrics rows to CSV in a deterministic order."""
    df = pd.DataFrame(rows, columns=metric_columns(metric_level))
    sort_columns = ["pid", "frame"] if metric_level == "volume" else ["pid", "frame", "z_idx"]
    if not df.empty:
        df = df.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    df.to_csv(output_path, index=False)
    return len(df)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = detect_device(args.device)
    model_name = derive_model_name(args)
    cache_dir = derive_cache_dir(args, model_name)

    base_name = f"{model_name}_{args.decoder}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = args.output_dir / f"{base_name}_{timestamp}.json"
    if json_path.exists():
        print(f"Skipping: {json_path} already exists.")
        return

    print(f"Dataset: {args.dataset}")
    print(f"Device: {device}")
    print(f"Backbone: {args.backbone} ({model_name})")
    print(f"Decoder: {args.decoder}")
    print(f"Cache dir: {cache_dir}")

    is_volume = args.decoder == "unetr"

    # ── Load backbone ──
    backbone_kwargs: dict = {}
    if args.backbone == "dinov3":
        backbone_kwargs["dinov3_model_name"] = args.dinov3_model_name
        backbone_kwargs["dinov3_repo_dir"] = args.dinov3_repo_dir
        if args.dinov3_weights_path:
            backbone_kwargs["dinov3_weights_path"] = args.dinov3_weights_path
    elif args.backbone == "sam2":
        backbone_kwargs["sam2_model_id"] = args.sam2_model_id
        backbone_kwargs["hf_cache_dir"] = str(args.hf_cache_dir)
        backbone_kwargs["auto_download"] = not args.no_auto_download
    elif args.backbone == "cinema":
        backbone_kwargs["hf_cache_dir"] = str(args.hf_cache_dir)
        backbone_kwargs["auto_download"] = not args.no_auto_download

    backbone, config = load_backbone(args.backbone, device, **backbone_kwargs)
    embed_dim = config["embed_dim"]
    layer_indices = tuple(config.get("layer_indices", (3, 6, 9, 11)))

    # Determine which layers to use for probe
    if args.use_layers is not None:
        use_layers = tuple(args.use_layers)
    elif args.backbone == "cinema" and not is_volume:
        # CineMA 2D features are single-layer (ViT output only)
        use_layers = (0,)
    elif args.decoder == "linear_probe":
        use_layers = (layer_indices[-1],)  # last layer only
    else:
        use_layers = layer_indices

    print(f"Embed dim: {embed_dim}")
    print(f"Cache layer indices: {layer_indices}")
    print(f"Probe layer indices: {use_layers}")

    # ── Load datasets ──
    train_ds, val_ds, test_ds, train_meta, val_meta, test_meta = (
        load_segmentation_datasets(
            args.dataset,
            args.data_dir,
            split_seed=args.seed,
        )
    )
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    # ── Cache features ──
    sam_processor = config.get("processor")

    if is_volume:
        # 3D volume caching
        if args.backbone == "dinov3":
            train_manifest = cache_dino_volume_features(
                backbone, train_ds, cache_dir / "train", layer_indices, device
            )
            val_manifest = cache_dino_volume_features(
                backbone, val_ds, cache_dir / "val", layer_indices, device
            )
            test_manifest = cache_dino_volume_features(
                backbone, test_ds, cache_dir / "test", layer_indices, device
            )
        elif args.backbone == "cinema":
            train_manifest = cache_cinema_volume_features(
                backbone, train_ds, cache_dir / "train", device
            )
            val_manifest = cache_cinema_volume_features(
                backbone, val_ds, cache_dir / "val", device
            )
            test_manifest = cache_cinema_volume_features(
                backbone, test_ds, cache_dir / "test", device
            )
        else:  # sam2
            train_manifest = cache_sam_volume_features(
                backbone,
                sam_processor,
                train_ds,
                cache_dir / "train",
                layer_indices,
                device,
            )
            val_manifest = cache_sam_volume_features(
                backbone,
                sam_processor,
                val_ds,
                cache_dir / "val",
                layer_indices,
                device,
            )
            test_manifest = cache_sam_volume_features(
                backbone,
                sam_processor,
                test_ds,
                cache_dir / "test",
                layer_indices,
                device,
            )
    else:
        # 2D slice caching
        if args.backbone == "dinov3":
            train_manifest = cache_features(
                backbone, train_ds, cache_dir / "train", layer_indices, device
            )
            val_manifest = cache_features(
                backbone, val_ds, cache_dir / "val", layer_indices, device
            )
            test_manifest = cache_features(
                backbone, test_ds, cache_dir / "test", layer_indices, device
            )
        elif args.backbone == "cinema":
            train_manifest = cache_cinema_2d_features(
                backbone, train_ds, cache_dir / "train", device
            )
            val_manifest = cache_cinema_2d_features(
                backbone, val_ds, cache_dir / "val", device
            )
            test_manifest = cache_cinema_2d_features(
                backbone, test_ds, cache_dir / "test", device
            )
        else:  # sam2
            train_manifest = cache_sam2_2d_features(
                backbone, sam_processor, train_ds, cache_dir / "train", device
            )
            val_manifest = cache_sam2_2d_features(
                backbone, sam_processor, val_ds, cache_dir / "val", device
            )
            test_manifest = cache_sam2_2d_features(
                backbone, sam_processor, test_ds, cache_dir / "test", device
            )

    print(
        f"Cached: train={len(train_manifest)}, val={len(val_manifest)}, "
        f"test={len(test_manifest)}"
    )

    # ── Build DataLoaders ──
    if is_volume:
        if args.backbone == "cinema":
            n_conv_skips = len(config.get("enc_conv_chans", (64, 128)))
            train_cached = CachedCinemaVolumeDataset(
                train_manifest, n_conv_skips=n_conv_skips
            )
            val_cached = CachedCinemaVolumeDataset(
                val_manifest, n_conv_skips=n_conv_skips
            )
            test_cached = CachedCinemaVolumeDataset(
                test_manifest, n_conv_skips=n_conv_skips
            )
        else:
            train_cached = CachedVolumeDataset(train_manifest, layer_indices)
            val_cached = CachedVolumeDataset(val_manifest, layer_indices)
            test_cached = CachedVolumeDataset(test_manifest, layer_indices)
    else:
        train_cached = CachedFeatureDataset(train_manifest)
        val_cached = CachedFeatureDataset(val_manifest)
        test_cached = CachedFeatureDataset(test_manifest)

    bs = args.batch_size
    if is_volume and bs > 1:
        print(
            "Warning: 3D volume segmentation is memory-intensive. "
            "If you encounter OOM errors, try reducing --batch-size."
        )
    g = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_cached, batch_size=bs, shuffle=True, num_workers=0, generator=g
    )
    val_loader = DataLoader(val_cached, batch_size=bs, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_cached, batch_size=bs, shuffle=False, num_workers=0)

    # ── Build decoder ──
    decoder_kwargs: dict = {}
    if args.decoder == "linear_probe":
        decoder_kwargs["dropout"] = args.dropout
        # Only set cached_layers when cache has more layers than the probe uses.
        # Skip for CineMA 2D: its cache is a single tensor, not multi-layer.
        if use_layers != layer_indices and not (
            args.backbone == "cinema" and not is_volume
        ):
            decoder_kwargs["cached_layers"] = layer_indices
    decoder = get_decoder(
        decoder_type=args.decoder,
        backbone_type=args.backbone,
        embed_dim=embed_dim,
        layer_indices=use_layers,
        **decoder_kwargs,
    ).to(device)

    n_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"Decoder trainable params: {n_params:,}")

    # ── Loss ──
    if is_volume:
        class_weights = compute_class_weights(train_manifest, device)
        print(f"Class weights: {class_weights.tolist()}")
        criterion = MaskedVolumeLoss(class_weights)
    elif args.decoder == "conv_decoder":
        class_weights = compute_class_weights(train_manifest, device)
        print(f"Class weights: {class_weights.tolist()}")
        criterion = WeightedCombinedLoss(class_weights)
    else:
        criterion = CombinedLoss()

    # ── Optimizer & Scheduler ──
    optimizer = torch.optim.AdamW(
        decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.n_epochs
    )

    # ── Train ──
    print("Training...")
    train_result = train_segmentation(
        model=decoder,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        n_epochs=args.n_epochs,
        patience=args.patience,
        is_volume=is_volume,
    )

    # ── Evaluate on test set ──
    eval_fn = evaluate_vol if is_volume else evaluate
    test_metrics = eval_fn(decoder, test_loader, device)
    per_sample_eval_fn = evaluate_vol_per_sample if is_volume else evaluate_per_sample
    per_sample_rows = per_sample_eval_fn(decoder, test_loader, device)
    per_sample_path = json_path.with_name(f"{json_path.stem}_per_sample.csv")
    metric_level = "volume" if is_volume else "slice"
    n_test_samples = save_metrics_csv(
        per_sample_rows,
        per_sample_path,
        metric_level=metric_level,
    )
    per_slice_path: Path | None = None
    n_test_slices: int | None = None
    if is_volume:
        per_slice_rows = evaluate_vol_per_slice(decoder, test_loader, device)
        per_slice_path = json_path.with_name(f"{json_path.stem}_per_slice.csv")
        n_test_slices = save_metrics_csv(
            per_slice_rows,
            per_slice_path,
            metric_level="slice",
        )

    print("\nTest Results:")
    print("Per-class Dice:")
    for name, d in test_metrics["per_class_dice"].items():
        print(f"  {name:>3s}: {d:.4f}")
    print(f"Macro Dice (excl. BG): {test_metrics['macro_dice']:.4f}")
    print(f"Per-sample metrics rows: {n_test_samples}")
    print(f"Per-sample metrics saved to: {per_sample_path}")
    if per_slice_path is not None and n_test_slices is not None:
        print(f"Per-slice metrics rows: {n_test_slices}")
        print(f"Per-slice metrics saved to: {per_slice_path}")

    # ── Save results ──
    results = {
        "config": {
            "dataset": args.dataset,
            "backbone": args.backbone,
            "model_name": model_name,
            "decoder": args.decoder,
            "embed_dim": embed_dim,
            "cache_layers": list(layer_indices),
            "probe_layers": list(use_layers),
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "n_epochs": args.n_epochs,
            "patience": args.patience,
            "seed": args.seed,
            "trainable_params": n_params,
        },
        "per_class_dice": test_metrics["per_class_dice"],
        "macro_dice": test_metrics["macro_dice"],
        "training_history": train_result["history"],
        "best_val_dice": train_result["best_val_dice"],
        "best_epoch": train_result["best_epoch"],
        "per_sample_metrics_file": per_sample_path.name,
        "n_test_samples": n_test_samples,
    }
    if per_slice_path is not None and n_test_slices is not None:
        results["per_slice_metrics_file"] = per_slice_path.name
        results["n_test_slices"] = n_test_slices
    json_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved results: {json_path}")


if __name__ == "__main__":
    main()
