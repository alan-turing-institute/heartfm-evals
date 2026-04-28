#!/usr/bin/env python3
"""Profile the feature extraction loop to diagnose CPU vs GPU bottlenecks.

Instruments each phase of the per-slice pipeline separately:
  - data loading (dataset.__getitem__)
  - CPU preprocessing (PIL conversion + processor call)
  - GPU upload (pixel_values.to(device))
  - GPU forward (vision_encoder)
  - GPU→CPU transfer (.cpu())

Prints per-phase mean ± std timing, thread counts, and a summary.

Usage:
    python scripts/segmentation/profiling/profile_feature_extraction.py \\
        --backbone sam2 --sam2-model-id facebook/sam2.1-hiera-tiny \\
        --dataset acdc --n-samples 20
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from heartfm_evals.backbones import load_backbone
from heartfm_evals.constants import GRID_SIZE, SAX_TARGET_DEPTH
from heartfm_evals.data import load_segmentation_datasets
from heartfm_evals.device import detect_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Feature extraction profiler")
    p.add_argument("--backbone", required=True, choices=["dinov3", "cinema", "sam2"])
    p.add_argument("--dataset", default="acdc", choices=["acdc", "mnm", "mnm2"])
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--dinov3-model-name", default="dinov3_vitb16")
    p.add_argument("--dinov3-repo-dir", default="models/dinov3/")
    p.add_argument("--sam2-model-id", default="facebook/sam2.1-hiera-tiny")
    p.add_argument("--hf-cache-dir", type=Path, default=Path("model_weights/hf"))
    p.add_argument(
        "--n-samples",
        type=int,
        default=20,
        help="Number of dataset samples to profile (default: 20)",
    )
    p.add_argument("--device", default=None)
    p.add_argument("--no-auto-download", action="store_true")
    args = p.parse_args()

    if args.data_dir is None:
        args.data_dir = Path(f"data/heartfm/processed/{args.dataset}")
    return args


# ── Timing helpers ──────────────────────────────────────────────────────────────

def cuda_sync_time() -> float:
    """Return wall time after synchronising CUDA (so GPU ops are fully counted)."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


class PhaseTimer:
    """Accumulates per-phase timing across many iterations."""

    def __init__(self, phases: list[str]):
        self.phases = phases
        self._times: dict[str, list[float]] = {p: [] for p in phases}
        self._start: float | None = None
        self._phase: str | None = None

    def start(self, phase: str) -> None:
        # Finish any in-flight GPU work before measuring the boundary
        t = cuda_sync_time()
        if self._phase is not None:
            self._times[self._phase].append(t - self._start)  # type: ignore[operator]
        self._phase = phase
        self._start = t

    def stop(self) -> None:
        if self._phase is not None:
            t = cuda_sync_time()
            self._times[self._phase].append(t - self._start)  # type: ignore[operator]
            self._phase = None
            self._start = None

    def summary(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for phase, times in self._times.items():
            if not times:
                continue
            out[phase] = {
                "n": len(times),
                "mean_ms": mean(times) * 1000,
                "std_ms": stdev(times) * 1000 if len(times) > 1 else 0.0,
                "total_s": sum(times),
            }
        return out


# ── Per-backbone profiling loops ────────────────────────────────────────────────

def profile_sam2(backbone, config, dataset, n_samples, device):
    processor = config["sam2_processor"]
    layer_indices = config["layer_indices"]
    target_depth = SAX_TARGET_DEPTH
    grid_size = GRID_SIZE

    timer = PhaseTimer(["data_load", "cpu_preproc", "gpu_upload", "gpu_forward", "cpu_transfer"])
    n = min(n_samples, len(dataset))

    for sample_idx in range(n):
        timer.start("data_load")
        sample = dataset[sample_idx]
        vol = sample["sax_image"]  # (1, H, W, z)
        n_slices = int(sample["n_slices"])
        if vol.shape[-1] < target_depth:
            vol = F.pad(vol, (0, target_depth - vol.shape[-1]))

        for z in range(min(n_slices, target_depth)):
            timer.start("cpu_preproc")
            image_2d = vol[0, :, :, z]
            img_np = (image_2d.clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
            pil = Image.fromarray(img_np, mode="L").convert("RGB")
            proc = processor(images=pil, return_tensors="pt")
            pixel_values = proc["pixel_values"]

            timer.start("gpu_upload")
            pixel_values = pixel_values.to(device)

            timer.start("gpu_forward")
            with torch.inference_mode():
                enc_out = backbone.vision_encoder(pixel_values, output_hidden_states=True)
                hidden_states = enc_out.hidden_states

            timer.start("cpu_transfer")
            for idx in layer_indices:
                feat = hidden_states[idx].permute(0, 3, 1, 2)
                feat = F.interpolate(feat, size=(grid_size, grid_size), mode="bilinear", align_corners=False)
                _ = feat.squeeze(0).cpu()

        print(f"  sample {sample_idx + 1}/{n} done ({n_slices} slices)")

    timer.stop()
    return timer.summary()


def profile_dinov3(backbone, config, dataset, n_samples, device):
    from heartfm_evals.features import preprocess_slice

    layer_indices = config["layer_indices"]
    target_depth = SAX_TARGET_DEPTH

    timer = PhaseTimer(["data_load", "cpu_preproc", "gpu_upload", "gpu_forward", "cpu_transfer"])
    n = min(n_samples, len(dataset))

    for sample_idx in range(n):
        timer.start("data_load")
        sample = dataset[sample_idx]
        vol = sample["sax_image"]
        n_slices = int(sample["n_slices"])
        if vol.shape[-1] < target_depth:
            vol = F.pad(vol, (0, target_depth - vol.shape[-1]))

        for z in range(min(n_slices, target_depth)):
            timer.start("cpu_preproc")
            image_2d = vol[0, :, :, z]
            img = preprocess_slice(image_2d)  # (1, 3, H, W) CPU tensor

            timer.start("gpu_upload")
            img = img.to(device)

            timer.start("gpu_forward")
            with torch.inference_mode():
                feats = backbone.get_intermediate_layers(img, n=list(layer_indices), reshape=True, norm=True)

            timer.start("cpu_transfer")
            _ = torch.cat(feats, dim=1).squeeze(0).cpu()

        print(f"  sample {sample_idx + 1}/{n} done ({n_slices} slices)")

    timer.stop()
    return timer.summary()


def profile_cinema(backbone, config, dataset, n_samples, device):
    target_depth = SAX_TARGET_DEPTH

    timer = PhaseTimer(["data_load", "gpu_upload", "gpu_forward", "cpu_transfer"])
    n = min(n_samples, len(dataset))

    for sample_idx in range(n):
        timer.start("data_load")
        sample = dataset[sample_idx]
        vol = sample["sax_image"]
        if vol.shape[-1] < target_depth:
            vol = F.pad(vol, (0, target_depth - vol.shape[-1]))
        batch_input = vol.unsqueeze(0)  # (1, 1, H, W, Z)

        timer.start("gpu_upload")
        batch_input = batch_input.to(device=device, dtype=torch.float32)

        timer.start("gpu_forward")
        with torch.inference_mode():
            skips_list, x_view = backbone.enc_down_dict["sax"](batch_input, mask=None)
            x = backbone.encoder(x_view)

        timer.start("cpu_transfer")
        _ = x.cpu()
        for skip in skips_list:
            _ = skip.cpu()

        print(f"  sample {sample_idx + 1}/{n} done")

    timer.stop()
    return timer.summary()


# ── Main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    device = detect_device(args.device)

    print("=" * 60)
    print("Feature extraction profiler")
    print("=" * 60)
    print(f"Backbone:          {args.backbone}")
    print(f"Dataset:           {args.dataset}")
    print(f"Device:            {device}")
    print(f"Samples:           {args.n_samples}")
    print(f"torch num_threads: {torch.get_num_threads()}")
    print(f"OMP_NUM_THREADS:   {os.environ.get('OMP_NUM_THREADS', '(not set)')}")
    print(f"MKL_NUM_THREADS:   {os.environ.get('MKL_NUM_THREADS', '(not set)')}")
    if torch.cuda.is_available():
        print(f"GPU:               {torch.cuda.get_device_name(0)}")
    print()

    # Load backbone
    backbone_kwargs: dict = {}
    if args.backbone == "dinov3":
        backbone_kwargs["dinov3_model_name"] = args.dinov3_model_name
        backbone_kwargs["dinov3_repo_dir"] = args.dinov3_repo_dir
    elif args.backbone == "sam2":
        backbone_kwargs["sam2_model_id"] = args.sam2_model_id
        backbone_kwargs["hf_cache_dir"] = str(args.hf_cache_dir)
        backbone_kwargs["auto_download"] = not args.no_auto_download

    print("Loading backbone...")
    backbone, config = load_backbone(args.backbone, device, **backbone_kwargs)
    print("Backbone loaded.\n")

    # Load dataset (train split only, we only need samples to profile)
    train_ds, _, _, _, _, _ = load_segmentation_datasets(args.dataset, args.data_dir)
    print(f"Dataset loaded: {len(train_ds)} train samples\n")

    # Warm-up: one sample, not timed
    print("Warming up (1 sample, untimed)...")
    _ = train_ds[0]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print("Warm-up done.\n")

    # Profile
    print(f"Profiling {args.n_samples} samples...")
    if args.backbone == "sam2":
        summary = profile_sam2(backbone, config, train_ds, args.n_samples, device)
    elif args.backbone == "dinov3":
        summary = profile_dinov3(backbone, config, train_ds, args.n_samples, device)
    else:
        summary = profile_cinema(backbone, config, train_ds, args.n_samples, device)

    # Print summary
    print()
    print("=" * 60)
    print("Timing summary (per slice)")
    print("=" * 60)
    total_mean_ms = 0.0
    for phase, stats in summary.items():
        print(
            f"  {phase:<16s}  {stats['mean_ms']:7.1f} ms  ±{stats['std_ms']:6.1f} ms"
            f"  (total {stats['total_s']:.1f}s, n={stats['n']})"
        )
        total_mean_ms += stats["mean_ms"]
    print(f"  {'TOTAL':<16s}  {total_mean_ms:7.1f} ms/slice")
    print()

    # Percentage breakdown
    if total_mean_ms > 0:
        print("Phase breakdown:")
        for phase, stats in summary.items():
            pct = 100.0 * stats["mean_ms"] / total_mean_ms
            bar = "#" * int(pct / 2)
            print(f"  {phase:<16s}  {pct:5.1f}%  {bar}")
    print()


if __name__ == "__main__":
    main()
