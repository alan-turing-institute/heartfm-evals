"""3D UNetR decoder probes for DINOv3 and CineMA backbone evaluation.

Uses CineMA's UpsampleDecoder to decode frozen backbone features for 3D
segmentation.  Two decoder models are provided:

- ``DINOv3UNetRDecoder``: takes per-slice DINOv3 features stacked along z,
  synthesises multi-scale skip connections via bilinear upsampling.
- ``CineMAUNetRDecoder``: takes CineMA's native multi-scale conv encoder
  skips + ViT features, wired exactly like ``ConvUNetR``'s decoder.

Both enable a fair comparison: DINOv3 encoder vs CineMA encoder with an
identical 3D UpsampleDecoder.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from cinema.conv import ConvResBlock
from cinema.segmentation.convunetr import UpsampleDecoder
from torch.utils.data import Dataset
from tqdm import tqdm

from heartfm_evals.dense_linear_probe import (
    CLASS_NAMES,
    IMAGE_SIZE,
    NUM_CLASSES,
    PATCH_SIZE,
    dice_score,
    macro_dice,
    preprocess_slice,
)

# ── Constants ──────────────────────────────────────────────────────────────────
SAX_TARGET_DEPTH = 16
GRID_SIZE = IMAGE_SIZE // PATCH_SIZE  # 12 for 192 / 16


# ── Feature Extraction ─────────────────────────────────────────────────────────
@torch.inference_mode()
def extract_dino_volume_features(
    backbone: nn.Module,
    sax_volume: torch.Tensor,
    layer_indices: tuple[int, ...],
    device: torch.device | None = None,
    target_depth: int = SAX_TARGET_DEPTH,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, int]:
    """Extract per-layer DINOv3 features for a SAX volume, stacked along z.

    For each slice, runs the frozen backbone and extracts features at the
    specified layers.  Features from all slices are stacked to form 3D volumes.
    Follows the same z-padding convention as CineMA (pad to *target_depth*).

    Args:
        backbone: Frozen DINOv3 backbone in eval mode.
        sax_volume: (1, H, W, z) tensor in [0, 1].
        layer_indices: Which intermediate layers to extract.
        device: Device for inference.
        target_depth: Pad z to this depth.

    Returns:
        Tuple of (features_dict, padded_image, n_slices):
            features_dict: ``{f"layer_{idx}": (embed_dim, g, g, target_depth)}``
            padded_image: ``(1, H, W, target_depth)``
            n_slices: actual number of slices before padding.
    """
    vol = sax_volume  # (1, H, W, z)
    n_slices = int(vol.shape[-1])

    if n_slices > target_depth:
        vol = vol[..., :target_depth]
        n_slices = target_depth
    if vol.shape[-1] < target_depth:
        vol = F.pad(vol, (0, target_depth - vol.shape[-1]), mode="constant", value=0.0)

    padded_image = vol  # (1, H, W, target_depth)

    per_layer: dict[int, list[torch.Tensor]] = {idx: [] for idx in layer_indices}

    for z in range(target_depth):
        image_2d = vol[0, :, :, z]  # (H, W)
        img = preprocess_slice(image_2d)  # (1, 3, H, W)
        if device is not None:
            img = img.to(device)

        feats = backbone.get_intermediate_layers(
            img, n=list(layer_indices), reshape=True, norm=True
        )
        for i, idx in enumerate(layer_indices):
            per_layer[idx].append(feats[i].squeeze(0).cpu())  # (embed_dim, h, w)

    features_dict: dict[str, torch.Tensor] = {}
    for idx in layer_indices:
        features_dict[f"layer_{idx}"] = torch.stack(per_layer[idx], dim=-1)
        # shape: (embed_dim, grid_h, grid_w, target_depth)

    return features_dict, padded_image, n_slices


# ── Feature Caching ─────────────────────────────────────────────────────────────
def cache_dino_volume_features(
    backbone: nn.Module,
    cinema_dataset,
    cache_dir: Path,
    layer_indices: tuple[int, ...] = (3, 6, 9, 11),
    device: torch.device | None = None,
    target_depth: int = SAX_TARGET_DEPTH,
) -> list[dict]:
    """Cache volume-level DINOv3 features for all patients/frames.

    Saves one ``.pt`` file per patient+frame containing per-layer features,
    the padded image, padded label, and the actual number of slices.

    Args:
        backbone: Frozen DINOv3 backbone in eval mode.
        cinema_dataset: CineMA ``EndDiastoleEndSystoleDataset``.
        cache_dir: Root cache directory.
        layer_indices: Which intermediate layers to extract.
        device: Device for inference.
        target_depth: Pad z to this depth.

    Returns:
        List of dicts with keys: ``path``, ``pid``, ``is_ed``, ``n_slices``.
    """
    layers_tag = "layers_" + "-".join(str(i) for i in sorted(layer_indices))
    out_dir = Path(cache_dir) / layers_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    for sample_idx in tqdm(range(len(cinema_dataset)), desc="Caching volume features"):
        sample = cinema_dataset[sample_idx]
        image_3d = sample["sax_image"]  # (1, H, W, z)
        label_3d = sample["sax_label"]  # (1, H, W, z)
        n_slices = int(sample["n_slices"])
        pid = sample["pid"]
        is_ed = sample["is_ed"]
        frame = "ed" if is_ed else "es"

        fname = f"{pid}_{frame}.pt"
        fpath = out_dir / fname

        if fpath.exists():
            manifest.append(
                {"path": fpath, "pid": pid, "is_ed": is_ed, "n_slices": n_slices}
            )
            continue

        features_dict, padded_image, actual_slices = extract_dino_volume_features(
            backbone, image_3d, layer_indices, device, target_depth
        )

        # Pad label along z (padded region stays 0 = background)
        label = label_3d
        if label.shape[-1] > target_depth:
            label = label[..., :target_depth]
        elif label.shape[-1] < target_depth:
            label = F.pad(
                label, (0, target_depth - label.shape[-1]), mode="constant", value=0.0
            )

        save_dict: dict = {
            "image": padded_image,  # (1, H, W, target_depth)
            "label": label.long(),  # (1, H, W, target_depth)
            "n_slices": actual_slices,
        }
        save_dict.update(features_dict)

        torch.save(save_dict, fpath)
        manifest.append(
            {"path": fpath, "pid": pid, "is_ed": is_ed, "n_slices": actual_slices}
        )

    return manifest


# ── Dataset ─────────────────────────────────────────────────────────────────────
class CachedVolumeDataset(Dataset):
    """Loads pre-cached volume-level features from ``.pt`` files."""

    def __init__(
        self,
        manifest: list[dict],
        layer_indices: tuple[int, ...] = (3, 6, 9, 11),
    ):
        self.manifest = manifest
        self.layer_indices = layer_indices

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> dict:
        entry = self.manifest[idx]
        data = torch.load(entry["path"], weights_only=True)
        result = {
            "image": data["image"],  # (1, H, W, Z)
            "label": data["label"],  # (1, H, W, Z)
            "n_slices": data["n_slices"],
            "pid": entry["pid"],
        }
        for layer_idx in self.layer_indices:
            key = f"layer_{layer_idx}"
            result[key] = data[key]  # (embed_dim, g, g, Z)
        return result


# ── Model ───────────────────────────────────────────────────────────────────────
class DINOv3UNetRDecoder(nn.Module):
    """3D UNetR decoder for DINOv3 backbone features.

    Takes per-layer DINOv3 features (stacked along z) and raw images,
    builds multi-scale skip connections, and decodes with CineMA's
    ``UpsampleDecoder`` to produce dense 3D segmentation logits.

    Skip-connection wiring (for default ``dec_chans=(32,64,128,256,512)``):

    ==========  =========  ============  ========  ==============
    Source      Feat grid  Upsample x/y  Channels  Decoder level
    ==========  =========  ============  ========  ==============
    image       192x192    -             32        block 4 skip
    (none)      96x96      -             -         block 3 skip
    layer[0]    12->48     4x            64        block 2 skip
    layer[1]    12->24     2x            128       block 1 skip
    layer[2]    12x12      -             256       block 0 skip
    layer[3]    12->6      down 2x       512       bottleneck
    ==========  =========  ============  ========  ==============
    """

    def __init__(
        self,
        embed_dim: int = 384,
        layer_indices: tuple[int, ...] = (3, 6, 9, 11),
        dec_chans: tuple[int, ...] = (32, 64, 128, 256, 512),
        dec_patch_size: tuple[int, int, int] = (2, 2, 1),
        dec_scale_factor: tuple[int, int, int] = (2, 2, 1),
        num_classes: int = NUM_CLASSES,
        norm: str = "instance",
        dropout: float = 0.1,
    ):
        super().__init__()
        if len(layer_indices) != 4:
            msg = f"Expected 4 layer_indices, got {len(layer_indices)}"
            raise ValueError(msg)
        self.layer_indices = layer_indices

        # Image path: raw image → shallowest skip
        self.image_conv = ConvResBlock(
            n_dims=3, in_chans=1, out_chans=dec_chans[0], norm=norm
        )

        # Skip adapters: DINOv3 features -> decoder skip channels
        #   layer_indices[0]: 4x xy spatial upsample -> dec_chans[1]
        #   layer_indices[1]: 2x xy spatial upsample -> dec_chans[2]
        #   layer_indices[2]: native (12x12)          -> dec_chans[3]
        self.skip_adapters = nn.ModuleDict(
            {
                f"layer_{layer_indices[0]}": ConvResBlock(
                    n_dims=3, in_chans=embed_dim, out_chans=dec_chans[1], norm=norm
                ),
                f"layer_{layer_indices[1]}": ConvResBlock(
                    n_dims=3, in_chans=embed_dim, out_chans=dec_chans[2], norm=norm
                ),
                f"layer_{layer_indices[2]}": ConvResBlock(
                    n_dims=3, in_chans=embed_dim, out_chans=dec_chans[3], norm=norm
                ),
            }
        )
        self._skip_scales: dict[str, tuple[int, ...] | None] = {
            f"layer_{layer_indices[0]}": (4, 4, 1),
            f"layer_{layer_indices[1]}": (2, 2, 1),
            f"layer_{layer_indices[2]}": None,
        }

        # Bottleneck: deepest layer → adapt channels → downsample spatially
        self.bottleneck_adapter = ConvResBlock(
            n_dims=3, in_chans=embed_dim, out_chans=dec_chans[-1], norm=norm
        )
        self.bottleneck_down = nn.Conv3d(
            dec_chans[-1],
            dec_chans[-1],
            kernel_size=tuple(dec_scale_factor),  # type: ignore[arg-type]
            stride=tuple(dec_scale_factor),  # type: ignore[arg-type]
        )

        # CineMA UpsampleDecoder
        self.decoder = UpsampleDecoder(
            n_dims=3,
            chans=dec_chans,
            patch_size=dec_patch_size,
            scale_factor=dec_scale_factor,
            norm=norm,
            dropout=dropout,
        )

        # Prediction head
        self.pred_head = nn.Conv3d(dec_chans[0], num_classes, kernel_size=1)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass.

        Args:
            batch: Dict with keys ``"image"`` (B, 1, H, W, Z) and
                   ``"layer_{idx}"`` (B, embed_dim, g, g, Z) for each layer.

        Returns:
            logits: (B, num_classes, H, W, Z)
        """
        image = batch["image"]  # (B, 1, H, W, Z)

        # Shallowest skip: image → conv
        img_skip = self.image_conv(image)  # (B, dec_chans[0], H, W, Z)

        # Mid-level skips: optionally upsample, then adapt channels
        skips = []
        for key in [
            f"layer_{self.layer_indices[0]}",
            f"layer_{self.layer_indices[1]}",
            f"layer_{self.layer_indices[2]}",
        ]:
            feat = batch[key]  # (B, embed_dim, g, g, Z)
            scale = self._skip_scales[key]
            if scale is not None:
                feat = F.interpolate(
                    feat,
                    scale_factor=[float(s) for s in scale],
                    mode="trilinear",
                    align_corners=False,
                )
            feat = self.skip_adapters[key](feat)
            skips.append(feat)

        # Bottleneck: deepest layer → adapt + downsample
        deep = batch[f"layer_{self.layer_indices[3]}"]
        bottleneck = self.bottleneck_adapter(deep)
        bottleneck = self.bottleneck_down(bottleneck)  # (B, dec_chans[-1], 6, 6, Z)

        # Build embeddings list (UpsampleDecoder pops from the end)
        embeddings: list[torch.Tensor | None] = [
            img_skip,  # (B, 32, 192, 192, Z) — block 4 skip
            None,  # block 3 skip (no 96x96 source)
            skips[0],  # (B, 64, 48, 48, Z)  — block 2 skip
            skips[1],  # (B, 128, 24, 24, Z) — block 1 skip
            skips[2],  # (B, 256, 12, 12, Z) — block 0 skip
            bottleneck,  # (B, 512, 6, 6, Z)  — initial x (popped first)
        ]

        x = self.decoder(embeddings)  # (B, dec_chans[0], H, W, Z)
        return self.pred_head(x)  # (B, num_classes, H, W, Z)


# ── CineMA Feature Extraction ──────────────────────────────────────────────────
def _pad_volume_z(vol: torch.Tensor, target_depth: int) -> tuple[torch.Tensor, int]:
    """Pad or truncate the last (z) dimension to *target_depth*.

    Returns (padded_volume, actual_slices_before_padding).
    """
    n_slices = int(vol.shape[-1])
    if n_slices > target_depth:
        vol = vol[..., :target_depth]
        n_slices = target_depth
    elif vol.shape[-1] < target_depth:
        vol = F.pad(vol, (0, target_depth - vol.shape[-1]), mode="constant", value=0.0)
    return vol, n_slices


@torch.inference_mode()
def extract_cinema_volume_features(
    backbone: nn.Module,
    sax_volume: torch.Tensor,
    device: torch.device | None = None,
    target_depth: int = SAX_TARGET_DEPTH,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, int]:
    """Extract CineMA conv-encoder skips and ViT features for a SAX volume.

    Runs the frozen CineMA encoder (``enc_down_dict["sax"]`` + ``encoder``)
    and returns the multi-scale conv skips together with the reshaped ViT
    output.  These are the ingredients needed by ``CineMAUNetRDecoder``.

    Args:
        backbone: Frozen CineMA model (``CineMA.from_pretrained()``).
        sax_volume: ``(1, H, W, z)`` tensor in [0, 1].
        device: Device for inference.
        target_depth: Pad z to this depth.

    Returns:
        Tuple of ``(features_dict, padded_image, n_slices)``:
            features_dict: ``{"conv_skip_0": ..., ..., "vit_features": ...}``
            padded_image: ``(1, H, W, target_depth)``
            n_slices: actual number of slices before padding.
    """
    vol, n_slices = _pad_volume_z(sax_volume, target_depth)

    # CineMA expects (B, 1, H, W, Z) — add batch dim
    batch_input = vol.unsqueeze(0)  # (1, 1, H, W, Z)
    if device is not None:
        batch_input = batch_input.to(device=device, dtype=torch.float32)

    # 1. Conv encoder: multi-scale skips + patch-embedded tokens
    skips_list, x_view = backbone.enc_down_dict["sax"](batch_input, mask=None)
    # skips_list: list of (1, ch, *spatial), x_view: (1, n_patches, embed_dim)

    # 2. Shared ViT encoder
    x = backbone.encoder(x_view)  # (1, 1+n_patches, embed_dim)
    x = x[:, 1:]  # strip cls token → (1, n_patches, embed_dim)

    # 3. Reshape ViT output to spatial grid
    grid_size = backbone.enc_down_dict["sax"].patch_embed.grid_size
    x = x.permute(0, 2, 1)  # (1, embed_dim, n_patches)
    vit_feat = x.reshape(1, x.shape[1], *grid_size)  # (1, embed_dim, gx, gy, gz)

    # Build return dict — move everything to CPU
    features_dict: dict[str, torch.Tensor] = {}
    for i, skip in enumerate(skips_list):
        features_dict[f"conv_skip_{i}"] = skip.squeeze(0).cpu()
    features_dict["vit_features"] = vit_feat.squeeze(0).cpu()

    return features_dict, vol, n_slices


def cache_cinema_volume_features(
    backbone: nn.Module,
    cinema_dataset,
    cache_dir: Path,
    device: torch.device | None = None,
    target_depth: int = SAX_TARGET_DEPTH,
) -> list[dict]:
    """Cache volume-level CineMA encoder features for all patients/frames.

    Saves one ``.pt`` file per patient+frame containing conv skips, ViT
    features, the padded image, padded label, and the actual slice count.

    Args:
        backbone: Frozen CineMA model in eval mode.
        cinema_dataset: CineMA ``EndDiastoleEndSystoleDataset``.
        cache_dir: Root cache directory.
        device: Device for inference.
        target_depth: Pad z to this depth.

    Returns:
        List of dicts with keys: ``path``, ``pid``, ``is_ed``, ``n_slices``.
    """
    out_dir = Path(cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    for sample_idx in tqdm(
        range(len(cinema_dataset)), desc="Caching CineMA volume features"
    ):
        sample = cinema_dataset[sample_idx]
        image_3d = sample["sax_image"]  # (1, H, W, z)
        label_3d = sample["sax_label"]  # (1, H, W, z)
        n_slices = int(sample["n_slices"])
        pid = sample["pid"]
        is_ed = sample["is_ed"]
        frame = "ed" if is_ed else "es"

        fname = f"{pid}_{frame}.pt"
        fpath = out_dir / fname

        if fpath.exists():
            manifest.append(
                {"path": fpath, "pid": pid, "is_ed": is_ed, "n_slices": n_slices}
            )
            continue

        features_dict, padded_image, actual_slices = extract_cinema_volume_features(
            backbone, image_3d, device, target_depth
        )

        # Pad label along z
        label, _ = _pad_volume_z(label_3d, target_depth)

        save_dict: dict = {
            "image": padded_image,  # (1, H, W, target_depth)
            "label": label.long(),  # (1, H, W, target_depth)
            "n_slices": actual_slices,
        }
        save_dict.update(features_dict)

        torch.save(save_dict, fpath)
        manifest.append(
            {"path": fpath, "pid": pid, "is_ed": is_ed, "n_slices": actual_slices}
        )

    return manifest


# ── CineMA Cached Dataset ──────────────────────────────────────────────────────
class CachedCinemaVolumeDataset(Dataset):
    """Loads pre-cached CineMA volume-level features from ``.pt`` files."""

    def __init__(
        self,
        manifest: list[dict],
        n_conv_skips: int = 3,
    ):
        self.manifest = manifest
        self.n_conv_skips = n_conv_skips

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> dict:
        entry = self.manifest[idx]
        data = torch.load(entry["path"], weights_only=True)
        result: dict = {
            "image": data["image"],  # (1, H, W, Z)
            "label": data["label"],  # (1, H, W, Z)
            "n_slices": data["n_slices"],
            "pid": entry["pid"],
            "vit_features": data["vit_features"],
        }
        for i in range(self.n_conv_skips):
            result[f"conv_skip_{i}"] = data[f"conv_skip_{i}"]
        return result


# ── CineMA UNetR Decoder ───────────────────────────────────────────────────────
class CineMAUNetRDecoder(nn.Module):
    """3D UNetR decoder for CineMA backbone features.

    Mirrors the decoder wiring in ``ConvUNetR.forward()``.  Takes CineMA's
    natural multi-scale conv-encoder skips + ViT features and decodes with
    ``UpsampleDecoder`` to produce dense 3D segmentation logits.

    Skip-connection wiring (default SAX config, 192x192x16 input):

    ========  ===========  =========  ============  ================
    Emb idx   Source       Spatial    Out channels  Note
    ========  ===========  =========  ============  ================
    0         image_conv   192x192    32            shallowest
    1         None         --         --            n_layers_wo_skip
    2         conv_skip_0  48x48      64            1st conv skip
    3         conv_skip_1  24x24      128           2nd conv skip
    4         vit_output   12x12      256           ViT features
    5         bottleneck   6x6        512           downsampled ViT
    ========  ===========  =========  ============  ================

    ``UpsampleDecoder`` pops from the end (bottleneck first).
    """

    def __init__(
        self,
        enc_embed_dim: int = 768,
        enc_conv_chans: tuple[int, ...] = (64, 128),
        dec_chans: tuple[int, ...] = (32, 64, 128, 256, 512),
        dec_patch_size: tuple[int, int, int] = (2, 2, 1),
        dec_scale_factor: tuple[int, int, int] = (2, 2, 1),
        num_classes: int = NUM_CLASSES,
        n_layers_wo_skip: int = 1,
        in_chans: int = 1,
        norm: str = "layer",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_layers_wo_skip = n_layers_wo_skip

        # Image path: raw image -> shallowest skip
        self.image_conv = ConvResBlock(
            n_dims=3, in_chans=in_chans, out_chans=dec_chans[0], norm=norm
        )

        # Conv-skip adapters.
        # The embeddings list has len(dec_chans)+1 entries.  UpsampleDecoder
        # pops from the end and each up-transpose produces channels that must
        # match the skip at that position.
        #
        # Positions:  0 = image_conv(dec_chans[0])
        #             1..n_layers_wo_skip = None  (no skip)
        #             n_layers_wo_skip+1 .. n_layers_wo_skip+n_conv_skips = conv skips
        #             next = ViT adapter
        #             last = bottleneck (popped first as initial x)
        #
        # UpsampleDecoder block k pops embedding at position (len(dec_chans)-k-1)
        # and its up-conv output has channels dec_chans[len-k-2].  So the skip
        # at position p (p>=1) must have dec_chans[p-1] channels.
        self.n_conv_skips = len(enc_conv_chans)

        self.skip_adapters = nn.ModuleList()
        for i, ch in enumerate(enc_conv_chans):
            # Embedding position = 1 + n_layers_wo_skip + i
            # Required = dec_chans[position - 1]
            target_ch = dec_chans[n_layers_wo_skip + i]
            self.skip_adapters.append(
                ConvResBlock(
                    n_dims=3,
                    in_chans=ch,
                    out_chans=target_ch,
                    norm=norm,
                    dropout=dropout,
                )
            )

        # ViT output adapter
        # Position = 1 + n_layers_wo_skip + n_conv_skips
        # Required = dec_chans[n_layers_wo_skip + n_conv_skips]
        vit_ch = dec_chans[n_layers_wo_skip + self.n_conv_skips]
        self.vit_adapter = ConvResBlock(
            n_dims=3,
            in_chans=enc_embed_dim,
            out_chans=vit_ch,
            norm=norm,
            dropout=dropout,
        )

        # Bottleneck: downsample ViT features spatially
        conv_cls = nn.Conv3d
        self.bottleneck_down = conv_cls(
            enc_embed_dim,
            enc_embed_dim,
            kernel_size=tuple(dec_scale_factor),  # type: ignore[arg-type]
            stride=tuple(dec_scale_factor),  # type: ignore[arg-type]
        )
        self.bottleneck_adapter = ConvResBlock(
            n_dims=3,
            in_chans=enc_embed_dim,
            out_chans=dec_chans[-1],
            norm=norm,
            dropout=dropout,
        )

        # CineMA UpsampleDecoder
        self.decoder = UpsampleDecoder(
            n_dims=3,
            chans=dec_chans,
            patch_size=dec_patch_size,
            scale_factor=dec_scale_factor,
            norm=norm,
            dropout=dropout,
        )

        # Prediction head
        self.pred_head = nn.Conv3d(dec_chans[0], num_classes, kernel_size=1)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass.

        Args:
            batch: Dict with keys ``"image"`` (B, 1, H, W, Z),
                   ``"conv_skip_0"``..``"conv_skip_{N-1}"`` from conv encoder,
                   and ``"vit_features"`` (B, embed_dim, gx, gy, gz).

        Returns:
            logits: (B, num_classes, H, W, Z)
        """
        image = batch["image"]  # (B, 1, H, W, Z)
        vit_feat = batch["vit_features"]  # (B, embed_dim, gx, gy, gz)

        # Shallowest skip: raw image
        img_skip = self.image_conv(image)  # (B, dec_chans[0], H, W, Z)

        # Conv-encoder skips: adapt channels
        conv_skips = []
        for i in range(self.n_conv_skips):
            skip = batch[f"conv_skip_{i}"]  # (B, ch_i, sx, sy, Z)
            conv_skips.append(self.skip_adapters[i](skip))

        # ViT output: adapt channels
        vit_skip = self.vit_adapter(vit_feat)

        # Bottleneck: downsample ViT features
        down = self.bottleneck_down(vit_feat)
        bottleneck = self.bottleneck_adapter(down)

        # Build embeddings list (UpsampleDecoder pops from the end)
        # Order: [shallowest, ..., deepest]
        embeddings: list[torch.Tensor | None] = [img_skip]
        # Decoder levels with no encoder skip
        for _ in range(self.n_layers_wo_skip):
            embeddings.append(None)
        # Conv-encoder skips
        for skip in conv_skips:
            embeddings.append(skip)
        embeddings.append(vit_skip)
        embeddings.append(bottleneck)

        x = self.decoder(embeddings)  # (B, dec_chans[0], H, W, Z)
        return self.pred_head(x)  # (B, num_classes, H, W, Z)


# ── Loss ────────────────────────────────────────────────────────────────────────
class MaskedVolumeLoss(nn.Module):
    """Weighted CE + Dice loss, with CE masked to exclude z-padded slices.

    Padded slices have label 0 (background).  Since the Dice component
    excludes the background class, padded slices contribute nothing to Dice.
    The CE component is explicitly masked to avoid biasing toward BG.
    """

    def __init__(
        self,
        ce_weight_tensor: torch.Tensor,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.register_buffer("class_weights", ce_weight_tensor)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        n_slices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute masked loss.

        Args:
            logits: (B, C, H, W, Z)
            targets: (B, 1, H, W, Z) — padded slices have label 0 (BG).
            n_slices: (B,) tensor of actual slice counts.
        """
        B, C, H, W, Z = logits.shape
        targets = targets.squeeze(1)  # (B, H, W, Z)

        # ── Masked CE: only valid (non-padded) z-slices ──
        z_idx = torch.arange(Z, device=logits.device)
        valid = z_idx.unsqueeze(0) < n_slices.unsqueeze(1)  # (B, Z)
        valid_mask = valid[:, None, None, :].expand(B, H, W, Z)  # (B, H, W, Z)

        logits_bhwzc = logits.permute(0, 2, 3, 4, 1)  # (B, H, W, Z, C)
        class_w: torch.Tensor = self.class_weights  # type: ignore[assignment]
        ce = F.cross_entropy(
            logits_bhwzc[valid_mask],  # (N_valid, C)
            targets[valid_mask].long(),  # (N_valid,)
            weight=class_w,
        )

        # ── Dice: padded slices are all-BG → excluded by foreground-only Dice ──
        probs = F.softmax(logits, dim=1)[:, 1:]  # (B, C-1, H, W, Z)
        targets_oh = (
            F.one_hot(targets.long(), num_classes=C)
            .permute(0, 4, 1, 2, 3)
            .float()[:, 1:]
        )  # (B, C-1, H, W, Z)

        dims = (0, 2, 3, 4)
        intersection = (probs * targets_oh).sum(dim=dims)
        cardinality = probs.sum(dim=dims) + targets_oh.sum(dim=dims)
        dice = (2.0 * intersection + 1.0) / (cardinality + 1.0)
        dice_loss = 1.0 - dice.mean()

        return self.ce_weight * ce + self.dice_weight * dice_loss


# ── Helpers ─────────────────────────────────────────────────────────────────────
_NON_MODEL_KEYS = {"label", "n_slices", "pid"}


def _batch_to_device(batch: dict, device: torch.device) -> dict[str, torch.Tensor]:
    """Move all tensor values (except label/n_slices/pid) to *device*."""
    return {
        k: v.to(device)
        for k, v in batch.items()
        if k not in _NON_MODEL_KEYS and isinstance(v, torch.Tensor)
    }


# ── Training ────────────────────────────────────────────────────────────────────
def train_one_epoch_vol(
    model: nn.Module,
    dataloader,
    criterion: MaskedVolumeLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    **_kwargs,
) -> float:
    """Train for one epoch on cached volume features.  Returns mean loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        batch_gpu = _batch_to_device(batch, device)

        labels = batch["label"].to(device)  # (B, 1, H, W, Z)
        n_slices = batch["n_slices"].to(device)  # (B,)

        logits = model(batch_gpu)  # (B, C, H, W, Z)
        loss = criterion(logits, labels, n_slices)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


# ── Evaluation ──────────────────────────────────────────────────────────────────
@torch.inference_mode()
def evaluate_vol(
    model: nn.Module,
    dataloader,
    device: torch.device,
    **_kwargs,
) -> dict:
    """Evaluate on cached volume features.

    Computes per-class Dice and macro Dice on valid (non-padded) slices only.
    """
    model.eval()
    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for batch in dataloader:
        batch_gpu = _batch_to_device(batch, device)

        labels = batch["label"]  # (B, 1, H, W, Z)
        n_slices = batch["n_slices"]  # (B,)

        logits = model(batch_gpu)
        preds = logits.argmax(dim=1).cpu()  # (B, H, W, Z)

        for i in range(preds.shape[0]):
            ns = int(n_slices[i])
            for z in range(ns):
                all_preds.append(preds[i, :, :, z].numpy())
                all_labels.append(labels[i, 0, :, :, z].numpy())

    all_preds_arr = np.stack(all_preds)
    all_labels_arr = np.stack(all_labels)

    results: dict = {"per_class_dice": {}}
    for c in range(NUM_CLASSES):
        name = CLASS_NAMES.get(c, f"C{c}")
        results["per_class_dice"][name] = float(
            dice_score(all_preds_arr, all_labels_arr, c)
        )
    results["macro_dice"] = float(macro_dice(all_preds_arr, all_labels_arr))

    return results
