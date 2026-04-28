"""Segmentation decoder heads.

Provides decoder models for 2D and 3D segmentation:

- ``DenseLinearProbe``: 1×1 Conv linear probe (DINOv3 evaluation protocol).
- ``ConvDecoderProbe``: Multi-layer CNN decoder for 2D features.
- ``DINOv3UNetRDecoder``: 3D UNetR decoder for DINOv3 volume features.
- ``CineMAUNetRDecoder``: 3D UNetR decoder for CineMA volume features.
- ``get_decoder()``: Factory function to instantiate decoders by name.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from cinema.conv import ConvResBlock
from cinema.segmentation.convunetr import UpsampleDecoder

from heartfm_evals.constants import IMAGE_SIZE, NUM_CLASSES


# ── Dense Linear Probe (2D) ──────────────────────────────────────────────────
class DenseLinearProbe(nn.Module):
    """Per-pixel linear classifier on frozen DINOv3 features.

    Matches the official DINOv3 linear evaluation protocol:
        features → Dropout2d → BatchNorm2d → 1×1 Conv2d → bilinear upsample.

    Supports selecting a subset of layers at forward time from a cache that
    stored more layers (e.g. cache has 4 layers, probe uses only the last).
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int = NUM_CLASSES,
        layer_indices: tuple[int, ...] = (11,),
        cached_layers: tuple[int, ...] | None = None,
        cached_embed_dims: tuple[int, ...] | None = None,
        output_size: tuple[int, int] = (IMAGE_SIZE, IMAGE_SIZE),
        dropout: float = 0.1,
    ):
        """
        Args:
            embed_dim: Channel count of each *probed* layer (uniform across
                ``layer_indices``).  For SAM2 linear probe this should be the
                Stage-4 channel count (``cls_embed_dim``), not the Stage-3 one.
            cached_embed_dims: Per-layer channel counts for *all* layers stored
                in the cache (i.e. one value per entry in ``cached_layers``).
                Required when the cache contains SAM2 multi-scale features where
                different stages have different channel widths.  When ``None``,
                ``embed_dim`` is assumed to be uniform across all cached layers.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.layer_indices = layer_indices
        self.cached_layers = (
            cached_layers if cached_layers is not None else layer_indices
        )
        self.output_size = output_size

        # Build channel-selection indices when cache differs from probe layers.
        # Supports non-uniform cached_embed_dims (e.g. SAM2 multi-scale).
        self._channel_indices: list[int] | None = None
        if self.cached_layers != self.layer_indices:
            cached_list = list(self.cached_layers)
            indices: list[int] = []
            for li in self.layer_indices:
                pos = cached_list.index(li)
                if cached_embed_dims is not None:
                    start = sum(cached_embed_dims[:pos])
                    width = cached_embed_dims[pos]
                else:
                    start = pos * embed_dim
                    width = embed_dim
                indices.extend(range(start, start + width))
            self._channel_indices = indices

        in_channels = embed_dim * len(layer_indices)
        self.dropout = nn.Dropout2d(dropout)
        self.batchnorm = nn.BatchNorm2d(in_channels)
        self.head = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        nn.init.normal_(self.head.weight, mean=0, std=0.01)
        nn.init.constant_(self.head.bias, 0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Pre-extracted concatenated features (B, C, h, w)
                      where C = embed_dim * n_cached_layers, h/w = patch grid.
        Returns:
            logits: (B, num_classes, H, W) at output_size resolution.
        """
        if self._channel_indices is not None:
            features = features[:, self._channel_indices]
        x = self.dropout(features)
        x = self.batchnorm(x)
        x = self.head(x)
        x = F.interpolate(
            x, size=self.output_size, mode="bilinear", align_corners=False
        )
        return x


# ── Conv Decoder Probe (2D) ──────────────────────────────────────────────────
class ConvDecoderProbe(nn.Module):
    """Multi-layer CNN decoder for 2D cached features.

    Architecture: bilinear upsample → 2× (Conv3x3 → BN → ReLU) → Conv1x1.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int = NUM_CLASSES,
        output_size: tuple[int, int] = (IMAGE_SIZE, IMAGE_SIZE),
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.output_size = output_size
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(hidden_dim, num_classes, kernel_size=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            features, size=self.output_size, mode="bilinear", align_corners=False
        )
        x = self.decoder(x)
        return self.head(x)


# ── DINOv3 UNetR Decoder (3D) ────────────────────────────────────────────────
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
        embed_dims: tuple[int, ...] | None = None,
        layer_indices: tuple[int, ...] = (3, 6, 9, 11),
        dec_chans: tuple[int, ...] = (32, 64, 128, 256, 512),
        dec_patch_size: tuple[int, int, int] = (2, 2, 1),
        dec_scale_factor: tuple[int, int, int] = (2, 2, 1),
        num_classes: int = NUM_CLASSES,
        norm: str = "instance",
        dropout: float = 0.1,
    ):
        """
        Args:
            embed_dim: Uniform channel count across all layers (DINOv3 / SAM v1).
                Ignored when ``embed_dims`` is provided.
            embed_dims: Per-layer channel counts, one per entry in
                ``layer_indices`` (Stage 1, Stage 2, Stage 3 end, Stage 4 end).
                Use this for SAM2 multi-scale features where each stage has a
                different channel width.  When ``None``, ``embed_dim`` is
                broadcast to all four layers.
        """
        super().__init__()
        if len(layer_indices) != 4:
            msg = f"Expected 4 layer_indices, got {len(layer_indices)}"
            raise ValueError(msg)
        self.layer_indices = layer_indices

        # Resolve per-layer channel counts
        _dims: tuple[int, ...] = (
            embed_dims if embed_dims is not None else (embed_dim,) * 4
        )

        # Image path: raw image → shallowest skip
        self.image_conv = ConvResBlock(
            n_dims=3, in_chans=1, out_chans=dec_chans[0], norm=norm
        )

        # Skip adapters: backbone features → decoder skip channels.
        # Each adapter uses the channel count for its respective stage.
        self.skip_adapters = nn.ModuleDict(
            {
                f"layer_{layer_indices[0]}": ConvResBlock(
                    n_dims=3, in_chans=_dims[0], out_chans=dec_chans[1], norm=norm
                ),
                f"layer_{layer_indices[1]}": ConvResBlock(
                    n_dims=3, in_chans=_dims[1], out_chans=dec_chans[2], norm=norm
                ),
                f"layer_{layer_indices[2]}": ConvResBlock(
                    n_dims=3, in_chans=_dims[2], out_chans=dec_chans[3], norm=norm
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
            n_dims=3, in_chans=_dims[3], out_chans=dec_chans[-1], norm=norm
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
        import torch

        image = batch["image"]  # (B, 1, H, W, Z)

        # Shallowest skip: image → conv
        img_skip = self.image_conv(image)

        # Mid-level skips: optionally upsample, then adapt channels
        skips = []
        for key in [
            f"layer_{self.layer_indices[0]}",
            f"layer_{self.layer_indices[1]}",
            f"layer_{self.layer_indices[2]}",
        ]:
            feat = batch[key]
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
        bottleneck = self.bottleneck_down(bottleneck)

        # Build embeddings list (UpsampleDecoder pops from the end)
        embeddings: list[torch.Tensor | None] = [
            img_skip,
            None,  # block 3 skip (no 96x96 source)
            skips[0],
            skips[1],
            skips[2],
            bottleneck,
        ]

        x = self.decoder(embeddings)
        return self.pred_head(x)


# ── CineMA UNetR Decoder (3D) ────────────────────────────────────────────────
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

        self.n_conv_skips = len(enc_conv_chans)

        self.skip_adapters = nn.ModuleList()
        for i, ch in enumerate(enc_conv_chans):
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
        vit_ch = dec_chans[n_layers_wo_skip + self.n_conv_skips]
        self.vit_adapter = ConvResBlock(
            n_dims=3,
            in_chans=enc_embed_dim,
            out_chans=vit_ch,
            norm=norm,
            dropout=dropout,
        )

        # Bottleneck: downsample ViT features spatially
        self.bottleneck_down = nn.Conv3d(
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
        import torch

        image = batch["image"]
        vit_feat = batch["vit_features"]

        # Shallowest skip: raw image
        img_skip = self.image_conv(image)

        # Conv-encoder skips: adapt channels
        conv_skips = []
        for i in range(self.n_conv_skips):
            skip = batch[f"conv_skip_{i}"]
            conv_skips.append(self.skip_adapters[i](skip))

        # ViT output: adapt channels
        vit_skip = self.vit_adapter(vit_feat)

        # Bottleneck: downsample ViT features
        down = self.bottleneck_down(vit_feat)
        bottleneck = self.bottleneck_adapter(down)

        # Build embeddings list (UpsampleDecoder pops from the end)
        embeddings: list[torch.Tensor | None] = [img_skip]
        for _ in range(self.n_layers_wo_skip):
            embeddings.append(None)
        for skip in conv_skips:
            embeddings.append(skip)
        embeddings.append(vit_skip)
        embeddings.append(bottleneck)

        x = self.decoder(embeddings)
        return self.pred_head(x)


# ── Decoder Factory ──────────────────────────────────────────────────────────
def get_decoder(
    decoder_type: str,
    backbone_type: str,
    embed_dim: int,
    num_classes: int = NUM_CLASSES,
    layer_indices: tuple[int, ...] = (3, 6, 9, 11),
    cached_layers: tuple[int, ...] | None = None,
    embed_dims: tuple[int, ...] | None = None,
    cached_embed_dims: tuple[int, ...] | None = None,
    **kwargs,
) -> nn.Module:
    """Instantiate a segmentation decoder by name.

    Args:
        decoder_type: One of ``"linear_probe"``, ``"conv_decoder"``, ``"unetr"``.
        backbone_type: One of ``"dinov3"``, ``"cinema"``, ``"sam2"``.
        embed_dim: Backbone embedding dimension (uniform across layers).
            For SAM2 linear probe, pass the channel count of the *probed* layer
            (i.e. ``cls_embed_dim``, Stage 4 channels).
        num_classes: Number of segmentation classes.
        layer_indices: Layer indices used for feature extraction.
        cached_layers: Layer indices present in the feature cache (may differ
            from ``layer_indices`` for linear_probe layer subset selection).
        embed_dims: Per-layer channel counts, one per ``layer_indices`` entry.
            Used for SAM2 multi-scale features (``conv_decoder`` and ``unetr``).
            When provided, overrides ``embed_dim`` for per-layer adapter sizing.
        cached_embed_dims: Per-layer channel counts for all layers in the cache,
            one per ``cached_layers`` entry.  Used by ``linear_probe`` channel
            selection when the cache contains SAM2 multi-scale features.
        **kwargs: Additional keyword arguments passed to the decoder constructor.

    Returns:
        An ``nn.Module`` decoder.
    """
    if decoder_type == "linear_probe":
        return DenseLinearProbe(
            embed_dim=embed_dim,
            num_classes=num_classes,
            layer_indices=layer_indices,
            cached_layers=cached_layers,
            cached_embed_dims=cached_embed_dims,
            **kwargs,
        )
    elif decoder_type == "conv_decoder":
        if embed_dims is not None:
            in_channels = sum(embed_dims)
        else:
            in_channels = embed_dim * len(layer_indices)
        return ConvDecoderProbe(
            in_channels=in_channels,
            num_classes=num_classes,
            **kwargs,
        )
    elif decoder_type == "unetr":
        if backbone_type == "cinema":
            return CineMAUNetRDecoder(
                enc_embed_dim=embed_dim,
                num_classes=num_classes,
                **kwargs,
            )
        else:
            # DINOv3, SAM2, SAM all use DINOv3UNetRDecoder
            return DINOv3UNetRDecoder(
                embed_dim=embed_dim,
                embed_dims=embed_dims,
                layer_indices=layer_indices,
                num_classes=num_classes,
                **kwargs,
            )
    else:
        msg = f"Unknown decoder_type: {decoder_type!r}. Expected 'linear_probe', 'conv_decoder', or 'unetr'."
        raise ValueError(msg)
