"""Tests for ``hidden_states`` layer indexing in the SAM feature extractors.

``hidden_states[0]`` is the initial patch embedding, so block *i*'s output lives
at ``hidden_states[i+1]``.  The two SAM families express ``layer_indices`` in
different spaces — SAM v1 in block indices, SAM2 in ``hidden_states`` indices —
and one extractor serves both, so these tests pin the convention down.

All stubs are synthetic: no model weights or datasets are downloaded.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from heartfm_evals.backbones import SAM_CONFIGS
from heartfm_evals.features import (
    _select_hidden_state,
    extract_sam2_2d_features,
    extract_sam_2d_features,
    extract_sam_volume_features,
)

GRID = 4


def _hidden_states(
    n: int, channels: int = 8, spatial: int = 4
) -> tuple[torch.Tensor, ...]:
    """``n`` channels-last states, entry ``i`` filled with the constant ``i``."""
    return tuple(
        torch.full((1, spatial, spatial, channels), float(i)) for i in range(n)
    )


class _StubEncoder:
    """Stands in for a SAM/SAM2 vision encoder returning fixed hidden states."""

    def __init__(self, n_states: int, channels: int = 8, spatial: int = 4):
        self.states = _hidden_states(n_states, channels, spatial)

    def __call__(self, _pixel_values, output_hidden_states: bool = False):
        assert output_hidden_states, "extractors must request hidden states"
        return SimpleNamespace(hidden_states=self.states)


def _stub_model(n_states: int, channels: int = 8, spatial: int = 4) -> SimpleNamespace:
    return SimpleNamespace(vision_encoder=_StubEncoder(n_states, channels, spatial))


def _stub_processor(**_kwargs) -> dict:
    """Stands in for a SamImageProcessor; the extractors only need the tensor."""
    return {"pixel_values": torch.zeros(1, 3, 8, 8)}


def _layer_values(feats: torch.Tensor, n_layers: int) -> list[float]:
    """Per-layer constant from a channel-concatenated feature tensor."""
    chunks = torch.chunk(feats, n_layers, dim=0)
    return [float(c.unique().item()) for c in chunks]


# ── _select_hidden_state ──────────────────────────────────────────────────────


def test_select_hidden_state_offsets() -> None:
    states = _hidden_states(13)
    # SAM v1: block index 11 is the final block, at hidden_states[12].
    assert _select_hidden_state(states, 11, 1).unique().item() == 12.0
    # SAM2: indices are already hidden_states indices.
    assert _select_hidden_state(states, 11, 0).unique().item() == 11.0


def test_select_hidden_state_out_of_range() -> None:
    states = _hidden_states(13)
    with pytest.raises(IndexError, match=r"hidden_states\[13\]"):
        _select_hidden_state(states, 12, 1)


# ── SAM v1 ────────────────────────────────────────────────────────────────────


def test_sam_v1_configs_reach_their_final_block() -> None:
    """Under the block-index convention, the last entry is the final block."""
    for model_id, cfg in SAM_CONFIGS.items():
        indices = cfg["layer_indices"]
        assert max(indices) == cfg["n_layers"] - 1, (
            f"{model_id}: layer_indices {indices} never reach the final block "
            f"{cfg['n_layers'] - 1}"
        )
        assert min(indices) >= 0


def test_sam_2d_features_read_blocks_not_raw_hidden_states() -> None:
    """``layer_indices`` are block indices, so entry i is read at hs[i+1]."""
    model = _stub_model(n_states=13)  # 12 blocks + patch embedding
    layer_indices = (3, 6, 9, 11)

    feats = extract_sam_2d_features(
        model,
        _stub_processor,
        torch.zeros(8, 8),
        layer_indices,
        grid_size=GRID,
    )

    assert feats.shape == (8 * len(layer_indices), GRID, GRID)
    assert _layer_values(feats, len(layer_indices)) == [4.0, 7.0, 10.0, 12.0]


def test_sam_volume_features_apply_offset() -> None:
    model = _stub_model(n_states=13)
    layer_indices = (3, 11)

    features, _, n_slices = extract_sam_volume_features(
        model,
        _stub_processor,
        torch.zeros(1, 8, 8, 2),
        layer_indices,
        target_depth=2,
        grid_size=GRID,
        hidden_state_offset=1,
    )

    assert n_slices == 2
    assert features["layer_3"].unique().item() == 4.0
    assert features["layer_11"].unique().item() == 12.0  # final block, was skipped


# ── SAM2 ──────────────────────────────────────────────────────────────────────


def test_sam2_2d_features_use_raw_hidden_state_indices() -> None:
    """SAM2 indices are already hidden_states indices — no shift."""
    model = _stub_model(n_states=25)  # 24 Hiera blocks + patch embedding
    layer_indices = (6, 11, 16, 21)

    feats = extract_sam2_2d_features(
        model,
        _stub_processor,
        torch.zeros(8, 8),
        layer_indices,
        grid_size=GRID,
    )

    assert _layer_values(feats, len(layer_indices)) == [6.0, 11.0, 16.0, 21.0]


def test_sam_volume_features_offset_zero_matches_sam2() -> None:
    """The shared 3D extractor must not shift SAM2 out of Stage 3."""
    model = _stub_model(n_states=25)

    features, _, _ = extract_sam_volume_features(
        model,
        _stub_processor,
        torch.zeros(1, 8, 8, 1),
        (6, 21),
        target_depth=1,
        grid_size=GRID,
        hidden_state_offset=0,
    )

    assert features["layer_6"].unique().item() == 6.0
    assert features["layer_21"].unique().item() == 21.0
