"""Tests for the cache split manifest / contamination guard in caching.py."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from heartfm_evals.caching import (
    SPLIT_MANIFEST_NAME,
    CacheSplitError,
    verify_cache_splits,
    verify_caches_consistent,
    write_or_check_split_manifest,
)

SPLITS = {"train": ["p001", "p002"], "val": ["p003"], "test": ["p004"]}


def _touch(root: Path, split: str, name: str, layers: str = "layers_3-6") -> None:
    d = root / split / layers
    d.mkdir(parents=True, exist_ok=True)
    torch.save({}, d / name)


def _fill(root: Path, splits: dict[str, list[str]] = SPLITS) -> None:
    for split, pids in splits.items():
        for pid in pids:
            _touch(root, split, f"{pid}_ed_z00.pt")
            _touch(root, split, f"{pid}_es_z00.pt")


def test_fresh_cache_writes_manifest_and_verifies(tmp_path):
    write_or_check_split_manifest(tmp_path, SPLITS, split_seed=0)
    assert (tmp_path / SPLIT_MANIFEST_NAME).exists()
    _fill(tmp_path)
    verify_cache_splits(tmp_path)


def test_same_split_is_accepted_again(tmp_path):
    write_or_check_split_manifest(tmp_path, SPLITS, split_seed=0)
    write_or_check_split_manifest(tmp_path, SPLITS, split_seed=0)


def test_different_split_raises(tmp_path):
    write_or_check_split_manifest(tmp_path, SPLITS, split_seed=0)
    other = {"train": ["p001", "p003"], "val": ["p002"], "test": ["p004"]}
    with pytest.raises(CacheSplitError, match="different split"):
        write_or_check_split_manifest(tmp_path, other, split_seed=1)


def test_unmanifested_cache_with_files_raises(tmp_path):
    _fill(tmp_path)
    with pytest.raises(CacheSplitError, match="no split_manifest.json"):
        write_or_check_split_manifest(tmp_path, SPLITS, split_seed=0)


def test_overlapping_requested_splits_raise(tmp_path):
    bad = {"train": ["p001", "p002"], "val": ["p002"], "test": ["p004"]}
    with pytest.raises(CacheSplitError, match="overlap"):
        write_or_check_split_manifest(tmp_path, bad, split_seed=0)


def test_stray_patient_in_split_dir_is_caught(tmp_path):
    """The real incident: a patient from another split's extraction appears in train/."""
    write_or_check_split_manifest(tmp_path, SPLITS, split_seed=0)
    _fill(tmp_path)
    _touch(tmp_path, "train", "p003_ed_z00.pt")  # belongs to val
    with pytest.raises(CacheSplitError, match=r"train: patients not in manifest.*p003"):
        verify_cache_splits(tmp_path)


def test_incomplete_cache_fails_unless_allowed(tmp_path):
    write_or_check_split_manifest(tmp_path, SPLITS, split_seed=0)
    _touch(tmp_path, "train", "p001_ed_z00.pt")
    with pytest.raises(CacheSplitError, match="missing from cache"):
        verify_cache_splits(tmp_path)
    verify_cache_splits(tmp_path, require_complete=False)


def test_missing_manifest_fails_verify(tmp_path):
    with pytest.raises(CacheSplitError, match="no split_manifest.json"):
        verify_cache_splits(tmp_path)


def test_volume_cache_file_names_parse(tmp_path):
    write_or_check_split_manifest(tmp_path, SPLITS, split_seed=0)
    for split, pids in SPLITS.items():
        for pid in pids:
            _touch(tmp_path, split, f"{pid}_ed.pt")
    verify_cache_splits(tmp_path)


def test_cross_cache_consistency(tmp_path):
    a, b, c = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    write_or_check_split_manifest(a, SPLITS, split_seed=0)
    write_or_check_split_manifest(b, SPLITS, split_seed=0)
    write_or_check_split_manifest(
        c, {"train": ["p001", "p003"], "val": ["p002"], "test": ["p004"]}, split_seed=1
    )
    verify_caches_consistent([a, b])
    with pytest.raises(CacheSplitError, match="different splits"):
        verify_caches_consistent([a, c])
