"""Dataset loading utilities for segmentation tasks.

Provides :func:`load_segmentation_datasets` which returns train / val / test
``EndDiastoleEndSystoleDataset`` instances for ACDC, M&M, or M&M2.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset

from heartfm_evals.constants import IMAGENET_MEAN, IMAGENET_STD, imagenet_normalize

# Re-usable ImageNet normaliser is imported from constants and also available
# directly here for convenience.
_imagenet_normalize = imagenet_normalize


# ── Public dataset loader ──────────────────────────────────────────────────────


def load_segmentation_datasets(
    dataset_name: str,
    data_dir: str | Path,
    split_seed: int = 0,
) -> tuple:
    """Load train / val / test ``EndDiastoleEndSystoleDataset`` for segmentation.

    The returned tuple is ``(train_ds, val_ds, test_ds, train_meta, val_meta, test_meta)``
    so that callers also have access to the metadata DataFrames.

    Supports ACDC, M&M (``"mnm"``), and M&M2 (``"mnm2"``).  All three share
    the same directory layout::

        {data_dir}/
            train/
            test/
            val/              (M&M and M&M2 only)
            train_metadata.csv
            test_metadata.csv
            val_metadata.csv  (M&M and M&M2 only)

    For **ACDC** (no ``val_metadata.csv``), a validation set is carved out of
    training by sampling two patients per pathology class.  For **M&M / M&M2**
    the existing ``val_metadata.csv`` is used directly.

    Parameters
    ----------
    dataset_name:
        ``"acdc"``, ``"mnm"``, or ``"mnm2"``.
    data_dir:
        Root of the preprocessed dataset.
    split_seed:
        Random seed for the ACDC validation split.  Ignored when
        ``val_metadata.csv`` exists.

    Returns
    -------
    train_ds, val_ds, test_ds, train_meta, val_meta, test_meta
    """
    from cinema.segmentation.dataset import EndDiastoleEndSystoleDataset
    from monai.transforms import ScaleIntensityd

    data_dir = Path(data_dir)
    dataset_name = dataset_name.lower()

    # ── read metadata ──
    train_meta_df = pd.read_csv(data_dir / "train_metadata.csv", dtype={"pid": str})
    test_meta_df = pd.read_csv(data_dir / "test_metadata.csv", dtype={"pid": str})

    val_meta_path = data_dir / "val_metadata.csv"
    has_val_split = val_meta_path.exists()

    if has_val_split:
        val_meta_df = pd.read_csv(val_meta_path, dtype={"pid": str})
        train_split_df = train_meta_df
    else:
        # ACDC-style: create val from train
        if "pathology" in train_meta_df.columns:
            val_pids = (
                train_meta_df.groupby("pathology")
                .sample(n=2, random_state=split_seed)["pid"]
                .tolist()
            )
        else:
            val_pids = train_meta_df.sample(frac=0.1, random_state=split_seed)[
                "pid"
            ].tolist()

        train_split_df = train_meta_df[
            ~train_meta_df["pid"].isin(val_pids)
        ].reset_index(drop=True)
        val_meta_df = train_meta_df[train_meta_df["pid"].isin(val_pids)].reset_index(
            drop=True
        )

    # ── shared transform ──
    transform = ScaleIntensityd(keys="sax_image", factor=1 / 255, channel_wise=False)

    # ── train / val / test datasets ──
    # For ACDC val comes from the *train* directory; for M&M/M&M2 it has its
    # own directory.
    val_data_subdir = "val" if has_val_split else "train"

    train_ds = EndDiastoleEndSystoleDataset(
        data_dir=data_dir / "train",
        meta_df=train_split_df,
        views="sax",
        transform=transform,
    )
    val_ds = EndDiastoleEndSystoleDataset(
        data_dir=data_dir / val_data_subdir,
        meta_df=val_meta_df,
        views="sax",
        transform=transform,
    )
    test_ds = EndDiastoleEndSystoleDataset(
        data_dir=data_dir / "test",
        meta_df=test_meta_df,
        views="sax",
        transform=transform,
    )

    return train_ds, val_ds, test_ds, train_split_df, val_meta_df, test_meta_df


# ── Slice-level wrapper ───────────────────────────────────────────────────────


class ACDCSliceDataset(Dataset):
    """Wraps a CineMA ``EndDiastoleEndSystoleDataset`` to yield individual 2-D slices.

    Each item is a dict with keys: ``image`` (3, H, W), ``label`` (H, W), ``pid``.

    Uses lazy indexing: only a lightweight ``(sample_idx, z, pid, is_ed)``
    index list is built at construction time; actual volume tensors are loaded
    on demand inside ``__getitem__``.
    """

    def __init__(self, cinema_dataset, augment: bool = False):
        self.cinema_dataset = cinema_dataset
        self.index: list[tuple[int, int, str, bool]] = []
        for sample_idx in range(len(cinema_dataset)):
            sample = cinema_dataset[sample_idx]
            n_slices = sample["n_slices"]
            pid = sample["pid"]
            is_ed = sample["is_ed"]
            for z in range(n_slices):
                self.index.append((sample_idx, z, pid, is_ed))
        self.augment = augment
        self._cache_key: int | None = None
        self._cache_value: dict | None = None

    def __len__(self) -> int:
        return len(self.index)

    def _load_sample(self, sample_idx: int) -> dict:
        """Load a volume sample, reusing the cached result when possible."""
        if self._cache_key != sample_idx:
            self._cache_key = sample_idx
            self._cache_value = self.cinema_dataset[sample_idx]
        assert self._cache_value is not None
        return self._cache_value

    def __getitem__(self, idx: int) -> dict:
        sample_idx, z, pid, _ = self.index[idx]
        sample = self._load_sample(sample_idx)
        image_2d = sample["sax_image"][0, :, :, z]  # (H, W) in [0,1]
        label_2d = sample["sax_label"][0, :, :, z]  # (H, W)

        # Repeat to 3 channels + ImageNet normalise
        img = image_2d.unsqueeze(0).repeat(3, 1, 1)  # (3, H, W)
        img = _imagenet_normalize(img)

        if self.augment and torch.rand(1).item() > 0.5:
            img = img.flip(-1)
            label_2d = label_2d.flip(-1)

        return {"image": img, "label": label_2d.long(), "pid": pid}
