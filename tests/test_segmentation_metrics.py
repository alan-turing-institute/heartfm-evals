from __future__ import annotations

import importlib.util
import json
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from heartfm_evals.metrics import (
    per_sample_dice_metrics,
    per_sample_dice_score,
    per_sample_macro_dice,
)
from heartfm_evals.training import (
    evaluate,
    evaluate_per_sample,
    evaluate_vol,
    evaluate_vol_per_sample,
)


class SliceDataset(Dataset):
    def __init__(self, samples: list[dict]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


class VolumeDataset(Dataset):
    def __init__(self, samples: list[dict]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


class DictPassthroughModel(torch.nn.Module):
    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return batch["logits"]


def _class_logits(labels: torch.Tensor, num_classes: int = 4) -> torch.Tensor:
    logits = torch.zeros((num_classes, *labels.shape), dtype=torch.float32)
    for class_idx in range(num_classes):
        logits[class_idx] = (labels == class_idx).float()
    return logits


def _load_run_segmentation_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "segmentation"
        / "run_segmentation.py"
    )
    spec = importlib.util.spec_from_file_location("run_segmentation", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        spec.loader.exec_module(module)
    return module


def test_per_sample_dice_metrics_support_nan_empty_gt() -> None:
    pred = np.array([[0, 1], [1, 1]])
    true = np.array([[0, 1], [1, 0]])

    assert np.isnan(per_sample_dice_score(pred, true, class_idx=2))
    assert np.isclose(per_sample_dice_score(pred, true, class_idx=0), 2 / 3)
    assert np.isclose(per_sample_dice_score(pred, true, class_idx=1), 0.8)
    assert np.isclose(per_sample_macro_dice(pred, true), 0.8)

    metrics = per_sample_dice_metrics(pred, true)
    assert np.isclose(metrics["dice_BG"], 2 / 3)
    assert np.isclose(metrics["dice_RV"], 0.8)
    assert np.isnan(metrics["dice_MYO"])
    assert np.isnan(metrics["dice_LV"])
    assert np.isclose(metrics["macro_dice"], 0.8)


def test_evaluate_per_sample_returns_one_row_per_slice() -> None:
    label_0 = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    label_1 = torch.tensor([[0, 0], [1, 1]], dtype=torch.long)
    samples = [
        {
            "features": _class_logits(label_0),
            "label": label_0,
            "pid": "001",
            "is_ed": True,
            "z_idx": 1,
        },
        {
            "features": _class_logits(label_1),
            "label": label_1,
            "pid": "001",
            "is_ed": False,
            "z_idx": 0,
        },
    ]
    loader = DataLoader(SliceDataset(samples), batch_size=2, shuffle=False)
    model = torch.nn.Identity()

    rows = evaluate_per_sample(model, loader, device=torch.device("cpu"))

    assert len(rows) == 2
    assert rows[0]["pid"] == "001"
    assert rows[0]["frame"] == "ed"
    assert rows[0]["is_ed"] is True
    assert rows[0]["z_idx"] == 1
    assert np.isclose(rows[0]["dice_BG"], 1.0)
    assert np.isclose(rows[0]["dice_RV"], 1.0)
    assert np.isclose(rows[0]["dice_MYO"], 1.0)
    assert np.isclose(rows[0]["dice_LV"], 1.0)
    assert np.isclose(rows[0]["macro_dice"], 1.0)

    aggregate = evaluate(model, loader, device=torch.device("cpu"))
    assert np.isclose(aggregate["macro_dice"], 1.0)
    assert all(np.isclose(score, 1.0) for score in aggregate["per_class_dice"].values())


def test_evaluate_vol_per_sample_uses_whole_stack_and_ignores_padding() -> None:
    label = torch.tensor(
        [
            [
                [[0, 1, 0], [1, 2, 0]],
                [[0, 3, 0], [3, 0, 0]],
            ]
        ],
        dtype=torch.long,
    )
    pred_labels = torch.tensor(
        [
            [[0, 1, 2], [1, 2, 2]],
            [[0, 3, 1], [3, 0, 1]],
        ],
        dtype=torch.long,
    )
    sample = {
        "logits": _class_logits(pred_labels),
        "label": label,
        "n_slices": 2,
        "pid": "002",
        "is_ed": False,
    }
    loader = DataLoader(VolumeDataset([sample]), batch_size=1, shuffle=False)
    model = DictPassthroughModel()

    rows = evaluate_vol_per_sample(model, loader, device=torch.device("cpu"))

    assert len(rows) == 1
    row = rows[0]
    assert row["pid"] == "002"
    assert row["frame"] == "es"
    assert row["is_ed"] is False
    assert row["n_slices"] == 2
    assert np.isclose(row["dice_BG"], 1.0)
    assert np.isclose(row["dice_RV"], 1.0)
    assert np.isclose(row["dice_MYO"], 1.0)
    assert np.isclose(row["dice_LV"], 1.0)
    assert np.isclose(row["macro_dice"], 1.0)

    aggregate = evaluate_vol(model, loader, device=torch.device("cpu"))
    assert np.isclose(aggregate["macro_dice"], 1.0)
    assert all(np.isclose(score, 1.0) for score in aggregate["per_class_dice"].values())


def test_run_segmentation_main_writes_json_and_per_sample_csv(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_run_segmentation_module()

    args = SimpleNamespace(
        backbone="dinov3",
        decoder="linear_probe",
        dataset="acdc",
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "results",
        cache_dir=tmp_path / "cache",
        dinov3_model_name="dinov3_vits16",
        dinov3_repo_dir="models/dinov3/",
        dinov3_weights_path=None,
        sam2_model_id="facebook/sam2.1-hiera-base-plus",
        hf_cache_dir=tmp_path / "hf",
        use_layers=None,
        batch_size=2,
        lr=1e-3,
        weight_decay=1e-3,
        n_epochs=1,
        patience=1,
        dropout=0.1,
        seed=0,
        device="cpu",
        no_auto_download=True,
    )

    class DummyCachedFeatureDataset(Dataset):
        def __init__(self, manifest: list[dict]):
            self.manifest = manifest

        def __len__(self) -> int:
            return len(self.manifest)

        def __getitem__(self, idx: int) -> dict:
            return {
                "features": torch.zeros((4, 2, 2), dtype=torch.float32),
                "label": torch.zeros((2, 2), dtype=torch.long),
                "pid": f"{idx:03d}",
                "is_ed": idx % 2 == 0,
                "z_idx": idx,
            }

    monkeypatch.setattr(module, "parse_args", lambda: args)
    monkeypatch.setattr(module, "set_seed", lambda _seed: None)
    monkeypatch.setattr(module, "detect_device", lambda _device: torch.device("cpu"))
    monkeypatch.setattr(
        module,
        "load_backbone",
        lambda *_args, **_kwargs: (
            torch.nn.Identity(),
            {"embed_dim": 4, "layer_indices": (3, 6, 9, 11)},
        ),
    )
    monkeypatch.setattr(
        module,
        "load_segmentation_datasets",
        lambda *_args, **_kwargs: (["train"], ["val"], ["test"], None, None, None),
    )
    monkeypatch.setattr(
        module,
        "cache_features",
        lambda *_args, **_kwargs: [{"path": Path("unused.pt"), "pid": "001", "is_ed": True, "z_idx": 0}],
    )
    monkeypatch.setattr(module, "CachedFeatureDataset", DummyCachedFeatureDataset)
    monkeypatch.setattr(
        module,
        "get_decoder",
        lambda **_kwargs: torch.nn.Conv2d(4, 4, kernel_size=1, bias=False),
    )
    monkeypatch.setattr(
        module,
        "train_segmentation",
        lambda **_kwargs: {
            "history": {"train_loss": [0.1], "val_macro_dice": [0.5], "lr": [1e-3]},
            "best_val_dice": 0.5,
            "best_epoch": 1,
        },
    )
    monkeypatch.setattr(
        module,
        "evaluate",
        lambda *_args, **_kwargs: {
            "per_class_dice": {"BG": 0.9, "RV": 0.8, "MYO": 0.7, "LV": 0.6},
            "macro_dice": 0.7,
        },
    )
    monkeypatch.setattr(
        module,
        "evaluate_per_sample",
        lambda *_args, **_kwargs: [
            {
                "pid": "002",
                "frame": "es",
                "is_ed": False,
                "z_idx": 1,
                "dice_BG": 0.8,
                "dice_RV": 0.7,
                "dice_MYO": np.nan,
                "dice_LV": 0.6,
                "macro_dice": 0.65,
            },
            {
                "pid": "001",
                "frame": "ed",
                "is_ed": True,
                "z_idx": 0,
                "dice_BG": 0.9,
                "dice_RV": 0.8,
                "dice_MYO": 0.7,
                "dice_LV": 0.6,
                "macro_dice": 0.7,
            },
        ],
    )

    module.main()

    json_files = sorted(args.output_dir.glob("*.json"))
    csv_files = sorted(args.output_dir.glob("*_per_sample.csv"))
    assert len(json_files) == 1
    assert len(csv_files) == 1

    results = json.loads(json_files[0].read_text())
    assert results["per_sample_metrics_file"] == csv_files[0].name
    assert results["n_test_samples"] == 2

    df = pd.read_csv(csv_files[0], dtype={"pid": str})
    assert list(df["pid"]) == ["001", "002"]
    assert list(df["frame"]) == ["ed", "es"]
    assert list(df["z_idx"]) == [0, 1]
