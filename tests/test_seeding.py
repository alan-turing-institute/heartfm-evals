"""Seeding of the segmentation pipeline (issue #86).

Covers ``set_seed`` itself, and the segmentation driver's two seeds:
``--seed-data-split`` (the ACDC val split, which feature caches are tied to) and
``--seed-training`` (decoder init, dropout, shuffle order).
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import subprocess
import sys
import warnings
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pandas as pd
import pytest
import torch
from torch.utils.data import Dataset

from heartfm_evals.caching import CacheSplitError
from heartfm_evals.decoders import get_decoder
from heartfm_evals.reproducibility import CUBLAS_WORKSPACE_CONFIG, set_seed

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_SEGMENTATION = REPO_ROOT / "scripts" / "segmentation" / "run_segmentation.py"


@pytest.fixture(autouse=True)
def _restore_global_torch_state(monkeypatch) -> Iterator[None]:
    """``set_seed`` flips process-wide torch flags; undo them after each test."""
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    det = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn_det = torch.backends.cudnn.deterministic
    cudnn_bench = torch.backends.cudnn.benchmark
    rng = torch.random.get_rng_state()
    yield
    torch.use_deterministic_algorithms(det, warn_only=warn_only)
    torch.backends.cudnn.deterministic = cudnn_det
    torch.backends.cudnn.benchmark = cudnn_bench
    torch.random.set_rng_state(rng)


# ── set_seed ─────────────────────────────────────────────────────────────────


def test_set_seed_sets_cublas_workspace_config_when_unset() -> None:
    set_seed(0)
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == CUBLAS_WORKSPACE_CONFIG


def test_set_seed_keeps_an_existing_cublas_workspace_config(monkeypatch) -> None:
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    set_seed(0)
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"


def test_set_seed_enables_deterministic_algorithms() -> None:
    set_seed(0)
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.deterministic
    assert not torch.backends.cudnn.benchmark


@pytest.mark.parametrize(
    ("decoder_type", "backbone_type"),
    [
        ("linear_probe", "dinov3"),
        ("conv_decoder", "dinov3"),
        ("unetr", "dinov3"),
        ("unetr", "cinema"),
    ],
)
def test_set_seed_fixes_decoder_init(decoder_type: str, backbone_type: str) -> None:
    def build(seed: int) -> dict[str, torch.Tensor]:
        set_seed(seed)
        decoder = get_decoder(
            decoder_type=decoder_type, backbone_type=backbone_type, embed_dim=8
        )
        return {k: v.clone() for k, v in decoder.state_dict().items()}

    first, again, other = build(0), build(0), build(1)
    assert all(torch.equal(first[k], again[k]) for k in first)
    # Some tensors (BatchNorm, biases) start constant; the conv weights do not.
    assert any(not torch.equal(first[k], other[k]) for k in first)


# ── run_segmentation.py ─────────────────────────────────────────────────────


def _load_run_segmentation_module():
    spec = importlib.util.spec_from_file_location("run_segmentation", RUN_SEGMENTATION)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        spec.loader.exec_module(module)
    return module


def _write_acdc_like_metadata(data_dir: Path) -> None:
    """ACDC layout: train/test metadata only, so the val split is carved by seed."""
    data_dir.mkdir(parents=True)
    pathologies = ["NOR", "DCM", "HCM", "MINF", "RV"]
    train = pd.DataFrame(
        {
            "pid": [f"p{i:03d}" for i in range(20)],
            "pathology": [c for c in pathologies for _ in range(4)],
            "n_slices": 4,
        }
    )
    test = pd.DataFrame(
        {"pid": ["t000", "t001"], "pathology": ["NOR", "DCM"], "n_slices": 4}
    )
    train.to_csv(data_dir / "train_metadata.csv", index=False)
    test.to_csv(data_dir / "test_metadata.csv", index=False)


class _FakeEndDiastoleEndSystoleDataset:
    """Stands in for CineMA's dataset, which only stores its arguments anyway."""

    def __init__(self, meta_df: pd.DataFrame, **_kwargs) -> None:
        self.meta_df = meta_df

    def __len__(self) -> int:
        return len(self.meta_df) * 2


def _stub_cinema_dataset_module(monkeypatch) -> None:
    """Let the real ``load_segmentation_datasets`` (and its split) run without SimpleITK.

    Importing ``cinema.segmentation.dataset`` pulls in SimpleITK, whose SWIG init
    emits a DeprecationWarning ("builtin type SwigPyPacked has no __module__
    attribute"); under this repo's ``filterwarnings = error`` that kills the
    interpreter with a segfault rather than raising.
    """
    fake = ModuleType("cinema.segmentation.dataset")
    fake.EndDiastoleEndSystoleDataset = _FakeEndDiastoleEndSystoleDataset  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cinema.segmentation.dataset", fake)


class _DummyCachedFeatureDataset(Dataset):
    def __init__(self, manifest: list[dict]):
        self.manifest = manifest

    def __len__(self) -> int:
        return 2

    def __getitem__(self, idx: int) -> dict:
        return {
            "features": torch.zeros((4, 2, 2), dtype=torch.float32),
            "label": torch.zeros((2, 2), dtype=torch.long),
            "pid": f"{idx:03d}",
            "is_ed": True,
            "z_idx": idx,
        }


def _run_main(
    tmp_path: Path,
    monkeypatch,
    *,
    seed_data_split: int = 0,
    seed_training: int = 0,
    backbone_rng_draws: int = 0,
    cache_dir: Path | None = None,
) -> tuple[dict[str, torch.Tensor], dict, Path]:
    """Run ``main()`` with the real seeding, split and split-guard code.

    Only the expensive parts (backbone, feature extraction, training, evaluation)
    are stubbed. The stub backbone draws ``backbone_rng_draws`` numbers from the
    global RNG, as constructing a real pretrained backbone does.

    Returns the decoder's initial weights, the results JSON and its path.
    """
    module = _load_run_segmentation_module()
    data_dir = tmp_path / "data"
    if not data_dir.exists():
        _write_acdc_like_metadata(data_dir)
    output_dir = tmp_path / f"results_{seed_data_split}_{seed_training}"

    args = SimpleNamespace(
        backbone="dinov3",
        decoder="linear_probe",
        dataset="acdc",
        data_dir=data_dir,
        output_dir=output_dir,
        cache_dir=cache_dir or tmp_path / f"cache_{seed_data_split}_{seed_training}",
        dinov3_model_name="dinov3_vits16",
        dinov3_repo_dir="models/dinov3/",
        dinov3_weights_path=None,
        sam_model_id="facebook/sam-vit-base",
        hf_cache_dir=tmp_path / "hf",
        cache_only=False,
        max_patients=None,
        use_layers=None,
        batch_size=2,
        lr=1e-3,
        weight_decay=1e-3,
        n_epochs=1,
        patience=1,
        dropout=0.1,
        seed_data_split=seed_data_split,
        seed_training=seed_training,
        device="cpu",
        no_auto_download=True,
    )

    def fake_load_backbone(*_args, **_kwargs):
        torch.randn(backbone_rng_draws)
        return torch.nn.Identity(), {"embed_dim": 4, "layer_indices": (3, 6, 9, 11)}

    captured: dict[str, dict[str, torch.Tensor]] = {}

    def fake_train_segmentation(**kwargs):
        model = kwargs["model"]
        captured["init"] = {k: v.clone() for k, v in model.state_dict().items()}
        return {"history": {}, "best_val_dice": 0.5, "best_epoch": 1}

    metrics = {"per_class_dice": {"BG": 1.0}, "macro_dice": 0.5}
    _stub_cinema_dataset_module(monkeypatch)
    monkeypatch.setattr(module, "parse_args", lambda: args)
    monkeypatch.setattr(module, "load_backbone", fake_load_backbone)
    monkeypatch.setattr(module, "verify_cache_splits", lambda _cache_dir: None)
    monkeypatch.setattr(module, "cache_features", lambda *_a, **_k: [])
    monkeypatch.setattr(module, "CachedFeatureDataset", _DummyCachedFeatureDataset)
    monkeypatch.setattr(module, "train_segmentation", fake_train_segmentation)
    monkeypatch.setattr(module, "evaluate", lambda *_a, **_k: metrics)
    monkeypatch.setattr(module, "evaluate_per_sample", lambda *_a, **_k: [])
    monkeypatch.setattr(module, "evaluate_per_stack", lambda *_a, **_k: [])

    module.main()

    (json_path,) = output_dir.glob("*.json")
    return captured["init"], json.loads(json_path.read_text()), json_path


def _same_weights(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_same_seed_training_gives_same_decoder_init(tmp_path, monkeypatch) -> None:
    first, _, _ = _run_main(tmp_path / "a", monkeypatch, seed_training=0)
    again, _, _ = _run_main(tmp_path / "b", monkeypatch, seed_training=0)
    other, _, _ = _run_main(tmp_path / "c", monkeypatch, seed_training=1)
    assert _same_weights(first, again)
    assert not _same_weights(first, other)


def test_decoder_init_does_not_depend_on_backbone_rng_use(
    tmp_path, monkeypatch
) -> None:
    """Building a backbone consumes the global RNG; the decoder init must not care."""
    none, _, _ = _run_main(tmp_path / "a", monkeypatch, backbone_rng_draws=0)
    many, _, _ = _run_main(tmp_path / "b", monkeypatch, backbone_rng_draws=1000)
    assert _same_weights(none, many)


def test_seed_training_does_not_change_the_split(tmp_path, monkeypatch) -> None:
    """Several training seeds can share one feature cache."""
    cache_dir = tmp_path / "shared_cache"
    _run_main(tmp_path, monkeypatch, seed_training=0, cache_dir=cache_dir)
    _run_main(tmp_path, monkeypatch, seed_training=1, cache_dir=cache_dir)
    manifest = json.loads((cache_dir / "split_manifest.json").read_text())
    assert manifest["split_seed"] == 0


def test_seed_data_split_still_guards_the_cache(tmp_path, monkeypatch) -> None:
    """A different split seed changes the ACDC val split, so the guard must refuse."""
    cache_dir = tmp_path / "shared_cache"
    _run_main(tmp_path, monkeypatch, seed_data_split=0, cache_dir=cache_dir)
    with pytest.raises(CacheSplitError):
        _run_main(tmp_path, monkeypatch, seed_data_split=1, cache_dir=cache_dir)


def test_results_record_both_seeds_and_filename_has_training_seed(
    tmp_path, monkeypatch
) -> None:
    _, results, json_path = _run_main(
        tmp_path, monkeypatch, seed_data_split=0, seed_training=3
    )
    cfg = results["config"]
    assert cfg["seed_data_split"] == 0
    assert cfg["seed_training"] == 3
    assert "seed" not in cfg
    assert cfg["device"] == "cpu"
    assert cfg["torch_version"] == torch.__version__
    assert cfg["deterministic_algorithms"] is True
    assert json_path.name.startswith("dinov3_vits16_linear_probe_seedtrain3_")


def test_old_seed_flag_is_rejected() -> None:
    """``--seed`` used to set the split too; it must fail, not silently change meaning."""
    proc = subprocess.run(
        [sys.executable, str(RUN_SEGMENTATION), "--backbone", "dinov3"]
        + ["--decoder", "linear_probe", "--seed", "1"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    # argparse rejects it as an ambiguous prefix of the two new flags.
    assert proc.returncode == 2, proc.stderr
    assert "--seed" in proc.stderr


# ── build_summary.py / aggregate_summary.py ──────────────────────────────────


def _load_script(name: str):
    path = REPO_ROOT / "scripts" / "segmentation" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_result(path: Path, macro_dice: float, **seeds) -> None:
    cfg = {
        "backbone": "dinov3",
        "model_name": "dinov3_vits16",
        "decoder": "unetr",
        "probe_layers": [3, 6, 9, 11],
        **seeds,
    }
    path.write_text(json.dumps({"config": cfg, "macro_dice": macro_dice}))


def test_build_summary_has_one_row_per_training_seed(tmp_path) -> None:
    build_summary = _load_script("build_summary")
    for seed in (2, 0, 1):
        _write_result(
            tmp_path / f"r{seed}.json", 0.8, seed_data_split=0, seed_training=seed
        )
    df = pd.read_csv(io.StringIO(build_summary.build_summary(tmp_path)))
    assert df["Seed Training"].tolist() == [0, 1, 2]
    assert df["Seed Data Split"].tolist() == [0, 0, 0]


def test_build_summary_reads_legacy_seed_as_data_split_seed(tmp_path) -> None:
    build_summary = _load_script("build_summary")
    _write_result(tmp_path / "legacy.json", 0.8, seed=0)
    df = pd.read_csv(io.StringIO(build_summary.build_summary(tmp_path)))
    assert df["Seed Data Split"].tolist() == [0]
    assert df["Seed Training"].isna().all()


def test_build_summary_rejects_duplicate_seeds(tmp_path) -> None:
    build_summary = _load_script("build_summary")
    for name in ("a", "b"):
        _write_result(
            tmp_path / f"{name}.json", 0.8, seed_data_split=0, seed_training=1
        )
    with pytest.raises(ValueError, match="Duplicate result"):
        build_summary.build_summary(tmp_path)


def test_aggregate_averages_training_seeds_within_each_dataset(tmp_path) -> None:
    aggregate_summary = _load_script("aggregate_summary")
    header = "Backbone,Model,Decoder,Seed Training,Macro Dice\n"
    rows = {
        "acdc": ["Dino,dinov3_vits16,UNETR,0,0.80", "Dino,dinov3_vits16,UNETR,1,0.90"],
        "mnm": ["Dino,dinov3_vits16,UNETR,0,0.70"],
        "mnm2": ["Dino,dinov3_vits16,UNETR,0,0.60"],
    }
    for dataset, lines in rows.items():
        (tmp_path / dataset).mkdir()
        (tmp_path / dataset / "summary.csv").write_text(header + "\n".join(lines))

    (row,) = aggregate_summary.aggregate(tmp_path).to_dict("records")
    assert row["acdc"] == pytest.approx(0.85)
    assert row["acdc SD"] == pytest.approx(0.0707, abs=1e-4)
    assert row["acdc N"] == 2
    assert row["mnm N"] == 1
    # each dataset weighs the same, however many seeds it has
    assert row["Mean Macro Dice"] == pytest.approx((0.85 + 0.70 + 0.60) / 3)
