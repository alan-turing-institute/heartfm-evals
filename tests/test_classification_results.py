"""Tests for loading committed classification result JSONs.

All tests use synthetic JSON files in a temp dir — no real results required.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.classification._common import load_classifiers
from scripts.classification.build_summary import build_summary


def _write_result(
    path: Path, pooling: str = "gap", head: str = "logreg", **extra_config
) -> None:
    labels = [0, 1, 2, 1]
    path.write_text(
        json.dumps(
            {
                "config": {
                    "backbone": "cinema",
                    "model_name": "cinema_pretrained",
                    "embed_dim": 768,
                    "head": head,
                    "pooling": pooling,
                    **extra_config,
                },
                "five_way": {
                    "accuracy": 1.0,
                    "macro_f1": 1.0,
                    "predictions": labels,
                    "true_labels": labels,
                    "patient_ids": ["a", "b", "c", "d"],
                },
            }
        )
    )


class TestDuplicateResults:
    def test_distinct_configs_load(self, tmp_path):
        _write_result(tmp_path / "cinema_pretrained_logreg_gap_1.json", "gap")
        _write_result(tmp_path / "cinema_pretrained_logreg_cls_1.json", "cls")

        assert len(load_classifiers(tmp_path, "five_way")) == 2
        assert len(build_summary(tmp_path).strip().splitlines()) == 3

    def test_load_classifiers_rejects_duplicate(self, tmp_path):
        _write_result(tmp_path / "cinema_pretrained_logreg_gap_1.json")
        _write_result(tmp_path / "cinema_pretrained_logreg_gap_2.json")

        with pytest.raises(ValueError, match="Duplicate result"):
            load_classifiers(tmp_path, "five_way")

    def test_build_summary_rejects_duplicate(self, tmp_path):
        _write_result(tmp_path / "cinema_pretrained_logreg_gap_1.json")
        _write_result(tmp_path / "cinema_pretrained_logreg_gap_2.json")

        with pytest.raises(ValueError, match="Duplicate result"):
            build_summary(tmp_path)


def _write_legacy_result(path: Path, eval_mode: str, **extra_config) -> None:
    """Write a result as run_classification did before issue #83."""
    _write_result(path, **extra_config)
    data = json.loads(path.read_text())
    del data["config"]["head"]
    data["config"]["eval_mode"] = eval_mode
    path.write_text(json.dumps(data))


class TestHeadLabels:
    @pytest.mark.parametrize("head", ["logreg", "linear"])
    def test_head_round_trips(self, tmp_path, head):
        _write_result(tmp_path / f"cinema_pretrained_{head}_gap_1.json", head=head)

        assert list(load_classifiers(tmp_path, "five_way")) == [
            f"cinema_pretrained_{head}_gap"
        ]
        assert f",{head}," in build_summary(tmp_path)

    def test_legacy_finetune_loads_as_linear(self, tmp_path):
        # Older JSONs may also carry freeze_backbone; it never selects a head.
        _write_legacy_result(
            tmp_path / "cinema_pretrained_ftfrozen_gap_1.json",
            eval_mode="finetune",
            freeze_backbone=False,
        )

        assert list(load_classifiers(tmp_path, "five_way")) == [
            "cinema_pretrained_linear_gap"
        ]
        assert ",linear," in build_summary(tmp_path)

    def test_legacy_logreg_loads(self, tmp_path):
        _write_legacy_result(
            tmp_path / "cinema_pretrained_logreg_gap_1.json", eval_mode="logreg"
        )

        assert list(load_classifiers(tmp_path, "five_way")) == [
            "cinema_pretrained_logreg_gap"
        ]
        assert ",logreg," in build_summary(tmp_path)

    def test_legacy_and_new_linear_are_duplicates(self, tmp_path):
        _write_legacy_result(
            tmp_path / "cinema_pretrained_ftfrozen_gap_1.json", eval_mode="finetune"
        )
        _write_result(tmp_path / "cinema_pretrained_linear_gap_2.json", head="linear")

        with pytest.raises(ValueError, match="Duplicate result"):
            load_classifiers(tmp_path, "five_way")
        with pytest.raises(ValueError, match="Duplicate result"):
            build_summary(tmp_path)

    def test_unknown_head_rejected(self, tmp_path):
        _write_result(tmp_path / "cinema_pretrained_x_gap_1.json", head="x")

        with pytest.raises(ValueError, match="Unknown head"):
            load_classifiers(tmp_path, "five_way")
        with pytest.raises(ValueError, match="Unknown head"):
            build_summary(tmp_path)
