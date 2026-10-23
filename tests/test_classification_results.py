"""Tests for loading committed classification result JSONs.

All tests use synthetic JSON files in a temp dir — no real results required.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.classification._common import load_classifiers
from scripts.classification.build_summary import build_summary


def _write_result(path: Path, pooling: str = "gap") -> None:
    labels = [0, 1, 2, 1]
    path.write_text(
        json.dumps(
            {
                "config": {
                    "backbone": "cinema",
                    "model_name": "cinema_pretrained",
                    "embed_dim": 768,
                    "eval_mode": "logreg",
                    "pooling": pooling,
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
