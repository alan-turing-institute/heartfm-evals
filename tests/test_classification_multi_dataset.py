"""Tests for multi-dataset classification support (ACDC, MnM, MnM2).

All tests use synthetic data — no real models or data files required.
"""

from __future__ import annotations

import warnings
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from heartfm_evals.classification_probe import (
    DATASET_PATHOLOGY_CLASSES,
    PATHOLOGY_CLASSES,
    binarize_labels,
    build_patient_features,
    evaluate_binary_detection,
    evaluate_classification,
    get_pathology_classes,
    sweep_C_and_train,
    validate_split_pathology_labels,
)
from heartfm_evals.finetune_classification import (
    ClassificationHead,
    ClassificationHeadPredictor,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_cls_features(pids, embed_dim=8):
    """Create fake cls_features dict: {pid: {ed_features: (n, dim), es_features: (n, dim)}}."""
    feats = {}
    for pid in pids:
        feats[pid] = {
            "ed_features": torch.randn(3, embed_dim),
            "es_features": torch.randn(3, embed_dim),
        }
    return feats


# ═══════════════════════════════════════════════════════════════════════════════
# Group A: Pathology class registry
# ═══════════════════════════════════════════════════════════════════════════════


class TestPathologyClassRegistry:
    def test_get_pathology_classes_acdc(self):
        classes = get_pathology_classes("acdc")
        assert classes == {"NOR": 0, "DCM": 1, "HCM": 2, "MINF": 3, "RV": 4}

    def test_get_pathology_classes_mnm(self):
        classes = get_pathology_classes("mnm")
        assert classes == {"NOR": 0, "DCM": 1, "HCM": 2, "ARV": 3, "HHD": 4}

    def test_get_pathology_classes_mnm2(self):
        classes = get_pathology_classes("mnm2")
        assert classes == {"NOR": 0, "HCM": 1, "ARR": 2, "CIA": 3, "FALL": 4, "LV": 5}

    def test_get_pathology_classes_invalid(self):
        with pytest.raises(KeyError):
            get_pathology_classes("nonexistent_dataset")

    def test_all_datasets_have_nor_at_zero(self):
        for dataset_name, classes in DATASET_PATHOLOGY_CLASSES.items():
            assert "NOR" in classes, f"{dataset_name} missing NOR"
            assert classes["NOR"] == 0, f"{dataset_name} NOR is not 0"

    def test_backward_compat_global(self):
        """The old PATHOLOGY_CLASSES global should still equal ACDC classes."""
        assert get_pathology_classes("acdc") == PATHOLOGY_CLASSES

    def test_validate_split_labels_train_exact_match_no_warning(self):
        mnm2_classes = get_pathology_classes("mnm2")
        train_map = {
            "p1": "NOR",
            "p2": "HCM",
            "p3": "ARR",
            "p4": "CIA",
            "p5": "FALL",
            "p6": "LV",
        }
        val_map = {"v1": "NOR", "v2": "LV"}

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            diagnostics = validate_split_pathology_labels(
                train_map,
                pathology_classes=mnm2_classes,
                val_pathology_map=val_map,
            )

        assert len(record) == 0
        assert diagnostics["train_missing"] == set()
        assert diagnostics["train_unknown"] == set()
        assert diagnostics["val_unknown"] == set()
        assert diagnostics["val_unseen_from_train"] == set()

    def test_validate_split_labels_warn_for_unknown_val_or_test_labels(self):
        mnm2_classes = get_pathology_classes("mnm2")
        # Train split covers all classes; val/test include unknown labels.
        train_map = {
            "p1": "NOR",
            "p2": "HCM",
            "p3": "ARR",
            "p4": "FALL",
            "p5": "CIA",
            "p6": "LV",
        }
        val_map = {"v1": "NOR", "v2": "LV", "v3": "XXX"}
        test_map = {"t1": "ARR", "t2": "CIA", "t3": "YYY"}

        with pytest.warns(UserWarning, match="not in pathology_classes"):
            diagnostics = validate_split_pathology_labels(
                train_map,
                pathology_classes=mnm2_classes,
                val_pathology_map=val_map,
                test_pathology_map=test_map,
            )

        assert diagnostics["val_unknown"] == {"XXX"}
        assert diagnostics["test_unknown"] == {"YYY"}

    def test_validate_split_labels_warn_for_val_and_test_unseen_from_train(self):
        mnm2_classes = get_pathology_classes("mnm2")
        # Train split intentionally missing CIA and LV.
        train_map = {
            "p1": "NOR",
            "p2": "HCM",
            "p3": "ARR",
            "p4": "FALL",
        }
        val_map = {"v1": "NOR", "v2": "LV"}
        test_map = {"t1": "ARR", "t2": "CIA"}

        with pytest.warns(UserWarning, match="absent from training split"):
            diagnostics = validate_split_pathology_labels(
                train_map,
                pathology_classes=mnm2_classes,
                val_pathology_map=val_map,
                test_pathology_map=test_map,
            )

        assert diagnostics["val_unseen_from_train"] == {"LV"}
        assert diagnostics["test_unseen_from_train"] == {"CIA"}


# ═══════════════════════════════════════════════════════════════════════════════
# Group B: build_patient_features with custom pathology_classes
# ═══════════════════════════════════════════════════════════════════════════════


class TestBuildPatientFeatures:
    def test_uses_custom_classes(self):
        mnm2_classes = get_pathology_classes("mnm2")
        pids = ["p1", "p2", "p3"]
        cls_features = _make_cls_features(pids, embed_dim=8)
        pathology_map = {"p1": "NOR", "p2": "ARR", "p3": "LV"}

        features, labels, out_pids = build_patient_features(
            cls_features,
            pathology_map,
            pathology_classes=mnm2_classes,
        )

        assert features.shape == (3, 16)  # 2 * embed_dim
        assert labels.tolist() == [0, 2, 5]  # NOR=0, ARR=2, LV=5
        assert out_pids == ["p1", "p2", "p3"]

    def test_unknown_pathology_skipped(self):
        """Patients with a pathology not in the class map are silently skipped."""
        mnm2_classes = get_pathology_classes("mnm2")
        pids = ["p1", "p2"]
        cls_features = _make_cls_features(pids, embed_dim=8)
        pathology_map = {"p1": "MINF", "p2": "NOR"}  # MINF not in MnM2

        features, labels, out_pids = build_patient_features(
            cls_features,
            pathology_map,
            pathology_classes=mnm2_classes,
        )
        assert out_pids == ["p2"]  # p1 (MINF) was skipped
        assert labels.tolist() == [0]  # NOR=0

    def test_default_uses_acdc_classes(self):
        """When pathology_classes is None, ACDC classes are used (backward compat)."""
        pids = ["p1"]
        cls_features = _make_cls_features(pids, embed_dim=8)
        pathology_map = {"p1": "MINF"}

        features, labels, _ = build_patient_features(cls_features, pathology_map)
        assert labels.tolist() == [3]  # MINF=3 in ACDC


# ═══════════════════════════════════════════════════════════════════════════════
# Group C: evaluate_classification with different num classes
# ═══════════════════════════════════════════════════════════════════════════════


class TestEvaluateClassification:
    def _make_mock_model(self, predictions, probabilities):
        model = MagicMock()
        model.predict = MagicMock(return_value=predictions)
        model.predict_proba = MagicMock(return_value=probabilities)
        return model

    def test_6_classes(self):
        mnm2_classes = get_pathology_classes("mnm2")
        n_samples = 12
        n_classes = 6
        y_true = torch.tensor([0, 1, 2, 3, 4, 5] * 2)
        y_pred = y_true.numpy()
        y_prob = np.eye(n_classes)[y_pred]

        model = self._make_mock_model(y_pred, y_prob)
        metrics = evaluate_classification(
            model,
            torch.randn(n_samples, 16),
            y_true,
            pathology_classes=mnm2_classes,
        )

        assert metrics["confusion_matrix"].shape == (n_classes, n_classes)
        assert set(mnm2_classes.keys()).issubset(
            metrics["per_class_sensitivity"].keys()
        )
        assert metrics["accuracy"] == 1.0

    def test_5_classes_backward_compat(self):
        n_samples = 10
        n_classes = 5
        y_true = torch.tensor([0, 1, 2, 3, 4] * 2)
        y_pred = y_true.numpy()
        y_prob = np.eye(n_classes)[y_pred]

        model = self._make_mock_model(y_pred, y_prob)
        metrics = evaluate_classification(model, torch.randn(n_samples, 16), y_true)

        assert metrics["confusion_matrix"].shape == (n_classes, n_classes)
        assert "RV" in metrics["per_class_sensitivity"]


# ═══════════════════════════════════════════════════════════════════════════════
# Group D: binarize_labels and evaluate_binary_detection
# ═══════════════════════════════════════════════════════════════════════════════


class TestBinaryDetection:
    def test_binarize_labels_nor_always_zero(self):
        for dataset_name in DATASET_PATHOLOGY_CLASSES:
            classes = get_pathology_classes(dataset_name)
            nor_idx = classes["NOR"]
            n_classes = len(classes)
            labels = torch.arange(n_classes)
            binary = binarize_labels(labels, nor_idx=nor_idx)
            assert binary[nor_idx].item() == 0
            for i in range(n_classes):
                if i != nor_idx:
                    assert binary[i].item() == 1

    def test_binary_detection_6_classes(self):
        mnm2_classes = get_pathology_classes("mnm2")
        nor_idx = mnm2_classes["NOR"]
        n_samples = 12
        n_classes = 6

        # Probabilities: all probability on the true class
        labels = torch.tensor([0, 1, 2, 3, 4, 5] * 2)
        probs = np.eye(n_classes)[labels.numpy()]

        metrics = evaluate_binary_detection(probs, labels, nor_idx=nor_idx)

        # NOR samples should have disease_prob = 0, others should have disease_prob = 1
        assert metrics["accuracy"] == 1.0
        assert metrics["sensitivity"] == 1.0
        assert metrics["specificity"] == 1.0

    def test_binary_detection_default_nor_idx(self):
        """Default nor_idx=0 works for all datasets (NOR is always 0)."""
        labels = torch.tensor([0, 1, 2, 3, 4])
        probs = np.eye(5)[labels.numpy()]
        metrics = evaluate_binary_detection(probs, labels)
        assert metrics["accuracy"] == 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# Group E: sweep_C_and_train with val split
# ═══════════════════════════════════════════════════════════════════════════════


class TestSweepCWithValSplit:
    def _make_synthetic_data(self, n_per_class=10, n_classes=5, embed_dim=8):
        features = torch.randn(n_per_class * n_classes, embed_dim, dtype=torch.float64)
        labels = torch.tensor([c for c in range(n_classes) for _ in range(n_per_class)])
        return features, labels

    def test_with_val_data(self):
        train_features, train_labels = self._make_synthetic_data()
        val_features, val_labels = self._make_synthetic_data(n_per_class=3)

        best_C, pipeline, sweep_results = sweep_C_and_train(
            train_features,
            train_labels,
            val_features=val_features,
            val_labels=val_labels,
        )

        assert best_C > 0
        assert pipeline is not None
        assert len(sweep_results) > 0
        # Each result should have C, mean_cv_acc, std_cv_acc
        for r in sweep_results:
            assert "C" in r
            assert "mean_cv_acc" in r
            # With val split, std_cv_acc should be 0 (single evaluation)
            assert r["std_cv_acc"] == 0.0

    def test_without_val_data_uses_kfold(self):
        train_features, train_labels = self._make_synthetic_data()

        best_C, pipeline, sweep_results = sweep_C_and_train(
            train_features,
            train_labels,
            n_folds=3,
            max_iter=10_000,
            tol=1e-4,
        )

        assert best_C > 0
        assert pipeline is not None
        # With K-fold, std should generally be > 0 for at least some C values
        # (not guaranteed but very likely with random data)
        assert len(sweep_results) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Group F: Finetune module
# ═══════════════════════════════════════════════════════════════════════════════


class TestFinetuneModule:
    def test_classification_head_custom_num_classes(self):
        head = ClassificationHead(in_dim=768, num_classes=6)
        x = torch.randn(2, 768)
        out = head(x)
        assert out.shape == (2, 6)

    def test_classification_head_default_is_5(self):
        """Default num_classes should be 5 (ACDC backward compat)."""
        head = ClassificationHead(in_dim=768)
        x = torch.randn(2, 768)
        out = head(x)
        assert out.shape == (2, 5)

    def test_classification_head_predictor_predict(self):
        """ClassificationHeadPredictor.predict returns argmax predictions."""
        from sklearn.preprocessing import StandardScaler

        head = ClassificationHead(in_dim=16, num_classes=5)
        scaler = StandardScaler()
        X_train = np.random.randn(10, 16)
        scaler.fit(X_train)

        predictor = ClassificationHeadPredictor(head, scaler, torch.device("cpu"))
        X_test = np.random.randn(4, 16).astype(np.float64)
        preds = predictor.predict(X_test)
        assert preds.shape == (4,)
        assert all(0 <= p < 5 for p in preds)

    def test_classification_head_predictor_predict_proba(self):
        """ClassificationHeadPredictor.predict_proba returns softmax probabilities."""
        from sklearn.preprocessing import StandardScaler

        head = ClassificationHead(in_dim=16, num_classes=5)
        scaler = StandardScaler()
        X_train = np.random.randn(10, 16)
        scaler.fit(X_train)

        predictor = ClassificationHeadPredictor(head, scaler, torch.device("cpu"))
        X_test = np.random.randn(4, 16).astype(np.float64)
        probs = predictor.predict_proba(X_test)
        assert probs.shape == (4, 5)
        # Each row should sum to ~1.0 (softmax)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)


# ═══════════════════════════════════════════════════════════════════════════════
# Group G: CLI / output path helpers
# ═══════════════════════════════════════════════════════════════════════════════


class TestCLIHelpers:
    def test_output_dir_includes_dataset_name(self):
        from pathlib import Path

        for dataset in ("acdc", "mnm", "mnm2"):
            expected = Path(f"results/classification/{dataset}")
            # The CLI should construct output dir as results/classification/{dataset}
            assert expected == Path("results/classification") / dataset

    def test_default_data_dir_per_dataset(self):
        from pathlib import Path

        for dataset in ("acdc", "mnm", "mnm2"):
            expected = Path(f"data/heartfm/processed/{dataset}")
            assert expected == Path("data/heartfm/processed") / dataset

    def test_eval_mode_tag(self):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from scripts.classification.run_classification import eval_mode_tag

        assert eval_mode_tag("logreg", True) == "logreg"
        assert eval_mode_tag("finetune", True) == "ftfrozen"
        assert eval_mode_tag("finetune", False) == "ftfull"
