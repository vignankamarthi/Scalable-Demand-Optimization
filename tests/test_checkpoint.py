"""
Tests for per-model checkpointing in scripts/02_train.py.
Validates save/load round-trips, resume logic, fresh-run override,
atomic write safety, and numpy array serialization.
"""

import json
import os

import numpy as np
import pytest

from src.checkpoint import load_checkpoint, save_checkpoint


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def sample_results():
    """Fake model results dict matching the structure produced by the training loop."""
    return {
        "decision_tree": {
            "macro_f1": 0.7512,
            "balanced_accuracy": 0.7400,
            "confusion_matrix": np.array([[50, 5, 2], [3, 45, 8], [1, 7, 42]]),
            "per_class": {
                "low": {"precision": 0.926, "recall": 0.877, "f1": 0.901, "support": 57},
                "medium": {"precision": 0.789, "recall": 0.804, "f1": 0.796, "support": 56},
                "high": {"precision": 0.808, "recall": 0.840, "f1": 0.824, "support": 50},
            },
            "train_time_s": 12.34,
            "predict_time_s": 0.56,
            "params": {"max_depth": None, "criterion": "gini"},
        },
        "random_forest": {
            "macro_f1": 0.8321,
            "balanced_accuracy": 0.8200,
            "confusion_matrix": np.array([[55, 2, 0], [1, 50, 5], [0, 3, 47]]),
            "per_class": {
                "low": {"precision": 0.982, "recall": 0.965, "f1": 0.973, "support": 57},
                "medium": {"precision": 0.909, "recall": 0.893, "f1": 0.901, "support": 56},
                "high": {"precision": 0.904, "recall": 0.940, "f1": 0.922, "support": 50},
            },
            "train_time_s": 95.67,
            "predict_time_s": 1.23,
            "params": {"n_estimators": 300, "max_depth": None, "n_jobs": -1},
        },
    }


@pytest.fixture
def single_result(sample_results):
    """Just the decision_tree result."""
    return {"decision_tree": sample_results["decision_tree"]}


# -----------------------------------------------------------------------
# save_checkpoint tests
# -----------------------------------------------------------------------

class TestSaveCheckpoint:
    def test_creates_file(self, tmp_path, sample_results):
        path = str(tmp_path / "checkpoint.json")
        save_checkpoint(sample_results, path=path)
        assert os.path.exists(path)

    def test_valid_json(self, tmp_path, sample_results):
        path = str(tmp_path / "checkpoint.json")
        save_checkpoint(sample_results, path=path)
        with open(path) as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_no_tmp_file_remains(self, tmp_path, sample_results):
        path = str(tmp_path / "checkpoint.json")
        save_checkpoint(sample_results, path=path)
        assert not os.path.exists(path + ".tmp")

    def test_confusion_matrix_serialized_as_list(self, tmp_path, sample_results):
        path = str(tmp_path / "checkpoint.json")
        save_checkpoint(sample_results, path=path)
        with open(path) as f:
            data = json.load(f)
        for model_name in sample_results:
            assert isinstance(data[model_name]["confusion_matrix"], list)

    def test_all_models_present(self, tmp_path, sample_results):
        path = str(tmp_path / "checkpoint.json")
        save_checkpoint(sample_results, path=path)
        with open(path) as f:
            data = json.load(f)
        assert set(data.keys()) == set(sample_results.keys())

    def test_overwrites_existing(self, tmp_path, single_result, sample_results):
        path = str(tmp_path / "checkpoint.json")
        save_checkpoint(single_result, path=path)
        save_checkpoint(sample_results, path=path)
        with open(path) as f:
            data = json.load(f)
        assert len(data) == 2

    def test_preserves_metric_values(self, tmp_path, sample_results):
        path = str(tmp_path / "checkpoint.json")
        save_checkpoint(sample_results, path=path)
        with open(path) as f:
            data = json.load(f)
        assert data["decision_tree"]["macro_f1"] == 0.7512
        assert data["random_forest"]["train_time_s"] == 95.67


# -----------------------------------------------------------------------
# load_checkpoint tests
# -----------------------------------------------------------------------

class TestLoadCheckpoint:
    def test_returns_empty_if_no_file(self, tmp_path):
        path = str(tmp_path / "nonexistent.json")
        result = load_checkpoint(path=path, fresh=False)
        assert result == {}

    def test_returns_empty_if_fresh(self, tmp_path, sample_results):
        path = str(tmp_path / "checkpoint.json")
        save_checkpoint(sample_results, path=path)
        result = load_checkpoint(path=path, fresh=True)
        assert result == {}

    def test_roundtrip_preserves_all_keys(self, tmp_path, sample_results):
        path = str(tmp_path / "checkpoint.json")
        save_checkpoint(sample_results, path=path)
        loaded = load_checkpoint(path=path, fresh=False)
        assert set(loaded.keys()) == set(sample_results.keys())

    def test_roundtrip_confusion_matrix_is_numpy(self, tmp_path, sample_results):
        path = str(tmp_path / "checkpoint.json")
        save_checkpoint(sample_results, path=path)
        loaded = load_checkpoint(path=path, fresh=False)
        for name in loaded:
            assert isinstance(loaded[name]["confusion_matrix"], np.ndarray)

    def test_roundtrip_confusion_matrix_values(self, tmp_path, sample_results):
        path = str(tmp_path / "checkpoint.json")
        save_checkpoint(sample_results, path=path)
        loaded = load_checkpoint(path=path, fresh=False)
        for name in sample_results:
            np.testing.assert_array_equal(
                loaded[name]["confusion_matrix"],
                sample_results[name]["confusion_matrix"],
            )

    def test_roundtrip_scalar_metrics(self, tmp_path, sample_results):
        path = str(tmp_path / "checkpoint.json")
        save_checkpoint(sample_results, path=path)
        loaded = load_checkpoint(path=path, fresh=False)
        for name in sample_results:
            assert loaded[name]["macro_f1"] == sample_results[name]["macro_f1"]
            assert loaded[name]["balanced_accuracy"] == sample_results[name]["balanced_accuracy"]
            assert loaded[name]["train_time_s"] == sample_results[name]["train_time_s"]
            assert loaded[name]["predict_time_s"] == sample_results[name]["predict_time_s"]

    def test_roundtrip_per_class(self, tmp_path, sample_results):
        path = str(tmp_path / "checkpoint.json")
        save_checkpoint(sample_results, path=path)
        loaded = load_checkpoint(path=path, fresh=False)
        for name in sample_results:
            assert loaded[name]["per_class"] == sample_results[name]["per_class"]

    def test_roundtrip_params(self, tmp_path, sample_results):
        path = str(tmp_path / "checkpoint.json")
        save_checkpoint(sample_results, path=path)
        loaded = load_checkpoint(path=path, fresh=False)
        for name in sample_results:
            assert loaded[name]["params"] == sample_results[name]["params"]


# -----------------------------------------------------------------------
# Incremental checkpoint (simulates training loop)
# -----------------------------------------------------------------------

class TestIncrementalCheckpointing:
    def test_incremental_save_and_resume(self, tmp_path, sample_results):
        """Simulate: train model 1 -> checkpoint -> 'crash' -> resume -> train model 2."""
        path = str(tmp_path / "checkpoint.json")

        # Phase 1: save first model only
        partial = {"decision_tree": sample_results["decision_tree"]}
        save_checkpoint(partial, path=path)

        # Phase 2: "resume" - load checkpoint, skip completed, add second model
        loaded = load_checkpoint(path=path, fresh=False)
        assert "decision_tree" in loaded
        assert "random_forest" not in loaded

        # Simulate training second model
        loaded["random_forest"] = sample_results["random_forest"]
        save_checkpoint(loaded, path=path)

        # Verify final state
        final = load_checkpoint(path=path, fresh=False)
        assert set(final.keys()) == {"decision_tree", "random_forest"}

    def test_skip_logic(self, tmp_path, sample_results):
        """Verify that model names in checkpoint would be skipped in training loop."""
        path = str(tmp_path / "checkpoint.json")
        save_checkpoint(sample_results, path=path)
        loaded = load_checkpoint(path=path, fresh=False)

        all_models = ["decision_tree", "random_forest", "knn", "mlp_small", "mlp_medium", "mlp_large"]
        to_train = [m for m in all_models if m not in loaded]
        assert to_train == ["knn", "mlp_small", "mlp_medium", "mlp_large"]
