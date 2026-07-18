from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def evaluate_mod():
    spec = importlib.util.spec_from_file_location(
        "_scripts_05_evaluate_persample",
        _PROJECT_ROOT / "scripts" / "05_evaluate.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod



def _make_eval_samples(n: int = 5):
    """Return a list of minimal objects with sample_id and patient_id."""
    return [
        SimpleNamespace(sample_id=f"s{i:03d}", patient_id=f"p{i // 2:03d}")
        for i in range(n)
    ]


def _make_task_arrays(n: int, task_names: list[str], *, binary_tasks, ordinal_tasks):
    """Return preds_np, targets, masks dicts for given tasks."""
    preds_np: dict = {}
    targets: dict = {}
    masks: dict = {}
    for t in task_names:
        if t in ordinal_tasks:
            n_classes = ordinal_tasks[t]
            preds_np[t] = np.random.randn(n, n_classes).astype(np.float32)
            targets[t]  = np.random.randint(0, n_classes, size=n).astype(np.float32)
        else:
            preds_np[t] = np.random.randn(n).astype(np.float32)
            targets[t]  = np.random.randint(0, 2, size=n).astype(np.float32)
        masks[t] = np.ones(n, dtype=np.float32)
    return preds_np, targets, masks



class TestSavePerSamplePredictions:
    """Unit tests for _save_per_sample_predictions (Stage 8D-3.5 A1 patch)."""

    BINARY_TASK = "diabetes"
    ORDINAL_TASK = "dr_grade"
    ORDINAL_CLASSES = 5
    TASK_NAMES = [ORDINAL_TASK, BINARY_TASK]

    N = 5

    def _run(self, evaluate_mod, tmp_path):
        eval_samples = _make_eval_samples(self.N)
        preds_np, targets, masks = _make_task_arrays(
            self.N,
            self.TASK_NAMES,
            binary_tasks={self.BINARY_TASK},
            ordinal_tasks={self.ORDINAL_TASK: self.ORDINAL_CLASSES},
        )
        evaluate_mod._save_per_sample_predictions(
            tmp_path, eval_samples, self.TASK_NAMES,
            preds_np, targets, masks, git_sha="test_sha_abc123",
        )
        return tmp_path, preds_np, targets, masks

    def test_predictions_npz_created(self, evaluate_mod, tmp_path):
        self._run(evaluate_mod, tmp_path)
        assert (tmp_path / "predictions.npz").exists(), "predictions.npz not created"

    def test_schema_json_created(self, evaluate_mod, tmp_path):
        self._run(evaluate_mod, tmp_path)
        assert (tmp_path / "predictions_schema.json").exists(), "predictions_schema.json not created"

    def test_npz_no_pickle(self, evaluate_mod, tmp_path):
        """npz must be loadable with allow_pickle=False (no object arrays beyond sample_id/patient_id)."""
        self._run(evaluate_mod, tmp_path)
        data = np.load(tmp_path / "predictions.npz", allow_pickle=True)
        assert "sample_id" in data
        assert "patient_id" in data

    def test_sample_id_present_and_correct_length(self, evaluate_mod, tmp_path):
        self._run(evaluate_mod, tmp_path)
        data = np.load(tmp_path / "predictions.npz", allow_pickle=True)
        assert len(data["sample_id"]) == self.N
        assert data["sample_id"][0] == "s000"
        assert data["sample_id"][-1] == f"s{self.N - 1:03d}"

    def test_patient_id_present_and_correct_length(self, evaluate_mod, tmp_path):
        self._run(evaluate_mod, tmp_path)
        data = np.load(tmp_path / "predictions.npz", allow_pickle=True)
        assert len(data["patient_id"]) == self.N

    def test_logit_arrays_present_for_all_tasks(self, evaluate_mod, tmp_path):
        self._run(evaluate_mod, tmp_path)
        data = np.load(tmp_path / "predictions.npz", allow_pickle=True)
        for t in self.TASK_NAMES:
            key = f"logit__{t}"
            assert key in data, f"Missing key {key}"
            assert data[key].shape[0] == self.N, f"Wrong n_samples for {key}"

    def test_label_arrays_present_for_all_tasks(self, evaluate_mod, tmp_path):
        self._run(evaluate_mod, tmp_path)
        data = np.load(tmp_path / "predictions.npz", allow_pickle=True)
        for t in self.TASK_NAMES:
            key = f"label__{t}"
            assert key in data, f"Missing key {key}"
            assert data[key].shape[0] == self.N

    def test_mask_arrays_present_for_all_tasks(self, evaluate_mod, tmp_path):
        self._run(evaluate_mod, tmp_path)
        data = np.load(tmp_path / "predictions.npz", allow_pickle=True)
        for t in self.TASK_NAMES:
            key = f"mask__{t}"
            assert key in data, f"Missing key {key}"
            assert data[key].shape[0] == self.N

    def test_binary_logit_dtype_preserved(self, evaluate_mod, tmp_path):
        """Float32 in → float32 out for binary task logits."""
        self._run(evaluate_mod, tmp_path)
        data = np.load(tmp_path / "predictions.npz", allow_pickle=True)
        assert data[f"logit__{self.BINARY_TASK}"].dtype == np.float32

    def test_ordinal_logit_shape_is_2d(self, evaluate_mod, tmp_path):
        """Ordinal (dr_grade) logit has shape (n_samples, n_classes)."""
        self._run(evaluate_mod, tmp_path)
        data = np.load(tmp_path / "predictions.npz", allow_pickle=True)
        logit = data[f"logit__{self.ORDINAL_TASK}"]
        assert logit.ndim == 2, f"Expected 2D ordinal logit, got {logit.shape}"
        assert logit.shape == (self.N, self.ORDINAL_CLASSES)

    def test_schema_json_required_keys(self, evaluate_mod, tmp_path):
        self._run(evaluate_mod, tmp_path)
        with open(tmp_path / "predictions_schema.json", encoding="utf-8") as fh:
            schema = json.load(fh)
        required_keys = [
            "n_eval_samples", "tasks", "task_types",
            "logit_dtype", "label_dtype", "mask_dtype",
            "score_orientation", "masking_convention",
            "field_name_convention", "sample_id_field", "patient_id_field",
            "evaluation_script_git_commit", "evaluation_script_path",
            "produced_by_patch",
        ]
        for k in required_keys:
            assert k in schema, f"Missing key {k!r} in predictions_schema.json"

    def test_schema_n_eval_samples(self, evaluate_mod, tmp_path):
        self._run(evaluate_mod, tmp_path)
        with open(tmp_path / "predictions_schema.json", encoding="utf-8") as fh:
            schema = json.load(fh)
        assert schema["n_eval_samples"] == self.N

    def test_schema_tasks_list(self, evaluate_mod, tmp_path):
        self._run(evaluate_mod, tmp_path)
        with open(tmp_path / "predictions_schema.json", encoding="utf-8") as fh:
            schema = json.load(fh)
        assert schema["tasks"] == self.TASK_NAMES

    def test_schema_git_sha(self, evaluate_mod, tmp_path):
        self._run(evaluate_mod, tmp_path)
        with open(tmp_path / "predictions_schema.json", encoding="utf-8") as fh:
            schema = json.load(fh)
        assert schema["evaluation_script_git_commit"] == "test_sha_abc123"

    def test_schema_score_orientation_binary(self, evaluate_mod, tmp_path):
        self._run(evaluate_mod, tmp_path)
        with open(tmp_path / "predictions_schema.json", encoding="utf-8") as fh:
            schema = json.load(fh)
        orient = schema["score_orientation"]
        assert self.BINARY_TASK in orient
        assert orient[self.BINARY_TASK]["score_format"] == "raw_logit"
        assert orient[self.BINARY_TASK]["higher_means_positive"] is True

    def test_schema_score_orientation_ordinal(self, evaluate_mod, tmp_path):
        self._run(evaluate_mod, tmp_path)
        with open(tmp_path / "predictions_schema.json", encoding="utf-8") as fh:
            schema = json.load(fh)
        orient = schema["score_orientation"]
        assert self.ORDINAL_TASK in orient
        assert orient[self.ORDINAL_TASK]["score_format"] == "raw_per_class_logit"

    def test_schema_masking_convention(self, evaluate_mod, tmp_path):
        self._run(evaluate_mod, tmp_path)
        with open(tmp_path / "predictions_schema.json", encoding="utf-8") as fh:
            schema = json.load(fh)
        conv = schema["masking_convention"]
        assert conv["missing_label_placeholder"] == -1.0
        assert "1.0 = observed" in conv["mask_meaning"]

    def test_alignment_error_raised_on_length_mismatch(self, evaluate_mod, tmp_path):
        """Helper must raise RuntimeError if any per-task array length != n_eval_samples."""
        eval_samples = _make_eval_samples(5)
        preds_np = {
            self.ORDINAL_TASK: np.zeros((5, self.ORDINAL_CLASSES), dtype=np.float32),
            self.BINARY_TASK: np.zeros(6, dtype=np.float32),
        }
        targets = {
            self.ORDINAL_TASK: np.zeros(5, dtype=np.float32),
            self.BINARY_TASK: np.zeros(5, dtype=np.float32),
        }
        masks = {
            self.ORDINAL_TASK: np.ones(5, dtype=np.float32),
            self.BINARY_TASK: np.ones(5, dtype=np.float32),
        }
        with pytest.raises(RuntimeError, match="alignment error"):
            evaluate_mod._save_per_sample_predictions(
                tmp_path, eval_samples, self.TASK_NAMES,
                preds_np, targets, masks, git_sha="test",
            )

    def test_double_underscore_field_convention(self, evaluate_mod, tmp_path):
        """All per-task keys must use double-underscore separator."""
        self._run(evaluate_mod, tmp_path)
        data = np.load(tmp_path / "predictions.npz", allow_pickle=True)
        for key in data.files:
            if key in ("sample_id", "patient_id"):
                continue
            assert "__" in key, f"Key {key!r} does not use double-underscore convention"
            prefix, task = key.split("__", 1)
            assert prefix in ("logit", "label", "mask"), f"Unknown prefix {prefix!r} in key {key!r}"
            assert task in self.TASK_NAMES, f"Unknown task {task!r} in key {key!r}"
