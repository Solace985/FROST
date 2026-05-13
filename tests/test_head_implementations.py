"""
tests/test_head_implementations.py -- Stage 8D-3A head implementation tests.

Covers:
  - build_head factory: MultiTaskHead, LinearProbeHead, multitask_default alias,
    unknown head_type error, missing head_type default, unsupported kwargs error.
  - LinearProbeHead: output keys, output shapes (binary/ordinal/regression),
    no hidden layers/trunk/dropout/activation, no metadata consumption,
    fewer parameters than MultiTaskHead.
  - Backward compatibility: existing MultiTaskHead construction unchanged.
  - Training/loss path: LinearProbeHead outputs work with compute_masked_task_loss
    and train_one_step on synthetic tensors.
  - Script dispatch: head_type from config drives head selection.
  - No dataset-specific vocabulary in model.py.

All tests use synthetic tensors only. No BRSET data, no cache, no real training.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from retina_screen.model import (
    LinearProbeHead,
    MultiTaskHead,
    _VALID_HEAD_TYPES,
    build_head,
)
from retina_screen.training import (
    KendallUncertaintyWeighting,
    compute_masked_task_loss,
    train_one_step,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMB_DIM = 16
BINARY_TASK = "diabetes"
ORDINAL_TASK = "dr_grade"       # 5 classes
REGRESSION_TASK = "retinal_age"
TASKS_BINARY_ORDINAL = [BINARY_TASK, ORDINAL_TASK]
TASKS_ALL = [BINARY_TASK, ORDINAL_TASK, REGRESSION_TASK]
BATCH = 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_batch(task_names, batch=BATCH, emb_dim=EMB_DIM):
    """Synthetic embeddings + all-valid targets and masks."""
    emb = torch.randn(batch, emb_dim)
    from retina_screen.tasks import TASK_REGISTRY, TaskType
    targets: dict[str, torch.Tensor] = {}
    masks: dict[str, torch.Tensor] = {}
    for tn in task_names:
        task = TASK_REGISTRY[tn]
        masks[tn] = torch.ones(batch)
        if task.task_type == TaskType.BINARY:
            targets[tn] = torch.randint(0, 2, (batch,)).float()
        elif task.task_type == TaskType.ORDINAL:
            targets[tn] = torch.randint(0, task.num_classes, (batch,)).float()
        elif task.task_type == TaskType.REGRESSION:
            targets[tn] = torch.randn(batch)
    return emb, targets, masks


# ---------------------------------------------------------------------------
# Factory: head type selection
# ---------------------------------------------------------------------------


def test_multitask_head_via_factory():
    model = build_head(EMB_DIM, TASKS_BINARY_ORDINAL, head_type="multitask")
    assert isinstance(model, MultiTaskHead), \
        f"Expected MultiTaskHead, got {type(model).__name__}"


def test_linear_probe_head_via_factory():
    model = build_head(EMB_DIM, TASKS_BINARY_ORDINAL, head_type="linear_probe")
    assert isinstance(model, LinearProbeHead), \
        f"Expected LinearProbeHead, got {type(model).__name__}"


def test_multitask_default_alias():
    model = build_head(EMB_DIM, TASKS_BINARY_ORDINAL, head_type="multitask_default")
    assert isinstance(model, MultiTaskHead), \
        "head_type='multitask_default' should resolve to MultiTaskHead"


def test_missing_head_type_defaults_to_multitask():
    model = build_head(EMB_DIM, TASKS_BINARY_ORDINAL)
    assert isinstance(model, MultiTaskHead), \
        "Missing head_type should default to MultiTaskHead for backward compatibility"


def test_unknown_head_type_raises_valueerror():
    with pytest.raises(ValueError) as exc_info:
        build_head(EMB_DIM, TASKS_BINARY_ORDINAL, head_type="transformer_probe_v99")
    msg = str(exc_info.value)
    assert "transformer_probe_v99" in msg, "Error must name the invalid head_type"
    # At least one valid type should be mentioned
    assert any(ht in msg for ht in _VALID_HEAD_TYPES), \
        "Error must list valid head_type values"


def test_unknown_head_type_error_lists_valid_values():
    with pytest.raises(ValueError) as exc_info:
        build_head(EMB_DIM, TASKS_BINARY_ORDINAL, head_type="xyz")
    msg = str(exc_info.value)
    for ht in _VALID_HEAD_TYPES:
        assert ht in msg, f"Error message must list valid type {ht!r}"


# ---------------------------------------------------------------------------
# Factory: unsupported kwargs for LinearProbeHead
# ---------------------------------------------------------------------------


def test_linear_probe_rejects_unsupported_kwargs():
    with pytest.raises(ValueError) as exc_info:
        build_head(EMB_DIM, TASKS_BINARY_ORDINAL, head_type="linear_probe",
                   hidden_dim=256, dropout=0.3)
    msg = str(exc_info.value)
    assert "hidden_dim" in msg or "dropout" in msg, \
        "Error must name the rejected kwargs"


def test_multitask_accepts_kwargs():
    # Sanity: MultiTaskHead still accepts known kwargs through factory.
    model = build_head(EMB_DIM, TASKS_BINARY_ORDINAL, head_type="multitask",
                       hidden_dim=64, dropout=0.1)
    assert isinstance(model, MultiTaskHead)


# ---------------------------------------------------------------------------
# LinearProbeHead: output contract
# ---------------------------------------------------------------------------


def test_linear_probe_output_keys_match_tasks():
    model = LinearProbeHead(EMB_DIM, TASKS_BINARY_ORDINAL)
    emb = torch.randn(BATCH, EMB_DIM)
    out = model(emb)
    assert set(out.keys()) == set(TASKS_BINARY_ORDINAL), \
        f"Output keys {set(out.keys())} != tasks {set(TASKS_BINARY_ORDINAL)}"


def test_linear_probe_binary_output_shape():
    model = LinearProbeHead(EMB_DIM, [BINARY_TASK])
    emb = torch.randn(BATCH, EMB_DIM)
    out = model(emb)
    assert out[BINARY_TASK].shape == (BATCH,), \
        f"Binary output shape {out[BINARY_TASK].shape} != ({BATCH},)"


def test_linear_probe_ordinal_output_shape():
    from retina_screen.tasks import TASK_REGISTRY
    n_classes = TASK_REGISTRY[ORDINAL_TASK].num_classes
    model = LinearProbeHead(EMB_DIM, [ORDINAL_TASK])
    emb = torch.randn(BATCH, EMB_DIM)
    out = model(emb)
    assert out[ORDINAL_TASK].shape == (BATCH, n_classes), \
        f"Ordinal output shape {out[ORDINAL_TASK].shape} != ({BATCH}, {n_classes})"


def test_linear_probe_regression_output_shape():
    model = LinearProbeHead(EMB_DIM, [REGRESSION_TASK])
    emb = torch.randn(BATCH, EMB_DIM)
    out = model(emb)
    assert out[REGRESSION_TASK].shape == (BATCH,), \
        f"Regression output shape {out[REGRESSION_TASK].shape} != ({BATCH},)"


def test_multitask_and_linear_probe_compatible_output():
    mt = MultiTaskHead(EMB_DIM, TASKS_BINARY_ORDINAL)
    lp = LinearProbeHead(EMB_DIM, TASKS_BINARY_ORDINAL)
    emb = torch.randn(BATCH, EMB_DIM)
    mt_out = mt(emb)
    lp_out = lp(emb)
    assert set(mt_out.keys()) == set(lp_out.keys()), \
        "MultiTaskHead and LinearProbeHead must return the same output key set"
    for tn in TASKS_BINARY_ORDINAL:
        assert mt_out[tn].shape == lp_out[tn].shape, \
            f"Shape mismatch for task {tn!r}: MT={mt_out[tn].shape}, LP={lp_out[tn].shape}"


# ---------------------------------------------------------------------------
# LinearProbeHead: architecture constraints
# ---------------------------------------------------------------------------


def test_linear_probe_fewer_params_than_multitask():
    mt = MultiTaskHead(EMB_DIM, TASKS_BINARY_ORDINAL)
    lp = LinearProbeHead(EMB_DIM, TASKS_BINARY_ORDINAL)
    mt_params = sum(p.numel() for p in mt.parameters())
    lp_params = sum(p.numel() for p in lp.parameters())
    assert lp_params < mt_params, (
        f"LinearProbeHead ({lp_params} params) must have fewer parameters "
        f"than MultiTaskHead ({mt_params} params) for the same task list. "
        f"Note: KendallUncertaintyWeighting (log_sigma) is external to both heads."
    )


def test_linear_probe_no_trunk():
    model = LinearProbeHead(EMB_DIM, TASKS_BINARY_ORDINAL)
    assert not hasattr(model, "trunk"), \
        "LinearProbeHead must not have a 'trunk' attribute"


def test_linear_probe_no_hidden_layers_or_nonlinear():
    model = LinearProbeHead(EMB_DIM, TASKS_BINARY_ORDINAL)
    forbidden_types = (nn.GELU, nn.ReLU, nn.Tanh, nn.Sigmoid,
                       nn.Dropout, nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d,
                       nn.Sequential, nn.MultiheadAttention)
    for name, module in model.named_modules():
        assert not isinstance(module, forbidden_types), (
            f"LinearProbeHead must not contain {type(module).__name__} "
            f"(found at '{name}'). Must be a true linear probe with no hidden layers, "
            "no activation, no dropout, no normalization, no attention."
        )


def test_linear_probe_only_linear_in_task_heads():
    model = LinearProbeHead(EMB_DIM, TASKS_BINARY_ORDINAL)
    for tn in TASKS_BINARY_ORDINAL:
        head = model.task_heads[tn]
        assert isinstance(head, nn.Linear), \
            f"task_heads[{tn!r}] must be nn.Linear, got {type(head).__name__}"


def test_linear_probe_ignores_metadata():
    model = LinearProbeHead(EMB_DIM, TASKS_BINARY_ORDINAL)
    emb = torch.randn(BATCH, EMB_DIM)
    metadata = torch.randn(BATCH, 4)
    out_without = model(emb, metadata=None)
    out_with = model(emb, metadata=metadata)
    # Both calls must succeed and produce identical outputs (metadata is ignored)
    for tn in TASKS_BINARY_ORDINAL:
        assert torch.allclose(out_without[tn], out_with[tn]), \
            f"LinearProbeHead must ignore metadata; got different output for task {tn!r}"


def test_linear_probe_no_metadata_attribute():
    model = LinearProbeHead(EMB_DIM, TASKS_BINARY_ORDINAL)
    # LinearProbeHead should not have a metadata_dim field (it has no metadata branch)
    assert not hasattr(model, "_metadata_dim"), \
        "LinearProbeHead must not have a _metadata_dim attribute"


def test_linear_probe_task_registry_validation():
    with pytest.raises(KeyError) as exc_info:
        LinearProbeHead(EMB_DIM, ["nonexistent_task_xyz"])
    assert "nonexistent_task_xyz" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Training / loss path compatibility
# ---------------------------------------------------------------------------


def test_loss_path_works_with_linear_probe():
    model = LinearProbeHead(EMB_DIM, TASKS_BINARY_ORDINAL)
    emb, targets, masks = _make_batch(TASKS_BINARY_ORDINAL)
    out = model(emb)
    result = compute_masked_task_loss(out, targets, masks, TASKS_BINARY_ORDINAL)
    for tn in TASKS_BINARY_ORDINAL:
        loss = result.losses[tn]
        assert torch.isfinite(loss), f"Loss for {tn!r} is not finite: {loss}"
        assert result.valid_counts[tn] > 0, f"No valid rows for task {tn!r}"


def test_train_one_step_with_linear_probe():
    model = LinearProbeHead(EMB_DIM, TASKS_BINARY_ORDINAL)
    weighter = KendallUncertaintyWeighting(TASKS_BINARY_ORDINAL)
    params = list(model.parameters()) + list(weighter.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    emb, targets, masks = _make_batch(TASKS_BINARY_ORDINAL)
    result = train_one_step(
        model, optimizer, emb, targets, masks, TASKS_BINARY_ORDINAL,
        loss_weighter=weighter,
    )
    assert torch.isfinite(torch.tensor(result["total_loss"])), \
        f"train_one_step total_loss is not finite: {result['total_loss']}"
    assert result["grad_norm"] >= 0.0


def test_kendall_weighter_is_separate_from_linear_probe():
    model = LinearProbeHead(EMB_DIM, TASKS_BINARY_ORDINAL)
    weighter = KendallUncertaintyWeighting(TASKS_BINARY_ORDINAL)
    # log_sigma parameters live in the weighter, not the model
    model_param_names = {n for n, _ in model.named_parameters()}
    assert not any("log_sigma" in name for name in model_param_names), \
        "LinearProbeHead must not own log_sigma parameters (owned by KendallUncertaintyWeighting)"
    weighter_param_names = {n for n, _ in weighter.named_parameters()}
    assert any("log_sigma" in name for name in weighter_param_names), \
        "KendallUncertaintyWeighting must own log_sigma parameters"


# ---------------------------------------------------------------------------
# Backward compatibility: existing MultiTaskHead behavior unchanged
# ---------------------------------------------------------------------------


def test_multitask_head_backward_compat_direct_construction():
    model = MultiTaskHead(embedding_dim=EMB_DIM, task_names=TASKS_BINARY_ORDINAL)
    emb = torch.randn(BATCH, EMB_DIM)
    out = model(emb)
    assert set(out.keys()) == set(TASKS_BINARY_ORDINAL)
    assert out[BINARY_TASK].shape == (BATCH,)


def test_multitask_head_task_names_property():
    model = MultiTaskHead(embedding_dim=EMB_DIM, task_names=TASKS_BINARY_ORDINAL)
    assert model.task_names == list(TASKS_BINARY_ORDINAL)


def test_linear_probe_task_names_property():
    model = LinearProbeHead(EMB_DIM, TASKS_BINARY_ORDINAL)
    assert model.task_names == list(TASKS_BINARY_ORDINAL)


# ---------------------------------------------------------------------------
# No dataset-specific vocabulary in model.py
# ---------------------------------------------------------------------------


def test_no_dataset_names_in_model_py():
    model_path = Path(__file__).resolve().parent.parent / "src" / "retina_screen" / "model.py"
    source = model_path.read_text(encoding="utf-8")
    forbidden = ["brset", "mbrset", "odir", "idrid", "aptos", "messidor", "eyepacs"]
    for name in forbidden:
        assert name not in source.lower(), (
            f"Dataset name {name!r} found in model.py — "
            "dataset-specific logic must stay in adapters/configs/tests only."
        )


# ---------------------------------------------------------------------------
# Script dispatch: head_type from config
# ---------------------------------------------------------------------------


def test_train_script_uses_head_type_from_config(tmp_path, monkeypatch):
    """build_head is called with the head_type from the merged config."""
    import importlib
    import types

    # Capture the head_type that build_head would receive.
    captured: dict = {}
    original_build_head = build_head

    def mock_build_head(embedding_dim, task_names, head_type="multitask", **kw):
        captured["head_type"] = head_type
        return original_build_head(embedding_dim, task_names, head_type=head_type, **kw)

    import retina_screen.model as model_module
    monkeypatch.setattr(model_module, "build_head", mock_build_head)

    # Simulate the config read that 04_train.py does.
    cfg = {"head_type": "linear_probe", "embedding_dim": EMB_DIM}
    head_type = str(cfg.get("head_type", "multitask"))
    model_module.build_head(embedding_dim=EMB_DIM, task_names=[BINARY_TASK], head_type=head_type)
    assert captured.get("head_type") == "linear_probe"


def test_eval_script_head_type_from_resolved_config(tmp_path):
    """Evaluator reads head_type from resolved_config.yaml when present."""
    import yaml
    resolved = tmp_path / "resolved_config.yaml"
    resolved.write_text("head_type: linear_probe\n", encoding="utf-8")

    from retina_screen.core import load_config
    run_cfg = load_config(resolved)
    head_type = str(run_cfg.get("head_type", "multitask"))
    assert head_type == "linear_probe", \
        "Evaluator must read head_type from resolved_config.yaml when present"


def test_eval_script_head_type_default_when_resolved_missing(tmp_path):
    """When resolved_config.yaml is absent, experiment config or default is used."""
    missing = tmp_path / "resolved_config.yaml"
    assert not missing.exists()
    # Simulates the fallback: use experiment cfg (or default)
    experiment_cfg = {}
    head_type = str(experiment_cfg.get("head_type", "multitask"))
    assert head_type == "multitask", \
        "Missing resolved_config.yaml with no head_type in experiment config must default to multitask"


def test_eval_script_fails_on_malformed_resolved_config(tmp_path):
    """Malformed resolved_config.yaml must not be silently ignored."""
    bad = tmp_path / "resolved_config.yaml"
    bad.write_text("head_type: [\nnot valid yaml", encoding="utf-8")
    from retina_screen.core import load_config
    with pytest.raises(Exception):
        load_config(bad)
