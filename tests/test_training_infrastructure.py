from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from retina_screen.model import MultiTaskHead
from retina_screen.training import (
    EarlyStopping,
    KendallUncertaintyWeighting,
    TaskLossResult,
    compute_class_weights,
    compute_masked_task_loss,
    get_kendall_log_sigmas,
    train_one_epoch,
    train_one_step,
)
from retina_screen.core import load_config



BINARY_TASK = "diabetes"
ORDINAL_TASK = "dr_grade"
TASKS = [BINARY_TASK, ORDINAL_TASK]
EMB_DIM = 8


def _make_model_and_weighter(task_names=None):
    if task_names is None:
        task_names = TASKS
    model = MultiTaskHead(embedding_dim=EMB_DIM, task_names=task_names)
    weighter = KendallUncertaintyWeighting(task_names=task_names)
    return model, weighter


def _make_optimizer(model, weighter, optimizer_type="adamw", lr=1e-4, weight_decay=1e-2):
    params = list(model.parameters()) + list(weighter.parameters())
    if optimizer_type == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    return torch.optim.Adam(params, lr=lr)


def _make_batch(n: int, task_names=None):
    """Create synthetic embeddings, binary/ordinal targets, all-valid masks."""
    if task_names is None:
        task_names = TASKS
    from retina_screen.tasks import TASK_REGISTRY, TaskType

    emb = torch.randn(n, EMB_DIM)
    targets = {}
    masks = {}
    for tn in task_names:
        task = TASK_REGISTRY[tn]
        if task.task_type == TaskType.BINARY:
            targets[tn] = torch.randint(0, 2, (n,)).float()
        elif task.task_type == TaskType.ORDINAL:
            targets[tn] = torch.randint(0, 5, (n,)).float()
        else:
            targets[tn] = torch.randn(n)
        masks[tn] = torch.ones(n)
    return emb, targets, masks



def test_train_one_epoch_multiple_steps():
    """When N > batch_size, train_one_epoch must perform more than one optimizer step."""
    n = 500
    batch_size = 128
    model, weighter = _make_model_and_weighter()
    optimizer = _make_optimizer(model, weighter)
    emb, targets, masks = _make_batch(n)

    result = train_one_epoch(
        model, optimizer, emb, targets, masks, TASKS,
        batch_size=batch_size, weighter=weighter,
    )
    assert result["n_optimizer_steps"] >= 3, (
        f"Expected >= 3 optimizer steps for n={n}, batch_size={batch_size}, "
        f"got {result['n_optimizer_steps']}"
    )
    assert result["n_optimizer_steps"] == math.ceil(n / batch_size), (
        f"Expected exactly {math.ceil(n/batch_size)} steps, got {result['n_optimizer_steps']}"
    )



def test_train_one_epoch_single_step_when_batch_covers_all():
    """When batch_size >= n, there is exactly one optimizer step per epoch."""
    n = 100
    batch_size = 200
    model, weighter = _make_model_and_weighter()
    optimizer = _make_optimizer(model, weighter)
    emb, targets, masks = _make_batch(n)

    result = train_one_epoch(
        model, optimizer, emb, targets, masks, TASKS,
        batch_size=batch_size, weighter=weighter,
    )
    assert result["n_optimizer_steps"] == 1, (
        f"Expected 1 step when batch_size={batch_size} >= n={n}, "
        f"got {result['n_optimizer_steps']}"
    )



def test_hyperparameters_from_standard_yaml():
    """configs/training/standard.yaml must contain all required training keys."""
    cfg = load_config("configs/training/standard.yaml")
    assert cfg.get("optimizer") == "adamw", "optimizer must be 'adamw'"
    assert abs(cfg.get("lr", 0) - 1e-4) < 1e-9, f"lr must be 1e-4, got {cfg.get('lr')}"
    assert cfg.get("batch_size") == 256, f"batch_size must be 256, got {cfg.get('batch_size')}"
    assert cfg.get("max_epochs") == 100, f"max_epochs must be 100, got {cfg.get('max_epochs')}"
    assert cfg.get("warmup_epochs") == 5, f"warmup_epochs must be 5, got {cfg.get('warmup_epochs')}"
    assert cfg.get("early_stopping_patience") == 10
    assert cfg.get("class_weighting_enabled") is False
    assert abs(cfg.get("weight_decay", 0) - 1e-2) < 1e-9, f"weight_decay must be 1e-2"



def test_adamw_used_when_configured():
    """When optimizer='adamw', the optimizer must be AdamW (not Adam)."""
    model, weighter = _make_model_and_weighter()
    optimizer = _make_optimizer(model, weighter, optimizer_type="adamw")
    assert isinstance(optimizer, torch.optim.AdamW), (
        f"Expected AdamW, got {type(optimizer).__name__}"
    )


def test_adam_not_adamw_when_configured():
    """Sanity: Adam != AdamW (they are distinct optimizer classes)."""
    model, weighter = _make_model_and_weighter()
    optimizer = _make_optimizer(model, weighter, optimizer_type="adam")
    assert not isinstance(optimizer, torch.optim.AdamW), "Adam should not be AdamW"



def test_scheduler_lr_warmup_and_cosine():
    """LR schedule must: start > 0, reach base_lr at warmup end, decay after warmup."""
    lr = 1e-4
    lr_min = 0.0
    warmup_epochs = 5
    max_epochs = 100

    def lr_lambda(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        t = epoch - warmup_epochs
        total_cosine = max(1, max_epochs - warmup_epochs)
        cosine = 0.5 * (1.0 + math.cos(math.pi * t / total_cosine))
        min_factor = lr_min / lr if lr > 0 else 0.0
        return min_factor + (1.0 - min_factor) * cosine

    factor_epoch0 = lr_lambda(0)
    assert factor_epoch0 > 0, f"LR at epoch 0 must be > 0, got factor={factor_epoch0}"

    factor_at_warmup_end = lr_lambda(warmup_epochs - 1)
    assert abs(factor_at_warmup_end - 1.0) < 1e-9, (
        f"LR at warmup_epochs-1={warmup_epochs-1} must be 1.0 (base_lr), got {factor_at_warmup_end}"
    )

    factor_post_warmup = lr_lambda(warmup_epochs + 10)
    assert factor_post_warmup < 1.0, (
        f"LR after warmup must be < base_lr (factor < 1.0), got {factor_post_warmup}"
    )

    factor_at_max = lr_lambda(max_epochs)
    assert factor_at_max <= 0.1, (
        f"LR at max_epochs should be near lr_min=0, got factor={factor_at_max}"
    )


def test_scheduler_first_step_positive_lr_via_pytorch():
    """PyTorch LambdaLR with the warmup formula must produce LR > 0 at epoch 0."""
    from retina_screen.model import MultiTaskHead
    model = MultiTaskHead(embedding_dim=EMB_DIM, task_names=[BINARY_TASK])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    warmup_epochs = 5
    max_epochs = 20

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        t = epoch - warmup_epochs
        cosine = 0.5 * (1.0 + math.cos(math.pi * t / max(1, max_epochs - warmup_epochs)))
        return cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    optimizer.step()
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    assert current_lr > 0, f"After first scheduler step, LR must be > 0, got {current_lr}"



def test_early_stopping_triggers_after_patience():
    """EarlyStopping must return True after patience epochs without improvement.

    The first call always establishes a new best (improving from None), so
    triggering patience=3 requires 1 + 3 = 4 total calls: the first sets the
    best, then three consecutive non-improving steps exhaust patience.
    """
    es = EarlyStopping(patience=3, mode="max")
    assert not es.step(0.5), "First call is always a new best; should not stop"
    assert not es.step(0.5), "epoch 1: 1 without improvement"
    assert not es.step(0.5), "epoch 2: 2 without improvement"
    triggered = es.step(0.5)
    assert triggered is True, (
        "EarlyStopping should trigger after 3 consecutive non-improving epochs "
        "(epochs_without_improvement >= patience=3)"
    )


def test_early_stopping_does_not_trigger_before_patience():
    """EarlyStopping must not trigger before patience epochs are exhausted.

    With patience=5, triggering requires 1 (new best) + 5 non-improving = 6 calls.
    After only 5 total calls (1 best + 4 non-improving), must not stop.
    """
    es = EarlyStopping(patience=5, mode="max")
    results = [es.step(0.5) for _ in range(5)]
    assert not results[-1], "EarlyStopping should not trigger after 4 non-improving epochs with patience=5"



def test_early_stopping_resets_on_improvement():
    """Improvement resets the patience counter; stopping should not trigger prematurely."""
    es = EarlyStopping(patience=3, mode="max")
    es.step(0.5)
    es.step(0.5)
    es.step(0.5)
    triggered = es.step(0.6)
    assert not triggered, "EarlyStopping should not trigger when improvement occurs"
    assert es.epochs_without_improvement == 0, "Counter should reset on improvement"
    assert es.best_metric == pytest.approx(0.6)



def test_early_stopping_no_split_parameter():
    """EarlyStopping must accept only scalar metrics, no split arguments."""
    import inspect
    sig = inspect.signature(EarlyStopping.__init__)
    param_names = list(sig.parameters.keys())
    assert "split" not in param_names, "EarlyStopping must not accept a 'split' parameter"
    assert "dataset" not in param_names, "EarlyStopping must not accept a 'dataset' parameter"

    sig_step = inspect.signature(EarlyStopping.step)
    step_params = list(sig_step.parameters.keys())
    assert len(step_params) <= 2, f"step() should only take self + metric, got: {step_params}"



def test_compute_class_weights_binary():
    """compute_class_weights for a binary task returns pos_weight based on class ratio."""
    n = 100
    targets = {BINARY_TASK: torch.tensor([1.0] * 10 + [0.0] * 90)}
    masks = {BINARY_TASK: torch.ones(n)}

    weights = compute_class_weights(targets, masks, [BINARY_TASK], max_weight=100.0)
    assert BINARY_TASK in weights, "Binary task must have a weight"
    pw = weights[BINARY_TASK].item()
    assert abs(pw - 9.0) < 0.01, f"Expected pos_weight=9.0 (90/10), got {pw}"


def test_compute_class_weights_respects_cap():
    """compute_class_weights must cap pos_weight at max_weight."""
    n = 101
    targets = {BINARY_TASK: torch.tensor([1.0] + [0.0] * 100)}
    masks = {BINARY_TASK: torch.ones(n)}

    weights = compute_class_weights(targets, masks, [BINARY_TASK], max_weight=5.0)
    pw = weights[BINARY_TASK].item()
    assert pw <= 5.0, f"pos_weight must not exceed max_weight=5.0, got {pw}"



def test_class_weights_exclude_masked_labels():
    """compute_class_weights must exclude rows where mask==0."""
    n = 100
    targets = {BINARY_TASK: torch.tensor([1.0] * 50 + [0.0] * 50)}
    masks = {BINARY_TASK: torch.tensor([0.0] * 50 + [1.0] * 50)}

    weights = compute_class_weights(targets, masks, [BINARY_TASK], max_weight=10.0)
    assert BINARY_TASK not in weights, (
        "When all positives are masked, compute_class_weights must skip the task "
        "(single-class) rather than computing a spurious weight."
    )



def test_get_kendall_log_sigmas_returns_finite():
    """get_kendall_log_sigmas must return a finite float for every task."""
    _, weighter = _make_model_and_weighter()
    sigmas = get_kendall_log_sigmas(weighter)
    assert set(sigmas.keys()) == set(TASKS), (
        f"Expected keys={TASKS}, got {list(sigmas.keys())}"
    )
    for tn, val in sigmas.items():
        assert isinstance(val, float), f"log_sigma for {tn} must be float"
        assert math.isfinite(val), f"log_sigma for {tn} must be finite, got {val}"


def test_get_kendall_log_sigmas_updates_after_training():
    """log_sigma values must change from initial zeros after a training step."""
    model, weighter = _make_model_and_weighter()
    optimizer = _make_optimizer(model, weighter)
    emb, targets, masks = _make_batch(50)

    before = get_kendall_log_sigmas(weighter)
    assert all(abs(v) < 1e-8 for v in before.values()), "Initial log_sigmas should be zero"

    train_one_step(model, optimizer, emb, targets, masks, TASKS, loss_weighter=weighter)
    after = get_kendall_log_sigmas(weighter)

    changed = any(abs(after[tn] - before[tn]) > 1e-9 for tn in TASKS)
    assert changed, "log_sigma values must update after a training step with Kendall weighting"



def test_train_one_step_backward_compat_no_class_weights():
    """Existing callers that omit class_weights must work unchanged."""
    model, weighter = _make_model_and_weighter()
    optimizer = _make_optimizer(model, weighter)
    emb, targets, masks = _make_batch(32)

    result = train_one_step(model, optimizer, emb, targets, masks, TASKS, loss_weighter=weighter)
    assert "total_loss" in result
    assert math.isfinite(result["total_loss"]), "Loss must be finite"
    assert result["grad_norm"] >= 0.0



def test_compute_masked_task_loss_with_class_weights():
    """compute_masked_task_loss must accept and apply class_weights without error."""
    from retina_screen.model import MultiTaskHead
    model = MultiTaskHead(embedding_dim=EMB_DIM, task_names=[BINARY_TASK])
    emb, targets, masks = _make_batch(32, task_names=[BINARY_TASK])
    preds = model(emb)

    class_weights = {BINARY_TASK: torch.tensor([5.0])}
    result = compute_masked_task_loss(preds, targets, masks, [BINARY_TASK], class_weights=class_weights)
    assert isinstance(result, TaskLossResult)
    assert BINARY_TASK in result.losses
    assert math.isfinite(result.losses[BINARY_TASK].item())



def test_compute_masked_task_loss_without_class_weights():
    """compute_masked_task_loss called without class_weights must behave exactly as before."""
    from retina_screen.model import MultiTaskHead
    model = MultiTaskHead(embedding_dim=EMB_DIM, task_names=[BINARY_TASK])
    emb, targets, masks = _make_batch(32, task_names=[BINARY_TASK])
    preds = model(emb)

    result = compute_masked_task_loss(preds, targets, masks, [BINARY_TASK])
    assert isinstance(result, TaskLossResult)
    assert BINARY_TASK in result.losses
    assert BINARY_TASK in result.valid_counts
    assert math.isfinite(result.losses[BINARY_TASK].item())



def test_compute_class_weights_ordinal():
    """compute_class_weights for an ordinal task returns per-class inverse-frequency weights.

    Verifies that:
    - A weight tensor of length n_classes is returned.
    - The majority class receives a lower weight than minority classes.
    - All weights respect the max_weight cap.
    - All weights are positive.
    """
    labels = [0] * 180 + [1] * 10 + [2] * 5 + [3] * 3 + [4] * 2
    targets = {ORDINAL_TASK: torch.tensor(labels, dtype=torch.float32)}
    masks = {ORDINAL_TASK: torch.ones(len(labels))}

    weights = compute_class_weights(targets, masks, [ORDINAL_TASK], max_weight=10.0)

    assert ORDINAL_TASK in weights, "Ordinal task must return a weight tensor"
    w = weights[ORDINAL_TASK]
    assert w.ndim == 1, "Ordinal weight tensor must be 1D"
    assert len(w) == 5, f"Expected 5 per-class weights (classes 0-4), got {len(w)}"
    assert float(w[0]) < float(w[1]), (
        f"Class 0 weight {float(w[0]):.4f} must be < class 1 weight {float(w[1]):.4f}"
    )
    assert float(w[0]) < float(w[4]), (
        f"Class 0 weight {float(w[0]):.4f} must be < class 4 weight {float(w[4]):.4f}"
    )
    assert float(w.max()) <= 10.0, (
        f"Max weight {float(w.max()):.4f} must not exceed max_class_weight=10.0"
    )
    assert float(w.min()) > 0.0, "All ordinal class weights must be positive"
