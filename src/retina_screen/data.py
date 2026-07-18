from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Sequence

from retina_screen.feature_policy import FeaturePolicy, ModelInputMode
from retina_screen.schema import CanonicalSample
from retina_screen.tasks import TASK_REGISTRY, TaskType, get_task

logger = logging.getLogger(__name__)


MISSING_CLASS_PLACEHOLDER: float = -1.0
MISSING_REGRESSION_PLACEHOLDER: float = float("nan")
MISSING_METADATA_PLACEHOLDER: float = 0.0



@dataclass
class EncodedTarget:
    """Encoded label and mask for a single (sample, task) pair.

    Attributes
    ----------
    task_name:
        Name of the task this encoding is for.
    value:
        Encoded target value.  For missing labels, this is the appropriate
        placeholder constant; downstream code must use ``mask`` to ignore it.
    mask:
        1.0 if the label is observed and valid; 0.0 if the label is missing
        or unmapped.  Loss functions must multiply by this mask (or filter by it
        for regression targets with NaN placeholders — see module docstring).
    """

    task_name: str
    value: float
    mask: float


@dataclass
class TaskTargetsBatch:
    """Batch of encoded targets and masks for a collection of samples.

    ``targets[task_name]`` and ``masks[task_name]`` are parallel lists of
    length == number of samples.  Missing labels have mask=0.0.
    """

    targets: dict[str, list[float]]
    masks: dict[str, list[float]]


@dataclass
class MetadataFeatures:
    """FeaturePolicy-filtered metadata for a single sample.

    Attributes
    ----------
    values:
        Raw Python values (enum, float, str, int, or None) for each field
        in ``allowed_fields``.  No numeric encoding is performed here;
        that is deferred to later data-layer batching/collate utilities.
    observation_mask:
        For each field in ``allowed_fields``, 1.0 if the field has a
        non-None value on this sample, else 0.0.
        This is INDEPENDENT of FeaturePolicy: a field allowed by policy but
        absent on this sample still gets mask=0.0.
    allowed_fields:
        The set of field names permitted by FeaturePolicy for this
        (task, mode, explicit_allow) combination.
    """

    values: dict[str, Any]
    observation_mask: dict[str, float]
    allowed_fields: frozenset[str]




def encode_task_target(
    sample: CanonicalSample,
    task_name: str,
) -> EncodedTarget:
    """Encode the label of ``sample`` for ``task_name`` into (value, mask).

    Rules
    -----
    - Unknown task name: raises ``KeyError`` (fail closed).
    - Missing label (None): ``mask=0.0``, placeholder value.
    - Observed binary label (0 or 1): ``value=float(label)``, ``mask=1.0``.
      Any other int observed in a binary field raises ``ValueError``.
    - Observed ordinal label (0..num_classes-1): ``value=float(label)``, ``mask=1.0``.
      Labels outside the valid range raise ``ValueError``.
    - Observed regression label: ``value=float(label)``, ``mask=1.0``.
      Observed NaN is treated as missing (``mask=0.0``).
    - Task with ``target_encoding`` (e.g. sex → Sex enum):
      - None: ``mask=0.0``.
      - Mapped value: ``value=encoding[label]``, ``mask=1.0``.
      - Unmapped value (e.g. Sex.UNKNOWN): ``mask=0.0`` (not a fabricated class).
    """
    task = get_task(task_name)
    raw = getattr(sample, task.target_column)

    if task.target_encoding is not None:
        if raw is None:
            return EncodedTarget(task_name, MISSING_CLASS_PLACEHOLDER, 0.0)
        encoded = task.target_encoding.get(raw)
        if encoded is None:
            logger.debug(
                "task=%s: target value %r not in target_encoding; masking.",
                task_name,
                raw,
            )
            return EncodedTarget(task_name, MISSING_CLASS_PLACEHOLDER, 0.0)
        return EncodedTarget(task_name, float(encoded), 1.0)

    if raw is None:
        if task.task_type == TaskType.REGRESSION:
            return EncodedTarget(task_name, MISSING_REGRESSION_PLACEHOLDER, 0.0)
        return EncodedTarget(task_name, MISSING_CLASS_PLACEHOLDER, 0.0)

    if task.task_type == TaskType.BINARY:
        if raw not in (0, 1):
            raise ValueError(
                f"Task {task_name!r}: expected binary label 0 or 1, "
                f"got {raw!r} for sample {sample.sample_id!r}."
            )
        return EncodedTarget(task_name, float(raw), 1.0)

    if task.task_type == TaskType.ORDINAL:
        if task.num_classes is None:
            raise ValueError(
                f"Ordinal task {task_name!r} has num_classes=None in TASK_REGISTRY."
            )
        if not isinstance(raw, int) or not (0 <= raw < task.num_classes):
            raise ValueError(
                f"Task {task_name!r}: expected ordinal label in "
                f"[0, {task.num_classes - 1}], got {raw!r} for sample "
                f"{sample.sample_id!r}."
            )
        return EncodedTarget(task_name, float(raw), 1.0)

    if task.task_type == TaskType.REGRESSION:
        f_raw = float(raw)
        if math.isnan(f_raw):
            return EncodedTarget(task_name, MISSING_REGRESSION_PLACEHOLDER, 0.0)
        return EncodedTarget(task_name, f_raw, 1.0)

    raise ValueError(
        f"Task {task_name!r} has unrecognised task_type={task.task_type!r}."
    )




def build_task_targets_and_masks(
    samples: Sequence[CanonicalSample],
    task_names: Sequence[str],
) -> TaskTargetsBatch:
    """Encode labels and masks for a batch of samples across multiple tasks.

    Parameters
    ----------
    samples:
        Sequence of canonical samples to encode.
    task_names:
        Sequence of task names to encode.  All must exist in TASK_REGISTRY.

    Returns
    -------
    TaskTargetsBatch
        ``targets[task_name]`` and ``masks[task_name]`` are lists of length
        ``len(samples)``.  Masks are 0.0 for missing/unmapped labels.
    """
    for tn in task_names:
        get_task(tn)

    targets: dict[str, list[float]] = {tn: [] for tn in task_names}
    masks: dict[str, list[float]] = {tn: [] for tn in task_names}

    for sample in samples:
        for tn in task_names:
            enc = encode_task_target(sample, tn)
            targets[tn].append(enc.value)
            masks[tn].append(enc.mask)

    return TaskTargetsBatch(targets=targets, masks=masks)




def build_metadata_features(
    sample: CanonicalSample,
    task_name: str,
    feature_policy: FeaturePolicy,
    mode: str | ModelInputMode,
    explicit_allow: frozenset[str] | None = None,
) -> MetadataFeatures:
    """Apply FeaturePolicy and return allowed metadata for a sample.

    FeaturePolicy determines WHICH fields are allowed (access control).
    ``observation_mask`` determines WHETHER an allowed field has a real value
    on this specific sample (data availability).  These are independent:
    a field allowed by policy but absent on this sample gets mask=0.0.

    Parameters
    ----------
    sample:
        The canonical sample to extract metadata from.
    task_name:
        Task name; used by FeaturePolicy to apply per-task leakage blocks.
    feature_policy:
        FeaturePolicy instance. Called before any metadata is exposed.
    mode:
        Model input mode (str or ModelInputMode enum).
    explicit_allow:
        Optional set of restricted fields to permit explicitly.

    Returns
    -------
    MetadataFeatures
        Raw field values (not numerically encoded) and per-field observation mask.
    """
    allowed = feature_policy.allowed_fields(task_name, mode, explicit_allow)

    values: dict[str, Any] = {}
    observation_mask: dict[str, float] = {}
    for field in sorted(allowed):
        val = getattr(sample, field, None)
        values[field] = val
        observation_mask[field] = 0.0 if val is None else 1.0

    return MetadataFeatures(
        values=values,
        observation_mask=observation_mask,
        allowed_fields=allowed,
    )
