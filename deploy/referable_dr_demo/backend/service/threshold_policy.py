from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .bundle import DeploymentBundle
from .provenance import LOCAL_DIR

logger = logging.getLogger(__name__)

REFERABLE = "REFERABLE"
NOT_REFERABLE = "NOT REFERABLE"


class ThresholdInvalidError(RuntimeError):
    """Raised when the operating-point artifact fails a provenance/validation gate."""


@dataclass(frozen=True)
class OperatingPoint:
    threshold: float
    derivation_split: str
    target_sensitivity: float
    validation_sensitivity: float
    validation_specificity: float
    validation_positive_count: int
    validation_negative_count: int
    heldout_test_sensitivity: float
    heldout_test_specificity: float
    test_positive_count: int
    test_negative_count: int
    native_protocol: str
    threshold_selection_algorithm: str
    manifest: dict[str, Any]

    def decide(self, score: float) -> str:
        """Apply the frozen decision rule: score >= threshold -> REFERABLE."""
        return REFERABLE if score >= self.threshold else NOT_REFERABLE


def default_threshold_path() -> Path:
    override = os.environ.get("RETINA_SCREEN_DEMO_THRESHOLD_PATH")
    if override:
        return Path(override)
    return LOCAL_DIR / "operating_point.local.json"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ThresholdInvalidError(message)


def load_operating_point(
    bundle: DeploymentBundle, path: Path | None = None
) -> OperatingPoint:
    """Load + provenance-validate the operating point against ``bundle``."""
    p = path or default_threshold_path()
    if not p.exists():
        raise ThresholdInvalidError(
            f"Operating-point artifact not found at {p}. Derive it with: "
            "uv run python deploy/referable_dr_demo/analysis/derive_threshold.py"
        )
    m: dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))

    _require(
        m.get("derivation_split") == "validation",
        "operating point was not derived from the validation split "
        f"(got {m.get('derivation_split')!r}); refusing to use it.",
    )

    _require(
        m.get("backbone_checkpoint_sha256") == bundle.backbone_sha256,
        "operating point backbone hash does not match the active bundle.",
    )
    _require(
        m.get("head_checkpoint_sha256") == bundle.head_sha256,
        "operating point head-checkpoint hash does not match the active bundle.",
    )
    _require(
        m.get("preprocessing_hash") == bundle.preprocessing_hash,
        "operating point preprocessing hash does not match the active bundle.",
    )
    _require(
        m.get("task_ordering_hash") == bundle.task_ordering_hash,
        "operating point task ordering does not match the active bundle.",
    )
    _require(
        m.get("model_family") == bundle.manifest.get("model_family"),
        "operating point model family does not match the active bundle.",
    )
    _require(
        m.get("native_protocol") == bundle.manifest.get("native_protocol"),
        "operating point native protocol does not match the active bundle.",
    )

    threshold = m.get("threshold")
    _require(
        isinstance(threshold, (int, float)) and 0.0 <= float(threshold) <= 1.0,
        f"operating point threshold {threshold!r} is not a probability in [0, 1].",
    )

    return OperatingPoint(
        threshold=float(threshold),
        derivation_split=str(m["derivation_split"]),
        target_sensitivity=float(m["target_sensitivity"]),
        validation_sensitivity=float(m["validation_sensitivity"]),
        validation_specificity=float(m["validation_specificity"]),
        validation_positive_count=int(m["validation_positive_count"]),
        validation_negative_count=int(m["validation_negative_count"]),
        heldout_test_sensitivity=float(m["heldout_test_sensitivity"]),
        heldout_test_specificity=float(m["heldout_test_specificity"]),
        test_positive_count=int(m["test_positive_count"]),
        test_negative_count=int(m["test_negative_count"]),
        native_protocol=str(m["native_protocol"]),
        threshold_selection_algorithm=str(m.get("threshold_selection_algorithm", "")),
        manifest=m,
    )
