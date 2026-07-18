from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image

from . import preprocessing_parity
from .bundle import DeploymentBundle
from .provenance import ensure_src_importable

ensure_src_importable()

import torch  # noqa: E402

from retina_screen.embeddings import BackboneConfig, load_backbone  # noqa: E402
from retina_screen.evaluation.referable_dr import (  # noqa: E402
    compute_referable_dr_score,
)
from retina_screen.model import build_head  # noqa: E402

logger = logging.getLogger(__name__)


def referable_dr_score(dr_grade_logits: np.ndarray) -> np.ndarray:
    """Canonical referable-DR score: softmax(logits)[:, 2:5].sum(axis=1).

    Thin re-export of the canonical implementation so callers (and tests) consume
    the exact study definition rather than a re-derived formula.
    """
    return compute_referable_dr_score(dr_grade_logits)


@dataclass
class InferenceResult:
    referable_score: float
    dr_grade_logits: list[float]
    dr_grade_probs: list[float]
    embedding_dim: int
    input_resolution: str
    timings_ms: dict[str, float] = field(default_factory=dict)


class InferenceService:
    """Frozen, load-once referable-DR inference service."""

    def __init__(self, bundle: DeploymentBundle, device: str = "cpu") -> None:
        self.bundle = bundle
        self.device = torch.device(device)
        self._load_count = 0

        cfg = BackboneConfig(**bundle.backbone_config_kwargs())
        self.backbone = load_backbone(cfg, self.device)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        if any(p.requires_grad for p in self.backbone.parameters()):
            raise RuntimeError("Backbone is not fully frozen after load.")

        state = torch.load(
            bundle.head_checkpoint, map_location=self.device, weights_only=True
        )
        self.head = build_head(
            embedding_dim=bundle.embedding_dim,
            task_names=list(bundle.task_order),
            head_type="multitask",
        )
        self.head.load_state_dict(state)
        self.head.eval()
        for p in self.head.parameters():
            p.requires_grad_(False)
        self.head.to(self.device)

        self.dr_grade_key = bundle.manifest["dr_grade_task_key"]
        self._load_count += 1

        self._warmup()
        logger.info(
            "InferenceService ready (backbone=%s, embedding_dim=%d, device=%s).",
            cfg.name, bundle.embedding_dim, self.device,
        )

    def _warmup(self) -> None:
        synthetic = Image.new("RGB", (512, 512), color=(127, 127, 127))
        _ = self.infer(synthetic)

    def embed(self, image: Image.Image) -> np.ndarray:
        """Return the frozen 384-d embedding for one image (float32, shape (384,)).

        Exactly reproduces the cache-extraction route:
        preprocess(mode="extract") -> backbone(...).squeeze(0).
        """
        tensor = preprocessing_parity.preprocess(image).to(self.device)
        with torch.inference_mode():
            emb = self.backbone(tensor).squeeze(0)
        return emb.detach().cpu().numpy().astype(np.float32)

    def infer(self, image: Image.Image) -> InferenceResult:
        """Full referable-DR inference for one PIL image."""
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        tensor = preprocessing_parity.preprocess(image).to(self.device)
        timings["preprocessing"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        with torch.inference_mode():
            embedding = self.backbone(tensor)
        timings["backbone"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        with torch.inference_mode():
            outputs = self.head(embedding)
            dr_logits = outputs[self.dr_grade_key]
        timings["head"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        dr_logits_np = dr_logits.detach().cpu().numpy().astype(np.float64)
        score = float(referable_dr_score(dr_logits_np)[0])
        probs = _softmax_rows(dr_logits_np)[0]
        timings["postprocessing"] = (time.perf_counter() - t0) * 1000.0

        emb_dim = int(embedding.shape[-1])
        if emb_dim != self.bundle.embedding_dim:
            raise RuntimeError(
                f"Embedding dim {emb_dim} != expected {self.bundle.embedding_dim}."
            )

        return InferenceResult(
            referable_score=score,
            dr_grade_logits=[float(x) for x in dr_logits_np[0]],
            dr_grade_probs=[float(x) for x in probs],
            embedding_dim=emb_dim,
            input_resolution=f"{preprocessing_parity.EXPECTED_IMAGE_SIZE}x"
            f"{preprocessing_parity.EXPECTED_IMAGE_SIZE}",
            timings_ms=timings,
        )

    @property
    def load_count(self) -> int:
        return self._load_count


def _softmax_rows(logits_2d: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax (matches the canonical score helper)."""
    arr = np.asarray(logits_2d, dtype=np.float64)
    shifted = arr - arr.max(axis=1, keepdims=True)
    exp_s = np.exp(shifted)
    return exp_s / exp_s.sum(axis=1, keepdims=True)


_SERVICE: InferenceService | None = None


def init_service(bundle: DeploymentBundle, device: str = "cpu") -> InferenceService:
    """Initialise the process-wide singleton exactly once."""
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = InferenceService(bundle, device=device)
    return _SERVICE


def get_service() -> InferenceService:
    if _SERVICE is None:
        raise RuntimeError("InferenceService not initialised; call init_service first.")
    return _SERVICE


def is_loaded() -> bool:
    return _SERVICE is not None


def reset_service_for_tests() -> None:
    """Test-only: drop the singleton so a fresh service can be constructed."""
    global _SERVICE
    _SERVICE = None
