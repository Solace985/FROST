"""preprocessing_parity.py -- exact native-392 preprocessing for FROST.

Wraps the canonical ``retina_screen.preprocessing`` route used to build the
accepted native-392 embedding cache. It does NOT re-implement or approximate any
transform: it constructs the canonical :class:`PreprocessingConfig` from the
canonical preprocessing YAML and calls the canonical ``preprocess_image`` with
``mode="extract"`` (the exact deterministic route the extractor used).

Fail-closed: the constructed config's preprocessing hash must equal the accepted
native-392 cache hash, or import-time validation raises.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image

from .provenance import REPO_ROOT, ensure_src_importable

ensure_src_importable()

from retina_screen.preprocessing import (  # noqa: E402  (after sys.path shim)
    PreprocessingConfig,
    get_preprocessing_hash,
    preprocess_image,
)

logger = logging.getLogger(__name__)

# Canonical native-392 preprocessing identity (Stage 8D-3.5 native-392 comparator).
NATIVE_392_PREPROCESSING_NAME = "retfound_green_native392"
NATIVE_392_PREPROCESSING_YAML = (
    REPO_ROOT / "configs" / "preprocessing" / f"{NATIVE_392_PREPROCESSING_NAME}.yaml"
)
# Accepted native-392 cache preprocessing hash (fail-closed anchor).
EXPECTED_PREPROCESSING_HASH = "003e0cecb1459266"
EXPECTED_IMAGE_SIZE = 392

# Fallback constants if the canonical YAML is unavailable; still validated against
# EXPECTED_PREPROCESSING_HASH so they cannot silently diverge from the cache.
_FALLBACK_MEAN = (0.485, 0.456, 0.406)
_FALLBACK_STD = (0.229, 0.224, 0.225)


def _load_native_392_config() -> PreprocessingConfig:
    """Build the canonical native-392 PreprocessingConfig from its YAML.

    Mirrors ``scripts/03_extract_embeddings._build_prep_config`` field coercion.
    Falls back to the known ImageNet/392 constants only if the YAML is absent.
    """
    if NATIVE_392_PREPROCESSING_YAML.exists():
        raw: dict[str, Any] = yaml.safe_load(
            NATIVE_392_PREPROCESSING_YAML.read_text(encoding="utf-8")
        )
        return PreprocessingConfig(
            image_size=int(raw.get("image_size", EXPECTED_IMAGE_SIZE)),
            mean=tuple(raw.get("mean", _FALLBACK_MEAN)),
            std=tuple(raw.get("std", _FALLBACK_STD)),
            use_clahe=bool(raw.get("use_clahe", False)),
            use_graham=bool(raw.get("use_graham", False)),
            interpolation=str(raw.get("interpolation", "bilinear")),
            random_hflip_p=float(raw.get("random_hflip_p", 0.0)),
            random_rotation_deg=float(raw.get("random_rotation_deg", 0.0)),
            color_jitter=bool(raw.get("color_jitter", False)),
        )
    logger.warning(
        "Canonical preprocessing YAML not found at %s; using validated fallback "
        "constants (hash-checked against the accepted cache).",
        NATIVE_392_PREPROCESSING_YAML,
    )
    return PreprocessingConfig(
        image_size=EXPECTED_IMAGE_SIZE,
        mean=_FALLBACK_MEAN,
        std=_FALLBACK_STD,
        interpolation="bilinear",
    )


# Build + validate once at import. Fail closed if the hash diverges from the
# accepted native-392 cache: this guarantees the demonstrator preprocesses
# uploads exactly as the study cache was built.
NATIVE_392_CONFIG: PreprocessingConfig = _load_native_392_config()
_ACTUAL_HASH = get_preprocessing_hash(NATIVE_392_CONFIG)
if _ACTUAL_HASH != EXPECTED_PREPROCESSING_HASH:
    raise RuntimeError(
        "BLOCKED_DEPLOYMENT_INFERENCE_PARITY_UNRESOLVED: native-392 preprocessing "
        f"hash {_ACTUAL_HASH!r} != accepted cache hash "
        f"{EXPECTED_PREPROCESSING_HASH!r}. The preprocessing route does not match "
        "the accepted study cache; refusing to proceed."
    )
if NATIVE_392_CONFIG.image_size != EXPECTED_IMAGE_SIZE:
    raise RuntimeError(
        "BLOCKED_DEPLOYMENT_INFERENCE_PARITY_UNRESOLVED: native-392 image_size "
        f"{NATIVE_392_CONFIG.image_size} != {EXPECTED_IMAGE_SIZE}."
    )


def preprocessing_hash() -> str:
    """Return the canonical native-392 preprocessing hash (validated at import)."""
    return _ACTUAL_HASH


def preprocessing_config_path() -> Path:
    """Path to the canonical preprocessing YAML (for provenance recording)."""
    return NATIVE_392_PREPROCESSING_YAML


def preprocess(image: Image.Image) -> torch.Tensor:
    """Exact native-392 preprocessing of one PIL image -> (1, 3, 392, 392) float32.

    Uses the canonical ``preprocess_image(..., mode="extract")`` route:
    RGB conversion -> Resize(392, bilinear) -> CenterCrop(392) -> ToTensor ->
    ImageNet Normalize. Identical to the route that produced the accepted cache.
    """
    tensor = preprocess_image(image, NATIVE_392_CONFIG, mode="extract")  # (3,392,392)
    return tensor.unsqueeze(0)  # (1,3,392,392)
