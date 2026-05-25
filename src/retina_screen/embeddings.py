"""
embeddings.py -- Backbone loading, frozen embedding extraction, and cache management.

Owns: backbone loading, one-image verification, frozen embedding extraction,
cache path construction, manifest writing/loading, checksum validation, cache repair.

Must not contain: task losses, fairness metrics, dashboard UI, native dataset parsing,
or any import of concrete adapter classes. Image loading is injected via an
image_loader callback to keep this module adapter-agnostic.

Cache namespace
---------------
    cache_root / backbone_name / dataset_source / preprocessing_hash /

Manifest columns
----------------
    sample_id, patient_id, dataset_source, cache_path, embedding_dim,
    backbone_name, backbone_version, preprocessing_hash, created_at, checksum

Silent cache skipping is FORBIDDEN (see docs/ai_context/04_forbidden_patterns.md).
Missing or corrupt cache entries raise CacheMissError / CacheCorruptError.

overwrite=False contract
------------------------
- Valid cache reuse requires a manifest row, matching namespace metadata, matching
  checksum, and exact one-dimensional shape (embedding_dim,).
- Existing orphan cache files without a manifest row are not trusted; they are
  re-extracted.
- Corrupt, missing, wrong-rank, or wrong-dim manifest-backed files raise
  CacheCorruptError / CacheMissError.

overwrite=True: always re-extract regardless of existing state.

Real backbone support (Stage 8A / 8D-3.5-B2)
---------------------------------------------
Supported model_type values: 'mock', 'resnet50', 'convnext_base', 'dinov2', 'retfound_green'.
Original RETFound/RETFound-MEH remains deferred/access-pending and is not part of B2.
RETFound-Green uses timm (vit_small_patch14_reg4_dinov2, 384-dim, matched-224, Decision 029).
Unknown model_type raises BackboneUnavailableError — never falls back silently to mock.
"""

from __future__ import annotations

import csv
import hashlib
import logging
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image

from retina_screen.preprocessing import (
    PreprocessingConfig,
    build_preprocessing_pipeline,
    get_preprocessing_hash,
)
from retina_screen.schema import CanonicalSample

logger = logging.getLogger(__name__)

MANIFEST_FIELDNAMES: list[str] = [
    "sample_id",
    "patient_id",
    "dataset_source",
    "cache_path",
    "embedding_dim",
    "backbone_name",
    "backbone_version",
    "preprocessing_hash",
    "created_at",
    "checksum",
]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class CacheMissError(FileNotFoundError):
    """Raised when an expected embedding file is absent from the cache."""


class CacheCorruptError(ValueError):
    """Raised when an embedding file fails checksum or dimension validation."""


class BackboneUnavailableError(RuntimeError):
    """Raised when a real backbone cannot be loaded.

    Causes include: missing dependencies, unavailable weights, network failure,
    or unknown model_type. NEVER falls back to MockBackbone silently.
    """


class BackboneDimensionError(ValueError):
    """Raised when a backbone produces an embedding with the wrong dimensionality."""


# ---------------------------------------------------------------------------
# BackboneConfig
# ---------------------------------------------------------------------------


@dataclass
class BackboneConfig:
    """Backbone identity and output specification."""

    name: str            # "mock", "dinov2_large", "retfound_green", etc.
    embedding_dim: int   # output dimensionality, e.g. 1024
    model_type: str      # "mock" | "dinov2" | "retfound_green" | "convnext_base" | "resnet50"
    version: str = ""    # version string; empty for mock
    checkpoint_path: str = ""  # local path to weights file; used by timm-loaded backbones
    input_size: int = 224      # image input resolution; 224 for matched-matrix, 392 for native
    global_pool: str = "token" # timm global_pool; "token" (CLS) or "avg" (native-392 protocol)


# ---------------------------------------------------------------------------
# MockBackbone
# ---------------------------------------------------------------------------


class MockBackbone(nn.Module):
    """Deterministic frozen backbone for testing without real weights.

    Architecture: AdaptiveAvgPool2d(1) → Flatten → Linear(in_channels, embedding_dim).
    Fixed seed=0 at init. All parameters have requires_grad=False.
    Creating a MockBackbone does NOT disturb the global RNG state.
    """

    def __init__(self, embedding_dim: int = 1024, in_channels: int = 3) -> None:
        super().__init__()
        self._embedding_dim = embedding_dim
        self.pool = nn.AdaptiveAvgPool2d(1)
        # Save and restore RNG state so MockBackbone construction is side-effect free.
        saved_state = torch.get_rng_state()
        torch.manual_seed(0)
        self.proj = nn.Linear(in_channels, embedding_dim, bias=True)
        torch.set_rng_state(saved_state)
        for param in self.parameters():
            param.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: (B, C, H, W) → (B, embedding_dim)."""
        pooled = self.pool(x)                   # (B, C, 1, 1)
        flat = pooled.view(pooled.size(0), -1)  # (B, C)
        return self.proj(flat)                  # (B, embedding_dim)


# ---------------------------------------------------------------------------
# Backbone loading helpers
# ---------------------------------------------------------------------------


def _freeze_backbone(model: nn.Module) -> None:
    """Set all parameters to requires_grad=False and switch to eval mode."""
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)


def _verify_embedding_dim(
    model: nn.Module, config: BackboneConfig, device: torch.device
) -> None:
    """Run a single test forward pass to confirm output dim matches config.embedding_dim.

    Must be called after model.to(device). Raises BackboneDimensionError on mismatch
    or if the forward pass itself fails.

    Uses config.input_size so that native-392 models are verified at their actual
    resolution rather than 224 (avoids spurious PE interpolation during verification).
    """
    input_size = getattr(config, "input_size", 224)
    dummy = torch.zeros(1, 3, input_size, input_size, device=device)
    with torch.no_grad():
        try:
            out = model(dummy)
        except Exception as exc:
            raise BackboneDimensionError(
                f"Test forward pass failed for backbone {config.name!r}: {exc}"
            ) from exc
    if out.ndim != 2 or out.shape[1] != config.embedding_dim:
        raise BackboneDimensionError(
            f"Backbone {config.name!r} produced shape {tuple(out.shape)}, "
            f"expected (1, {config.embedding_dim}). "
            f"Verify that the classifier head has been removed correctly and "
            f"that embedding_dim in the backbone config matches the actual output."
        )


def _load_resnet50(config: BackboneConfig, device: torch.device) -> nn.Module:
    try:
        from torchvision.models import ResNet50_Weights, resnet50  # noqa: PLC0415
    except ImportError as exc:
        raise BackboneUnavailableError(
            "torchvision is required for ResNet-50. "
            "Install with: pip install torchvision>=0.18"
        ) from exc
    try:
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    except Exception as exc:
        raise BackboneUnavailableError(
            f"Failed to load ResNet-50 weights (IMAGENET1K_V2): {exc}. "
            f"Check network access or pre-cached weights."
        ) from exc
    # Replace fc with Identity to expose the 2048-dim pooled features, not ImageNet logits.
    model.fc = nn.Identity()
    _freeze_backbone(model)
    model.to(device)
    _verify_embedding_dim(model, config, device)
    logger.info(
        "Loaded ResNet-50 backbone (embedding_dim=2048, frozen=True, device=%s)", device
    )
    return model


def _load_convnext_base(config: BackboneConfig, device: torch.device) -> nn.Module:
    try:
        from torchvision.models import ConvNeXt_Base_Weights, convnext_base  # noqa: PLC0415
    except ImportError as exc:
        raise BackboneUnavailableError(
            "torchvision is required for ConvNeXt-Base. "
            "Install with: pip install torchvision>=0.18"
        ) from exc
    try:
        model = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
    except Exception as exc:
        raise BackboneUnavailableError(
            f"Failed to load ConvNeXt-Base weights (IMAGENET1K_V1): {exc}. "
            f"Check network access or pre-cached weights."
        ) from exc
    # classifier = Sequential([LayerNorm2d(1024), Flatten, Linear(1024, 1000)])
    # Replace the final Linear with Identity → 1024-dim output after LayerNorm2d + Flatten.
    model.classifier[-1] = nn.Identity()
    _freeze_backbone(model)
    model.to(device)
    _verify_embedding_dim(model, config, device)
    logger.info(
        "Loaded ConvNeXt-Base backbone (embedding_dim=1024, frozen=True, device=%s)", device
    )
    return model


def _load_dinov2(config: BackboneConfig, device: torch.device) -> nn.Module:
    model_id = config.version  # e.g. "dinov2_vitb14" or "dinov2_vitl14"
    if not model_id:
        raise BackboneUnavailableError(
            f"DINOv2 requires a non-empty 'version' field specifying the hub model "
            f"identifier (e.g. 'dinov2_vitb14', 'dinov2_vitl14'). "
            f"Got version={config.version!r} for name={config.name!r}."
        )
    try:
        model = torch.hub.load(
            "facebookresearch/dinov2",
            model_id,
            trust_repo=True,
            verbose=False,
        )
    except Exception as exc:
        raise BackboneUnavailableError(
            f"Failed to load DINOv2 model {model_id!r} via torch.hub "
            f"(repo=facebookresearch/dinov2): {exc}. "
            f"Ensure weights are cached or network access is available."
        ) from exc
    # DINOv2 hub base models output CLS token embeddings directly; no head removal needed.
    _freeze_backbone(model)
    model.to(device)
    _verify_embedding_dim(model, config, device)
    logger.info(
        "Loaded DINOv2 backbone %s (embedding_dim=%d, frozen=True, device=%s)",
        model_id, config.embedding_dim, device,
    )
    return model


def _load_retfound_green(config: BackboneConfig, device: torch.device) -> nn.Module:
    """Load RETFound-Green (ViT-S/14-reg4-dinov2) via timm.

    Architecture: vit_small_patch14_reg4_dinov2, embedding_dim=384.

    Protocol is config-driven via BackboneConfig fields:
      - input_size=224, global_pool='token'  → matched-224/CLS protocol (B2 matrix row)
      - input_size=392, global_pool='avg'    → native-392/avg-pool comparator

    Matched-224 (Decision 029):
      The PE-interpolated checkpoint (retfoundgreen_statedict_224.pth) must be used
      because timm 1.0.27 checkpoint_path uses strict-load and does NOT interpolate PE.
      The pre-interpolated checkpoint has pos_embed=[1,256,384] matching 16×16 patches.

    Native-392:
      The original checkpoint (retfoundgreen_statedict.pth) is used directly.
      pos_embed=[1,784,384] matches 28×28 patches at 392px exactly — no interpolation.

    Checkpoint
    ----------
    Set BackboneConfig.checkpoint_path (populated from backbone YAML 'checkpoint_path'
    field by scripts/_build_backbone_config) or the environment variable
    RETFOUND_GREEN_CHECKPOINT.  Both empty → BackboneUnavailableError (no silent fallback).

    Raises
    ------
    BackboneUnavailableError
        timm not installed; checkpoint path empty; checkpoint file absent;
        timm model creation fails; output dim mismatch.
    """
    try:
        import timm  # noqa: PLC0415
    except ImportError as exc:
        raise BackboneUnavailableError(
            "timm is required for RETFound-Green. "
            "Install with: pip install 'timm>=0.9' (or add to the 'torch' extras group)."
        ) from exc

    _TIMM_MODEL_NAME = "vit_small_patch14_reg4_dinov2"

    # Verify the model is available in the installed timm version.
    available = [m for m in timm.list_models() if _TIMM_MODEL_NAME in m]
    if not available:
        raise BackboneUnavailableError(
            f"timm {timm.__version__} does not list {_TIMM_MODEL_NAME!r}. "
            f"Upgrade timm: pip install 'timm>=0.9'."
        )

    # Resolve checkpoint path: YAML field takes precedence, then env var.
    checkpoint_path = config.checkpoint_path or os.environ.get(
        "RETFOUND_GREEN_CHECKPOINT", ""
    )
    if not checkpoint_path:
        raise BackboneUnavailableError(
            "RETFound-Green requires a checkpoint. Set 'checkpoint_path' in "
            "configs/backbone/retfound_green_matched224.yaml to the local path of "
            "retfoundgreen_statedict.pth, or set the environment variable "
            "RETFOUND_GREEN_CHECKPOINT=/path/to/retfoundgreen_statedict.pth. "
            "Download (Apache-2.0, no access gate): "
            "https://github.com/justinengelmann/RETFound_Green/releases/download/"
            "v0.1/retfoundgreen_statedict.pth"
        )
    if not Path(checkpoint_path).exists():
        raise BackboneUnavailableError(
            f"RETFound-Green checkpoint not found at {checkpoint_path!r}. "
            "Download from: https://github.com/justinengelmann/RETFound_Green/"
            "releases/download/v0.1/retfoundgreen_statedict.pth"
        )

    _input_size = config.input_size   # 224 for matched-224, 392 for native
    _gpool = config.global_pool       # 'token' for CLS, 'avg' for native avg-pool
    try:
        if _gpool == "avg":
            # Native-392 avg-pool path: timm names the final LayerNorm 'fc_norm' when
            # global_pool='avg', but the checkpoint uses 'norm' (CLS-token naming).
            # Remap in-memory before load_state_dict.  No checkpoint file is modified.
            # This is a 1:1 layer rename — 'norm' and 'fc_norm' are the same LayerNorm,
            # just accessed via a different attribute name depending on pooling mode.
            _state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            for _k in ("norm.weight", "norm.bias"):
                if _k in _state:
                    _state[_k.replace("norm.", "fc_norm.", 1)] = _state.pop(_k)
            model = timm.create_model(
                _TIMM_MODEL_NAME,
                img_size=(_input_size, _input_size),
                num_classes=0,
                global_pool=_gpool,
            )
            model.load_state_dict(_state, strict=True)
        else:
            # Matched-224 CLS-token path: checkpoint key names match the 'token' pool
            # model exactly — timm checkpoint_path strict-load works directly.
            model = timm.create_model(
                _TIMM_MODEL_NAME,
                img_size=(_input_size, _input_size),
                num_classes=0,
                global_pool=_gpool,
                checkpoint_path=checkpoint_path,
            )
    except Exception as exc:
        raise BackboneUnavailableError(
            f"Failed to load RETFound-Green via timm ({_TIMM_MODEL_NAME!r}): {exc}. "
            f"Verify checkpoint integrity and timm compatibility (installed: {timm.__version__})."
        ) from exc

    _freeze_backbone(model)
    model.to(device)
    _verify_embedding_dim(model, config, device)
    _protocol = (
        "native-392/avg-pool" if _input_size == 392 and _gpool == "avg"
        else f"matched-{_input_size}/{_gpool}"
    )
    logger.info(
        "Loaded RETFound-Green backbone %s (embedding_dim=%d, input_size=%d, "
        "protocol=%s, frozen=True, device=%s)",
        _TIMM_MODEL_NAME, config.embedding_dim, _input_size, _protocol, device,
    )
    return model


# ---------------------------------------------------------------------------
# Backbone loading
# ---------------------------------------------------------------------------


def load_backbone(config: BackboneConfig, device: torch.device) -> nn.Module:
    """Return a ready frozen backbone on the given device.

    Supported model_type values: 'mock', 'resnet50', 'convnext_base', 'dinov2',
    'retfound_green'.

    Raises
    ------
    BackboneUnavailableError
        If a real backbone cannot be loaded (missing deps, unavailable weights,
        network failure, or unknown model_type). NEVER falls back to mock silently.
    BackboneDimensionError
        If the backbone produces a different embedding_dim than config.embedding_dim.
    """
    if config.model_type == "mock":
        backbone = MockBackbone(embedding_dim=config.embedding_dim)
        backbone.eval()
        backbone.to(device)
        logger.info(
            "Loaded MockBackbone (embedding_dim=%d, device=%s)",
            config.embedding_dim, device,
        )
        return backbone

    if config.model_type == "resnet50":
        return _load_resnet50(config, device)

    if config.model_type == "convnext_base":
        return _load_convnext_base(config, device)

    if config.model_type == "dinov2":
        return _load_dinov2(config, device)

    if config.model_type == "retfound_green":
        return _load_retfound_green(config, device)

    raise BackboneUnavailableError(
        f"Unknown or unsupported backbone model_type={config.model_type!r} "
        f"for backbone name={config.name!r}. "
        f"Supported types: 'mock', 'resnet50', 'convnext_base', 'dinov2', 'retfound_green'. "
        f"Do not add a mock fallback here — unknown backbones must fail loudly."
    )


# ---------------------------------------------------------------------------
# Cache path helpers
# ---------------------------------------------------------------------------


def get_cache_dir(
    cache_root: Path | str,
    backbone_name: str,
    dataset_source: str,
    preprocessing_hash: str,
) -> Path:
    """Return the canonical cache directory, creating it if needed.

    Structure: cache_root / backbone_name / dataset_source / preprocessing_hash /
    """
    cache_dir = Path(cache_root) / backbone_name / dataset_source / preprocessing_hash
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _sample_id_to_filename(sample_id: str) -> str:
    """Convert sample_id to a collision-safe cache filename.

    Keeps a sanitized, readable prefix and appends a 12-char SHA256 hash of
    the original sample_id to prevent collisions from samples that differ only
    in special characters. The manifest stores the original sample_id.
    """
    safe_stem = re.sub(r"[^\w\-.]", "_", sample_id)[:64]
    short_hash = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:12]
    return f"{safe_stem}_{short_hash}.pt"


# ---------------------------------------------------------------------------
# Checksum
# ---------------------------------------------------------------------------


def compute_tensor_checksum(tensor: torch.Tensor) -> str:
    """Return the SHA256 hex digest of the tensor's raw bytes (CPU, contiguous)."""
    data = tensor.cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def _validate_embedding_shape(
    tensor: torch.Tensor,
    expected_dim: int,
    cache_path: Path | str,
) -> None:
    """Validate that a cached per-sample embedding is exactly 1-D."""
    if tensor.ndim != 1 or tensor.shape != (expected_dim,):
        raise CacheCorruptError(
            f"Embedding dim/shape mismatch for {cache_path}: expected "
            f"({expected_dim},), got {tuple(tensor.shape)}"
        )


# ---------------------------------------------------------------------------
# Embedding compaction
# ---------------------------------------------------------------------------


def _compact_embedding(raw: torch.Tensor) -> torch.Tensor:
    """Detach, move to CPU, and clone to compact independent storage before saving.

    ViT-based backbones (e.g. DINOv2) return the CLS token as a view into the full
    token-sequence backing storage.  Saving that view with torch.save serialises the
    entire backing buffer (e.g. 257 × 768 × 4 = 789 KB) instead of the 768 visible
    elements (3 KB).  .clone() breaks the storage link; .contiguous() guarantees a
    sequential layout so the saved file is exactly numel × element_size bytes.
    """
    return raw.detach().cpu().clone().contiguous()


# ---------------------------------------------------------------------------
# Save / load individual embeddings
# ---------------------------------------------------------------------------


def save_embedding(
    embedding: torch.Tensor,
    sample_id: str,
    cache_dir: Path | str,
) -> tuple[Path, str]:
    """Save a 1-D embedding tensor to cache.

    Returns
    -------
    (absolute_path, checksum)
    """
    cache_dir = Path(cache_dir)
    filename = _sample_id_to_filename(sample_id)
    path = cache_dir / filename
    torch.save(embedding.cpu(), path)
    checksum = compute_tensor_checksum(embedding)
    logger.debug(
        "Saved embedding sample_id=%s → %s (checksum=%s...)",
        sample_id, path, checksum[:12],
    )
    return path, checksum


def load_embedding(
    cache_path: Path | str,
    expected_checksum: str,
    expected_dim: int,
) -> torch.Tensor:
    """Load and validate an embedding from its manifest cache_path.

    The manifest cache_path is the authoritative source for the file location;
    do not reconstruct from sample_id.

    Raises
    ------
    CacheMissError
        If the file does not exist.
    CacheCorruptError
        If the checksum mismatches OR the last dimension != expected_dim.
    """
    path = Path(cache_path)
    if not path.exists():
        raise CacheMissError(f"Cache file absent: {path}")

    try:
        tensor = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise CacheCorruptError(f"Could not load embedding cache file {path}: {exc}") from exc

    if not isinstance(tensor, torch.Tensor):
        raise CacheCorruptError(
            f"Cache file {path} did not contain a torch.Tensor; got "
            f"{type(tensor).__name__}"
        )

    _validate_embedding_shape(tensor, expected_dim, path)

    actual_checksum = compute_tensor_checksum(tensor)
    if actual_checksum != expected_checksum:
        raise CacheCorruptError(
            f"Checksum mismatch for {path}: "
            f"expected {expected_checksum[:12]}..., got {actual_checksum[:12]}..."
        )

    return tensor


# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------


def write_embedding_manifest(records: list[dict], manifest_path: Path | str) -> None:
    """Write embedding manifest CSV.

    All records must contain all MANIFEST_FIELDNAMES columns.
    """
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDNAMES)
        writer.writeheader()
        writer.writerows(records)
    logger.info(
        "Wrote embedding manifest: %s (%d rows)", manifest_path, len(records)
    )


def load_embedding_manifest(manifest_path: Path | str) -> list[dict]:
    """Load embedding manifest CSV as a list of dicts.

    Raises
    ------
    FileNotFoundError
        If the manifest file does not exist.
    """
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Embedding manifest not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return list(reader)


def _manifest_index_by_sample_id(manifest_path: Path) -> dict[str, dict]:
    """Load a manifest into a sample_id -> row map, rejecting duplicate rows."""
    if not manifest_path.exists():
        return {}

    rows = load_embedding_manifest(manifest_path)
    index: dict[str, dict] = {}
    duplicates: list[str] = []
    for row in rows:
        sample_id = row["sample_id"]
        if sample_id in index:
            duplicates.append(sample_id)
            continue
        index[sample_id] = row

    if duplicates:
        raise CacheCorruptError(
            f"Embedding manifest {manifest_path} contains duplicate sample_id "
            f"values, including {sorted(set(duplicates))[:5]}"
        )
    return index


def _validate_manifest_row_matches_request(
    row: dict,
    sample: CanonicalSample,
    backbone_config: BackboneConfig,
    preprocessing_hash: str,
) -> None:
    """Validate manifest metadata before reusing a cached embedding."""
    checks = {
        "dataset_source": sample.dataset_source,
        "backbone_name": backbone_config.name,
        "backbone_version": backbone_config.version,
        "preprocessing_hash": preprocessing_hash,
        "embedding_dim": str(backbone_config.embedding_dim),
    }
    mismatches = [
        f"{field}: expected {expected!r}, got {row.get(field)!r}"
        for field, expected in checks.items()
        if str(row.get(field, "")) != expected
    ]
    if mismatches:
        raise CacheCorruptError(
            f"Manifest row for sample_id={sample.sample_id!r} does not match "
            f"the requested embedding namespace: {mismatches}"
        )


def _validate_unique_sample_ids(samples: list[CanonicalSample]) -> None:
    """Reject duplicate sample IDs before cache paths are derived."""
    seen: set[str] = set()
    duplicates: list[str] = []
    for sample in samples:
        if sample.sample_id in seen:
            duplicates.append(sample.sample_id)
        seen.add(sample.sample_id)

    if duplicates:
        raise ValueError(
            "Duplicate sample_id values are not allowed during embedding "
            f"extraction. Examples: {sorted(set(duplicates))[:5]}"
        )


# ---------------------------------------------------------------------------
# Batch extraction
# ---------------------------------------------------------------------------


def extract_embeddings(
    manifest: list[CanonicalSample],
    backbone: nn.Module,
    backbone_config: BackboneConfig,
    preprocessing_config: PreprocessingConfig,
    cache_root: Path | str,
    device: torch.device,
    image_loader: Callable[[CanonicalSample], Image.Image],
    batch_size: int = 32,
    overwrite: bool = False,
    limit: int | None = None,
) -> Path:
    """Extract embeddings for all samples (up to limit) and write manifest CSV.

    Parameters
    ----------
    manifest:
        List of CanonicalSample objects to process.
    backbone:
        Frozen backbone module. Must be in eval mode and on the correct device.
    backbone_config:
        Backbone identity and embedding_dim specification.
    preprocessing_config:
        Preprocessing configuration used to derive the preprocessing_hash.
    cache_root:
        Root directory for the embedding cache.
    device:
        Torch device to run inference on.
    image_loader:
        Callable(CanonicalSample) → PIL.Image.Image. Called once per sample.
        Must not raise on valid samples. Injected by the caller to keep this
        module adapter-agnostic.
    batch_size:
        Reserved for future batching (not used in Stage 6 sequential path).
    overwrite:
        False (default): reuse valid cache entries; raise on corrupt/wrong-dim.
        True: always re-extract regardless of existing state.
    limit:
        Process only the first `limit` samples (None = all).

    Returns
    -------
    Path
        Absolute path to the written manifest.csv.

    Notes
    -----
    - Always calls backbone.eval() and uses torch.no_grad().
    - Always uses mode='extract' (deterministic preprocessing).
    - Manifest is written atomically after all samples are processed.
    - Unreadable images raise immediately; no fake manifest rows are written.
    - overwrite=False reuses cache only through validated manifest rows.
    """
    if not manifest:
        raise ValueError("Cannot extract embeddings from an empty manifest.")

    prep_hash = get_preprocessing_hash(preprocessing_config)
    pipeline = build_preprocessing_pipeline(preprocessing_config, mode="extract")

    samples = list(manifest)
    if limit is not None:
        samples = samples[:limit]
    _validate_unique_sample_ids(samples)

    records: list[dict] = []
    backbone.eval()
    manifest_indexes: dict[Path, dict[str, dict]] = {}

    with torch.no_grad():
        for sample in samples:
            cache_dir = get_cache_dir(
                cache_root, backbone_config.name, sample.dataset_source, prep_hash
            )
            filename = _sample_id_to_filename(sample.sample_id)
            cache_path = cache_dir / filename
            existing_manifest_path = cache_dir / "manifest.csv"
            if existing_manifest_path not in manifest_indexes:
                manifest_indexes[existing_manifest_path] = _manifest_index_by_sample_id(
                    existing_manifest_path
                )
            existing_row = manifest_indexes[existing_manifest_path].get(sample.sample_id)

            if not overwrite and existing_row is not None:
                # Manifest row is the source of truth for validating cache reuse.
                _validate_manifest_row_matches_request(
                    existing_row, sample, backbone_config, prep_hash
                )
                cache_path = Path(existing_row["cache_path"])
                load_embedding(
                    cache_path,
                    existing_row["checksum"],
                    backbone_config.embedding_dim,
                )
                checksum = existing_row["checksum"]
                logger.debug("Validated cache hit: sample_id=%s", sample.sample_id)
            else:
                if not overwrite and cache_path.exists():
                    logger.warning(
                        "Existing orphan cache file has no manifest row; "
                        "re-extracting sample_id=%s path=%s",
                        sample.sample_id, cache_path,
                    )
                # Extract embedding.
                img = image_loader(sample)
                tensor = pipeline(img).unsqueeze(0).to(device)   # (1, C, H, W)
                embedding = _compact_embedding(backbone(tensor).squeeze(0))    # (embedding_dim,)
                _validate_embedding_shape(
                    embedding, backbone_config.embedding_dim, cache_path
                )
                cache_path, checksum = save_embedding(
                    embedding, sample.sample_id, cache_dir
                )

            records.append({
                "sample_id": sample.sample_id,
                "patient_id": sample.patient_id,
                "dataset_source": sample.dataset_source,
                "cache_path": str(cache_path),
                "embedding_dim": backbone_config.embedding_dim,
                "backbone_name": backbone_config.name,
                "backbone_version": backbone_config.version,
                "preprocessing_hash": prep_hash,
                "created_at": datetime.now(tz=timezone.utc).isoformat(),
                "checksum": checksum,
            })

    # Derive manifest path from first sample's cache directory.
    first_cache_dir = get_cache_dir(
        cache_root, backbone_config.name, samples[0].dataset_source, prep_hash
    )
    manifest_path = first_cache_dir / "manifest.csv"
    write_embedding_manifest(records, manifest_path)
    logger.info(
        "Extraction complete: %d samples processed, manifest at %s",
        len(records), manifest_path,
    )
    return manifest_path


# ---------------------------------------------------------------------------
# Cache integrity verification
# ---------------------------------------------------------------------------


def verify_cache_integrity(
    manifest_path: Path | str,
    expected_dim: int,
) -> list[str]:
    """Verify all entries in the manifest.

    For each row, validates:
    - File existence (cache_path)
    - Checksum (from manifest)
    - Embedding dimension (expected_dim)
    - backbone_name appears in cache_path directory components (if present)
    - preprocessing_hash appears in cache_path directory components (if present)

    Returns
    -------
    list[str]
        sample_ids of missing or corrupt entries. Empty list means all valid.
    """
    rows = load_embedding_manifest(manifest_path)
    failed: list[str] = []

    for row in rows:
        sample_id = row["sample_id"]
        cache_path = Path(row["cache_path"])
        expected_checksum = row["checksum"]
        row_backbone = row.get("backbone_name", "")
        row_prep_hash = row.get("preprocessing_hash", "")

        try:
            load_embedding(cache_path, expected_checksum, expected_dim)
        except (CacheMissError, CacheCorruptError) as exc:
            logger.warning(
                "Cache integrity failure for sample_id=%s: %s", sample_id, exc
            )
            failed.append(sample_id)
            continue

        # Structural validation: path components should include backbone and hash.
        path_parts = set(cache_path.parts)
        if row_backbone and row_backbone not in path_parts:
            logger.warning(
                "backbone_name %r not found in cache path components for sample_id=%s "
                "(path=%s)",
                row_backbone, sample_id, cache_path,
            )
            failed.append(sample_id)
            continue
        if row_prep_hash and row_prep_hash not in path_parts:
            logger.warning(
                "preprocessing_hash %r not found in cache path components "
                "for sample_id=%s (path=%s)",
                row_prep_hash, sample_id, cache_path,
            )
            failed.append(sample_id)
            continue

    return failed
