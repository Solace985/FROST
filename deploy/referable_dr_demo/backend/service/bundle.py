"""bundle.py -- local deployment-bundle discovery and fail-closed validation.

Discovers the accepted RETFound-Green native-392 + MultiTaskHead artifacts from
their existing local locations (resolved via environment variables, with
canonical fallbacks), validates them against the frozen expected specification,
and produces an immutable :class:`DeploymentBundle`.

Nothing here trains, extracts, copies, moves, or re-saves any artifact. It reads
checkpoints to hash them and to read the head's task structure; it never writes
to any pipeline location. The generated manifest is written only to the ignored
``deploy/referable_dr_demo/.local/`` directory by the analysis builder.

Fail-closed conditions (raise ``BundleValidationError``):
  - a required checkpoint is missing
  - a checkpoint SHA-256 does not match the expected/recorded value
  - expected embedding_dim is not 384
  - pooling is not native average pooling
  - native input size is not 392
  - task ordering differs from the accepted/canonical ordering
  - the head checkpoint does not load into a MultiTaskHead
  - dr_grade output shape is not 5
  - the canonical loader / preprocessing route cannot be verified
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from . import preprocessing_parity
from .provenance import (
    REPO_ROOT,
    ensure_src_importable,
    git_commit,
    python_version,
    sha256_file,
    task_ordering_hash,
    timm_version,
    torch_version,
    utc_timestamp,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Frozen expected specification (the accepted deployment candidate)
# --------------------------------------------------------------------------
BUNDLE_VERSION = "frost-referable-dr-1"
BACKBONE_IDENTIFIER = "retfound_green_native392"
MODEL_FAMILY = "retfound_green"
NATIVE_PROTOCOL = "native_392_avg_pool"
NATIVE_INPUT_SIZE = 392
NATIVE_POOLING = "average"          # timm global_pool="avg"
EXPECTED_EMBEDDING_DIM = 384
HEAD_TYPE = "MultiTaskHead"
DR_GRADE_TASK_KEY = "dr_grade"
DR_GRADE_OUTPUT_SHAPE = 5
BACKBONE_MODEL_TYPE = "retfound_green"   # retina_screen BackboneConfig.model_type
BACKBONE_GLOBAL_POOL = "avg"             # retina_screen BackboneConfig.global_pool

# --------------------------------------------------------------------------
# Canonical artifact locations (relative to repo root; runs/, models/, cache/,
# outputs/ are all git-ignored, so no private data is committed). Environment
# variables take precedence so private paths are never hardcoded as the source.
# --------------------------------------------------------------------------
ENV_BACKBONE_CKPT = "RETINA_SCREEN_RETF_GREEN_CHECKPOINT"
ENV_HEAD_CKPT = "RETINA_SCREEN_RETF_NATIVE392_MT_CHECKPOINT"
ENV_CANONICAL_BACKBONE_CKPT = "RETFOUND_GREEN_CHECKPOINT"  # canonical pipeline var

_DEFAULT_BACKBONE_CKPT = REPO_ROOT / "models" / "retfoundgreen_statedict.pth"
_DEFAULT_HEAD_RUN_DIR = REPO_ROOT / "runs" / "train" / "brset_20260525_113425"
_DEFAULT_HEAD_CKPT = _DEFAULT_HEAD_RUN_DIR / "model_checkpoint.pt"
_BACKBONE_YAML = REPO_ROOT / "configs" / "backbone" / f"{BACKBONE_IDENTIFIER}.yaml"
_TASKS_YAML = REPO_ROOT / "configs" / "tasks" / "brset_default.yaml"


class BundleValidationError(RuntimeError):
    """Raised when a deployment artifact fails a fail-closed validation gate."""


@dataclass(frozen=True)
class ArtifactPaths:
    backbone_checkpoint: Path
    head_checkpoint: Path
    resolved_config: Path | None
    backbone_yaml: Path
    preprocessing_yaml: Path
    tasks_yaml: Path


@dataclass(frozen=True)
class DeploymentBundle:
    """Validated, immutable description of the deployed artifacts."""

    manifest: dict[str, Any]
    backbone_checkpoint: Path
    head_checkpoint: Path
    task_order: tuple[str, ...]

    # ---- typed accessors -------------------------------------------------
    @property
    def bundle_version(self) -> str:
        return self.manifest["bundle_version"]

    @property
    def embedding_dim(self) -> int:
        return self.manifest["expected_embedding_dim"]

    @property
    def dr_grade_index(self) -> int:
        return self.manifest["dr_grade_task_index"]

    @property
    def preprocessing_hash(self) -> str:
        return self.manifest["preprocessing_hash"]

    @property
    def task_ordering_hash(self) -> str:
        return self.manifest["task_ordering_hash"]

    @property
    def backbone_sha256(self) -> str:
        return self.manifest["backbone_checkpoint_sha256"]

    @property
    def head_sha256(self) -> str:
        return self.manifest["head_checkpoint_sha256"]

    def backbone_config_kwargs(self) -> dict[str, Any]:
        """Kwargs for retina_screen.embeddings.BackboneConfig (native-392)."""
        return {
            "name": BACKBONE_IDENTIFIER,
            "embedding_dim": EXPECTED_EMBEDDING_DIM,
            "model_type": BACKBONE_MODEL_TYPE,
            "version": self.manifest.get("backbone_version", ""),
            "checkpoint_path": str(self.backbone_checkpoint),
            "input_size": NATIVE_INPUT_SIZE,
            "global_pool": BACKBONE_GLOBAL_POOL,
        }


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------
def _first_existing(*candidates: Path | str | None) -> Path | None:
    for c in candidates:
        if not c:
            continue
        p = Path(c)
        if p.exists():
            return p
    return None


def discover_artifacts() -> ArtifactPaths:
    """Resolve the local artifact paths (no hashing/loading)."""
    backbone = _first_existing(
        os.environ.get(ENV_BACKBONE_CKPT),
        os.environ.get(ENV_CANONICAL_BACKBONE_CKPT),
        _DEFAULT_BACKBONE_CKPT,
    )
    head = _first_existing(
        os.environ.get(ENV_HEAD_CKPT),
        _DEFAULT_HEAD_CKPT,
    )
    if backbone is None:
        raise BundleValidationError(
            "Backbone checkpoint not found. Set "
            f"{ENV_BACKBONE_CKPT} (or {ENV_CANONICAL_BACKBONE_CKPT}) to the local "
            "RETFound-Green state-dict path, or place it at "
            f"{_DEFAULT_BACKBONE_CKPT}."
        )
    if head is None:
        raise BundleValidationError(
            "Native-392 MultiTaskHead checkpoint not found. Set "
            f"{ENV_HEAD_CKPT} to the accepted native-392 run's model_checkpoint.pt, "
            f"or place it at {_DEFAULT_HEAD_CKPT}."
        )
    resolved_config = head.parent / "resolved_config.yaml"
    return ArtifactPaths(
        backbone_checkpoint=backbone,
        head_checkpoint=head,
        resolved_config=resolved_config if resolved_config.exists() else None,
        backbone_yaml=_BACKBONE_YAML,
        preprocessing_yaml=preprocessing_parity.preprocessing_config_path(),
        tasks_yaml=_TASKS_YAML,
    )


# --------------------------------------------------------------------------
# Head-checkpoint introspection
# --------------------------------------------------------------------------
def _load_head_state_dict(head_ckpt: Path) -> dict[str, Any]:
    ensure_src_importable()
    import torch  # noqa: PLC0415

    state = torch.load(head_ckpt, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise BundleValidationError(
            f"Head checkpoint {head_ckpt} is not a state_dict (got "
            f"{type(state).__name__})."
        )
    return state


def _task_order_from_state_dict(state: dict[str, Any]) -> list[str]:
    """Extract task-head names in ModuleDict insertion order (= training order)."""
    order: list[str] = []
    for key in state.keys():
        if key.startswith("task_heads."):
            name = key.split(".", 2)[1]
            if name not in order:
                order.append(name)
    if not order:
        raise BundleValidationError(
            "Head checkpoint contains no 'task_heads.*' parameters; it is not a "
            "MultiTaskHead checkpoint."
        )
    return order


def _canonical_task_set() -> set[str] | None:
    """Canonical supported-task set from the committed tasks config (cross-check)."""
    if not _TASKS_YAML.exists():
        return None
    raw = yaml.safe_load(_TASKS_YAML.read_text(encoding="utf-8"))
    tasks = raw.get("supported_tasks")
    return set(tasks) if tasks else None


def _expected_backbone_sha() -> str | None:
    """Read the published backbone SHA-256 from the canonical backbone YAML."""
    if not _BACKBONE_YAML.exists():
        return None
    raw = yaml.safe_load(_BACKBONE_YAML.read_text(encoding="utf-8"))
    sha = raw.get("checkpoint_sha256_original")
    return str(sha).upper() if sha else None


# --------------------------------------------------------------------------
# Build (full validation) — used by the analysis builder
# --------------------------------------------------------------------------
def build_bundle(*, validate_backbone_forward: bool = True) -> dict[str, Any]:
    """Discover, validate, and return the deployment-bundle manifest dict.

    Performs every fail-closed gate. When ``validate_backbone_forward`` is True it
    loads the frozen backbone and runs a single synthetic forward to confirm the
    live embedding dimension is exactly 384 (the strongest pooling/dim check).
    """
    paths = discover_artifacts()

    # --- backbone hash gate ---
    backbone_sha = sha256_file(paths.backbone_checkpoint)
    expected_sha = _expected_backbone_sha()
    if expected_sha and backbone_sha != expected_sha:
        raise BundleValidationError(
            "Backbone checkpoint SHA-256 mismatch. "
            f"expected {expected_sha[:16]}..., got {backbone_sha[:16]}.... "
            "This is not the accepted RETFound-Green checkpoint."
        )

    # --- head structure gate ---
    state = _load_head_state_dict(paths.head_checkpoint)
    task_order = _task_order_from_state_dict(state)
    if DR_GRADE_TASK_KEY not in task_order:
        raise BundleValidationError(
            f"Head checkpoint has no {DR_GRADE_TASK_KEY!r} task head; cannot "
            "produce a referable-DR score."
        )
    canonical_set = _canonical_task_set()
    if canonical_set is not None and set(task_order) != canonical_set:
        raise BundleValidationError(
            "Head task set differs from the canonical supported-task set. "
            f"checkpoint={sorted(task_order)}, canonical={sorted(canonical_set)}."
        )

    # --- build the head and confirm strict load + dr_grade output dim == 5 ---
    ensure_src_importable()
    import torch  # noqa: PLC0415

    from retina_screen.model import build_head  # noqa: PLC0415

    head = build_head(
        embedding_dim=EXPECTED_EMBEDDING_DIM,
        task_names=task_order,
        head_type="multitask",
    )
    if type(head).__name__ != HEAD_TYPE:
        raise BundleValidationError(
            f"Constructed head is {type(head).__name__}, expected {HEAD_TYPE}."
        )
    try:
        head.load_state_dict(state)  # strict=True
    except Exception as exc:  # noqa: BLE001
        raise BundleValidationError(
            f"Head checkpoint does not load strictly into a {HEAD_TYPE} with "
            f"embedding_dim={EXPECTED_EMBEDDING_DIM}: {exc}"
        ) from exc
    head.eval()
    with torch.inference_mode():
        out = head(torch.zeros(1, EXPECTED_EMBEDDING_DIM, dtype=torch.float32))
    dr_logits = out[DR_GRADE_TASK_KEY]
    if tuple(dr_logits.shape) != (1, DR_GRADE_OUTPUT_SHAPE):
        raise BundleValidationError(
            f"{DR_GRADE_TASK_KEY} output shape is {tuple(dr_logits.shape)}, "
            f"expected (1, {DR_GRADE_OUTPUT_SHAPE})."
        )
    head_sha = sha256_file(paths.head_checkpoint)

    # --- backbone live forward gate (pooling + dim) ---
    if validate_backbone_forward:
        _validate_backbone_forward(paths.backbone_checkpoint, task_order)

    # --- provenance of configs ---
    resolved_config_sha = (
        sha256_file(paths.resolved_config) if paths.resolved_config else None
    )
    preprocessing_yaml_sha = (
        sha256_file(paths.preprocessing_yaml)
        if paths.preprocessing_yaml.exists()
        else None
    )
    backbone_version = ""
    if paths.backbone_yaml.exists():
        backbone_version = str(
            yaml.safe_load(paths.backbone_yaml.read_text(encoding="utf-8")).get(
                "version", ""
            )
        )

    manifest: dict[str, Any] = {
        "bundle_version": BUNDLE_VERSION,
        "backbone_identifier": BACKBONE_IDENTIFIER,
        "model_family": MODEL_FAMILY,
        "native_protocol": NATIVE_PROTOCOL,
        "native_input_size": NATIVE_INPUT_SIZE,
        "native_pooling": NATIVE_POOLING,
        "expected_embedding_dim": EXPECTED_EMBEDDING_DIM,
        "head_type": HEAD_TYPE,
        "backbone_version": backbone_version,
        "backbone_checkpoint_path": str(paths.backbone_checkpoint),
        "head_checkpoint_path": str(paths.head_checkpoint),
        "backbone_checkpoint_sha256": backbone_sha,
        "head_checkpoint_sha256": head_sha,
        "resolved_config_path": (
            str(paths.resolved_config) if paths.resolved_config else None
        ),
        "resolved_config_sha256": resolved_config_sha,
        "preprocessing_config_path": str(paths.preprocessing_yaml),
        "preprocessing_config_sha256": preprocessing_yaml_sha,
        "preprocessing_hash": preprocessing_parity.preprocessing_hash(),
        "model_task_ordering": list(task_order),
        "task_ordering_hash": task_ordering_hash(task_order),
        "dr_grade_task_key": DR_GRADE_TASK_KEY,
        "dr_grade_task_index": task_order.index(DR_GRADE_TASK_KEY),
        "dr_grade_output_shape": DR_GRADE_OUTPUT_SHAPE,
        "python_version": python_version(),
        "torch_version": torch_version(),
        "timm_version": timm_version(),
        "git_commit": git_commit(),
        "timestamp": utc_timestamp(),
        "parity_status": "unverified",
        "network_access_required": False,
    }
    return manifest


def _validate_backbone_forward(backbone_ckpt: Path, task_order: list[str]) -> None:
    """Load the frozen backbone via the canonical loader and confirm dim 384."""
    ensure_src_importable()
    import torch  # noqa: PLC0415

    from retina_screen.embeddings import BackboneConfig, load_backbone  # noqa: PLC0415

    cfg = BackboneConfig(
        name=BACKBONE_IDENTIFIER,
        embedding_dim=EXPECTED_EMBEDDING_DIM,
        model_type=BACKBONE_MODEL_TYPE,
        version="",
        checkpoint_path=str(backbone_ckpt),
        input_size=NATIVE_INPUT_SIZE,
        global_pool=BACKBONE_GLOBAL_POOL,
    )
    backbone = load_backbone(cfg, torch.device("cpu"))
    backbone.eval()
    # Confirm frozen.
    if any(p.requires_grad for p in backbone.parameters()):
        raise BundleValidationError("Backbone parameters are not all frozen.")
    with torch.inference_mode():
        emb = backbone(torch.zeros(1, 3, NATIVE_INPUT_SIZE, NATIVE_INPUT_SIZE))
    emb = emb.squeeze(0) if emb.ndim == 2 else emb
    if emb.shape[-1] != EXPECTED_EMBEDDING_DIM:
        raise BundleValidationError(
            f"Backbone produced embedding dim {emb.shape[-1]}, expected "
            f"{EXPECTED_EMBEDDING_DIM}. Pooling/protocol mismatch."
        )


# --------------------------------------------------------------------------
# Persist / load the local manifest
# --------------------------------------------------------------------------
def write_bundle(manifest: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def load_bundle_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise BundleValidationError(
            f"Local deployment bundle not found at {path}. Build it with: "
            "uv run python deploy/referable_dr_demo/analysis/"
            "build_local_deployment_bundle.py"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise BundleValidationError(message)


def validate_bundle_against_artifacts(manifest: dict[str, Any]) -> DeploymentBundle:
    """Re-validate a loaded manifest against the live local artifacts.

    Fail-closed on any drift (missing file, hash mismatch, dim/pooling/size/task
    differences). Returns an immutable :class:`DeploymentBundle` on success.
    """
    # frozen-spec invariants
    _check(manifest.get("expected_embedding_dim") == EXPECTED_EMBEDDING_DIM,
           f"bundle embedding_dim must be {EXPECTED_EMBEDDING_DIM}.")
    _check(manifest.get("native_input_size") == NATIVE_INPUT_SIZE,
           f"bundle native_input_size must be {NATIVE_INPUT_SIZE}.")
    _check(manifest.get("native_pooling") == NATIVE_POOLING,
           f"bundle native_pooling must be {NATIVE_POOLING!r}.")
    _check(manifest.get("head_type") == HEAD_TYPE,
           f"bundle head_type must be {HEAD_TYPE!r}.")
    _check(manifest.get("model_family") == MODEL_FAMILY,
           f"bundle model_family must be {MODEL_FAMILY!r}.")
    _check(manifest.get("native_protocol") == NATIVE_PROTOCOL,
           f"bundle native_protocol must be {NATIVE_PROTOCOL!r}.")
    _check(manifest.get("dr_grade_output_shape") == DR_GRADE_OUTPUT_SHAPE,
           f"bundle dr_grade_output_shape must be {DR_GRADE_OUTPUT_SHAPE}.")
    _check(manifest.get("preprocessing_hash") == preprocessing_parity.preprocessing_hash(),
           "bundle preprocessing_hash does not match the canonical native-392 route.")

    backbone_ckpt = Path(manifest["backbone_checkpoint_path"])
    head_ckpt = Path(manifest["head_checkpoint_path"])
    _check(backbone_ckpt.exists(), f"backbone checkpoint missing: {backbone_ckpt}")
    _check(head_ckpt.exists(), f"head checkpoint missing: {head_ckpt}")

    # live hash gates
    live_backbone_sha = sha256_file(backbone_ckpt)
    _check(live_backbone_sha == manifest["backbone_checkpoint_sha256"],
           "backbone checkpoint SHA-256 changed since the bundle was built.")
    live_head_sha = sha256_file(head_ckpt)
    _check(live_head_sha == manifest["head_checkpoint_sha256"],
           "head checkpoint SHA-256 changed since the bundle was built.")

    # task ordering gate (intrinsic to the live checkpoint)
    state = _load_head_state_dict(head_ckpt)
    live_order = _task_order_from_state_dict(state)
    _check(live_order == list(manifest["model_task_ordering"]),
           "head task ordering changed since the bundle was built.")
    _check(task_ordering_hash(live_order) == manifest["task_ordering_hash"],
           "task_ordering_hash mismatch.")

    return DeploymentBundle(
        manifest=manifest,
        backbone_checkpoint=backbone_ckpt,
        head_checkpoint=head_ckpt,
        task_order=tuple(live_order),
    )


def _local_dir() -> Path:
    from .provenance import LOCAL_DIR  # noqa: PLC0415

    return LOCAL_DIR


def default_bundle_path() -> Path:
    """Resolve the local bundle path (env override or default .local location)."""
    override = os.environ.get("RETINA_SCREEN_DEMO_BUNDLE_PATH")
    if override:
        return Path(override)
    return _local_dir() / "deployment_bundle.local.json"


def load_validated_bundle(path: Path | None = None) -> DeploymentBundle:
    """Load the local manifest and re-validate it against live artifacts."""
    p = path or default_bundle_path()
    manifest = load_bundle_file(p)
    return validate_bundle_against_artifacts(manifest)
