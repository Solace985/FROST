from __future__ import annotations

import logging
import os

logger = logging.getLogger("frost.weights_source")

ENV_BACKBONE_CKPT = "RETINA_SCREEN_RETF_GREEN_CHECKPOINT"
ENV_CANONICAL_BACKBONE_CKPT = "RETFOUND_GREEN_CHECKPOINT"
ENV_HEAD_CKPT = "RETINA_SCREEN_RETF_NATIVE392_MT_CHECKPOINT"

BACKBONE_SHA256 = "431DE5DBC1BEBBB32F60E2C0BCF8DAA4F8BCBF06F7CB1E1DC97EC589713942E1"

DEFAULT_BACKBONE_FILENAME = "retfoundgreen_statedict.pth"
DEFAULT_HEAD_FILENAME = "mt_head_native392.pt"


def resolve() -> None:
    """Resolve weights for the hosted run. No-op unless ``FROST_WEIGHTS_REPO`` set.

    When the HF-repo route is used, downloads the backbone + head from the model
    repo and exports the discovery env vars so the existing bundle builder finds
    them. Safe to call unconditionally at startup.
    """
    repo = os.environ.get("FROST_WEIGHTS_REPO", "").strip()
    if not repo:
        logger.info(
            "FROST_WEIGHTS_REPO unset; using baked-in backbone + committed head "
            "(paths from the container env vars)."
        )
        return

    try:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "FROST_WEIGHTS_REPO is set but huggingface_hub is not installed."
        ) from exc

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    backbone_file = os.environ.get("FROST_BACKBONE_FILENAME", DEFAULT_BACKBONE_FILENAME)
    head_file = os.environ.get("FROST_HEAD_FILENAME", DEFAULT_HEAD_FILENAME)

    logger.info("Downloading weights from HF model repo %s ...", repo)
    backbone_path = hf_hub_download(repo_id=repo, filename=backbone_file, token=token)
    head_path = hf_hub_download(repo_id=repo, filename=head_file, token=token)

    os.environ[ENV_BACKBONE_CKPT] = backbone_path
    os.environ[ENV_CANONICAL_BACKBONE_CKPT] = backbone_path
    os.environ[ENV_HEAD_CKPT] = head_path
    logger.info(
        "Weights resolved from HF repo (backbone=%s, head=%s). SHA/shape gates are "
        "re-verified by the bundle builder at startup.",
        backbone_file, head_file,
    )
