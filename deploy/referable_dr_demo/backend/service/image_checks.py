from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

MAX_BYTES = 25 * 1024 * 1024
MAX_PIXELS = 50_000_000
MIN_DIMENSION = 96
SUPPORTED_FORMATS = frozenset({"JPEG", "PNG"})

LOW_CONTRAST_STD = 6.0
EXTREME_DARK_MEAN = 8.0
EXTREME_BRIGHT_MEAN = 247.0
EXTREME_BLUR_LAPLACIAN_VAR = 2.0


class ImageCheckError(Exception):
    """Raised when an upload fails a technical input check.

    ``category`` is a safe, fixed token suitable for client display and logging
    (never contains the filename, bytes, or any user content).
    """

    def __init__(self, category: str, message: str) -> None:
        super().__init__(message)
        self.category = category
        self.message = message


@dataclass
class ImageCheckResult:
    width: int
    height: int
    byte_size: int
    image_format: str
    mean_intensity: float
    contrast_std: float
    warnings: list[str] = field(default_factory=list)


def check_and_decode(raw: bytes) -> tuple[Image.Image, ImageCheckResult]:
    """Validate and decode an upload to an RGB image + technical-check result.

    Raises
    ------
    ImageCheckError
        On any rejection (too large/small, unsupported, corrupt, bomb, bad mode).
    """
    if not raw:
        raise ImageCheckError("empty_upload", "No image data was received.")
    if len(raw) > MAX_BYTES:
        raise ImageCheckError(
            "file_too_large",
            f"Upload exceeds the local size limit ({MAX_BYTES // (1024 * 1024)} MB).",
        )

    try:
        probe = Image.open(io.BytesIO(raw))
        fmt = probe.format
        width, height = probe.size
    except UnidentifiedImageError:
        raise ImageCheckError("not_an_image", "The file is not a recognised image.") from None
    except Exception as exc:  # noqa: BLE001
        raise ImageCheckError("corrupt_image", "The image could not be read.") from exc

    if fmt not in SUPPORTED_FORMATS:
        raise ImageCheckError(
            "unsupported_format",
            f"Unsupported image format {fmt!r}. Use JPEG or PNG.",
        )
    if width <= 0 or height <= 0:
        raise ImageCheckError("invalid_dimensions", "The image has invalid dimensions.")
    if width * height > MAX_PIXELS:
        raise ImageCheckError(
            "decompression_bomb",
            "The image has too many pixels and was rejected as a safety measure.",
        )
    if width < MIN_DIMENSION or height < MIN_DIMENSION:
        raise ImageCheckError(
            "below_min_dimensions",
            f"The image is smaller than the minimum {MIN_DIMENSION}x{MIN_DIMENSION}.",
        )

    mode = probe.mode
    if mode in ("A",):
        raise ImageCheckError(
            "alpha_only", "Alpha-only images are not supported."
        )

    try:
        probe.load()
        rgb = probe.convert("RGB")
    except ImageCheckError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ImageCheckError(
            "corrupt_image", "The image could not be decoded."
        ) from exc

    arr = np.asarray(rgb, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ImageCheckError(
            "invalid_channels", "The decoded image does not have 3 RGB channels."
        )

    mean_intensity = float(arr.mean())
    contrast_std = float(arr.std())

    warnings: list[str] = []
    if contrast_std < LOW_CONTRAST_STD:
        warnings.append("very_low_contrast")
    if mean_intensity < EXTREME_DARK_MEAN:
        warnings.append("extremely_dark")
    if mean_intensity > EXTREME_BRIGHT_MEAN:
        warnings.append("extremely_bright")
    if _laplacian_variance(arr) < EXTREME_BLUR_LAPLACIAN_VAR:
        warnings.append("possible_extreme_blur")

    result = ImageCheckResult(
        width=width,
        height=height,
        byte_size=len(raw),
        image_format=fmt,
        mean_intensity=mean_intensity,
        contrast_std=contrast_std,
        warnings=warnings,
    )
    return rgb, result


def _laplacian_variance(arr_rgb: np.ndarray) -> float:
    """Deterministic blur proxy: variance of a discrete Laplacian on luma."""
    luma = arr_rgb.mean(axis=2)
    lap = (
        -4.0 * luma[1:-1, 1:-1]
        + luma[:-2, 1:-1]
        + luma[2:, 1:-1]
        + luma[1:-1, :-2]
        + luma[1:-1, 2:]
    )
    return float(lap.var()) if lap.size else 0.0
