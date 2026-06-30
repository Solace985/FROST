"""Technical input-check tests (FROST test 19)."""

from __future__ import annotations

import io

import pytest
from PIL import Image

from deploy.referable_dr_demo.backend.service import image_checks


def _png(w: int, h: int, fmt: str = "PNG") -> bytes:
    img = Image.new("RGB", (w, h), (40, 80, 120))
    # add structure so it is not a flat image
    for y in range(0, h, 8):
        for x in range(0, w, 8):
            img.putpixel((x, y), (200, 30, 30))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def test_technical_input_rejection():
    """19. Valid JPEG/PNG accepted; malformed/unsupported/oversize inputs rejected."""
    # valid PNG accepted
    rgb, result = image_checks.check_and_decode(_png(420, 380))
    assert rgb.mode == "RGB"
    assert result.image_format == "PNG"
    assert result.width == 420 and result.height == 380

    # valid JPEG accepted
    rgb_j, result_j = image_checks.check_and_decode(_png(300, 300, fmt="JPEG"))
    assert result_j.image_format == "JPEG"

    # empty upload
    with pytest.raises(image_checks.ImageCheckError) as e_empty:
        image_checks.check_and_decode(b"")
    assert e_empty.value.category == "empty_upload"

    # not an image
    with pytest.raises(image_checks.ImageCheckError):
        image_checks.check_and_decode(b"this is definitely not an image file")

    # below minimum dimensions
    with pytest.raises(image_checks.ImageCheckError) as e_small:
        image_checks.check_and_decode(_png(20, 20))
    assert e_small.value.category == "below_min_dimensions"

    # unsupported format (BMP)
    with pytest.raises(image_checks.ImageCheckError) as e_fmt:
        image_checks.check_and_decode(_png(200, 200, fmt="BMP"))
    assert e_fmt.value.category == "unsupported_format"

    # oversize byte payload
    with pytest.raises(image_checks.ImageCheckError) as e_big:
        image_checks.check_and_decode(b"\x89PNG" + b"0" * (image_checks.MAX_BYTES + 1))
    assert e_big.value.category == "file_too_large"
