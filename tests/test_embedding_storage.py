"""
test_embedding_storage.py -- Regression tests for embedding cache storage compaction.

Guards against the W1 storage-view bug where ViT-based backbones (e.g. DINOv2) return
the CLS token as a view into the full token-sequence backing storage.  When torch.save()
serialises a view it writes the entire backing buffer rather than only the visible elements,
inflating each .pt file from ~3 KB to ~773 KB for DINOv2-Base (ViT-B/14, 257 tokens).

These tests verify:
  1. The pre-patch view-storage condition is detectable (simulates the bug).
  2. _compact_embedding fixes the storage ratio to <= 2.0, preserves shape/dtype/values.
  3. save_embedding + load_embedding round-trip produces a compact file on disk.

No real backbone, no real images, no network access required.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from retina_screen.embeddings import (
    _compact_embedding,
    load_embedding,
    save_embedding,
)

# ViT-B/14: 16*16 spatial patches + 1 CLS token = 257 sequence positions
_DIM = 768
_SEQ_LEN = 257


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cls_view() -> torch.Tensor:
    """Return a (768,) tensor that is a view into a (1, 257, 768) token sequence.

    This simulates the shape returned by `backbone(tensor).squeeze(0).cpu()` when the
    backbone is DINOv2 ViT-B/14 and the hub model exposes the CLS token as a slice.
    """
    tokens = torch.randn(1, _SEQ_LEN, _DIM)
    # Slice the CLS token: tokens[:, 0] has shape (1, 768), squeeze(0) → (768,).
    # The result shares backing storage with `tokens` (all 257 * 768 * 4 bytes).
    return tokens[:, 0].squeeze(0)


def _storage_ratio(t: torch.Tensor) -> float:
    compact_bytes = t.numel() * t.element_size()
    storage_bytes = t.untyped_storage().nbytes()
    return storage_bytes / compact_bytes


# ---------------------------------------------------------------------------
# Class 1: verify the view-storage condition exists before compaction
# ---------------------------------------------------------------------------


class TestViewStorageConditionDetected:
    """Confirms the simulated pre-patch state exhibits bloated backing storage."""

    def test_shape_is_correct_before_compaction(self) -> None:
        cls_view = _make_cls_view()
        assert cls_view.shape == (_DIM,), f"Expected ({_DIM},), got {cls_view.shape}"

    def test_storage_ratio_exceeds_threshold_before_compaction(self) -> None:
        """The view tensor's backing storage must be >> its visible elements."""
        cls_view = _make_cls_view()
        ratio = _storage_ratio(cls_view)
        # For ViT-B/14 the ratio should be ~257; require at least 10 to be robust.
        assert ratio > 10.0, (
            f"Expected storage_ratio > 10.0 to simulate the view-storage leak; "
            f"got {ratio:.2f}.  Check that _make_cls_view() still produces a view."
        )


# ---------------------------------------------------------------------------
# Class 2: _compact_embedding fixes storage while preserving semantics
# ---------------------------------------------------------------------------


class TestCompactionFixesStorage:
    """Verifies that _compact_embedding resolves the storage leak."""

    def test_shape_preserved(self) -> None:
        result = _compact_embedding(_make_cls_view())
        assert result.shape == (_DIM,)

    def test_dtype_preserved(self) -> None:
        result = _compact_embedding(_make_cls_view())
        assert result.dtype == torch.float32

    def test_no_nan(self) -> None:
        result = _compact_embedding(_make_cls_view())
        assert not result.isnan().any()

    def test_no_inf(self) -> None:
        result = _compact_embedding(_make_cls_view())
        assert not result.isinf().any()

    def test_storage_ratio_le_two(self) -> None:
        """After compaction the backing storage must cover only the visible elements."""
        result = _compact_embedding(_make_cls_view())
        ratio = _storage_ratio(result)
        assert ratio <= 2.0, (
            f"After _compact_embedding, storage_ratio must be <= 2.0; got {ratio:.2f}"
        )

    def test_values_match_original(self) -> None:
        """Compaction must not alter numerical values."""
        cls_view = _make_cls_view()
        result = _compact_embedding(cls_view)
        assert torch.allclose(cls_view, result)

    def test_not_grad_fn(self) -> None:
        """Compacted tensor must be detached from the autograd graph."""
        cls_view = _make_cls_view()
        result = _compact_embedding(cls_view)
        assert result.grad_fn is None

    def test_is_contiguous(self) -> None:
        result = _compact_embedding(_make_cls_view())
        assert result.is_contiguous()


# ---------------------------------------------------------------------------
# Class 3: save_embedding + load_embedding round-trip is compact
# ---------------------------------------------------------------------------


class TestSaveReloadCompactStorage:
    """Verifies the full save/load cycle produces a small, correctly-shaped file."""

    def test_saved_file_size_is_compact(self, tmp_path: Path) -> None:
        """The .pt file must be well under 10 KB for a compact 768-dim embedding."""
        compacted = _compact_embedding(_make_cls_view())
        path, _checksum = save_embedding(compacted, "test_compact_size", tmp_path)
        file_bytes = path.stat().st_size
        assert file_bytes < 10 * 1024, (
            f"Expected .pt file < 10 KB for a compact {_DIM}-dim embedding; "
            f"got {file_bytes / 1024:.2f} KB.  W1 storage compaction may not be active."
        )

    def test_reloaded_shape(self, tmp_path: Path) -> None:
        compacted = _compact_embedding(_make_cls_view())
        path, checksum = save_embedding(compacted, "test_reload_shape", tmp_path)
        reloaded = load_embedding(path, checksum, _DIM)
        assert reloaded.shape == (_DIM,)

    def test_reloaded_storage_ratio_le_two(self, tmp_path: Path) -> None:
        """Reloaded tensor must also have compact backing storage."""
        compacted = _compact_embedding(_make_cls_view())
        path, checksum = save_embedding(compacted, "test_reload_storage", tmp_path)
        reloaded = load_embedding(path, checksum, _DIM)
        ratio = _storage_ratio(reloaded)
        assert ratio <= 2.0, (
            f"Reloaded tensor storage_ratio must be <= 2.0; got {ratio:.2f}"
        )

    def test_reloaded_values_match(self, tmp_path: Path) -> None:
        compacted = _compact_embedding(_make_cls_view())
        path, checksum = save_embedding(compacted, "test_reload_values", tmp_path)
        reloaded = load_embedding(path, checksum, _DIM)
        assert torch.allclose(compacted, reloaded)
