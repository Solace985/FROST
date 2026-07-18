from __future__ import annotations

from pathlib import Path

import pytest
import torch

from retina_screen.embeddings import (
    _compact_embedding,
    load_embedding,
    save_embedding,
)

_DIM = 768
_SEQ_LEN = 257




def _make_cls_view() -> torch.Tensor:
    """Return a (768,) tensor that is a view into a (1, 257, 768) token sequence.

    This simulates the shape returned by `backbone(tensor).squeeze(0).cpu()` when the
    backbone is DINOv2 ViT-B/14 and the hub model exposes the CLS token as a slice.
    """
    tokens = torch.randn(1, _SEQ_LEN, _DIM)
    return tokens[:, 0].squeeze(0)


def _storage_ratio(t: torch.Tensor) -> float:
    compact_bytes = t.numel() * t.element_size()
    storage_bytes = t.untyped_storage().nbytes()
    return storage_bytes / compact_bytes




class TestViewStorageConditionDetected:
    """Confirms the simulated pre-patch state exhibits bloated backing storage."""

    def test_shape_is_correct_before_compaction(self) -> None:
        cls_view = _make_cls_view()
        assert cls_view.shape == (_DIM,), f"Expected ({_DIM},), got {cls_view.shape}"

    def test_storage_ratio_exceeds_threshold_before_compaction(self) -> None:
        """The view tensor's backing storage must be >> its visible elements."""
        cls_view = _make_cls_view()
        ratio = _storage_ratio(cls_view)
        assert ratio > 10.0, (
            f"Expected storage_ratio > 10.0 to simulate the view-storage leak; "
            f"got {ratio:.2f}.  Check that _make_cls_view() still produces a view."
        )




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
