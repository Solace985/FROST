from __future__ import annotations

import numpy as np

from deploy.referable_dr_demo.backend.service.inference import referable_dr_score


def _np_softmax(row: np.ndarray) -> np.ndarray:
    z = row - row.max()
    e = np.exp(z)
    return e / e.sum()


def test_referable_score_formula_uses_softmax_classes_2_to_4():
    """9. Score equals softmax(logits)[2:5].sum for arbitrary logit rows."""
    logits = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [2.0, 1.0, -1.0, 0.5, 3.0],
            [-3.0, 4.0, 0.0, 1.0, -2.0],
        ],
        dtype=np.float64,
    )
    got = referable_dr_score(logits)
    expected = np.array([_np_softmax(r)[2:].sum() for r in logits])
    np.testing.assert_allclose(got, expected, atol=1e-12)
    assert abs(got[0] - 0.6) < 1e-12


def test_grade_1_is_not_referable():
    """10. Mass concentrated on grade 0 or 1 yields a near-zero referable score."""
    grade1 = np.array([[0.0, 12.0, 0.0, 0.0, 0.0]])
    grade0 = np.array([[12.0, 0.0, 0.0, 0.0, 0.0]])
    assert float(referable_dr_score(grade1)[0]) < 1e-3
    assert float(referable_dr_score(grade0)[0]) < 1e-3


def test_argmax_is_not_used_as_score():
    """11. Score is probability mass on grades 2-4, distinct from an argmax rule."""
    logits = np.array([[3.0, 0.0, 1.0, 1.0, 1.0]])
    score = float(referable_dr_score(logits)[0])
    expected = float(_np_softmax(logits[0])[2:].sum())

    argmax_class = int(np.argmax(logits[0]))
    argmax_referable_indicator = 1.0 if argmax_class >= 2 else 0.0

    assert argmax_class == 0
    assert abs(score - expected) < 1e-12
    assert score > 0.2
    assert abs(score - argmax_referable_indicator) > 0.2
