"""Frozen-inference and parity tests (FROST tests 7, 8, 22, 23)."""

from __future__ import annotations

import pytest

from deploy.referable_dr_demo.analysis import verify_parity


def test_backbone_is_frozen(service):
    """7. Every backbone parameter has requires_grad == False."""
    grads = [p.requires_grad for p in service.backbone.parameters()]
    assert grads, "backbone has no parameters?"
    assert not any(grads), "backbone is not fully frozen"


def test_inference_uses_eval_and_inference_mode(service):
    """8. Backbone and head are in eval mode; inference yields a valid score."""
    assert service.backbone.training is False
    assert service.head.training is False
    img = verify_parity.deterministic_synthetic_image()
    result = service.infer(img)
    assert 0.0 <= result.referable_score <= 1.0
    assert result.embedding_dim == 384
    assert len(result.dr_grade_probs) == 5
    assert abs(sum(result.dr_grade_probs) - 1.0) < 1e-6


def test_canonical_synthetic_inference_parity(service, built_bundle):
    """22. App inference matches an independent canonical reconstruction."""
    res = verify_parity.canonical_synthetic_parity(service, built_bundle)
    assert res["status"] == "pass", res
    assert res["max_abs_logit_diff"] <= verify_parity.LOGIT_TOL
    assert res["max_abs_score_diff"] <= verify_parity.SCORE_TOL
    assert res["embedding_dim"] == 384
    assert res["dr_grade_logits_shape"] == [5]


def test_study_linked_parity_when_local_fixture_available(service, built_bundle):
    """23. App scores match accepted predictions.npz on local BRSET test images.

    Skipped only when the local BRSET parity fixture is unavailable.
    """
    res = verify_parity.study_linked_parity(service, built_bundle, write_cases=False)
    if res["status"] == "unavailable":
        pytest.skip(f"study-linked parity fixture unavailable: {res.get('reason')}")
    assert res["status"] == "pass", res
    assert res["n_cases"] >= 3
    assert res["max_abs_score_diff"] < verify_parity.STUDY_TOL
