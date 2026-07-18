from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from deploy.referable_dr_demo.backend.service import bundle as bundle_mod  # noqa: E402
from deploy.referable_dr_demo.backend.service import inference as inference_mod  # noqa: E402
from deploy.referable_dr_demo.backend.service import preprocessing_parity  # noqa: E402
from deploy.referable_dr_demo.backend.service.provenance import (  # noqa: E402
    LOCAL_DIR,
    REPO_ROOT,
    ensure_src_importable,
)

logger = logging.getLogger(__name__)

LOGIT_TOL = 1e-5
SCORE_TOL = 1e-6
STUDY_TOL = 1e-4
N_STUDY_CASES = 5

ensure_src_importable()


def deterministic_synthetic_image(size: tuple[int, int] = (500, 480)) -> Image.Image:
    """A fixed, RNG-free RGB image (non-square, to exercise resize+centercrop)."""
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w]
    r = (xx * 255 // max(1, w - 1)).astype(np.uint8)
    g = (yy * 255 // max(1, h - 1)).astype(np.uint8)
    b = (((xx + yy) // 2) % 256).astype(np.uint8)
    check = (((xx // 16) + (yy // 16)) % 2).astype(np.uint8) * 40
    arr = np.stack([np.clip(r + check, 0, 255),
                    np.clip(g, 0, 255),
                    np.clip(b + check, 0, 255)], axis=-1).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _canonical_reference_infer(
    image: Image.Image, bundle: bundle_mod.DeploymentBundle
) -> tuple[np.ndarray, float]:
    """Independent inline reference: canonical primitives, no InferenceService."""
    import torch  # noqa: PLC0415

    from retina_screen.embeddings import BackboneConfig, load_backbone  # noqa: PLC0415
    from retina_screen.evaluation.referable_dr import (  # noqa: PLC0415
        compute_referable_dr_score,
    )
    from retina_screen.model import build_head  # noqa: PLC0415

    cfg = BackboneConfig(**bundle.backbone_config_kwargs())
    backbone = load_backbone(cfg, torch.device("cpu"))
    backbone.eval()
    head = build_head(
        embedding_dim=bundle.embedding_dim,
        task_names=list(bundle.task_order),
        head_type="multitask",
    )
    head.load_state_dict(
        torch.load(bundle.head_checkpoint, map_location="cpu", weights_only=True)
    )
    head.eval()

    tensor = preprocessing_parity.preprocess(image)
    with torch.inference_mode():
        emb = backbone(tensor)
        logits = head(emb)[bundle.manifest["dr_grade_task_key"]]
    logits_np = logits.detach().cpu().numpy().astype(np.float64)
    score = float(compute_referable_dr_score(logits_np)[0])
    return logits_np[0], score


def canonical_synthetic_parity(
    service: inference_mod.InferenceService, bundle: bundle_mod.DeploymentBundle
) -> dict[str, Any]:
    image = deterministic_synthetic_image()
    app = service.infer(image)
    app_logits = np.asarray(app.dr_grade_logits, dtype=np.float64)
    ref_logits, ref_score = _canonical_reference_infer(image, bundle)

    logit_diff = float(np.max(np.abs(app_logits - ref_logits)))
    score_diff = float(abs(app.referable_score - ref_score))
    passed = (logit_diff <= LOGIT_TOL) and (score_diff <= SCORE_TOL)
    return {
        "status": "pass" if passed else "fail",
        "preprocessing_output_shape": [1, 3, preprocessing_parity.EXPECTED_IMAGE_SIZE,
                                       preprocessing_parity.EXPECTED_IMAGE_SIZE],
        "input_resolution": app.input_resolution,
        "embedding_dim": app.embedding_dim,
        "dr_grade_logits_shape": [5],
        "max_abs_logit_diff": logit_diff,
        "max_abs_score_diff": score_diff,
        "logit_tol": LOGIT_TOL,
        "score_tol": SCORE_TOL,
    }


def integrity_self_check(
    service: inference_mod.InferenceService, bundle: bundle_mod.DeploymentBundle
) -> dict[str, Any]:
    """Model-load integrity check that uses NO real image and NO study data.

    Runs a single forward through the already-loaded service on a deterministic
    RANDOM (non-patient) RGB array and asserts the outputs are well-formed:
    expected embedding dim (384), 5 dr_grade class probabilities that are finite
    and sum to 1, and a finite referable score in [0, 1]. This is the server's
    startup self-check: it proves the frozen model loaded and produces in-range
    outputs, without depending on any credentialed BRSET image. The stronger
    study-score reproduction is proven separately by the LOCAL pre-deploy parity
    gate (:func:`study_linked_parity`).
    """
    rng = np.random.default_rng(20260702)
    arr = rng.integers(0, 256, size=(480, 500, 3), dtype=np.uint8)
    image = Image.fromarray(arr, mode="RGB")
    result = service.infer(image)
    probs = np.asarray(result.dr_grade_probs, dtype=np.float64)
    score = float(result.referable_score)
    checks = {
        "embedding_dim_ok": bool(result.embedding_dim == bundle.embedding_dim),
        "n_class_probs_ok": bool(probs.shape == (bundle.manifest["dr_grade_output_shape"],)),
        "probs_finite": bool(np.all(np.isfinite(probs))),
        "probs_sum_to_one": bool(abs(float(probs.sum()) - 1.0) <= 1e-5),
        "score_finite": bool(np.isfinite(score)),
        "score_in_unit_interval": bool(0.0 <= score <= 1.0),
    }
    passed = all(checks.values())
    return {
        "status": "pass" if passed else "fail",
        "checks": checks,
        "embedding_dim": result.embedding_dim,
        "uses_real_image": False,
    }


def _import_from_string(path: str) -> Any:
    if ":" in path:
        mod_name, _, attr = path.partition(":")
    else:
        mod_name, _, attr = path.rpartition(".")
    import importlib  # noqa: PLC0415

    module = importlib.import_module(mod_name)
    return getattr(module, attr)


def _resolve_repo_path(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else REPO_ROOT / path


def _build_adapter_from_run(bundle: bundle_mod.DeploymentBundle):
    """Generic, provenance-driven adapter construction (no dataset hardcoding)."""
    import yaml  # noqa: PLC0415

    rc = bundle.manifest.get("resolved_config_path")
    if not rc or not Path(rc).exists():
        return None, "resolved_config.yaml unavailable"
    run_cfg = yaml.safe_load(Path(rc).read_text(encoding="utf-8"))
    dataset_config = run_cfg.get("dataset_config")
    if not dataset_config:
        return None, "dataset_config not recorded in resolved_config"
    dc_path = _resolve_repo_path(dataset_config)
    if not dc_path.exists():
        return None, f"dataset config missing: {dc_path}"
    ds_cfg = yaml.safe_load(dc_path.read_text(encoding="utf-8"))
    adapter_class = ds_cfg.get("adapter_class")
    if not adapter_class:
        return None, "adapter_class not declared in dataset config"
    try:
        cls = _import_from_string(str(adapter_class))
        adapter = cls()
    except Exception as exc:  # noqa: BLE001 - data may be unavailable locally
        return None, f"adapter could not be constructed: {exc}"
    return adapter, "ok"


def _discover_test_predictions(bundle: bundle_mod.DeploymentBundle) -> Path | None:
    env = os.environ.get("RETINA_SCREEN_DEMO_TEST_PREDICTIONS")
    if env and Path(env).exists():
        return Path(env)
    run_name = Path(bundle.head_checkpoint).parent.name
    ev_root = REPO_ROOT / "outputs" / "evaluation"
    if ev_root.exists():
        for d in sorted(ev_root.iterdir(), reverse=True):
            pn = d / "predictions.npz"
            if not pn.exists():
                continue
            for j in d.glob("*.json"):
                try:
                    if run_name in j.read_text(encoding="utf-8"):
                        return pn
                except OSError:
                    continue
    default = ev_root / "20260525_113924" / "predictions.npz"
    return default if default.exists() else None


def study_linked_parity(
    service: inference_mod.InferenceService,
    bundle: bundle_mod.DeploymentBundle,
    n_cases: int = N_STUDY_CASES,
    write_cases: bool = True,
) -> dict[str, Any]:
    from retina_screen.evaluation.referable_dr import (  # noqa: PLC0415
        compute_referable_dr_score,
    )

    preds_path = _discover_test_predictions(bundle)
    if preds_path is None:
        return {"status": "unavailable", "reason": "accepted test predictions.npz not found"}
    adapter, why = _build_adapter_from_run(bundle)
    if adapter is None:
        return {"status": "unavailable", "reason": f"adapter/data unavailable: {why}"}

    data = np.load(preds_path, allow_pickle=True)
    if "sample_id" not in data or "logit__dr_grade" not in data:
        return {"status": "unavailable", "reason": "predictions.npz missing required keys"}
    sample_ids = [str(s) for s in data["sample_id"]]
    logits = np.asarray(data["logit__dr_grade"], dtype=np.float64)

    cases: list[dict[str, Any]] = []
    for sid, lg in zip(sample_ids, logits):
        if len(cases) >= n_cases:
            break
        try:
            image = adapter.load_image(sid)
        except Exception:  # noqa: BLE001 - image may be absent locally
            continue
        app_score = service.infer(image).referable_score
        expected = float(compute_referable_dr_score(lg.reshape(1, 5))[0])
        cases.append({
            "sample_id": sid,
            "app_score": app_score,
            "expected_score": expected,
            "abs_diff": abs(app_score - expected),
        })

    if len(cases) < 3:
        return {
            "status": "unavailable",
            "reason": f"only {len(cases)} local BRSET test images resolvable (need >=3)",
        }

    max_diff = max(c["abs_diff"] for c in cases)
    passed = max_diff < STUDY_TOL
    if write_cases:
        _write_parity_cases(preds_path, cases)

    return {
        "status": "pass" if passed else "fail",
        "n_cases": len(cases),
        "max_abs_score_diff": max_diff,
        "tol": STUDY_TOL,
        "abs_diffs": [c["abs_diff"] for c in cases],
    }


def _write_parity_cases(preds_path: Path, cases: list[dict[str, Any]]) -> None:
    override = os.environ.get("RETINA_SCREEN_DEMO_PARITY_CASES_PATH")
    out = Path(override) if override else LOCAL_DIR / "parity_cases.local.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "_warning": "LOCAL ONLY. Contains BRSET sample identifiers. Never commit.",
        "predictions_path": str(preds_path),
        "tol": STUDY_TOL,
        "cases": cases,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_all(write_cases: bool = True) -> dict[str, Any]:
    bundle = bundle_mod.load_validated_bundle()
    service = inference_mod.init_service(bundle)
    synthetic = canonical_synthetic_parity(service, bundle)
    study = study_linked_parity(service, bundle, write_cases=write_cases)
    overall = (
        synthetic["status"] == "pass"
        and study["status"] in ("pass", "unavailable")
    )
    return {
        "overall": "pass" if overall else "fail",
        "canonical_synthetic_parity": synthetic,
        "study_linked_parity": study,
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = run_all()
    print(json.dumps(result, indent=2))
    syn = result["canonical_synthetic_parity"]
    study = result["study_linked_parity"]
    print("\nSUMMARY")
    print(f"  canonical synthetic parity : {syn['status']} "
          f"(max abs logit diff={syn['max_abs_logit_diff']:.2e}, "
          f"max abs score diff={syn['max_abs_score_diff']:.2e})")
    if study["status"] in ("pass", "fail"):
        print(f"  study-linked parity        : {study['status']} "
              f"(n={study['n_cases']}, max abs diff={study['max_abs_score_diff']:.2e})")
    else:
        print(f"  study-linked parity        : {study['status']} ({study.get('reason')})")
    return 0 if result["overall"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
