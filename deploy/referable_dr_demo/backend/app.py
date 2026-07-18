from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from deploy.referable_dr_demo import __version__ as APP_VERSION  # noqa: E402
from deploy.referable_dr_demo.analysis import verify_parity  # noqa: E402
from deploy.referable_dr_demo.backend import schemas, static_server  # noqa: E402
from deploy.referable_dr_demo.backend.service import bundle as bundle_mod  # noqa: E402
from deploy.referable_dr_demo.backend.service import image_checks  # noqa: E402
from deploy.referable_dr_demo.backend.service import inference as inference_mod  # noqa: E402
from deploy.referable_dr_demo.backend.service import privacy  # noqa: E402
from deploy.referable_dr_demo.backend.service import threshold_policy  # noqa: E402

logger = logging.getLogger("frost.app")

BACKBONE_DISPLAY = "RETFound-Green native-392"
BACKBONE_PARAMS = "~22M"
REFERABLE_FORMULA = "p2+p3+p4"


@dataclass
class AppState:
    ready: bool = False
    blocked_reason: str | None = None
    bundle: Any = None
    service: Any = None
    operating_point: Any = None
    parity_status: str = "unavailable"
    threshold_status: str = "blocked"
    bundle_version: str = "unknown"
    parity_detail: dict[str, Any] = field(default_factory=dict)


STATE = AppState()


def _startup() -> None:
    """Run every gate; populate STATE. Never raises (records blocked reason)."""
    try:
        if bundle_mod.default_bundle_path().exists():
            STATE.bundle = bundle_mod.load_validated_bundle()
        else:
            STATE.bundle = bundle_mod.build_validated_bundle()
        STATE.bundle_version = STATE.bundle.bundle_version
    except Exception as exc:  # noqa: BLE001
        STATE.blocked_reason = f"bundle validation failed: {exc}"
        logger.error("Startup blocked: %s", STATE.blocked_reason)
        return

    try:
        STATE.service = inference_mod.init_service(STATE.bundle)
    except Exception as exc:  # noqa: BLE001
        STATE.blocked_reason = f"model load failed: {exc}"
        logger.error("Startup blocked: %s", STATE.blocked_reason)
        return

    try:
        STATE.operating_point = threshold_policy.load_operating_point(STATE.bundle)
        STATE.threshold_status = "validated"
    except Exception as exc:  # noqa: BLE001
        STATE.threshold_status = "blocked"
        STATE.blocked_reason = f"operating point invalid: {exc}"
        logger.error("Startup blocked: %s", STATE.blocked_reason)
        return

    try:
        syn = verify_parity.canonical_synthetic_parity(STATE.service, STATE.bundle)
        study = verify_parity.study_linked_parity(
            STATE.service, STATE.bundle, write_cases=False
        )
        STATE.parity_detail = {"synthetic": syn, "study_linked": study}
        if syn["status"] != "pass":
            STATE.parity_status = "fail"
            STATE.blocked_reason = "canonical synthetic parity failed"
        elif study["status"] == "pass":
            STATE.parity_status = "pass"
        elif study["status"] == "fail":
            STATE.parity_status = "fail"
            STATE.blocked_reason = "study-linked parity failed"
        else:
            if os.environ.get("FROST_ALLOW_SYNTHETIC_ONLY_READINESS") == "1":
                integ = verify_parity.integrity_self_check(STATE.service, STATE.bundle)
                STATE.parity_detail["integrity"] = integ
                if integ["status"] == "pass":
                    STATE.parity_status = "pass"
                else:
                    STATE.parity_status = "fail"
                    STATE.blocked_reason = "non-credentialed integrity self-check failed"
            else:
                STATE.parity_status = "unavailable"
                STATE.blocked_reason = (
                    "study-linked parity fixture unavailable; refusing to emit real "
                    "predictions"
                )
    except Exception as exc:  # noqa: BLE001
        STATE.parity_status = "fail"
        STATE.blocked_reason = f"parity self-check error: {exc}"
        logger.error("Startup blocked: %s", STATE.blocked_reason)
        return

    STATE.ready = (
        STATE.parity_status == "pass"
        and STATE.threshold_status == "validated"
        and STATE.service is not None
    )
    if STATE.ready:
        STATE.blocked_reason = None
        logger.info("FROST ready (bundle=%s).", STATE.bundle_version)
    else:
        logger.error("FROST not ready: %s", STATE.blocked_reason)


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN201
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    _startup()
    yield


app = FastAPI(
    title="FROST — Frozen Representation for Ocular Screening and Triage",
    description="Local referable diabetic retinopathy research demonstrator.",
    version=APP_VERSION,
    lifespan=lifespan,
)
static_server.mount_frontend(app)


@app.get("/health", response_model=schemas.HealthResponse)
def health() -> schemas.HealthResponse:
    return schemas.HealthResponse(
        status="ready" if STATE.ready else "blocked",
        bundle_version=STATE.bundle_version,
        model_loaded=STATE.service is not None,
        parity_status=STATE.parity_status,
        threshold_status=STATE.threshold_status,
        network_access_required=False,
    )


def _error(category: str, message: str, http_status: int) -> JSONResponse:
    return JSONResponse(
        status_code=http_status,
        content={"error": message, "category": category},
    )


@app.post("/predict")
async def predict(image: UploadFile = File(...)) -> Any:  # noqa: ANN401
    if not STATE.ready:
        privacy.safe_request_log(
            success=False, app_version=APP_VERSION,
            bundle_version=STATE.bundle_version,
            error_category="server_not_ready", parity_status=STATE.parity_status,
        )
        return _error(
            "server_not_ready",
            "The demonstrator is not ready (bundle, threshold, or parity gate not "
            "satisfied). Real predictions are disabled.",
            503,
        )

    t_total0 = time.perf_counter()

    t0 = time.perf_counter()
    raw = await image.read()
    decode_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    try:
        rgb, checks = image_checks.check_and_decode(raw)
    except image_checks.ImageCheckError as exc:
        privacy.safe_request_log(
            success=False, app_version=APP_VERSION,
            bundle_version=STATE.bundle_version,
            error_category=exc.category, parity_status=STATE.parity_status,
        )
        return _error(exc.category, exc.message, 400)
    finally:
        raw = b""
    checks_ms = (time.perf_counter() - t0) * 1000.0

    try:
        result = STATE.service.infer(rgb)
    except Exception:  # noqa: BLE001 - never leak a stack trace to the client
        privacy.safe_request_log(
            success=False, app_version=APP_VERSION,
            bundle_version=STATE.bundle_version,
            error_category="inference_error", parity_status=STATE.parity_status,
        )
        return _error("inference_error", "Inference failed for this image.", 500)

    op = STATE.operating_point
    decision = op.decide(result.referable_score)

    total_ms = (time.perf_counter() - t_total0) * 1000.0
    timings = {
        "decode": round(decode_ms, 3),
        "technical_checks": round(checks_ms, 3),
        "preprocessing": round(result.timings_ms.get("preprocessing", 0.0), 3),
        "backbone": round(result.timings_ms.get("backbone", 0.0), 3),
        "head": round(result.timings_ms.get("head", 0.0), 3),
        "postprocessing": round(result.timings_ms.get("postprocessing", 0.0), 3),
        "total": round(total_ms, 3),
    }

    privacy.safe_request_log(
        success=True, app_version=APP_VERSION, bundle_version=STATE.bundle_version,
        total_ms=total_ms, parity_status=STATE.parity_status,
    )

    response = schemas.PredictResponse(
        referable_dr_score=result.referable_score,
        decision=decision,
        threshold=op.threshold,
        operating_point=schemas.OperatingPointInfo(
            derivation_split=op.derivation_split,
            target_sensitivity=op.target_sensitivity,
            validation_sensitivity=op.validation_sensitivity,
            validation_specificity=op.validation_specificity,
            heldout_test_sensitivity=op.heldout_test_sensitivity,
            heldout_test_specificity=op.heldout_test_specificity,
        ),
        pipeline_trace=schemas.PipelineTrace(
            input_resolution=result.input_resolution,
            backbone=BACKBONE_DISPLAY,
            backbone_params=BACKBONE_PARAMS,
            backbone_frozen=True,
            pooling="average",
            embedding_dim=result.embedding_dim,
            head="MultiTaskHead",
            dr_grade_class_probs=result.dr_grade_probs,
            referable_mass_formula=REFERABLE_FORMULA,
        ),
        timings_ms=schemas.Timings(**timings),
        bundle_version=STATE.bundle_version,
        technical_checks={
            "width": checks.width,
            "height": checks.height,
            "byte_size": checks.byte_size,
            "format": checks.image_format,
            "mean_intensity": round(checks.mean_intensity, 2),
            "contrast_std": round(checks.contrast_std, 2),
        },
        warnings=checks.warnings,
    )
    return response


if __name__ == "__main__":  # pragma: no cover - convenience local runner
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
