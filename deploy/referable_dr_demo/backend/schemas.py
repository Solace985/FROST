"""schemas.py -- FastAPI response models for FROST.

Pydantic models for /health and /predict. Imported only when the web app runs
(FastAPI/pydantic are app-local dependencies). The schemas deliberately exclude
raw embeddings, the five-class grade label, and any systemic / other-ocular task
output: the demonstrator surfaces only the referable-DR triage result.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., description='"ready" or "blocked"')
    bundle_version: str
    model_loaded: bool
    parity_status: str = Field(..., description='"pass" | "fail" | "unavailable"')
    threshold_status: str = Field(..., description='"validated" | "blocked"')
    network_access_required: bool = False


class OperatingPointInfo(BaseModel):
    derivation_split: str
    target_sensitivity: float
    validation_sensitivity: float
    validation_specificity: float
    heldout_test_sensitivity: float
    heldout_test_specificity: float


class PipelineTrace(BaseModel):
    input_resolution: str
    backbone: str
    backbone_params: str
    backbone_frozen: bool
    pooling: str
    embedding_dim: int
    head: str
    dr_grade_class_probs: list[float]
    referable_mass_formula: str


class Timings(BaseModel):
    decode: float
    technical_checks: float
    preprocessing: float
    backbone: float
    head: float
    postprocessing: float
    total: float


class PredictResponse(BaseModel):
    referable_dr_score: float
    decision: str = Field(..., description='"REFERABLE" or "NOT REFERABLE"')
    threshold: float
    operating_point: OperatingPointInfo
    pipeline_trace: PipelineTrace
    timings_ms: Timings
    bundle_version: str
    technical_checks: dict
    warnings: list[str] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    error: str
    category: str
