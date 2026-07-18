from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, field_validator




class EyeLaterality(str, Enum):
    """Which eye the image was taken from."""

    LEFT = "left"
    RIGHT = "right"
    UNKNOWN = "unknown"


class Sex(str, Enum):
    """Biological sex as recorded in the dataset."""

    MALE = "male"
    FEMALE = "female"
    UNKNOWN = "unknown"


class DeviceClass(str, Enum):
    """Broad class of imaging device."""

    CLINICAL = "clinical"
    SMARTPHONE = "smartphone"
    UNKNOWN = "unknown"


class MappingConfidence(str, Enum):
    """Confidence of a label-mapping from native to canonical grading scheme."""

    EXACT = "exact"
    APPROXIMATE = "approximate"
    UNKNOWN = "unknown"


class ImageQualityLabel(str, Enum):
    """Categorical image-quality assessment."""

    GOOD = "good"
    USABLE = "usable"
    REJECT = "reject"
    UNKNOWN = "unknown"



DR_GRADE_MIN: int = 0
DR_GRADE_MAX: int = 4



CANONICAL_IDENTIFIER_FIELDS: frozenset[str] = frozenset(
    {"sample_id", "patient_id", "dataset_source", "image_path"}
)

CANONICAL_METADATA_FIELDS: frozenset[str] = frozenset(
    {
        "eye_laterality",
        "age_years",
        "sex",
        "ethnicity",
        "camera_type",
        "device_class",
        "hospital_site",
        "education_level",
        "insurance_status",
        "image_quality_score",
        "image_quality_label",
        "dataset_source",
    }
)

CANONICAL_LABEL_FIELDS: frozenset[str] = frozenset(
    {
        "dr_grade",
        "dr_grade_source_scheme",
        "dr_grade_mapping_confidence",
        "glaucoma",
        "cataract",
        "amd",
        "macular_edema",
        "pathological_myopia",
        "hypertensive_retinopathy",
        "drusen",
        "other_ocular",
        "diabetes",
        "hypertension",
        "smoking",
        "obesity",
        "insulin_use",
        "cardiovascular_composite",
        "retinal_age",
        "diabetes_duration_years",
    }
)




class CanonicalSample(BaseModel):
    """Canonical representation of one retinal image sample.

    All adapters must produce instances that validate against this model.
    Missing labels are represented as None (never as 0).
    """

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    patient_id: str
    dataset_source: str
    image_path: str

    eye_laterality: Optional[EyeLaterality] = None

    age_years: Optional[float] = None
    sex: Optional[Sex] = None
    ethnicity: Optional[str] = None

    camera_type: Optional[str] = None
    device_class: Optional[DeviceClass] = None
    hospital_site: Optional[str] = None

    education_level: Optional[str] = None
    insurance_status: Optional[str] = None

    image_quality_score: Optional[float] = None
    image_quality_label: Optional[ImageQualityLabel] = None

    dr_grade: Optional[int] = None
    dr_grade_source_scheme: Optional[str] = None
    dr_grade_mapping_confidence: Optional[MappingConfidence] = None

    glaucoma: Optional[int] = None
    cataract: Optional[int] = None
    amd: Optional[int] = None
    macular_edema: Optional[int] = None
    pathological_myopia: Optional[int] = None
    hypertensive_retinopathy: Optional[int] = None
    drusen: Optional[int] = None
    other_ocular: Optional[int] = None

    diabetes: Optional[int] = None
    hypertension: Optional[int] = None
    smoking: Optional[int] = None
    obesity: Optional[int] = None
    insulin_use: Optional[int] = None

    cardiovascular_composite: Optional[float] = None
    retinal_age: Optional[float] = None
    diabetes_duration_years: Optional[float] = None


    @field_validator("sample_id", "patient_id", "dataset_source", "image_path")
    @classmethod
    def _validate_required_text(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("required identifier fields must be non-empty strings")
        return v

    @field_validator("dr_grade")
    @classmethod
    def _validate_dr_grade(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and not (DR_GRADE_MIN <= v <= DR_GRADE_MAX):
            raise ValueError(
                f"dr_grade must be in [{DR_GRADE_MIN}, {DR_GRADE_MAX}], got {v}"
            )
        return v

    @field_validator(
        "glaucoma",
        "cataract",
        "amd",
        "macular_edema",
        "pathological_myopia",
        "hypertensive_retinopathy",
        "drusen",
        "other_ocular",
        "diabetes",
        "hypertension",
        "smoking",
        "obesity",
        "insulin_use",
    )
    @classmethod
    def _validate_binary_label(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v not in (0, 1):
            raise ValueError(f"Binary label must be 0 or 1, got {v}")
        return v

    @field_validator("cardiovascular_composite")
    @classmethod
    def _validate_cardiovascular_composite(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError(
                f"cardiovascular_composite must be in [0.0, 1.0], got {v}"
            )
        return v




def validate_sample(data: dict) -> CanonicalSample:
    """Construct and validate a CanonicalSample from a raw dict.

    Adapters should call this to confirm their output satisfies the schema.
    Raises pydantic.ValidationError on failure.
    """
    return CanonicalSample(**data)
