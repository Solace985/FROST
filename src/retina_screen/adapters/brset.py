"""
adapters/brset.py -- BRSET dataset adapter.

BRSET-specific native vocabulary is confined to this file and BRSET configs/tests.
Downstream pipeline code consumes only CanonicalSample fields and task registry names.

Stage 8C scope
--------------
BRSET (Brazilian Multilabel Ophthalmological Dataset, PhysioNet v1.0.1) is the
primary scientific dataset. This adapter implements the full canonical mapping
for Stage 8C.

Label policy
-----------
- DR_SDRG -> canonical dr_grade (0-4). DR_ICDR retained in audit metadata only.
  This is a project canonicalization decision for reproducibility (Stage 8C locked).
- macular_edema: direct binary ophthalmologist-labeled retinal finding (HIGH quality).
  Not derived from dr_grade. Not confused with diabetic_retinopathy severity.
- hypertensive_retinopathy: direct fundoscopic retinal finding, NOT systemic hypertension.
- diabetes: clinical/medical-record label (PROXY). Do not overclaim.
- increased_cup_disc is NOT glaucoma. Glaucoma is unsupported for BRSET.
- pathological_myopia: DEFERRED. myopic_fundus is an anatomical proxy (indirect).
  Per-dataset label-quality override is not yet enforced downstream. Set to None.
- comorbidities: free text; dropped entirely and never exposed.
- image_id and patient_id are pseudonymised before CanonicalSample creation.
  Canonical patient_id = brset_pNNNNNN and sample_id = brset_sNNNNNN based on
  deterministic sorted-index mapping. Raw native IDs are never embedded in
  canonical IDs, split files, cache manifests, or cache filenames.
- NaN values map to None, never to 0.
- Sex encoding: 1=Male, 2=Female (BRSET PhysioNet v1.0.1 documentation confirmed).
- Laterality: exam_eye 1=right, 2=left (BRSET PhysioNet v1.0.1 documentation confirmed).
- Unknown camera values map to DeviceClass.UNKNOWN (fail-closed), not CLINICAL.
"""

from __future__ import annotations

import logging
import os
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from retina_screen.adapters.base import DatasetAdapter
from retina_screen.core import project_root
from retina_screen.schema import (
    CanonicalSample,
    DeviceClass,
    DR_GRADE_MAX,
    DR_GRADE_MIN,
    EyeLaterality,
    ImageQualityLabel,
    MappingConfidence,
    Sex,
)
from retina_screen.tasks import TASK_REGISTRY, TaskType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Native BRSET column names (private to this adapter boundary)
# ---------------------------------------------------------------------------

_COL_IMAGE_ID = "image_id"
_COL_PATIENT_ID = "patient_id"
_COL_CAMERA = "camera"
_COL_AGE = "patient_age"
_COL_SEX = "patient_sex"
_COL_EYE = "exam_eye"
_COL_QUALITY = "quality"
_COL_DR_SDRG = "DR_SDRG"
_COL_DR_ICDR = "DR_ICDR"
_COL_AMD = "amd"
_COL_HTR = "hypertensive_retinopathy"
_COL_DRUSENS = "drusens"
_COL_MYOPIC = "myopic_fundus"
_COL_MACULAR_EDEMA = "macular_edema"
_COL_SCAR = "scar"
_COL_NEVUS = "nevus"
_COL_VASCULAR_OCC = "vascular_occlusion"
_COL_HEMORRHAGE = "hemorrhage"
_COL_RETINAL_DET = "retinal_detachment"
_COL_OTHER = "other"
_COL_DIABETES = "diabetes"
_COL_INCREASED_CUP = "increased_cup_disc"

_REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {
        _COL_IMAGE_ID,
        _COL_PATIENT_ID,
        _COL_CAMERA,
        _COL_AGE,
        _COL_SEX,
        _COL_EYE,
        _COL_QUALITY,
        _COL_DR_SDRG,
        _COL_DR_ICDR,
        _COL_AMD,
        _COL_HTR,
        _COL_DRUSENS,
        _COL_MYOPIC,
        _COL_MACULAR_EDEMA,
        _COL_SCAR,
        _COL_NEVUS,
        _COL_VASCULAR_OCC,
        _COL_HEMORRHAGE,
        _COL_RETINAL_DET,
        _COL_OTHER,
        _COL_DIABETES,
        _COL_INCREASED_CUP,
    }
)

# Components that feed into the other_ocular computed label
_OTHER_OCULAR_COLS: tuple[str, ...] = (
    _COL_SCAR,
    _COL_NEVUS,
    _COL_VASCULAR_OCC,
    _COL_HEMORRHAGE,
    _COL_RETINAL_DET,
    _COL_OTHER,
)

# Known BRSET camera names (lowercase) -> DeviceClass.
# Unrecognised cameras map to DeviceClass.UNKNOWN (fail-closed).
_CAMERA_TO_DEVICE_CLASS: dict[str, DeviceClass] = {
    "canon cr": DeviceClass.CLINICAL,
    "nikon nf5050": DeviceClass.CLINICAL,
}

# Native quality string (lowercase) -> ImageQualityLabel
_QUALITY_MAP: dict[str, ImageQualityLabel] = {
    "adequate": ImageQualityLabel.GOOD,
    "inadequate": ImageQualityLabel.REJECT,
}


# ---------------------------------------------------------------------------
# Private parsing helpers
# ---------------------------------------------------------------------------


def _parse_binary_int(value: Any, col_name: str) -> int | None:
    """Parse a BRSET binary (0/1) column value as 0 / 1 / None.

    Handles integer, float, and missing (NaN / None).
    Out-of-range or non-numeric values are logged and returned as None.
    """
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        logger.warning(
            "BRSET: unexpected value %r in column %r; treating as None", value, col_name
        )
        return None
    if parsed not in (0, 1):
        logger.warning(
            "BRSET: non-binary value %r in column %r; treating as None", parsed, col_name
        )
        return None
    return parsed


def _parse_dr_grade(value: Any, col_name: str) -> int | None:
    """Parse a BRSET DR grade column (0-4 ordinal) as int or None."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        grade = int(value)
    except (TypeError, ValueError):
        logger.warning(
            "BRSET: unexpected value %r in column %r; treating as None", value, col_name
        )
        return None
    if not (DR_GRADE_MIN <= grade <= DR_GRADE_MAX):
        logger.warning(
            "BRSET: out-of-range DR grade %r in column %r; treating as None", grade, col_name
        )
        return None
    return grade


def _parse_diabetes(value: Any) -> int | None:
    """Parse BRSET diabetes column (may be 'yes'/'no' string or 0/1 integer)."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    lowered = str(value).strip().lower()
    if lowered in ("yes", "1", "1.0"):
        return 1
    if lowered in ("no", "0", "0.0"):
        return 0
    logger.warning("BRSET: unrecognised diabetes value %r; treating as None", value)
    return None


def _parse_age(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_sex(value: Any) -> Sex:
    """Map BRSET patient_sex: 1=Male, 2=Female (BRSET PhysioNet v1.0.1 confirmed)."""
    if value is None:
        return Sex.UNKNOWN
    try:
        if pd.isna(value):
            return Sex.UNKNOWN
    except (TypeError, ValueError):
        pass
    try:
        code = int(value)
    except (TypeError, ValueError):
        logger.warning("BRSET: unexpected patient_sex value %r; using UNKNOWN", value)
        return Sex.UNKNOWN
    if code == 1:
        return Sex.MALE
    if code == 2:
        return Sex.FEMALE
    logger.warning("BRSET: out-of-range patient_sex %r; using UNKNOWN", code)
    return Sex.UNKNOWN


def _parse_laterality(value: Any) -> EyeLaterality | None:
    """Map BRSET exam_eye: 1=right, 2=left (BRSET PhysioNet v1.0.1 confirmed)."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        code = int(value)
    except (TypeError, ValueError):
        logger.warning("BRSET: unexpected exam_eye value %r; using None", value)
        return None
    if code == 1:
        return EyeLaterality.RIGHT
    if code == 2:
        return EyeLaterality.LEFT
    logger.warning("BRSET: out-of-range exam_eye %r; using None", code)
    return None


def _parse_quality(value: Any) -> ImageQualityLabel:
    if value is None:
        return ImageQualityLabel.UNKNOWN
    try:
        if pd.isna(value):
            return ImageQualityLabel.UNKNOWN
    except (TypeError, ValueError):
        pass
    return _QUALITY_MAP.get(str(value).strip().lower(), ImageQualityLabel.UNKNOWN)


def _parse_camera(value: Any) -> tuple[str | None, DeviceClass]:
    """Parse BRSET camera column.

    Unknown or unrecognised camera names map to DeviceClass.UNKNOWN (fail-closed).
    Only explicitly listed camera models map to DeviceClass.CLINICAL.
    """
    if value is None:
        return None, DeviceClass.UNKNOWN
    try:
        if pd.isna(value):
            return None, DeviceClass.UNKNOWN
    except (TypeError, ValueError):
        pass
    raw = str(value).strip()
    device_class = _CAMERA_TO_DEVICE_CLASS.get(raw.lower(), DeviceClass.UNKNOWN)
    return raw, device_class


def _compute_other_ocular(row: pd.Series) -> int | None:
    """Return 1 if any other_ocular component is positive, 0 if all observed negative,
    or None if no component is positive and at least one is missing/invalid.

    Returning None when components are missing prevents silently encoding
    unobserved absence as a confirmed negative label.
    """
    any_missing = False
    for col in _OTHER_OCULAR_COLS:
        val = _parse_binary_int(row.get(col), col)
        if val == 1:
            return 1  # short-circuit: confirmed positive
        if val is None:
            any_missing = True
    if any_missing:
        return None  # missing data: cannot confirm negative
    return 0  # all components are confirmed observed negatives


# ---------------------------------------------------------------------------
# BRSETAdapter
# ---------------------------------------------------------------------------


class BRSETAdapter(DatasetAdapter):
    """Adapter for BRSET (PhysioNet v1.0.1) retinal fundus dataset.

    Reads labels_brset.csv and fundus_photos/ to produce canonical samples.
    All native BRSET column names are private to this class.

    Pseudonymisation:
        Native patient_id and image_id values are never embedded in canonical
        sample_id or patient_id. Canonical IDs use deterministic index-based
        pseudonyms (brset_pNNNNNN / brset_sNNNNNN) derived from sorted native
        IDs. Patient grouping is preserved: both eyes of the same native patient
        receive the same canonical patient_id.
    """

    DATASET_SOURCE: str = "brset"

    def __init__(
        self,
        dataset_root: str | Path = "data/brset",
        metadata_file: str = "labels_brset.csv",
        images_dir: str = "fundus_photos",
    ) -> None:
        env_root = os.environ.get("RETINA_SCREEN_BRSET_ROOT")
        resolved_root = Path(env_root) if env_root else Path(dataset_root)
        if not resolved_root.is_absolute():
            resolved_root = project_root() / resolved_root

        self._root = resolved_root
        self._metadata_path = resolved_root / metadata_file
        self._images_dir = resolved_root / images_dir

        if not self._root.exists():
            raise FileNotFoundError(
                f"BRSET dataset root not found: {self._root}. "
                "Set RETINA_SCREEN_BRSET_ROOT or pass dataset_root."
            )
        if not self._metadata_path.exists():
            raise FileNotFoundError(
                f"BRSET metadata file not found: {self._metadata_path}"
            )
        if not self._images_dir.exists():
            raise FileNotFoundError(
                f"BRSET image directory not found: {self._images_dir}"
            )

        self._df = pd.read_csv(
            self._metadata_path,
            dtype={_COL_IMAGE_ID: str, _COL_PATIENT_ID: str},
        )

        missing_cols = sorted(_REQUIRED_COLUMNS - set(self._df.columns))
        if missing_cols:
            raise ValueError(f"BRSET metadata missing required columns: {missing_cols}")

        # Audit state (populated during manifest build)
        self._excluded_by_reason: Counter[str] = Counter()
        self._multi_row_patient_count: int = 0
        self._dr_sdrg_dist: Counter[int] = Counter()
        self._dr_icdr_dist: Counter[int] = Counter()
        self._dr_disagreement_count: int = 0

        self._supported_tasks: list[str] = self._resolve_supported_tasks()
        self._manifest: list[CanonicalSample] = self._build_manifest_internal()
        self._sample_lookup: dict[str, CanonicalSample] = {
            s.sample_id: s for s in self._manifest
        }

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def build_manifest(self) -> list[CanonicalSample]:
        return list(self._manifest)

    def load_sample(self, sample_id: str) -> CanonicalSample:
        try:
            return self._sample_lookup[sample_id]
        except KeyError:
            raise KeyError(f"BRSET sample_id not found: {sample_id!r}") from None

    def load_image(self, sample_id: str) -> Image.Image:
        sample = self.load_sample(sample_id)
        image_path = Path(sample.image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"BRSET image not found: {image_path}")
        return Image.open(image_path).convert("RGB")

    def get_supported_tasks(self) -> list[str]:
        return list(self._supported_tasks)

    def get_stratification_columns(self) -> list[str]:
        return ["sex", "age_years", "eye_laterality", "device_class"]

    def get_quality_columns(self) -> list[str]:
        return ["image_quality_label"]

    def get_dataset_audit(self) -> dict[str, Any]:
        ages = [s.age_years for s in self._manifest if s.age_years is not None]
        sex_counts: Counter[str] = Counter(
            s.sex.value if s.sex is not None else "missing"
            for s in self._manifest
        )
        quality_counts: Counter[str] = Counter(
            s.image_quality_label.value if s.image_quality_label is not None else "missing"
            for s in self._manifest
        )

        # Task-type-aware label coverage
        label_coverage: dict[str, dict] = {}
        for task in self._supported_tasks:
            task_def = TASK_REGISTRY[task]
            target_col = task_def.target_column
            if task_def.task_type == TaskType.BINARY:
                positives = sum(
                    1 for s in self._manifest if getattr(s, target_col, None) == 1
                )
                negatives = sum(
                    1 for s in self._manifest if getattr(s, target_col, None) == 0
                )
                missing = sum(
                    1 for s in self._manifest if getattr(s, target_col, None) is None
                )
                label_coverage[task] = {
                    "positives": positives,
                    "negatives": negatives,
                    "missing": missing,
                    "total": len(self._manifest),
                }
            elif task_def.task_type == TaskType.ORDINAL:
                class_dist: dict[str, int] = {}
                missing_ord = 0
                for s in self._manifest:
                    val = getattr(s, target_col, None)
                    if val is None:
                        missing_ord += 1
                    else:
                        class_dist[str(val)] = class_dist.get(str(val), 0) + 1
                observed = sum(class_dist.values())
                label_coverage[task] = {
                    "class_distribution": dict(
                        sorted(class_dist.items(), key=lambda kv: int(kv[0]))
                    ),
                    "observed": observed,
                    "missing": missing_ord,
                    "total": len(self._manifest),
                }
            elif task_def.task_type == TaskType.REGRESSION:
                vals = [
                    getattr(s, target_col)
                    for s in self._manifest
                    if getattr(s, target_col, None) is not None
                ]
                missing_reg = len(self._manifest) - len(vals)
                label_coverage[task] = {
                    "observed": len(vals),
                    "missing": missing_reg,
                    "total": len(self._manifest),
                }
            else:
                label_coverage[task] = {"total": len(self._manifest)}

        patient_image_counts: Counter[str] = Counter(
            s.patient_id for s in self._manifest
        )
        total_rows = len(self._df)
        disagree_pct = (
            round(self._dr_disagreement_count / total_rows * 100, 4)
            if total_rows > 0
            else 0.0
        )

        return {
            "dataset_root_used": str(self._root),
            "metadata_path_used": str(self._metadata_path),
            "image_dir_used": str(self._images_dir),
            "total_csv_rows": total_rows,
            "valid_image_samples": len(self._manifest),
            "excluded_samples_by_reason": dict(self._excluded_by_reason),
            "unique_patients": len(patient_image_counts),
            "bilateral_patients": sum(
                1 for c in patient_image_counts.values() if c == 2
            ),
            "unilateral_patients": sum(
                1 for c in patient_image_counts.values() if c == 1
            ),
            "multi_row_patients_count": self._multi_row_patient_count,
            "subgroup_coverage": {
                "sex_counts": dict(sex_counts),
                "age": {
                    "available": len(ages),
                    "missing": len(self._manifest) - len(ages),
                    "min": min(ages) if ages else None,
                    "max": max(ages) if ages else None,
                    "mean": round(sum(ages) / len(ages), 2) if ages else None,
                },
                "image_quality": dict(quality_counts),
            },
            "label_coverage": label_coverage,
            "dr_grading": {
                "canonical_source": "DR_SDRG",
                "audit_retained": "DR_ICDR",
                "dr_sdrg_distribution": dict(sorted(self._dr_sdrg_dist.items())),
                "dr_icdr_distribution": dict(sorted(self._dr_icdr_dist.items())),
                "dr_sdrg_vs_icdr_disagreement_count": self._dr_disagreement_count,
                "dr_sdrg_vs_icdr_disagreement_pct": disagree_pct,
                "canonicalization_note": (
                    "DR_SDRG is used as the canonical dr_grade source per Stage 8C "
                    "locked decision. This is a project canonicalization decision for "
                    "reproducibility. DR_ICDR is retained here for audit and potential "
                    "later ablation. Not a claim of medical superiority."
                ),
            },
            "label_quality_notes": {
                "macular_edema": (
                    "HIGH. Direct binary ophthalmologist-labeled retinal finding."
                ),
                "hypertensive_retinopathy": (
                    "HIGH. Direct fundoscopic retinal finding. "
                    "NOT systemic hypertension."
                ),
                "diabetes": "PROXY. Clinical/medical-record label. Do not overclaim.",
                "other_ocular": (
                    "retinal_detachment component has ~7 positives; "
                    "subgroup evaluation will trigger NA safeguard."
                ),
                "glaucoma": (
                    "UNSUPPORTED for BRSET. increased_cup_disc is an anatomical "
                    "marker, not a confirmed glaucoma diagnosis."
                ),
                "pathological_myopia": (
                    "DEFERRED for Stage 8C. myopic_fundus is a fundoscopic/anatomical "
                    "proxy (indirect). Per-dataset label-quality override is not yet "
                    "enforced downstream. Will reconsider in Stage 8D+."
                ),
            },
            "unsupported_tasks": {
                "glaucoma": (
                    "increased_cup_disc is an anatomical marker, not confirmed glaucoma."
                ),
                "cataract": "No cataract column in BRSET metadata.",
                "hypertension": (
                    "comorbidities is free text, 50% missing; "
                    "not parsed in Stage 8C."
                ),
                "insulin_use": "insuline column is 89% missing; deferred to Stage 8D+.",
                "diabetes_duration_years": (
                    "diabetes_time_y is 88% missing; deferred to Stage 8D+."
                ),
                "pathological_myopia": (
                    "DEFERRED. myopic_fundus is an anatomical fundoscopic proxy. "
                    "Per-dataset label-quality override not yet enforced downstream."
                ),
            },
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_id_mappings(
        df: pd.DataFrame,
    ) -> tuple[dict[str, str], dict[str, str]]:
        """Build deterministic pseudonymous ID mappings.

        Native patient_ids and image_ids are sorted, then assigned sequential
        zero-padded canonical IDs. Raw native IDs are not present in output.

        Returns
        -------
        patient_map
            native patient_id string -> canonical patient_id (brset_pNNNNNN)
        sample_map
            native image_id string -> canonical sample_id (brset_sNNNNNN)
        """
        raw_pids = df[_COL_PATIENT_ID].astype(str).unique().tolist()
        try:
            sorted_pids = sorted(raw_pids, key=int)
        except ValueError:
            sorted_pids = sorted(raw_pids)
        patient_map = {pid: f"brset_p{i + 1:06d}" for i, pid in enumerate(sorted_pids)}

        raw_iids = df[_COL_IMAGE_ID].astype(str).unique().tolist()
        sorted_iids = sorted(raw_iids)
        sample_map = {iid: f"brset_s{i + 1:06d}" for i, iid in enumerate(sorted_iids)}

        return patient_map, sample_map

    def _resolve_supported_tasks(self) -> list[str]:
        required = [
            "dr_grade",
            "macular_edema",
            "hypertensive_retinopathy",
            "amd",
            "drusen",
            "other_ocular",
            "diabetes",
            # pathological_myopia deferred: myopic_fundus is anatomical proxy;
            # per-dataset label-quality override not yet enforced downstream.
        ]
        tasks: list[str] = []
        for name in required:
            if name not in TASK_REGISTRY:
                raise ValueError(
                    f"BRSET required task {name!r} is not in TASK_REGISTRY. "
                    "Register it in tasks.py first."
                )
            tasks.append(name)
        return tasks

    def _build_manifest_internal(self) -> list[CanonicalSample]:
        patient_map, sample_map = self._build_id_mappings(self._df)

        patient_row_counts: Counter[str] = Counter(self._df[_COL_PATIENT_ID].astype(str))
        self._multi_row_patient_count = sum(
            1 for c in patient_row_counts.values() if c >= 3
        )
        if self._multi_row_patient_count > 0:
            logger.info(
                "BRSET: %d patients have 3+ rows (repeated exams or quality rescans). "
                "All rows included; patient-level split groups them correctly.",
                self._multi_row_patient_count,
            )

        samples: list[CanonicalSample] = []

        for _, row in self._df.iterrows():
            image_id = str(row[_COL_IMAGE_ID]).strip()
            patient_id_raw = str(row[_COL_PATIENT_ID]).strip()
            sample_id = sample_map[image_id]
            patient_id = patient_map[patient_id_raw]

            image_path = self._images_dir / f"{image_id}.jpg"
            if not image_path.exists():
                self._excluded_by_reason["missing_image_file"] += 1
                logger.debug(
                    "BRSET: image not found for sample_id %r; excluded.", sample_id
                )
                continue

            dr_sdrg = _parse_dr_grade(row.get(_COL_DR_SDRG), _COL_DR_SDRG)
            dr_icdr = _parse_dr_grade(row.get(_COL_DR_ICDR), _COL_DR_ICDR)

            # Track DR distributions for audit metadata
            if dr_sdrg is not None:
                self._dr_sdrg_dist[dr_sdrg] += 1
            if dr_icdr is not None:
                self._dr_icdr_dist[dr_icdr] += 1
            if dr_sdrg is not None and dr_icdr is not None and dr_sdrg != dr_icdr:
                self._dr_disagreement_count += 1

            camera_type, device_class = _parse_camera(row.get(_COL_CAMERA))

            sample = CanonicalSample(
                sample_id=sample_id,
                patient_id=patient_id,
                dataset_source=self.DATASET_SOURCE,
                image_path=str(image_path),
                eye_laterality=_parse_laterality(row.get(_COL_EYE)),
                age_years=_parse_age(row.get(_COL_AGE)),
                sex=_parse_sex(row.get(_COL_SEX)),
                camera_type=camera_type,
                device_class=device_class,
                image_quality_label=_parse_quality(row.get(_COL_QUALITY)),
                dr_grade=dr_sdrg,
                dr_grade_source_scheme="DR_SDRG",
                dr_grade_mapping_confidence=MappingConfidence.EXACT,
                amd=_parse_binary_int(row.get(_COL_AMD), _COL_AMD),
                macular_edema=_parse_binary_int(
                    row.get(_COL_MACULAR_EDEMA), _COL_MACULAR_EDEMA
                ),
                hypertensive_retinopathy=_parse_binary_int(
                    row.get(_COL_HTR), _COL_HTR
                ),
                drusen=_parse_binary_int(row.get(_COL_DRUSENS), _COL_DRUSENS),
                # pathological_myopia deferred: myopic_fundus is anatomical proxy.
                pathological_myopia=None,
                other_ocular=_compute_other_ocular(row),
                diabetes=_parse_diabetes(row.get(_COL_DIABETES)),
                # Unsupported for BRSET — always None
                glaucoma=None,
                cataract=None,
                hypertension=None,
                smoking=None,
                obesity=None,
                insulin_use=None,
            )
            samples.append(sample)

        logger.info(
            "BRSET manifest built: %d valid samples from %d CSV rows (%d excluded).",
            len(samples),
            len(self._df),
            sum(self._excluded_by_reason.values()),
        )
        return samples
