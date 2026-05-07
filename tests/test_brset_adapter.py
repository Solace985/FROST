"""
Synthetic-fixture tests for BRSETAdapter.

All tests use tmp_path with minimal CSV rows and fake .jpg files.
No real BRSET data is required.  Real-data tests are guarded with
pytest.mark.skipif and only run when RETINA_SCREEN_BRSET_ROOT is set.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from retina_screen.adapters.brset import BRSETAdapter
from retina_screen.schema import (
    CanonicalSample,
    DeviceClass,
    EyeLaterality,
    ImageQualityLabel,
    Sex,
)
from retina_screen.tasks import TASK_REGISTRY

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

_BRSET_REAL_ROOT_AVAILABLE = bool(os.environ.get("RETINA_SCREEN_BRSET_ROOT")) or Path(
    "data/brset/labels_brset.csv"
).exists()


def _make_brset_fixture(
    tmp_path: Path,
    rows: list[dict],
    *,
    create_images: bool = True,
) -> BRSETAdapter:
    """Create a minimal BRSET fixture directory and return an adapter."""
    images_dir = tmp_path / "fundus_photos"
    images_dir.mkdir()

    if create_images:
        for row in rows:
            img_file = images_dir / f"{row['image_id']}.jpg"
            Image.new("RGB", (32, 32), color=(128, 128, 128)).save(img_file)

    csv_path = tmp_path / "labels_brset.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    return BRSETAdapter(
        dataset_root=tmp_path,
        metadata_file="labels_brset.csv",
        images_dir="fundus_photos",
    )


def _default_row(**overrides) -> dict:
    """Return a minimal valid BRSET CSV row."""
    base = {
        "image_id": "00001",
        "patient_id": "1001",
        "camera": "Canon CR",
        "patient_age": 55.0,
        "patient_sex": 1,
        "exam_eye": 1,
        "quality": "Adequate",
        "DR_SDRG": 0,
        "DR_ICDR": 0,
        "amd": 0,
        "hypertensive_retinopathy": 0,
        "drusens": 0,
        "myopic_fundus": 0,
        "macular_edema": 0,
        "scar": 0,
        "nevus": 0,
        "vascular_occlusion": 0,
        "hemorrhage": 0,
        "retinal_detachment": 0,
        "other": 0,
        "diabetes": "no",
        "increased_cup_disc": 0,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Tests — manifest and basic structure
# ---------------------------------------------------------------------------


def test_brset_adapter_synthetic_fixture_manifest(tmp_path: Path) -> None:
    rows = [_default_row(image_id="00001", patient_id="1001")]
    adapter = _make_brset_fixture(tmp_path, rows)
    manifest = adapter.build_manifest()
    assert len(manifest) == 1
    sample = manifest[0]
    assert isinstance(sample, CanonicalSample)
    assert sample.dataset_source == "brset"


def test_brset_adapter_sample_id_format(tmp_path: Path) -> None:
    rows = [_default_row(image_id="00042", patient_id="2000")]
    adapter = _make_brset_fixture(tmp_path, rows)
    sample = adapter.build_manifest()[0]
    assert re.match(r"brset_s\d{6}$", sample.sample_id), (
        f"sample_id must be brset_sNNNNNN format, got {sample.sample_id!r}"
    )


def test_brset_adapter_patient_id_format(tmp_path: Path) -> None:
    rows = [_default_row(image_id="00001", patient_id="9999")]
    adapter = _make_brset_fixture(tmp_path, rows)
    sample = adapter.build_manifest()[0]
    assert re.match(r"brset_p\d{6}$", sample.patient_id), (
        f"patient_id must be brset_pNNNNNN format, got {sample.patient_id!r}"
    )


def test_brset_adapter_sample_id_not_raw_derived(tmp_path: Path) -> None:
    """Native image_id must NOT appear in canonical sample_id."""
    native_image_id = "raw_img_55555"
    images_dir = tmp_path / "fundus_photos"
    images_dir.mkdir()
    Image.new("RGB", (32, 32)).save(images_dir / f"{native_image_id}.jpg")
    row = _default_row(image_id=native_image_id, patient_id="raw_pid_77777")
    csv_path = tmp_path / "labels_brset.csv"
    pd.DataFrame([row]).to_csv(csv_path, index=False)
    adapter = BRSETAdapter(tmp_path, "labels_brset.csv", "fundus_photos")
    sample = adapter.build_manifest()[0]
    assert "raw_img_55555" not in sample.sample_id, (
        "Native image_id must not appear in canonical sample_id."
    )
    assert "raw_pid_77777" not in sample.sample_id, (
        "Native patient_id must not appear in canonical sample_id."
    )


def test_brset_adapter_patient_id_not_raw_derived(tmp_path: Path) -> None:
    """Native patient_id must NOT appear in canonical patient_id."""
    native_image_id = "raw_img_66666"
    images_dir = tmp_path / "fundus_photos"
    images_dir.mkdir()
    Image.new("RGB", (32, 32)).save(images_dir / f"{native_image_id}.jpg")
    row = _default_row(image_id=native_image_id, patient_id="raw_pid_88888")
    csv_path = tmp_path / "labels_brset.csv"
    pd.DataFrame([row]).to_csv(csv_path, index=False)
    adapter = BRSETAdapter(tmp_path, "labels_brset.csv", "fundus_photos")
    sample = adapter.build_manifest()[0]
    assert "raw_pid_88888" not in sample.patient_id, (
        "Native patient_id must not appear in canonical patient_id."
    )
    assert "raw_img_66666" not in sample.patient_id, (
        "Native image_id must not appear in canonical patient_id."
    )


def test_brset_adapter_bilateral_same_patient_id(tmp_path: Path) -> None:
    rows = [
        _default_row(image_id="00001", patient_id="2001", exam_eye=1),
        _default_row(image_id="00002", patient_id="2001", exam_eye=2),
    ]
    adapter = _make_brset_fixture(tmp_path, rows)
    manifest = adapter.build_manifest()
    assert len(manifest) == 2
    patient_ids = {s.patient_id for s in manifest}
    assert len(patient_ids) == 1, "Both eyes must share the same canonical patient_id."
    canonical_pid = list(patient_ids)[0]
    assert re.match(r"brset_p\d{6}$", canonical_pid), (
        f"Bilateral patient_id must be brset_pNNNNNN, got {canonical_pid!r}"
    )


def test_brset_adapter_missing_age_produces_none(tmp_path: Path) -> None:
    rows = [_default_row(image_id="00001", patient_id="3001", patient_age=float("nan"))]
    adapter = _make_brset_fixture(tmp_path, rows)
    sample = adapter.build_manifest()[0]
    assert sample.age_years is None, "NaN age must map to None, not 0."


# ---------------------------------------------------------------------------
# Tests — field mappings
# ---------------------------------------------------------------------------


def test_brset_adapter_sex_encoding(tmp_path: Path) -> None:
    rows = [
        _default_row(image_id="00001", patient_id="4001", patient_sex=1),
        _default_row(image_id="00002", patient_id="4002", patient_sex=2),
    ]
    adapter = _make_brset_fixture(tmp_path, rows)
    manifest = adapter.build_manifest()
    # image "00001" sorts before "00002"; patient "4001" sorts before "4002"
    sex_map = {s.sample_id: s.sex for s in manifest}
    assert sex_map["brset_s000001"] == Sex.MALE
    assert sex_map["brset_s000002"] == Sex.FEMALE


def test_brset_adapter_laterality_encoding(tmp_path: Path) -> None:
    rows = [
        _default_row(image_id="00001", patient_id="5001", exam_eye=1),
        _default_row(image_id="00002", patient_id="5001", exam_eye=2),
    ]
    adapter = _make_brset_fixture(tmp_path, rows)
    manifest = adapter.build_manifest()
    lat_map = {s.sample_id: s.eye_laterality for s in manifest}
    assert lat_map["brset_s000001"] == EyeLaterality.RIGHT
    assert lat_map["brset_s000002"] == EyeLaterality.LEFT


def test_brset_adapter_image_quality_label_mapping(tmp_path: Path) -> None:
    rows = [
        _default_row(image_id="00001", patient_id="6001", quality="Adequate"),
        _default_row(image_id="00002", patient_id="6002", quality="Inadequate"),
    ]
    adapter = _make_brset_fixture(tmp_path, rows)
    manifest = adapter.build_manifest()
    quality_map = {s.sample_id: s.image_quality_label for s in manifest}
    assert quality_map["brset_s000001"] == ImageQualityLabel.GOOD
    assert quality_map["brset_s000002"] == ImageQualityLabel.REJECT


# ---------------------------------------------------------------------------
# Tests — other_ocular computed label (Patch 4: missingness safety)
# ---------------------------------------------------------------------------


def test_brset_adapter_other_ocular_computed_correctly(tmp_path: Path) -> None:
    rows = [
        _default_row(image_id="00001", patient_id="7001", scar=1, other=0),
        _default_row(image_id="00002", patient_id="7002", scar=0, other=0, nevus=0,
                     vascular_occlusion=0, hemorrhage=0, retinal_detachment=0),
    ]
    adapter = _make_brset_fixture(tmp_path, rows)
    manifest = adapter.build_manifest()
    oo_map = {s.sample_id: s.other_ocular for s in manifest}
    assert oo_map["brset_s000001"] == 1, "scar=1 should yield other_ocular=1."
    assert oo_map["brset_s000002"] == 0, "All zeros should yield other_ocular=0."


def test_brset_adapter_other_ocular_all_nan_returns_none(tmp_path: Path) -> None:
    """All component columns NaN: cannot confirm negative => other_ocular=None."""
    rows = [_default_row(
        image_id="00001", patient_id="7003",
        scar=float("nan"), nevus=float("nan"),
        vascular_occlusion=float("nan"), hemorrhage=float("nan"),
        retinal_detachment=float("nan"), other=float("nan"),
    )]
    adapter = _make_brset_fixture(tmp_path, rows)
    sample = adapter.build_manifest()[0]
    assert sample.other_ocular is None, (
        "All-NaN other_ocular components must yield None, not 0."
    )


def test_brset_adapter_other_ocular_all_zeros_returns_zero(tmp_path: Path) -> None:
    """All components are confirmed 0: other_ocular=0 (observed negative)."""
    rows = [_default_row(
        image_id="00001", patient_id="7004",
        scar=0, nevus=0, vascular_occlusion=0, hemorrhage=0,
        retinal_detachment=0, other=0,
    )]
    adapter = _make_brset_fixture(tmp_path, rows)
    sample = adapter.build_manifest()[0]
    assert sample.other_ocular == 0, (
        "All-zero other_ocular components must yield 0."
    )


def test_brset_adapter_other_ocular_one_positive_rest_missing(tmp_path: Path) -> None:
    """One positive component short-circuits to 1 regardless of other missing values."""
    rows = [_default_row(
        image_id="00001", patient_id="7005",
        scar=float("nan"), hemorrhage=1,
        nevus=float("nan"), vascular_occlusion=float("nan"),
        retinal_detachment=float("nan"), other=float("nan"),
    )]
    adapter = _make_brset_fixture(tmp_path, rows)
    sample = adapter.build_manifest()[0]
    assert sample.other_ocular == 1, (
        "One positive component must yield other_ocular=1 even if others are missing."
    )


# ---------------------------------------------------------------------------
# Tests — macular_edema
# ---------------------------------------------------------------------------


def test_brset_adapter_macular_edema_direct_label(tmp_path: Path) -> None:
    rows = [
        _default_row(image_id="00001", patient_id="8001", macular_edema=1),
        _default_row(image_id="00002", patient_id="8002", macular_edema=0),
    ]
    adapter = _make_brset_fixture(tmp_path, rows)
    manifest = adapter.build_manifest()
    me_map = {s.sample_id: s.macular_edema for s in manifest}
    assert me_map["brset_s000001"] == 1
    assert me_map["brset_s000002"] == 0


def test_brset_adapter_macular_edema_nan_produces_none(tmp_path: Path) -> None:
    rows = [_default_row(image_id="00001", patient_id="8003", macular_edema=float("nan"))]
    adapter = _make_brset_fixture(tmp_path, rows)
    sample = adapter.build_manifest()[0]
    assert sample.macular_edema is None, "NaN macular_edema must map to None, not 0."


# ---------------------------------------------------------------------------
# Tests — unsupported labels
# ---------------------------------------------------------------------------


def test_brset_adapter_cataract_is_none(tmp_path: Path) -> None:
    rows = [_default_row(image_id="00001", patient_id="9001")]
    adapter = _make_brset_fixture(tmp_path, rows)
    sample = adapter.build_manifest()[0]
    assert sample.cataract is None, "cataract must be None for all BRSET samples."


def test_brset_adapter_glaucoma_is_none(tmp_path: Path) -> None:
    rows = [_default_row(image_id="00001", patient_id="9002", increased_cup_disc=1)]
    adapter = _make_brset_fixture(tmp_path, rows)
    sample = adapter.build_manifest()[0]
    assert sample.glaucoma is None, (
        "glaucoma must be None for BRSET. "
        "increased_cup_disc is an anatomical marker, not mapped to glaucoma."
    )


def test_brset_adapter_pathological_myopia_deferred(tmp_path: Path) -> None:
    """pathological_myopia is deferred; must be None and not in supported_tasks."""
    rows = [_default_row(image_id="00001", patient_id="9003", myopic_fundus=1)]
    adapter = _make_brset_fixture(tmp_path, rows)
    sample = adapter.build_manifest()[0]
    assert sample.pathological_myopia is None, (
        "pathological_myopia must be None for Stage 8C BRSET "
        "(myopic_fundus is an anatomical proxy; task is deferred)."
    )
    assert "pathological_myopia" not in adapter.get_supported_tasks(), (
        "pathological_myopia must not be in BRSET supported_tasks for Stage 8C."
    )


# ---------------------------------------------------------------------------
# Tests — diabetes parsing
# ---------------------------------------------------------------------------


def test_brset_adapter_diabetes_yes_no_string(tmp_path: Path) -> None:
    rows = [
        _default_row(image_id="00001", patient_id="10001", diabetes="yes"),
        _default_row(image_id="00002", patient_id="10002", diabetes="no"),
    ]
    adapter = _make_brset_fixture(tmp_path, rows)
    manifest = adapter.build_manifest()
    dm_map = {s.sample_id: s.diabetes for s in manifest}
    assert dm_map["brset_s000001"] == 1
    assert dm_map["brset_s000002"] == 0


# ---------------------------------------------------------------------------
# Tests — DR grading
# ---------------------------------------------------------------------------


def test_brset_adapter_dr_grade_from_sdrg(tmp_path: Path) -> None:
    rows = [
        _default_row(image_id="00001", patient_id="11001", DR_SDRG=3, DR_ICDR=2),
    ]
    adapter = _make_brset_fixture(tmp_path, rows)
    sample = adapter.build_manifest()[0]
    assert sample.dr_grade == 3, "dr_grade must use DR_SDRG, not DR_ICDR."
    assert sample.dr_grade_source_scheme == "DR_SDRG"


# ---------------------------------------------------------------------------
# Tests — task registry and public interface
# ---------------------------------------------------------------------------


def test_brset_adapter_supported_tasks_registered(tmp_path: Path) -> None:
    rows = [_default_row(image_id="00001", patient_id="12001")]
    adapter = _make_brset_fixture(tmp_path, rows)
    for task_name in adapter.get_supported_tasks():
        assert task_name in TASK_REGISTRY, (
            f"Supported task {task_name!r} not in TASK_REGISTRY."
        )


def test_brset_adapter_stratification_columns_in_schema(tmp_path: Path) -> None:
    rows = [_default_row(image_id="00001", patient_id="13001")]
    adapter = _make_brset_fixture(tmp_path, rows)
    schema_fields = set(CanonicalSample.model_fields.keys())
    for col in adapter.get_stratification_columns():
        assert col in schema_fields, (
            f"Stratification column {col!r} not in CanonicalSample."
        )


def test_brset_adapter_missing_image_excluded(tmp_path: Path) -> None:
    images_dir = tmp_path / "fundus_photos"
    images_dir.mkdir()
    # Create image for first row only
    Image.new("RGB", (32, 32)).save(images_dir / "00001.jpg")
    rows = [
        _default_row(image_id="00001", patient_id="14001"),
        _default_row(image_id="99999", patient_id="14002"),  # no image file
    ]
    csv_path = tmp_path / "labels_brset.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    adapter = BRSETAdapter(tmp_path, "labels_brset.csv", "fundus_photos")
    assert len(adapter.build_manifest()) == 1, "Row with missing image must be excluded."


def test_brset_adapter_no_brset_columns_in_public_interface(tmp_path: Path) -> None:
    rows = [_default_row(image_id="00001", patient_id="15001")]
    adapter = _make_brset_fixture(tmp_path, rows)
    brset_native_cols = {
        "image_id", "patient_id", "camera", "patient_age", "patient_sex",
        "exam_eye", "DR_SDRG", "DR_ICDR", "drusens", "myopic_fundus",
        "increased_cup_disc", "comorbidities", "insuline", "diabetes_time_y",
    }
    for col in adapter.get_stratification_columns():
        assert col not in brset_native_cols, (
            f"Stratification column {col!r} is a native BRSET name; "
            "must use canonical field names only."
        )
    for col in adapter.get_quality_columns():
        assert col not in brset_native_cols, (
            f"Quality column {col!r} is a native BRSET name."
        )


# ---------------------------------------------------------------------------
# Tests — camera/device handling (Patch 8: fail-closed for unknown cameras)
# ---------------------------------------------------------------------------


def test_brset_adapter_known_camera_maps_to_clinical(tmp_path: Path) -> None:
    rows = [_default_row(image_id="00001", patient_id="16001", camera="Canon CR")]
    adapter = _make_brset_fixture(tmp_path, rows)
    sample = adapter.build_manifest()[0]
    assert sample.device_class == DeviceClass.CLINICAL


def test_brset_adapter_unknown_camera_maps_to_unknown(tmp_path: Path) -> None:
    """Unknown camera names must map to DeviceClass.UNKNOWN, not CLINICAL."""
    rows = [_default_row(image_id="00001", patient_id="16002", camera="Unknown Camera X9")]
    adapter = _make_brset_fixture(tmp_path, rows)
    sample = adapter.build_manifest()[0]
    assert sample.device_class == DeviceClass.UNKNOWN, (
        "Unknown camera names must be DeviceClass.UNKNOWN, not CLINICAL."
    )


# ---------------------------------------------------------------------------
# Guarded real-data tests (skip if BRSET data unavailable)
# ---------------------------------------------------------------------------

_SKIP_REAL = pytest.mark.skipif(
    not _BRSET_REAL_ROOT_AVAILABLE,
    reason="RETINA_SCREEN_BRSET_ROOT not set and data/brset/ not found.",
)


@_SKIP_REAL
def test_brset_real_manifest_row_count() -> None:
    adapter = BRSETAdapter()
    manifest = adapter.build_manifest()
    assert len(manifest) == 16266, f"Expected 16266 samples, got {len(manifest)}."


@_SKIP_REAL
def test_brset_real_unique_patients() -> None:
    adapter = BRSETAdapter()
    patient_ids = {s.patient_id for s in adapter.build_manifest()}
    assert len(patient_ids) == 8524, f"Expected 8524 unique patients, got {len(patient_ids)}."


@_SKIP_REAL
def test_brset_real_all_images_exist() -> None:
    adapter = BRSETAdapter()
    missing = [
        s.image_path for s in adapter.build_manifest()
        if not Path(s.image_path).exists()
    ]
    assert not missing, f"{len(missing)} image files missing from disk."


@_SKIP_REAL
def test_brset_real_dr_disagreement_count() -> None:
    adapter = BRSETAdapter()
    audit = adapter.get_dataset_audit()
    assert audit["dr_grading"]["dr_sdrg_vs_icdr_disagreement_count"] == 327
    assert audit["dr_grading"]["canonical_source"] == "DR_SDRG"


@_SKIP_REAL
def test_brset_real_macular_edema_positives() -> None:
    adapter = BRSETAdapter()
    manifest = adapter.build_manifest()
    positives = sum(1 for s in manifest if s.macular_edema == 1)
    assert positives == 401, f"Expected 401 macular_edema positives, got {positives}."


@_SKIP_REAL
def test_brset_real_canonical_ids_are_pseudonymous() -> None:
    """No canonical ID may contain a raw native integer patient or image ID."""
    adapter = BRSETAdapter()
    manifest = adapter.build_manifest()
    for sample in manifest[:100]:  # spot-check first 100
        assert re.match(r"brset_s\d{6}$", sample.sample_id), (
            f"sample_id format invalid: {sample.sample_id!r}"
        )
        assert re.match(r"brset_p\d{6}$", sample.patient_id), (
            f"patient_id format invalid: {sample.patient_id!r}"
        )


@_SKIP_REAL
def test_brset_real_validate() -> None:
    adapter = BRSETAdapter()
    adapter.validate()
