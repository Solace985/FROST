from __future__ import annotations

from enum import Enum

from retina_screen.schema import CANONICAL_METADATA_FIELDS
from retina_screen.tasks import TASK_REGISTRY




class ModelInputMode(str, Enum):
    """Controls which metadata fields may be sent to the model."""

    IMAGE_ONLY = "image_only"
    IMAGE_PLUS_METADATA = "image_plus_metadata"
    CLINICAL_DEPLOYMENT = "clinical_deployment"
    FAIRNESS_ABLATION = "fairness_ablation"



_RESTRICTED_FIELDS: frozenset[str] = frozenset({"dataset_source", "camera_type"})

_DEFAULT_METADATA: frozenset[str] = CANONICAL_METADATA_FIELDS - _RESTRICTED_FIELDS

_MODE_BASES: dict[str, frozenset[str]] = {
    ModelInputMode.IMAGE_ONLY: frozenset(),
    ModelInputMode.IMAGE_PLUS_METADATA: _DEFAULT_METADATA,
    ModelInputMode.CLINICAL_DEPLOYMENT: frozenset(
        {"age_years", "sex", "eye_laterality", "image_quality_score", "image_quality_label"}
    ),
    ModelInputMode.FAIRNESS_ABLATION: _DEFAULT_METADATA,
}

_TASK_BLOCKS: dict[str, frozenset[str]] = {
    "retinal_age": frozenset({"age_years"}),
    "sex": frozenset({"sex"}),
}




class FeaturePolicy:
    """Determines which metadata fields may be used as model inputs.

    Usage::

        policy = FeaturePolicy()
        allowed = policy.allowed_fields("glaucoma", "image_plus_metadata")

    The policy fails closed:
    - Unknown task names raise ValueError.
    - Unknown fields in explicit_allow raise ValueError.
    - image_only always returns an empty set.
    """

    RESTRICTED_FIELDS: frozenset[str] = _RESTRICTED_FIELDS

    def allowed_fields(
        self,
        task_name: str,
        mode: str,
        explicit_allow: frozenset[str] | None = None,
    ) -> frozenset[str]:
        """Return the set of metadata field names permitted as model inputs.

        Parameters
        ----------
        task_name:
            Registered task name (must exist in TASK_REGISTRY).
        mode:
            One of the ModelInputMode values (string or enum).
        explicit_allow:
            Additional restricted fields to permit (e.g. for ablation studies).
            Every field in this set must be in CANONICAL_METADATA_FIELDS;
            unrecognised fields raise ValueError.

        Returns
        -------
        frozenset[str]
            Names of CanonicalSample metadata fields allowed as model inputs.

        Raises
        ------
        ValueError
            If task_name, mode, or any explicit_allow field is not recognised.
        """
        mode_str = mode.value if isinstance(mode, ModelInputMode) else mode
        if mode_str not in _MODE_BASES:
            valid = [m.value for m in ModelInputMode]
            raise ValueError(
                f"Unknown mode {mode_str!r}. Valid modes: {valid}"
            )

        if task_name not in TASK_REGISTRY:
            raise ValueError(
                f"Unknown task {task_name!r} — FeaturePolicy fails closed. "
                f"Register the task in tasks.py first."
            )

        if explicit_allow:
            unknown = explicit_allow - CANONICAL_METADATA_FIELDS
            if unknown:
                raise ValueError(
                    f"explicit_allow contains unrecognised metadata fields: {sorted(unknown)}. "
                    f"Valid metadata fields: {sorted(CANONICAL_METADATA_FIELDS)}"
                )

        if mode_str == ModelInputMode.IMAGE_ONLY.value:
            return frozenset()

        base: frozenset[str] = _MODE_BASES[mode_str]

        if explicit_allow:
            base = base | (explicit_allow & _RESTRICTED_FIELDS)

        task_blocks = _TASK_BLOCKS.get(task_name, frozenset())
        return base - task_blocks

    def is_field_allowed(
        self,
        field_name: str,
        task_name: str,
        mode: str,
        explicit_allow: frozenset[str] | None = None,
    ) -> bool:
        """Return True if *field_name* is permitted for *task_name* in *mode*."""
        return field_name in self.allowed_fields(task_name, mode, explicit_allow)
