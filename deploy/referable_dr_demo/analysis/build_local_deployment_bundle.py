from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from deploy.referable_dr_demo.backend.service import bundle as bundle_mod  # noqa: E402


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    log = logging.getLogger("frost.build_bundle")

    log.info("Discovering and validating accepted native-392 artifacts ...")
    manifest = bundle_mod.build_bundle(validate_backbone_forward=True)

    out = bundle_mod.default_bundle_path()
    bundle_mod.write_bundle(manifest, out)

    bundle_mod.validate_bundle_against_artifacts(manifest)

    print("OK: wrote local deployment bundle")
    print(f"  path                 : {out}")
    print(f"  bundle_version       : {manifest['bundle_version']}")
    print(f"  backbone_identifier  : {manifest['backbone_identifier']}")
    print(f"  native_input_size    : {manifest['native_input_size']}")
    print(f"  native_pooling       : {manifest['native_pooling']}")
    print(f"  expected_embedding_dim: {manifest['expected_embedding_dim']}")
    print(f"  head_type            : {manifest['head_type']}")
    print(f"  backbone_sha256      : {manifest['backbone_checkpoint_sha256'][:16]}...")
    print(f"  head_sha256          : {manifest['head_checkpoint_sha256'][:16]}...")
    print(f"  preprocessing_hash   : {manifest['preprocessing_hash']}")
    print(f"  task_ordering        : {manifest['model_task_ordering']}")
    print(f"  dr_grade_index/shape : {manifest['dr_grade_task_index']} / "
          f"{manifest['dr_grade_output_shape']}")
    print(f"  torch / timm         : {manifest['torch_version']} / {manifest['timm_version']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
