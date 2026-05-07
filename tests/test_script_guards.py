"""
tests/test_script_guards.py -- Verify smoke-limit enforcement guards in scripts/03.

Stage 8C smoke configs must require --limit when running extract_embeddings.
These tests run scripts/03_extract_embeddings.py as a subprocess and verify
the guard exits with a nonzero code and an informative message.

The guard runs before any adapter or backbone loading, so no BRSET data or
GPU is required for the first test.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_STAGE8C_CONFIG = "configs/experiment/stage8c_brset_resnet50.yaml"
_SCRIPT = "scripts/03_extract_embeddings.py"


def test_stage8c_guard_exits_nonzero_without_limit() -> None:
    """Stage 8C smoke config without --limit must exit nonzero with an informative message."""
    result = subprocess.run(
        [sys.executable, _SCRIPT, "--config", _STAGE8C_CONFIG],
        capture_output=True,
        text=True,
        cwd=str(_PROJECT_ROOT),
    )
    assert result.returncode != 0, (
        "scripts/03_extract_embeddings.py should exit nonzero when --limit is "
        "omitted for a stage8c_brset_smoke config."
    )
    combined = result.stdout + result.stderr
    assert "Stage 8C" in combined or "smoke" in combined.lower(), (
        f"Exit message should mention 'Stage 8C' or 'smoke', got: {combined[:300]!r}"
    )


def test_stage8c_guard_does_not_trigger_with_limit() -> None:
    """Stage 8C smoke config with --limit must not trigger the guard.

    The script may fail for other reasons (missing data, etc.) after the guard
    passes, but it must not exit due to the limit guard specifically.
    """
    result = subprocess.run(
        [sys.executable, _SCRIPT, "--config", _STAGE8C_CONFIG, "--limit", "1"],
        capture_output=True,
        text=True,
        cwd=str(_PROJECT_ROOT),
    )
    combined = result.stdout + result.stderr
    assert "Stage 8C smoke configs require --limit" not in combined, (
        "Guard must not trigger when --limit is provided. "
        f"Got output: {combined[:300]!r}"
    )
