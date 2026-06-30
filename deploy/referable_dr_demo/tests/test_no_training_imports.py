"""No-training / no-optimization tests (FROST tests 16-17).

The demonstrator is inference-only. It must not import the training module and
must not contain optimizer/backward/fit constructs anywhere in its source.
"""

from __future__ import annotations

import ast
from pathlib import Path

DEMO_ROOT = Path(__file__).resolve().parents[1]

# Precise training-API constructs (chosen so prose/docstrings that merely state
# "no optimizer is created" do not trip the scan).
FORBIDDEN_CALL_TOKENS = (
    ".backward(",
    "torch.optim",
    "optim.Adam",
    "optim.SGD",
    "optim.AdamW",
    "optimizer.step",
    "optimizer.zero_grad",
    "create_optimizer",
    "lr_scheduler",
    "GradScaler",
    "loss.backward",
    ".fit(",
    ".train()",
)


def _non_test_sources() -> list[Path]:
    files: list[Path] = []
    for p in DEMO_ROOT.rglob("*.py"):
        parts = p.relative_to(DEMO_ROOT).parts
        if parts and parts[0] == "tests":
            continue
        if "__pycache__" in parts:
            continue
        files.append(p)
    return files


def test_no_training_imports():
    """16. No demonstrator module imports retina_screen.training."""
    bad: list[tuple[str, str]] = []
    for f in _non_test_sources():
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("retina_screen.training"):
                        bad.append((f.name, alias.name))
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith("retina_screen.training"):
                    bad.append((f.name, node.module))
    assert not bad, f"forbidden training imports found: {bad}"


def test_no_optimizer_or_backward_call():
    """17. No optimizer/backward/fit constructs appear in demonstrator source."""
    hits: list[tuple[str, str]] = []
    for f in _non_test_sources():
        text = f.read_text(encoding="utf-8")
        for tok in FORBIDDEN_CALL_TOKENS:
            if tok in text:
                hits.append((f.name, tok))
    assert not hits, f"forbidden training-style tokens found: {hits}"
