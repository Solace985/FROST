from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
HF_SPACE_DIR = Path(__file__).resolve().parent

CONFIGS = [
    "configs/backbone/retfound_green_native392.yaml",
    "configs/preprocessing/retfound_green_native392.yaml",
    "configs/tasks/brset_default.yaml",
]

_DEPLOY_IGNORE = shutil.ignore_patterns(
    "__pycache__", "*.pyc", "*.pyo",
    ".local", "tests", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "graphical_changes.md",
    "*.local.json", "*.local.yaml",
)

DEFAULT_HEAD = REPO_ROOT / "runs" / "train" / "brset_20260525_113425" / "model_checkpoint.pt"
DEFAULT_OP = HF_SPACE_DIR.parent / ".local" / "operating_point.local.json"


def _resolve_head(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    env = os.environ.get("RETINA_SCREEN_RETF_NATIVE392_MT_CHECKPOINT")
    return Path(env) if env else DEFAULT_HEAD


def _resolve_op(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    env = os.environ.get("RETINA_SCREEN_DEMO_THRESHOLD_PATH")
    return Path(env) if env else DEFAULT_OP


def main() -> int:
    ap = argparse.ArgumentParser(description="Assemble a FROST Hugging Face Docker Space.")
    ap.add_argument("--space-dir", required=True, help="Path to your cloned (empty) Space repo.")
    ap.add_argument("--head", default=None, help="Trained head checkpoint (.pt). Default: accepted run.")
    ap.add_argument("--operating-point", default=None, help="operating_point JSON. Default: .local artifact.")
    ap.add_argument("--force", action="store_true", help="Overwrite an existing non-empty space dir.")
    args = ap.parse_args()

    space = Path(args.space_dir).resolve()
    head = _resolve_head(args.head)
    op = _resolve_op(args.operating_point)

    problems = []
    if not head.exists():
        problems.append(f"head checkpoint not found: {head}\n    (pass --head or set "
                        "RETINA_SCREEN_RETF_NATIVE392_MT_CHECKPOINT)")
    if not op.exists():
        problems.append(f"operating-point JSON not found: {op}\n    (run analysis/derive_threshold.py "
                        "first, or pass --operating-point)")
    for c in CONFIGS:
        if not (REPO_ROOT / c).exists():
            problems.append(f"missing required config: {c}")
    if not (REPO_ROOT / "src" / "retina_screen").exists():
        problems.append("missing src/retina_screen")
    if problems:
        print("ERROR — cannot assemble the Space:\n  - " + "\n  - ".join(problems), file=sys.stderr)
        return 1

    space.mkdir(parents=True, exist_ok=True)
    if any(space.iterdir()) and not args.force:
        print(f"ERROR — target {space} is not empty (use --force to overwrite matching paths).",
              file=sys.stderr)
        return 1

    shutil.copy2(HF_SPACE_DIR / "Dockerfile", space / "Dockerfile")
    shutil.copy2(HF_SPACE_DIR / ".dockerignore", space / ".dockerignore")
    shutil.copy2(HF_SPACE_DIR / "README_space.md", space / "README.md")

    _copytree(REPO_ROOT / "src" / "retina_screen", space / "src" / "retina_screen",
              ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    for c in CONFIGS:
        dst = space / c
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / c, dst)

    _copytree(REPO_ROOT / "deploy" / "referable_dr_demo",
              space / "deploy" / "referable_dr_demo", ignore=_DEPLOY_IGNORE)

    _unignore_space_artifacts(space / "deploy" / "referable_dr_demo" / ".gitignore")

    weights_dir = space / "deploy" / "referable_dr_demo" / "hf_space" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(head, weights_dir / "mt_head_native392.pt")
    shutil.copy2(op, space / "deploy" / "referable_dr_demo" / "hf_space" / "operating_point.json")

    head_kb = (weights_dir / "mt_head_native392.pt").stat().st_size / 1024
    print("Assembled FROST Space at:", space)
    print(f"  head    -> hf_space/weights/mt_head_native392.pt ({head_kb:.0f} KB)")
    print(f"  op json -> hf_space/operating_point.json")
    print("  backbone: fetched + SHA-verified by the Dockerfile at build (not copied).")
    print("\nNext:")
    print(f"  cd {space}")
    print("  git lfs install && git lfs track '*.pt'   # (optional; head is small but LFS is fine)")
    print("  git add -A && git commit -m 'FROST Space' && git push")
    return 0


def _copytree(src: Path, dst: Path, ignore) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=ignore)


_SPACE_ALLOW = {"hf_space/weights/*.pt", "hf_space/operating_point.json"}


def _unignore_space_artifacts(gitignore: Path) -> None:
    if not gitignore.exists():
        return
    kept = []
    for line in gitignore.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s in _SPACE_ALLOW:
            continue
        if s.startswith("# HF Space travelling artifacts"):
            continue
        kept.append(line)
    gitignore.write_text("\n".join(kept) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
