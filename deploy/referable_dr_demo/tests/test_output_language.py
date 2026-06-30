"""Output-language tests (FROST test 24).

The result-facing frontend must use screening/triage language, not diagnostic or
rule-out claims, and must surface the required research-framing statements.
"""

from __future__ import annotations

from pathlib import Path

FRONTEND = Path(__file__).resolve().parents[1] / "frontend"

# Phrases that must never appear in the result-facing frontend.
FORBIDDEN_PHRASES = (
    "no disease",
    "no referral needed",
    "no referral",
    "calibrated risk",
    "clinical certainty",
    "probability of disease",
    "final decision",
    "diagnosis",
)

# Framing statements that must be present.
REQUIRED_PHRASES = (
    "not a medical device",
    "operating-point triage result",
    "not a rule-out result",
    "research alert",
)


def _frontend_text() -> str:
    parts = []
    for name in ("index.html", "app.js"):
        parts.append((FRONTEND / name).read_text(encoding="utf-8").lower())
    return "\n".join(parts)


def test_output_language_does_not_use_rule_out_or_diagnosis_claims():
    text = _frontend_text()
    present = [p for p in FORBIDDEN_PHRASES if p in text]
    assert present == [], f"forbidden result language present: {present}"
    missing = [p for p in REQUIRED_PHRASES if p not in text]
    assert missing == [], f"required research-framing phrases missing: {missing}"
