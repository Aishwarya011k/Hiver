"""Deterministic safety/policy layer; it never calls an LLM."""
from __future__ import annotations

RISK_KEYWORDS = {
    "high": ("billing", "refund", "charge", "legal", "lawsuit", "privacy", "account hacked", "fraud"),
    "medium": ("cancel", "delivery", "late", "missing", "replacement", "damaged"),
}


def risk_tier(intent: str, message: str) -> str:
    text = f"{intent} {message}".lower()
    if any(word in text for word in RISK_KEYWORDS["high"]):
        return "high"
    if any(word in text for word in RISK_KEYWORDS["medium"]):
        return "medium"
    return "low"


def decide(message: str, intent: str, classifier_confidence: float, reply_confidence: float) -> dict:
    tier = risk_tier(intent, message)
    reasons = []
    if tier == "high":
        reasons.append("high-risk intent requires human review")
    if classifier_confidence < 0.65:
        reasons.append(f"classifier confidence {classifier_confidence:.2f} is below 0.65")
    if reply_confidence < 0.60:
        reasons.append(f"reply grounding confidence {reply_confidence:.2f} is below 0.60")
    decision = "escalate-to-human" if reasons else "auto-handle"
    if not reasons:
        reasons.append("low/medium risk with sufficient classifier and grounding confidence")
    return {"decision": decision, "risk_tier": tier, "reason": "; ".join(reasons)}
