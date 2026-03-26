"""
Sentinel-Graph API — Natural Language Explanation Engine
Template-based NL explanation engine (no LLM required).
Converts signal scores and decision context into human-readable sentences.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


# Decision key constants (mirrors scoring/decision_engine.py)
ALLOW = "ALLOW"
MFA = "STEP_UP_MFA"
BLOCK = "BLOCK_ALERT"
CRITICAL = "BLOCK_SOC"


# ---------------------------------------------------------------------------
# Helper: Signal phrase builders
# ---------------------------------------------------------------------------

def _peer_phrase(peer_group: Optional[Dict[str, Any]], resource_id: str, peer_score: float) -> Optional[str]:
    """Build a peer group anomaly phrase if the signal is elevated."""
    if peer_score < 0.3:
        return None
    if peer_group is None:
        return "behavior deviates from known peer groups"

    label = peer_group.get("label", "their peer group")
    size = peer_group.get("size", 0)
    size_str = f", {size} users" if size else ""

    if peer_score >= 0.80:
        return f"this user's peer group ({label}{size_str}) has never accessed {resource_id}"
    elif peer_score >= 0.55:
        return f"this user is significantly deviating from their peer group ({label}{size_str})"
    else:
        return f"mild deviation from peer group ({label}{size_str})"


def _geo_phrase(geo_score: float, event_dict: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Build a geo-velocity anomaly phrase if the signal is elevated."""
    if geo_score < 0.25:
        return None
    city = (event_dict or {}).get("ip_city", "an unusual location")
    if geo_score >= 0.80:
        return f"impossible travel detected (login from {city})"
    elif geo_score >= 0.50:
        return f"login from distant location ({city})"
    else:
        return f"login from {city}"


def _device_phrase(device_score: float, event_dict: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Build a device trust phrase if the signal is elevated."""
    if device_score < 0.20:
        return None
    is_registered = (event_dict or {}).get("device_registered", True)
    if device_score >= 0.60:
        return "an unregistered device" if not is_registered else "an untrusted device"
    elif device_score >= 0.35:
        return "a previously unseen device"
    else:
        return "a device with minor trust concerns"


def _temporal_phrase(temporal_score: float, event_dict: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Build a temporal anomaly phrase if the signal is elevated."""
    if temporal_score < 0.25:
        return None
    ts_str = (event_dict or {}).get("timestamp", "")
    try:
        from datetime import datetime
        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        hour = ts.hour
        time_str = f"a {hour}:{ts.minute:02d} {'AM' if hour < 12 else 'PM'} login"
    except Exception:
        time_str = "login at an unusual hour"

    if temporal_score >= 0.70:
        return time_str
    elif temporal_score >= 0.40:
        return "login outside normal business hours"
    else:
        return "mildly unusual login time"


def _entity_phrase(entity_score: float, resource_id: str) -> Optional[str]:
    """Build an entity affinity phrase if the signal is elevated."""
    if entity_score < 0.30:
        return None
    if entity_score >= 0.55:
        return f"first-ever access to {resource_id}"
    else:
        return f"infrequent access to {resource_id}"


# ---------------------------------------------------------------------------
# Decision-level prefix templates
# ---------------------------------------------------------------------------

DECISION_PREFIXES = {
    ALLOW: "Allowed",
    MFA: "Step-up MFA required",
    BLOCK: "Blocked",
    CRITICAL: "Blocked and SOC alerted",
}


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def generate_explanation(
    user_id: str,
    resource_id: str,
    signals: Dict[str, float],
    decision_key: str,
    peer_group: Optional[Dict[str, Any]] = None,
    event_dict: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Build a human-readable NL explanation from signal scores and context.

    Args:
        user_id:      The user being scored.
        resource_id:  The resource being accessed.
        signals:      Dict with keys peer_deviation, geo_velocity, device_trust,
                      temporal, entity_affinity — all in the 0-1 range.
        decision_key: One of ALLOW / STEP_UP_MFA / BLOCK_ALERT / BLOCK_SOC.
        peer_group:   Optional dict with 'label', 'size' keys.
        event_dict:   Optional raw event dict for richer phrase generation.

    Returns:
        A single human-readable sentence.

    Examples:
        "Blocked and SOC alerted: This user's peer group (HR Operations, 23 users)
         has never accessed finance_db_prod; impossible travel detected (login from
         Beijing); an unregistered device."

        "Step-up MFA required: First-ever access to analytics_db; login outside
         normal business hours."

        "Allowed: All behavioral signals within normal bounds for this peer group."
    """
    prefix = DECISION_PREFIXES.get(decision_key, "Decision made")

    # Build individual signal phrases in priority order
    phrases = []

    peer_p = _peer_phrase(peer_group, resource_id, signals.get("peer_deviation", 0.0))
    geo_p = _geo_phrase(signals.get("geo_velocity", 0.0), event_dict)
    device_p = _device_phrase(signals.get("device_trust", 0.0), event_dict)
    temporal_p = _temporal_phrase(signals.get("temporal", 0.0), event_dict)
    entity_p = _entity_phrase(signals.get("entity_affinity", 0.0), resource_id)

    # Lead with peer group finding (highest-value signal for this system)
    if peer_p:
        phrases.append(peer_p.capitalize())
    if geo_p:
        phrases.append(geo_p)
    if device_p:
        phrases.append(device_p)
    if temporal_p:
        phrases.append(temporal_p)
    if entity_p:
        phrases.append(entity_p)

    if not phrases:
        body = "all behavioral signals within normal bounds for this peer group"
        return f"{prefix}: {body.capitalize()}."

    # Join: first phrase follows colon, rest joined with semicolons/commas
    if len(phrases) == 1:
        body = phrases[0]
    elif len(phrases) == 2:
        body = f"{phrases[0]}; {phrases[1]}"
    else:
        # Lead phrase + "combined with" + rest
        rest = "; ".join(phrases[1:])
        body = f"{phrases[0]}. Combined with {rest}"

    return f"{prefix}: {body}."
