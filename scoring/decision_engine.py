"""
Sentinel-Graph Engine — Decision Engine
Maps composite risk scores to adaptive access control decisions.
"""

import yaml
from pathlib import Path
from datetime import datetime

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def _load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


# Decision constants
ALLOW = "ALLOW"
MFA = "STEP_UP_MFA"
BLOCK = "BLOCK_ALERT"
CRITICAL = "BLOCK_SOC"

DECISION_DETAILS = {
    ALLOW: {
        "action": "Allow",
        "color": "#00C853",
        "icon": "✅",
        "description": "Access granted — normal risk level",
        "severity": "low",
    },
    MFA: {
        "action": "Step-Up MFA",
        "color": "#FFD600",
        "icon": "🔶",
        "description": "Multi-factor authentication required",
        "severity": "medium",
    },
    BLOCK: {
        "action": "Block + Admin Alert",
        "color": "#FF1744",
        "icon": "🔴",
        "description": "Access blocked — administrator notified",
        "severity": "high",
    },
    CRITICAL: {
        "action": "Block + SOC + Session Terminate",
        "color": "#000000",
        "icon": "🚨",
        "description": "Critical threat — SOC team alerted, all sessions terminated",
        "severity": "critical",
    },
}


class DecisionEngine:
    """Maps risk scores to adaptive authentication decisions."""

    def __init__(self, config=None):
        if config is None:
            config = _load_config()

        thresholds = config["scoring"]["thresholds"]
        self.allow_max = thresholds["allow"]          # 0-30
        self.mfa_max = thresholds["mfa"]              # 31-60
        self.block_max = thresholds["block"]           # 61-85
        # 86-100 = critical

    def decide(self, risk_result):
        """
        Make an adaptive access control decision.
        
        Args:
            risk_result: dict from RiskScorer.score() containing 'total' and 'scores'
        
        Returns:
            decision dict with action, color, explanation, and full breakdown
        """
        total = risk_result.get("total", 0)
        scores = risk_result.get("scores", {})
        explanations = risk_result.get("explanations", {})
        event = risk_result.get("event", {})

        # Determine decision
        if total <= self.allow_max:
            decision_key = ALLOW
        elif total <= self.mfa_max:
            decision_key = MFA
        elif total <= self.block_max:
            decision_key = BLOCK
        else:
            decision_key = CRITICAL

        decision = DECISION_DETAILS[decision_key].copy()

        # Build full result
        result = {
            "decision": decision["action"],
            "decision_key": decision_key,
            "severity": decision["severity"],
            "color": decision["color"],
            "icon": decision["icon"],
            "description": decision["description"],
            "risk_score": total,
            "score_breakdown": scores,
            "explanations": explanations,
            "timestamp": datetime.now().isoformat(),
            "user_id": event.get("user_id", ""),
            "resource_id": event.get("resource_id", ""),
        }

        return result

    def format_risk_report(self, risk_result):
        """
        Generate a human-readable risk report.
        
        Example output:
        Risk Score: 87 (Critical)
        Breakdown:
          - Temporal anomaly: +18 (Login at 3:12 AM)
          - Geo-velocity: +25 (Impossible travel)
          - Device trust: +15 (Unregistered device)
          - Entity affinity: +12 (First-time resource)
          - Peer deviation: +17 (3.2σ from centroid)
        Recommended action: BLOCK + SOC ALERT
        """
        decision = self.decide(risk_result)
        scores = risk_result.get("scores", {})
        explanations = risk_result.get("explanations", {})
        total = risk_result.get("total", 0)

        lines = [
            f"Risk Score: {total:.1f} ({decision['severity'].upper()})",
            f"Decision: {decision['icon']} {decision['decision']}",
            f"",
            f"Score Breakdown:",
        ]

        signal_names = {
            "temporal": "Temporal anomaly",
            "geo_velocity": "Geo-velocity",
            "device_trust": "Device trust",
            "entity_affinity": "Entity affinity",
            "peer_alignment": "Peer deviation",
        }

        for key, name in signal_names.items():
            s = scores.get(key, 0)
            exp = explanations.get(key, "")
            lines.append(f"  • {name}: {s:.1f}/100 — {exp}")

        lines.append(f"\n{decision['description']}")

        return "\n".join(lines)
