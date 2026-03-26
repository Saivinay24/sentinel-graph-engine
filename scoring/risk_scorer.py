"""
Sentinel-Graph Engine — Composite Risk Scorer
Combines 5 signals into a weighted risk score (0-100).
"""

import yaml
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def _load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


class RiskScorer:
    """
    Composite risk scoring engine.
    
    Combines 5 weighted signals:
    - Temporal velocity: 20%
    - Geo-velocity: 25%
    - Device trust: 15%
    - Entity affinity: 15%
    - Peer alignment: 25%
    """

    def __init__(self, temporal_analyzer=None, geo_analyzer=None,
                 device_scorer=None, peer_scorer=None, config=None):
        if config is None:
            config = _load_config()

        self.weights = config["scoring"]["weights"]
        self.temporal = temporal_analyzer
        self.geo = geo_analyzer
        self.device = device_scorer
        self.peer = peer_scorer

        # Entity affinity scorer is built into this class
        self.user_resource_history = {}  # user_id → set of resource_ids

    def fit_entity_affinity(self, events_df):
        """Build entity affinity profiles (what resources each user normally accesses)."""
        for uid, group in events_df.groupby("user_id"):
            self.user_resource_history[uid] = set(group["resource_id"].unique())

    def _score_entity_affinity(self, event):
        """Score entity affinity anomaly: accessing resources the user never touches."""
        uid = event.get("user_id")
        rid = event.get("resource_id")
        history = self.user_resource_history.get(uid, set())

        if not history:
            return 30.0  # New user → moderate score

        if rid not in history:
            # Never accessed this resource before
            return 60.0
        else:
            return 0.0

    def score(self, event):
        """
        Compute composite risk score for a single event.
        
        Returns:
            dict with 'total', individual signal scores, decision, and explanations
        """
        scores = {}

        # 1. Temporal
        if self.temporal:
            scores["temporal"] = self.temporal.score(event)
        else:
            scores["temporal"] = 0.0

        # 2. Geo-velocity
        if self.geo:
            scores["geo_velocity"] = self.geo.score(event)
        else:
            scores["geo_velocity"] = 0.0

        # 3. Device trust
        if self.device:
            scores["device_trust"] = self.device.score(event)
        else:
            scores["device_trust"] = 0.0

        # 4. Entity affinity
        scores["entity_affinity"] = self._score_entity_affinity(event)

        # 5. Peer alignment
        if self.peer:
            scores["peer_alignment"] = self.peer.score(event.get("user_id"), event)
        else:
            scores["peer_alignment"] = 0.0

        # Max-Boosted Composite Score
        # Pure weighted average dilutes strong signals. Instead:
        # total = 0.55 * weighted_avg + 0.35 * max_signal + 0.10 * multi_signal_bonus
        weighted_avg = (
            scores["temporal"]       * self.weights["temporal"]
            + scores["geo_velocity"]   * self.weights["geo_velocity"]
            + scores["device_trust"]   * self.weights["device_trust"]
            + scores["entity_affinity"]* self.weights.get("entity_affinity", 0.15)
            + scores["peer_alignment"] * self.weights["peer_alignment"]
        )

        signal_values = list(scores.values())
        max_signal = max(signal_values)

        # Multi-signal bonus: if 2+ signals are elevated (>30), add a bonus
        elevated_count = sum(1 for s in signal_values if s > 30)
        multi_bonus = min(20, elevated_count * 7) if elevated_count >= 2 else 0

        # Failed login penalty
        if not event.get("login_success", True):
            multi_bonus += 15

        total = 0.55 * weighted_avg + 0.35 * max_signal + 0.10 * multi_bonus * 10
        total = round(min(100, max(0, total)), 1)

        # Collect explanations
        explanations = {}
        if self.temporal:
            explanations["temporal"] = self.temporal.get_explanation(event)
        if self.geo:
            explanations["geo_velocity"] = self.geo.get_explanation(event)
        if self.device:
            explanations["device_trust"] = self.device.get_explanation(event)

        uid = event.get("user_id")
        rid = event.get("resource_id")
        if rid not in self.user_resource_history.get(uid, set()):
            explanations["entity_affinity"] = f"First-time access to resource {rid}"
        else:
            explanations["entity_affinity"] = "Normal resource access pattern"

        if self.peer:
            explanations["peer_alignment"] = self.peer.get_explanation(uid)

        return {
            "total": total,
            "scores": scores,
            "explanations": explanations,
            "event": event,
        }

    def score_batch(self, events_df):
        """Score a batch of events."""
        results = []
        for _, event in events_df.iterrows():
            result = self.score(event.to_dict())
            results.append(result)
        return results
