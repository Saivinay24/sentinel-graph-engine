"""
Sentinel-Graph Engine — Approach Comparison & Evaluation
Compares Rule-Based vs ML-Only vs Sentinel-Graph on the same attack dataset.
"""

import numpy as np
import pandas as pd
from collections import defaultdict


class RuleBasedScorer:
    """
    Baseline: Simple rule-based scoring (what 80% of teams will build).
    Hardcoded rules with fixed point penalties.
    """

    def __init__(self):
        self.rules = {
            "off_hours": 15,
            "new_device": 20,
            "new_country": 30,
            "failed_login": 10,
            "weekend": 10,
        }

    def score(self, event):
        total = 0
        # Off hours
        ts = pd.to_datetime(event.get("timestamp"))
        hour = ts.hour
        if hour < 8 or hour > 18:
            total += self.rules["off_hours"]
        # Weekend
        if ts.dayofweek >= 5:
            total += self.rules["weekend"]
        # New device
        if not event.get("device_registered", True):
            total += self.rules["new_device"]
        # Login failure
        if not event.get("login_success", True):
            total += self.rules["failed_login"]
        return min(100, total)


class MLOnlyScorer:
    """
    Baseline: ML-only approach (Isolation Forest on flat features).
    No graph, no peer groups.
    """

    def __init__(self):
        self.model = None

    def fit(self, events_df):
        from sklearn.ensemble import IsolationForest

        # Extract flat features
        features = self._extract_features(events_df)
        self.model = IsolationForest(contamination=0.05, random_state=42, n_estimators=100)
        self.model.fit(features)

    def _extract_features(self, events_df):
        df = events_df.copy()
        timestamps = pd.to_datetime(df["timestamp"])
        features = pd.DataFrame({
            "hour": timestamps.dt.hour,
            "day_of_week": timestamps.dt.dayofweek,
            "lat": df["ip_lat"],
            "lon": df["ip_lon"],
            "login_success": df["login_success"].astype(int),
            "device_registered": df["device_registered"].astype(int),
        })
        return features.fillna(0)

    def score(self, event):
        if self.model is None:
            return 50.0

        features = pd.DataFrame([{
            "hour": pd.to_datetime(event.get("timestamp")).hour,
            "day_of_week": pd.to_datetime(event.get("timestamp")).dayofweek,
            "lat": event.get("ip_lat", 0),
            "lon": event.get("ip_lon", 0),
            "login_success": int(event.get("login_success", True)),
            "device_registered": int(event.get("device_registered", True)),
        }])
        raw_score = self.model.decision_function(features)[0]
        # Normalize: lower raw → higher risk
        normalized = max(0, min(100, (0.5 - raw_score) * 100))
        return round(normalized, 1)


def evaluate_approaches(events_df, sentinel_scorer, decision_engine):
    """
    Compare all three approaches on the same dataset.
    
    Returns:
        comparison_df: DataFrame with per-approach metrics
        detailed_results: list of per-event results
    """
    # Initialize baselines
    rule_scorer = RuleBasedScorer()
    ml_scorer = MLOnlyScorer()

    # Train ML baseline on normal events only
    normal_events = events_df[events_df["is_attack"] == False]
    ml_scorer.fit(normal_events)

    # Score all events
    results = []
    for idx, event in events_df.iterrows():
        evt = event.to_dict()
        is_attack = evt.get("is_attack", False)
        attack_type = evt.get("attack_type", "none")

        rule_score = rule_scorer.score(evt)
        ml_score = ml_scorer.score(evt)

        sentinel_result = sentinel_scorer.score(evt)
        sg_score = sentinel_result["total"]

        results.append({
            "event_id": evt.get("event_id"),
            "user_id": evt.get("user_id"),
            "is_attack": is_attack,
            "attack_type": attack_type,
            "rule_based_score": rule_score,
            "ml_only_score": ml_score,
            "sentinel_graph_score": sg_score,
            "rule_based_decision": "FLAG" if rule_score > 50 else "ALLOW",
            "ml_only_decision": "FLAG" if ml_score > 50 else "ALLOW",
            "sentinel_decision": decision_engine.decide(sentinel_result)["decision"],
        })

    results_df = pd.DataFrame(results)

    # Calculate metrics
    threshold = 50  # Score above this = flagged
    metrics = {}

    for approach in ["rule_based", "ml_only", "sentinel_graph"]:
        score_col = f"{approach}_score"
        flagged = results_df[score_col] > threshold
        actual_attack = results_df["is_attack"]

        tp = (flagged & actual_attack).sum()
        fp = (flagged & ~actual_attack).sum()
        fn = (~flagged & actual_attack).sum()
        tn = (~flagged & ~actual_attack).sum()

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 0.001)

        metrics[approach] = {
            "precision": round(precision * 100, 1),
            "recall": round(recall * 100, 1),
            "f1_score": round(f1 * 100, 1),
            "false_positive_rate": round(fpr * 100, 1),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_negatives": int(tn),
        }

    comparison = pd.DataFrame(metrics).T
    comparison.index.name = "approach"

    # Additional capabilities comparison
    capabilities = pd.DataFrame({
        "approach": ["Rule-Based", "ML-Only", "Sentinel-Graph"],
        "can_score_new_users": ["❌ No", "❌ No", "✅ Yes (inductive GraphSAGE)"],
        "detects_insider_threats": ["❌ No", "❌ No", "✅ Yes (peer deviation)"],
        "privacy_compliant": ["❌ No", "❌ No", "✅ Yes (ε-DP, k-anonymity)"],
        "uses_relationships": ["❌ No", "❌ No", "✅ Yes (knowledge graph)"],
        "explainable_decisions": ["✅ Rules", "❌ Black box", "✅ Full breakdown"],
    })

    return comparison, capabilities, results_df
