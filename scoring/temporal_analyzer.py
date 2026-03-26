"""
Sentinel-Graph Engine — Temporal Analyzer
Scores login time anomalies using z-score deviation from user's typical patterns.
Weight: 20% of composite score.
"""

import numpy as np
import pandas as pd
from datetime import datetime


class TemporalAnalyzer:
    """Detects temporal anomalies in login patterns."""

    def __init__(self, business_hours_start=8, business_hours_end=18):
        self.bh_start = business_hours_start
        self.bh_end = business_hours_end
        self.user_profiles = {}

    def fit(self, events_df):
        """Build temporal profiles for each user from historical events."""
        events = events_df.copy()
        events["hour"] = pd.to_datetime(events["timestamp"]).dt.hour
        events["day_of_week"] = pd.to_datetime(events["timestamp"]).dt.dayofweek

        for uid, group in events.groupby("user_id"):
            hours = group["hour"].values
            self.user_profiles[uid] = {
                "mean_hour": float(np.mean(hours)),
                "std_hour": float(np.std(hours)) if len(hours) > 1 else 4.0,
                "typical_days": group["day_of_week"].mode().tolist(),
                "event_count": len(group),
                "login_frequency_daily": len(group) / max(1, group["day_of_week"].nunique()),
            }

        print(f"   ✅ Temporal profiles built for {len(self.user_profiles)} users")

    def score(self, event):
        """
        Score a single login event for temporal anomaly (0-100).
        
        High scores indicate anomalous timing.
        """
        uid = event.get("user_id")
        ts = pd.to_datetime(event.get("timestamp"))
        hour = ts.hour
        day_of_week = ts.dayofweek

        profile = self.user_profiles.get(uid)

        score = 0.0

        if profile is None:
            # New user — moderate suspicion
            if hour < self.bh_start or hour > self.bh_end:
                score += 40
            if day_of_week >= 5:  # Weekend
                score += 20
            return min(100, score)

        # Z-score from typical login hour
        mean_h = profile["mean_hour"]
        std_h = max(profile["std_hour"], 1.0)
        z = abs(hour - mean_h) / std_h
        # Map z-score to 0-60 range
        time_score = min(60, z * 20)

        # Weekend penalty
        if day_of_week >= 5 and day_of_week not in profile.get("typical_days", []):
            time_score += 20

        # Very late night / early morning penalty (2-5 AM)
        if 2 <= hour <= 5:
            time_score += 25

        # Frequency spike (many logins in short period)
        # This would be checked at a higher level; placeholder score
        score = min(100, time_score)

        return round(score, 1)

    def get_explanation(self, event):
        """Generate human-readable explanation for the score."""
        uid = event.get("user_id")
        ts = pd.to_datetime(event.get("timestamp"))
        hour = ts.hour
        profile = self.user_profiles.get(uid, {})
        mean_h = profile.get("mean_hour", 13)

        explanations = []
        if hour < self.bh_start or hour > self.bh_end:
            explanations.append(f"Login at {hour}:00 is outside business hours ({self.bh_start}:00-{self.bh_end}:00)")
        if ts.dayofweek >= 5:
            explanations.append("Login on weekend")
        if profile:
            z = abs(hour - mean_h) / max(profile.get("std_hour", 1), 1)
            if z > 2:
                explanations.append(f"Login time deviates {z:.1f}σ from user's typical hour ({mean_h:.0f}:00)")

        return "; ".join(explanations) if explanations else "Normal login time"
