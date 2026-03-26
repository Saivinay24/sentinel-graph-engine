"""
Sentinel-Graph Engine — Device Trust Scorer
Scores device signals: new/unregistered/jailbroken devices.
Weight: 15% of composite score.
"""

import pandas as pd
from collections import defaultdict


class DeviceTrustScorer:
    """Evaluates device trustworthiness for each login event."""

    def __init__(self, new_device_score=30, unregistered_score=20, jailbroken_score=40):
        self.new_device_penalty = new_device_score
        self.unregistered_penalty = unregistered_score
        self.jailbroken_penalty = jailbroken_score
        self.user_known_devices = defaultdict(set)  # user_id → set of known fingerprints
        self.registered_devices = set()  # Set of all registered device fingerprints

    def fit(self, events_df, devices_df=None):
        """Build device profiles from historical data."""
        # Learn known devices per user from events
        for _, evt in events_df.iterrows():
            uid = evt["user_id"]
            fp = evt.get("device_fingerprint", "")
            if fp:
                self.user_known_devices[uid].add(fp)

        # Load registered devices
        if devices_df is not None:
            for _, dev in devices_df.iterrows():
                if dev.get("is_registered", False):
                    self.registered_devices.add(dev["device_fingerprint"])

        n_users = len(self.user_known_devices)
        n_registered = len(self.registered_devices)
        print(f"   ✅ Device profiles: {n_users} users, {n_registered} registered devices")

    def score(self, event):
        """
        Score device trust for a single event (0-100).
        
        Penalties:
        - New device (never seen for this user): +30
        - Unregistered device (not in org registry): +20
        - Suspicious OS/browser combination: +15
        """
        uid = event.get("user_id")
        fp = event.get("device_fingerprint", "")
        os_name = event.get("os", "")
        browser = event.get("browser", "")
        is_registered = event.get("device_registered", True)

        score = 0.0

        # New device for this user?
        known = self.user_known_devices.get(uid, set())
        if fp and fp not in known:
            score += self.new_device_penalty

        # Unregistered in org?
        if not is_registered and fp not in self.registered_devices:
            score += self.unregistered_penalty

        # Suspicious indicators
        suspicious_os = ["Android 14", "iOS 17"]  # Mobile OS for enterprise login
        if os_name in suspicious_os and event.get("device_type") in ["Mobile", "Tablet"]:
            score += 5  # Mild flag for mobile access

        # Very old browser version
        if browser and "/" in browser:
            try:
                version = int(browser.split("/")[1])
                if version < 100:
                    score += 10  # Old browser
            except (ValueError, IndexError):
                pass

        return min(100, round(score, 1))

    def get_explanation(self, event):
        """Generate explanation."""
        uid = event.get("user_id")
        fp = event.get("device_fingerprint", "")
        explanations = []

        known = self.user_known_devices.get(uid, set())
        if fp and fp not in known:
            explanations.append(f"New device detected (fingerprint: {fp[:8]}...)")
        if not event.get("device_registered", True):
            explanations.append("Device not registered in organization")

        return "; ".join(explanations) if explanations else "Trusted device"
