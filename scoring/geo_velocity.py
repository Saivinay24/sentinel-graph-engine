"""
Sentinel-Graph Engine — Geo-Velocity Analyzer
Detects impossible travel: distance/time > max plausible speed.
Weight: 25% of composite score.
"""

import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2
from collections import defaultdict


def haversine_km(lat1, lon1, lat2, lon2):
    """Calculate great-circle distance between two points in km."""
    R = 6371.0  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


class GeoVelocityAnalyzer:
    """Detects impossible travel patterns."""

    def __init__(self, max_speed_kmh=900):
        self.max_speed = max_speed_kmh  # Commercial aviation speed
        self.user_last_login = {}  # user_id → (timestamp, lat, lon, city)
        self.user_home_locations = {}

    def fit(self, events_df, users_df=None):
        """Build location profiles from historical events."""
        events = events_df.sort_values("timestamp")

        # Store last known location per user
        for _, evt in events.iterrows():
            uid = evt["user_id"]
            self.user_last_login[uid] = {
                "timestamp": pd.to_datetime(evt["timestamp"]),
                "lat": evt["ip_lat"],
                "lon": evt["ip_lon"],
                "city": evt.get("ip_city", "Unknown"),
            }

        # Store home locations from user data
        if users_df is not None:
            for _, user in users_df.iterrows():
                self.user_home_locations[user["user_id"]] = {
                    "lat": user["office_lat"],
                    "lon": user["office_lon"],
                    "city": user["office_city"],
                }

        print(f"   ✅ Geo profiles built for {len(self.user_last_login)} users")

    def score(self, event):
        """
        Score a login event for geo-velocity anomaly (0-100).
        
        Checks:
        1. Impossible travel from last known location
        2. Login from unusual country/city
        """
        uid = event.get("user_id")
        ts = pd.to_datetime(event.get("timestamp"))
        lat = event.get("ip_lat", 0)
        lon = event.get("ip_lon", 0)
        city = event.get("ip_city", "Unknown")

        score = 0.0

        # Check impossible travel
        last = self.user_last_login.get(uid)
        if last:
            distance = haversine_km(last["lat"], last["lon"], lat, lon)
            time_diff_hours = max(
                (ts - last["timestamp"]).total_seconds() / 3600,
                0.001  # avoid division by zero
            )
            speed = distance / time_diff_hours

            if speed > self.max_speed and distance > 100:
                # Impossible travel detected
                severity = min(1.0, speed / (self.max_speed * 3))
                score += 60 * severity
                score += min(30, distance / 500)  # Distance bonus

        # Check if login is from a city the user has never been in
        home = self.user_home_locations.get(uid)
        if home:
            home_distance = haversine_km(home["lat"], home["lon"], lat, lon)
            if home_distance > 5000:  # > 5000 km from office
                score += 15
            elif home_distance > 2000:
                score += 8

        # Update last login
        self.user_last_login[uid] = {
            "timestamp": ts,
            "lat": lat,
            "lon": lon,
            "city": city,
        }

        return min(100, round(score, 1))

    def get_explanation(self, event):
        """Generate explanation for the geo-velocity score."""
        uid = event.get("user_id")
        ts = pd.to_datetime(event.get("timestamp"))
        lat = event.get("ip_lat", 0)
        lon = event.get("ip_lon", 0)
        city = event.get("ip_city", "Unknown")

        explanations = []
        last = self.user_last_login.get(uid)

        if last:
            distance = haversine_km(last["lat"], last["lon"], lat, lon)
            time_diff = max((ts - last["timestamp"]).total_seconds() / 3600, 0.001)
            speed = distance / time_diff

            if speed > self.max_speed and distance > 100:
                explanations.append(
                    f"Impossible travel: {last['city']} → {city} "
                    f"({distance:.0f} km in {time_diff:.1f}h = {speed:.0f} km/h)"
                )

        home = self.user_home_locations.get(uid)
        if home:
            d = haversine_km(home["lat"], home["lon"], lat, lon)
            if d > 2000:
                explanations.append(f"Login from {city}, {d:.0f} km from office ({home['city']})")

        return "; ".join(explanations) if explanations else "Normal location"

    def get_travel_path(self, event):
        """Get the travel path for visualization (impossible travel arcs)."""
        uid = event.get("user_id")
        last = self.user_last_login.get(uid)
        if not last:
            return None

        return {
            "from_lat": last["lat"],
            "from_lon": last["lon"],
            "from_city": last["city"],
            "to_lat": event.get("ip_lat"),
            "to_lon": event.get("ip_lon"),
            "to_city": event.get("ip_city", "Unknown"),
        }
