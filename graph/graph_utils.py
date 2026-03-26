"""
Sentinel-Graph Engine — Graph Utilities
Edge weight calculation and graph analysis helpers.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict


def compute_edge_weights(events_df, method="frequency_recency", decay=0.95):
    """
    Compute behavioral edge weights for User→Resource edges.
    
    Args:
        events_df: DataFrame of login events
        method: 'frequency_recency' or 'frequency_only'
        decay: exponential decay factor for recency weighting
    
    Returns:
        DataFrame with (user_id, resource_id, weight, access_count, last_access)
    """
    # Group by user-resource pairs
    grouped = events_df.groupby(["user_id", "resource_id"]).agg(
        access_count=("event_id", "count"),
        last_access=("timestamp", "max"),
        first_access=("timestamp", "min"),
    ).reset_index()

    if method == "frequency_only":
        # Normalize frequency to 0-1
        max_count = grouped["access_count"].max()
        grouped["weight"] = grouped["access_count"] / max_count
    else:
        # Frequency × Recency weighting
        max_count = grouped["access_count"].max()
        freq_norm = grouped["access_count"] / max_count

        # Recency: days since last access, decayed
        ref_date = pd.to_datetime(events_df["timestamp"]).max()
        days_since = (ref_date - pd.to_datetime(grouped["last_access"])).dt.days
        recency_weight = decay ** days_since

        grouped["weight"] = (freq_norm * 0.6 + recency_weight * 0.4).round(4)

    return grouped[["user_id", "resource_id", "weight", "access_count", "last_access"]]


def compute_user_access_profile(events_df, user_id):
    """Get a user's access profile: resources, times, devices."""
    user_events = events_df[events_df["user_id"] == user_id]

    if len(user_events) == 0:
        return None

    timestamps = pd.to_datetime(user_events["timestamp"])

    return {
        "user_id": user_id,
        "total_events": len(user_events),
        "unique_resources": user_events["resource_id"].nunique(),
        "unique_devices": user_events["device_fingerprint"].nunique(),
        "unique_cities": user_events["ip_city"].nunique(),
        "avg_hour": timestamps.dt.hour.mean(),
        "std_hour": timestamps.dt.hour.std(),
        "login_success_rate": user_events["login_success"].mean(),
        "most_common_city": user_events["ip_city"].mode().iloc[0] if len(user_events) > 0 else "Unknown",
        "resources_accessed": user_events["resource_id"].unique().tolist(),
    }


def find_stale_permissions(user_roles_df, permissions_df, events_df, days_threshold=90):
    """Find permissions not used in the last N days."""
    ref_date = pd.to_datetime(events_df["timestamp"]).max()
    cutoff = ref_date - pd.Timedelta(days=days_threshold)

    # Get all user-resource access after cutoff
    recent = events_df[pd.to_datetime(events_df["timestamp"]) >= cutoff]
    active_pairs = set(zip(recent["user_id"], recent["resource_id"]))

    # Get all user-resource permissions (through roles)
    stale = []
    for _, ur in user_roles_df.iterrows():
        uid = ur["user_id"]
        role_perms = permissions_df[permissions_df["role_id"] == ur["role_id"]]
        for _, perm in role_perms.iterrows():
            if (uid, perm["resource_id"]) not in active_pairs:
                stale.append({
                    "user_id": uid,
                    "role_id": ur["role_id"],
                    "resource_id": perm["resource_id"],
                    "action": perm["action"],
                    "days_unused": days_threshold,
                })

    return pd.DataFrame(stale)
