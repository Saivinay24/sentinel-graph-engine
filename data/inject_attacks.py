"""
Sentinel-Graph Engine — Attack Pattern Injector
Injects 5 realistic attack patterns into the synthetic login event dataset.
"""

import hashlib
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def _load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Attack Pattern Generators
# ---------------------------------------------------------------------------

def _impossible_travel(users_df, resources_df, base_events_df, rng, n=50):
    """User logs in from city A, then city B impossibly fast."""
    events = []
    user_ids = users_df["user_id"].tolist()
    city_coords = [
        ("Mumbai",    19.0760,  72.8777),
        ("London",    51.5074,  -0.1278),
        ("New York",  40.7128, -74.0060),
        ("São Paulo",-23.5505, -46.6333),
        ("Singapore",  1.3521, 103.8198),
        ("Tokyo",     35.6762, 139.6503),
        ("Sydney",   -33.8688, 151.2093),
    ]

    for i in range(n):
        uid = rng.choice(user_ids)
        ts = datetime(2025, 11, 15) + timedelta(
            days=int(rng.integers(0, 60)),
            hours=int(rng.integers(8, 20)),
            minutes=int(rng.integers(0, 60)),
        )
        # First login from home city
        c1 = city_coords[rng.integers(0, len(city_coords))]
        # Second login from distant city just 5-15 minutes later
        c2_choices = [c for c in city_coords if c[0] != c1[0]]
        c2 = c2_choices[rng.integers(0, len(c2_choices))]
        gap_minutes = int(rng.integers(5, 15))

        res_id = rng.choice(resources_df["resource_id"].tolist())
        base_evt = {
            "user_id": uid,
            "resource_id": res_id,
            "device_fingerprint": hashlib.sha256(f"{uid}-attack-{i}".encode()).hexdigest()[:16],
            "device_type": "Laptop",
            "browser": "Chrome/121",
            "os": "Windows 11",
            "device_registered": False,
            "login_success": True,
            "is_attack": True,
            "attack_type": "impossible_travel",
        }
        events.append({
            **base_evt,
            "event_id": f"ATK-IT-{i:04d}-A",
            "timestamp": ts.isoformat(),
            "ip_lat": c1[1] + rng.normal(0, 0.01),
            "ip_lon": c1[2] + rng.normal(0, 0.01),
            "ip_city": c1[0],
        })
        events.append({
            **base_evt,
            "event_id": f"ATK-IT-{i:04d}-B",
            "timestamp": (ts + timedelta(minutes=gap_minutes)).isoformat(),
            "ip_lat": c2[1] + rng.normal(0, 0.01),
            "ip_lon": c2[2] + rng.normal(0, 0.01),
            "ip_city": c2[0],
        })

    return pd.DataFrame(events)


def _privilege_escalation(users_df, resources_df, rng, n=40):
    """Junior user from one dept suddenly accesses critical resources of another dept."""
    events = []
    # Find critical resources
    critical = resources_df[resources_df["sensitivity"] == "Critical"]

    for i in range(n):
        uid_idx = rng.integers(0, len(users_df))
        user = users_df.iloc[uid_idx]
        ts = datetime(2025, 11, 20) + timedelta(
            days=int(rng.integers(0, 40)),
            hours=int(rng.integers(0, 6)),  # off hours
            minutes=int(rng.integers(0, 60)),
        )
        res = critical.iloc[rng.integers(0, len(critical))]

        events.append({
            "event_id": f"ATK-PE-{i:04d}",
            "timestamp": ts.isoformat(),
            "user_id": user["user_id"],
            "resource_id": res["resource_id"],
            "ip_lat": user["office_lat"] + rng.normal(0, 0.01),
            "ip_lon": user["office_lon"] + rng.normal(0, 0.01),
            "ip_city": user["office_city"],
            "device_fingerprint": hashlib.sha256(f"{user['user_id']}-priv-{i}".encode()).hexdigest()[:16],
            "device_type": "Laptop",
            "browser": "Chrome/121",
            "os": "Windows 11",
            "device_registered": True,
            "login_success": True,
            "is_attack": True,
            "attack_type": "privilege_escalation",
        })

    return pd.DataFrame(events)


def _credential_stuffing(users_df, resources_df, rng, n=30):
    """Many failed logins followed by one success from a new device."""
    events = []

    for i in range(n):
        uid = users_df.iloc[rng.integers(0, len(users_df))]["user_id"]
        ts = datetime(2025, 12, 1) + timedelta(
            days=int(rng.integers(0, 30)),
            hours=int(rng.integers(1, 5)),  # very early morning
        )
        res_id = rng.choice(resources_df["resource_id"].tolist())
        fake_fp = hashlib.sha256(f"botnet-{i}-{uid}".encode()).hexdigest()[:16]

        # 10-30 failed attempts
        n_failed = int(rng.integers(10, 30))
        for f in range(n_failed):
            events.append({
                "event_id": f"ATK-CS-{i:04d}-F{f:02d}",
                "timestamp": (ts + timedelta(seconds=f*3)).isoformat(),
                "user_id": uid,
                "resource_id": res_id,
                "ip_lat": 55.7558 + rng.normal(0, 0.1),  # Moscow area
                "ip_lon": 37.6173 + rng.normal(0, 0.1),
                "ip_city": "Moscow",
                "device_fingerprint": fake_fp,
                "device_type": "Desktop",
                "browser": rng.choice(["Chrome/120","Firefox/121"]),
                "os": "Linux",
                "device_registered": False,
                "login_success": False,
                "is_attack": True,
                "attack_type": "credential_stuffing",
            })
        # One success
        events.append({
            "event_id": f"ATK-CS-{i:04d}-S",
            "timestamp": (ts + timedelta(seconds=n_failed*3 + 5)).isoformat(),
            "user_id": uid,
            "resource_id": res_id,
            "ip_lat": 55.7558 + rng.normal(0, 0.1),
            "ip_lon": 37.6173 + rng.normal(0, 0.1),
            "ip_city": "Moscow",
            "device_fingerprint": fake_fp,
            "device_type": "Desktop",
            "browser": "Chrome/120",
            "os": "Linux",
            "device_registered": False,
            "login_success": True,
            "is_attack": True,
            "attack_type": "credential_stuffing",
        })

    return pd.DataFrame(events)


def _insider_lateral_movement(users_df, resources_df, rng, n=35):
    """HR user browsing financial databases on weekends."""
    events = []
    hr_users = users_df[users_df["department"] == "Human Resources"]
    finance_resources = resources_df[resources_df["data_type"].isin(["financial","executive"])]

    if len(hr_users) == 0 or len(finance_resources) == 0:
        return pd.DataFrame()

    for i in range(n):
        user = hr_users.iloc[rng.integers(0, len(hr_users))]
        res = finance_resources.iloc[rng.integers(0, len(finance_resources))]
        # Weekend
        base = datetime(2025, 11, 22)  # Saturday
        week_offset = int(rng.integers(0, 8))
        day_offset = int(rng.choice([0, 1]))  # Saturday or Sunday
        hour = int(rng.integers(22, 24)) if rng.random() < 0.5 else int(rng.integers(0, 5))
        ts = base + timedelta(weeks=week_offset, days=day_offset, hours=hour,
                              minutes=int(rng.integers(0, 60)))

        events.append({
            "event_id": f"ATK-IL-{i:04d}",
            "timestamp": ts.isoformat(),
            "user_id": user["user_id"],
            "resource_id": res["resource_id"],
            "ip_lat": user["office_lat"] + rng.normal(0, 0.5),
            "ip_lon": user["office_lon"] + rng.normal(0, 0.5),
            "ip_city": user["office_city"],
            "device_fingerprint": hashlib.sha256(f"{user['user_id']}-insider-{i}".encode()).hexdigest()[:16],
            "device_type": "Laptop",
            "browser": "Chrome/121",
            "os": "macOS 14",
            "device_registered": True,
            "login_success": True,
            "is_attack": True,
            "attack_type": "insider_lateral_movement",
        })

    return pd.DataFrame(events)


def _account_takeover(users_df, resources_df, rng, n=25):
    """Login from never-seen device + new location + off-hours — everything new."""
    events = []
    exotic_locs = [
        ("Lagos",    6.5244,  3.3792),
        ("Jakarta", -6.2088, 106.8456),
        ("Cairo",   30.0444,  31.2357),
        ("Lima",   -12.0464, -77.0428),
        ("Nairobi", -1.2921,  36.8219),
    ]

    for i in range(n):
        user = users_df.iloc[rng.integers(0, len(users_df))]
        res = resources_df.iloc[rng.integers(0, len(resources_df))]
        loc = exotic_locs[rng.integers(0, len(exotic_locs))]
        ts = datetime(2025, 12, 5) + timedelta(
            days=int(rng.integers(0, 20)),
            hours=int(rng.integers(2, 5)),
        )
        events.append({
            "event_id": f"ATK-AO-{i:04d}",
            "timestamp": ts.isoformat(),
            "user_id": user["user_id"],
            "resource_id": res["resource_id"],
            "ip_lat": loc[1] + rng.normal(0, 0.1),
            "ip_lon": loc[2] + rng.normal(0, 0.1),
            "ip_city": loc[0],
            "device_fingerprint": hashlib.sha256(f"takeover-{i}".encode()).hexdigest()[:16],
            "device_type": rng.choice(["Mobile", "Tablet"]),
            "browser": "Chrome/118",
            "os": rng.choice(["Android 14", "iOS 17"]),
            "device_registered": False,
            "login_success": True,
            "is_attack": True,
            "attack_type": "account_takeover",
        })

    return pd.DataFrame(events)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def inject_attacks(events_df, users_df, resources_df):
    """Inject all 5 attack patterns into the events DataFrame."""
    rng = np.random.default_rng(123)

    print("🔄 Injecting attack patterns...")

    attack_dfs = [
        ("impossible_travel",       _impossible_travel(users_df, resources_df, events_df, rng)),
        ("privilege_escalation",    _privilege_escalation(users_df, resources_df, rng)),
        ("credential_stuffing",     _credential_stuffing(users_df, resources_df, rng)),
        ("insider_lateral_movement",_insider_lateral_movement(users_df, resources_df, rng)),
        ("account_takeover",        _account_takeover(users_df, resources_df, rng)),
    ]

    for name, df in attack_dfs:
        print(f"   💉 {name}: {len(df)} events")

    all_attacks = pd.concat([df for _, df in attack_dfs], ignore_index=True)
    combined = pd.concat([events_df, all_attacks], ignore_index=True)
    combined = combined.sort_values("timestamp").reset_index(drop=True)

    n_attacks = combined["is_attack"].sum()
    print(f"   ✅ Total: {len(combined)} events ({n_attacks} attacks, {n_attacks/len(combined)*100:.1f}%)")

    return combined


if __name__ == "__main__":
    from generate_synthetic_data import generate_all

    data = generate_all()
    events_with_attacks = inject_attacks(
        data["events"], data["users"], data["resources"]
    )

    out_path = Path(__file__).resolve().parent / "generated" / "events_with_attacks.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    events_with_attacks.to_csv(out_path, index=False)
    print(f"   💾 Saved {out_path}")
