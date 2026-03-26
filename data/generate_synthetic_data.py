"""
Sentinel-Graph Engine — Synthetic Identity Data Generator
Generates realistic enterprise IAM data: users, roles, resources, permissions, and login events.
"""

import os
import sys
import json
import random
import hashlib
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

def _load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

# ---------------------------------------------------------------------------
# Deterministic ID helpers
# ---------------------------------------------------------------------------
def _user_id(i):   return f"USR-{i:04d}"
def _role_id(i):   return f"ROLE-{i:03d}"
def _res_id(i):    return f"RES-{i:04d}"
def _perm_id(i):   return f"PERM-{i:04d}"
def _device_fp(user_id, idx):
    raw = f"{user_id}-device-{idx}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

# ---------------------------------------------------------------------------
# USERS
# ---------------------------------------------------------------------------
FIRST_NAMES = [
    "Aarav","Priya","Rahul","Sneha","Vikram","Ananya","Karan","Meera",
    "Rohan","Divya","Arjun","Neha","Siddharth","Pooja","Aditya","Kavya",
    "Nikhil","Riya","Varun","Shruti","Dev","Isha","Amit","Tanvi",
    "Raj","Simran","Kunal","Tara","Manish","Naina","Akash","Jaya",
    "Suresh","Lakshmi","Deepak","Preeti","Gaurav","Sakshi","Harsh","Pallavi",
    "James","Sarah","Michael","Emily","David","Jessica","Robert","Ashley",
    "William","Jennifer","Carlos","Maria","Chen","Wei","Yuki","Kenji",
    "Hans","Eva","Pierre","Claire","Ali","Fatima","Omar","Layla",
]

LAST_NAMES = [
    "Sharma","Patel","Kumar","Singh","Gupta","Reddy","Nair","Iyer",
    "Rao","Mehta","Joshi","Shah","Agarwal","Mishra","Das","Verma",
    "Malhotra","Chakraborty","Bhat","Pillai","Smith","Johnson","Williams",
    "Brown","Jones","Garcia","Miller","Davis","Rodriguez","Martinez",
    "Anderson","Taylor","Thomas","Hernandez","Moore","Martin","Lee","Chen",
    "Wang","Kim","Nakamura","Tanaka","Mueller","Fischer","Dupont","Bernard",
]

# Role templates per department
ROLE_TEMPLATES = {
    "Engineering":    ["Backend Developer","Frontend Developer","DevOps Engineer","QA Engineer","Data Engineer","ML Engineer","Platform Engineer"],
    "Finance":        ["Financial Analyst","Accountant","Treasury Manager","Compliance Officer"],
    "Human Resources":["HR Generalist","Recruiter","Compensation Analyst","HR Business Partner"],
    "Marketing":      ["Content Strategist","SEO Specialist","Campaign Manager","Brand Analyst"],
    "Sales":          ["Account Executive","Sales Engineer","Business Development","Customer Success"],
    "IT Operations":  ["Sys Admin","Network Engineer","DBA","Cloud Architect","Security Analyst"],
    "Legal":          ["Corporate Counsel","Contract Manager","IP Specialist"],
    "Executive":      ["CEO","CTO","CFO","CISO","VP Engineering","VP Sales"],
}

# Resource templates
RESOURCE_TEMPLATES = [
    # (name_prefix, sensitivity, data_type, count)
    ("SourceCode-Repo",       "Medium", "code",      15),
    ("CI-CD-Pipeline",        "Medium", "infra",     5),
    ("Production-DB",         "Critical","data",     8),
    ("Staging-DB",            "Medium", "data",      5),
    ("HR-Records",            "Critical","pii",      4),
    ("Financial-Ledger",      "Critical","financial", 6),
    ("Marketing-Dashboard",   "Low",    "analytics", 4),
    ("Customer-CRM",          "High",   "pii",       6),
    ("Internal-Wiki",         "Low",    "docs",      3),
    ("Admin-Console",         "Critical","admin",     4),
    ("Email-Server",          "Medium", "comms",     2),
    ("VPN-Gateway",           "High",   "network",   3),
    ("Cloud-Console-AWS",     "Critical","infra",     5),
    ("Cloud-Console-GCP",     "Critical","infra",     3),
    ("Monitoring-Grafana",    "Medium", "monitoring", 4),
    ("Log-Aggregator-ELK",    "Medium", "monitoring", 3),
    ("Salary-Portal",         "Critical","financial", 2),
    ("Leave-Management",      "Low",    "hr",        2),
    ("Expense-Reports",       "Medium", "financial", 3),
    ("Board-Reports",         "Critical","executive", 3),
    ("API-Gateway",           "High",   "infra",     4),
    ("OAuth-Server",          "Critical","auth",      2),
    ("Secrets-Manager",       "Critical","security",  2),
    ("Backup-Storage",        "High",   "data",      3),
]

BROWSERS = ["Chrome/121","Firefox/122","Safari/17","Edge/121","Chrome/120","Firefox/121"]
OS_LIST  = ["Windows 11","macOS 14","Ubuntu 22.04","Windows 10","macOS 13","iOS 17","Android 14"]
DEVICE_TYPES = ["Laptop","Desktop","Mobile","Tablet"]

# ---------------------------------------------------------------------------
# Generator functions
# ---------------------------------------------------------------------------

def generate_users(config):
    """Generate synthetic users across departments."""
    cfg = config["data"]
    n = cfg["num_users"]
    departments = cfg["departments"]
    seniority_levels = cfg["seniority_levels"]
    offices = cfg["offices"]

    rng = np.random.default_rng(42)
    users = []

    # Assign rough department sizes (weighted)
    dept_weights = {
        "Engineering": 0.30, "Finance": 0.10, "Human Resources": 0.08,
        "Marketing": 0.10, "Sales": 0.12, "IT Operations": 0.15,
        "Legal": 0.05, "Executive": 0.02,
    }
    # Normalise weights for available departments
    total = sum(dept_weights.get(d, 0.1) for d in departments)
    dept_probs = [dept_weights.get(d, 0.1)/total for d in departments]

    for i in range(n):
        dept = rng.choice(departments, p=dept_probs)
        seniority = rng.choice(seniority_levels, p=[0.35, 0.30, 0.20, 0.10, 0.05])
        office = offices[rng.integers(0, len(offices))]
        hire_date = datetime(2018, 1, 1) + timedelta(days=int(rng.integers(0, 2500)))

        users.append({
            "user_id": _user_id(i),
            "first_name": random.choice(FIRST_NAMES),
            "last_name": random.choice(LAST_NAMES),
            "department": dept,
            "seniority": seniority,
            "office_city": office["city"],
            "office_lat": office["lat"],
            "office_lon": office["lon"],
            "hire_date": hire_date.strftime("%Y-%m-%d"),
            "is_active": True,
        })

    return pd.DataFrame(users)


def generate_roles(config):
    """Generate roles with permission bundles per department."""
    roles = []
    idx = 0
    for dept, titles in ROLE_TEMPLATES.items():
        for title in titles:
            roles.append({
                "role_id": _role_id(idx),
                "role_name": title,
                "department": dept,
                "seniority_level": "Mid",  # default, overridden during assignment
            })
            idx += 1
    return pd.DataFrame(roles)


def generate_resources(config):
    """Generate enterprise resources with sensitivity levels."""
    resources = []
    idx = 0
    for prefix, sensitivity, data_type, count in RESOURCE_TEMPLATES:
        for j in range(count):
            resources.append({
                "resource_id": _res_id(idx),
                "resource_name": f"{prefix}-{j+1:02d}",
                "sensitivity": sensitivity,
                "data_type": data_type,
            })
            idx += 1
    return pd.DataFrame(resources)


def generate_permissions(roles_df, resources_df, config):
    """Generate role→permission→resource mappings."""
    rng = np.random.default_rng(42)
    permissions = []
    perm_idx = 0

    # Department→resource type affinity
    dept_resource_affinity = {
        "Engineering":     ["code","infra","data","monitoring"],
        "Finance":         ["financial","analytics","data"],
        "Human Resources": ["pii","hr","docs"],
        "Marketing":       ["analytics","docs","comms"],
        "Sales":           ["pii","analytics","comms"],
        "IT Operations":   ["infra","network","monitoring","admin","security","auth","data"],
        "Legal":           ["docs","pii","financial"],
        "Executive":       ["executive","financial","analytics","admin","data"],
    }

    for _, role in roles_df.iterrows():
        dept = role["department"]
        affinities = dept_resource_affinity.get(dept, ["docs"])

        # Filter resources by affinity
        relevant = resources_df[resources_df["data_type"].isin(affinities)]
        # Each role gets access to 20-60% of relevant resources
        n_access = max(2, int(len(relevant) * rng.uniform(0.2, 0.6)))
        selected = relevant.sample(n=min(n_access, len(relevant)), random_state=int(rng.integers(0,9999)))

        for _, res in selected.iterrows():
            action = rng.choice(["read","write","admin"], p=[0.50, 0.35, 0.15])
            permissions.append({
                "permission_id": _perm_id(perm_idx),
                "role_id": role["role_id"],
                "resource_id": res["resource_id"],
                "action": action,
            })
            perm_idx += 1

    return pd.DataFrame(permissions)


def assign_roles_to_users(users_df, roles_df, config):
    """Assign users to roles based on their department."""
    rng = np.random.default_rng(42)
    assignments = []

    for _, user in users_df.iterrows():
        dept = user["department"]
        dept_roles = roles_df[roles_df["department"] == dept]
        if len(dept_roles) == 0:
            continue
        # Each user gets 1-3 roles from their department
        n_roles = min(len(dept_roles), rng.integers(1, 4))
        selected = dept_roles.sample(n=n_roles, random_state=int(rng.integers(0,9999)))
        for _, role in selected.iterrows():
            assignments.append({
                "user_id": user["user_id"],
                "role_id": role["role_id"],
            })

    return pd.DataFrame(assignments)


def generate_user_devices(users_df, config):
    """Generate 1-3 known devices per user."""
    rng = np.random.default_rng(42)
    devices = []

    for _, user in users_df.iterrows():
        n_devices = rng.integers(1, 4)
        for d in range(n_devices):
            devices.append({
                "user_id": user["user_id"],
                "device_fingerprint": _device_fp(user["user_id"], d),
                "device_type": random.choice(DEVICE_TYPES),
                "os": random.choice(OS_LIST),
                "browser": random.choice(BROWSERS),
                "is_registered": True,
                "registered_date": (datetime(2020,1,1) + timedelta(days=int(rng.integers(0,1800)))).strftime("%Y-%m-%d"),
            })

    return pd.DataFrame(devices)


def generate_login_events(users_df, resources_df, permissions_df, 
                           user_roles_df, devices_df, config):
    """Generate realistic login/access events (normal behavior only)."""
    cfg = config["data"]
    n_events = cfg["num_events"]
    offices = cfg["offices"]
    rng = np.random.default_rng(42)

    # Pre-compute user→permitted resources mapping
    user_perms = {}
    for _, ur in user_roles_df.iterrows():
        uid = ur["user_id"]
        rid = ur["role_id"]
        role_perms = permissions_df[permissions_df["role_id"] == rid]
        if uid not in user_perms:
            user_perms[uid] = set()
        user_perms[uid].update(role_perms["resource_id"].tolist())

    # Pre-compute user→devices
    user_devices = {}
    for _, dev in devices_df.iterrows():
        uid = dev["user_id"]
        if uid not in user_devices:
            user_devices[uid] = []
        user_devices[uid].append(dev.to_dict())

    events = []
    base_date = datetime(2025, 11, 1)
    user_ids = users_df["user_id"].tolist()
    user_lookup = users_df.set_index("user_id").to_dict("index")

    for i in range(n_events):
        uid = rng.choice(user_ids)
        user = user_lookup[uid]

        # Time: mostly business hours with some variation
        day_offset = int(rng.integers(0, 90))
        if rng.random() < 0.85:  # 85% during business hours
            hour = int(rng.normal(13, 3))  # centered around 1 PM
            hour = max(8, min(18, hour))
        else:
            hour = int(rng.integers(0, 24))
        minute = int(rng.integers(0, 60))
        second = int(rng.integers(0, 60))
        ts = base_date + timedelta(days=day_offset, hours=hour, minutes=minute, seconds=second)

        # Location: 90% from office, 10% from known remote locations
        if rng.random() < 0.90:
            lat = user["office_lat"] + rng.normal(0, 0.01)
            lon = user["office_lon"] + rng.normal(0, 0.01)
            city = user["office_city"]
        else:
            remote_office = offices[rng.integers(0, len(offices))]
            lat = remote_office["lat"] + rng.normal(0, 0.5)
            lon = remote_office["lon"] + rng.normal(0, 0.5)
            city = remote_office["city"]

        # Resource: pick from permitted resources
        permitted = list(user_perms.get(uid, set()))
        if not permitted:
            continue
        resource_id = rng.choice(permitted)

        # Device: use a registered device
        devs = user_devices.get(uid, [])
        if devs:
            device = rng.choice(devs)
            device_fp = device["device_fingerprint"]
            device_type = device["device_type"]
            browser = device["browser"]
            os_name = device["os"]
            is_registered = True
        else:
            device_fp = hashlib.sha256(f"{uid}-unknown".encode()).hexdigest()[:16]
            device_type = "Unknown"
            browser = rng.choice(BROWSERS)
            os_name = rng.choice(OS_LIST)
            is_registered = False

        # Login success: 97% success for normal events
        success = rng.random() < 0.97

        events.append({
            "event_id": f"EVT-{i:06d}",
            "timestamp": ts.isoformat(),
            "user_id": uid,
            "resource_id": resource_id,
            "ip_lat": round(lat, 4),
            "ip_lon": round(lon, 4),
            "ip_city": city,
            "device_fingerprint": device_fp,
            "device_type": device_type,
            "browser": browser,
            "os": os_name,
            "device_registered": is_registered,
            "login_success": success,
            "is_attack": False,
            "attack_type": "none",
        })

    return pd.DataFrame(events)


# ---------------------------------------------------------------------------
# Master generator
# ---------------------------------------------------------------------------

def generate_all(output_dir=None):
    """Generate the complete synthetic dataset and optionally save to disk."""
    config = _load_config()

    print("🔄 Generating users...")
    users_df = generate_users(config)
    print(f"   ✅ {len(users_df)} users")

    print("🔄 Generating roles...")
    roles_df = generate_roles(config)
    print(f"   ✅ {len(roles_df)} roles")

    print("🔄 Generating resources...")
    resources_df = generate_resources(config)
    print(f"   ✅ {len(resources_df)} resources")

    print("🔄 Generating permissions...")
    permissions_df = generate_permissions(roles_df, resources_df, config)
    print(f"   ✅ {len(permissions_df)} permission mappings")

    print("🔄 Assigning roles to users...")
    user_roles_df = assign_roles_to_users(users_df, roles_df, config)
    print(f"   ✅ {len(user_roles_df)} role assignments")

    print("🔄 Generating device registry...")
    devices_df = generate_user_devices(users_df, config)
    print(f"   ✅ {len(devices_df)} registered devices")

    print("🔄 Generating login events...")
    events_df = generate_login_events(
        users_df, resources_df, permissions_df,
        user_roles_df, devices_df, config
    )
    print(f"   ✅ {len(events_df)} login events")

    data = {
        "users": users_df,
        "roles": roles_df,
        "resources": resources_df,
        "permissions": permissions_df,
        "user_roles": user_roles_df,
        "devices": devices_df,
        "events": events_df,
    }

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name, df in data.items():
            path = out / f"{name}.csv"
            df.to_csv(path, index=False)
            print(f"   💾 Saved {path}")

    return data


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent / "generated"
    generate_all(output_dir=out_dir)
    print("\n✅ Synthetic data generation complete!")
