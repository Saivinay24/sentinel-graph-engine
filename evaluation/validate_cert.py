"""
Sentinel-Graph Engine — CERT Insider Threat Dataset r6.2 Validation
====================================================================
Validates Sentinel-Graph against the CERT Insider Threat benchmark by
comparing precision, recall, F1, and false-positive rate against a
rule-based baseline and a Random Forest baseline.

Usage
-----
    # Against real CERT data (place files in data/cert/):
    python evaluation/validate_cert.py

    # Demo mode (auto-generates mock CERT data):
    python evaluation/validate_cert.py --mock

    # Tune detection threshold:
    python evaluation/validate_cert.py --threshold 45
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Resolve project root regardless of CWD ────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CERT_DIR = PROJECT_ROOT / "data" / "cert"
OUTPUT_PATH = PROJECT_ROOT / "data" / "generated" / "cert_validation_results.json"

# ─────────────────────────────────────────────────────────────────────────────
# CERT schema → Sentinel-Graph event mapping
# ─────────────────────────────────────────────────────────────────────────────

def map_cert_row_to_event(row: pd.Series) -> dict:
    """Convert one CERT logon.csv row to a Sentinel-Graph event dict."""
    return {
        "user_id": row["user"],
        "resource_id": row["pc"],        # computer treated as resource
        "timestamp": row["date"],
        "device_id": row["pc"],
        "action": row["activity"],
        "ip_lat": 40.4,                  # Pittsburgh / CMU
        "ip_lon": -79.9,
        "ip_city": "Pittsburgh",
        "device_fingerprint": row["pc"],
        "device_registered": True,
        "os": "Windows",
        "browser": "Chrome/120",
        "login_success": str(row["activity"]).strip().lower() == "logon",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Mock CERT data generator (demo / CI mode)
# ─────────────────────────────────────────────────────────────────────────────

def generate_mock_cert_data(
    n_users: int = 120,
    n_events: int = 5_000,
    malicious_ratio: float = 0.08,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic dataset that mirrors CERT logon.csv schema so the
    full validation pipeline can run without the proprietary CERT files.
    """
    rng = np.random.default_rng(seed)
    n_malicious = max(1, int(n_users * malicious_ratio))
    users = [f"USER{i:04d}" for i in range(n_users)]
    malicious_users = set(users[:n_malicious])
    pcs = [f"PC{i:04d}" for i in range(30)]

    records = []
    base_dt = pd.Timestamp("2023-01-01")

    for _ in range(n_events):
        user = rng.choice(users)
        is_malicious_user = user in malicious_users

        # Malicious users log in more at odd hours
        if is_malicious_user and rng.random() < 0.55:
            hour = int(rng.choice([0, 1, 2, 3, 22, 23]))
        else:
            hour = int(rng.integers(7, 20))

        day_offset = int(rng.integers(0, 180))
        ts = base_dt + pd.Timedelta(days=day_offset, hours=hour,
                                    minutes=int(rng.integers(0, 60)))

        records.append({
            "id": f"EVT{len(records):07d}",
            "date": ts.isoformat(),
            "user": user,
            "pc": rng.choice(pcs),
            "activity": rng.choice(["Logon", "LogOff"], p=[0.55, 0.45]),
            "malicious": is_malicious_user,
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Baselines
# ─────────────────────────────────────────────────────────────────────────────

def rule_based_score(event: dict) -> float:
    """
    Heuristic baseline: penalise off-hours and weekend logins.

    Score range: 0–100 (higher = more suspicious).
    """
    ts = pd.to_datetime(event["timestamp"])
    hour = ts.hour
    dow = ts.dayofweek   # 0 = Monday … 6 = Sunday
    score = 0
    if hour < 8 or hour > 18:
        score += 50
    if dow >= 5:
        score += 30
    return float(min(100, score))


def _build_rf_features(events_df: pd.DataFrame) -> pd.DataFrame:
    """Extract temporal + device features for the RF baseline."""
    ts = pd.to_datetime(events_df["timestamp"])
    return pd.DataFrame({
        "hour": ts.dt.hour,
        "day_of_week": ts.dt.dayofweek,
        "is_logon": (events_df["action"].str.strip().str.lower() == "logon").astype(int),
        "device_registered": events_df.get("device_registered",
                                           pd.Series([True] * len(events_df))).astype(int),
        "login_success": events_df.get("login_success",
                                       pd.Series([True] * len(events_df))).astype(int),
    }, index=events_df.index).fillna(0)


def train_rf_baseline(events_df: pd.DataFrame, labels: pd.Series):
    """Train a Random Forest classifier on temporal + device features."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.utils import resample

    X = _build_rf_features(events_df)
    y = labels.astype(int)

    # Balance classes via over-sampling minority
    pos_idx = y[y == 1].index
    neg_idx = y[y == 0].index

    if len(pos_idx) == 0 or len(neg_idx) == 0:
        # Degenerate dataset — return a dummy model
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X, y)
        return clf

    n_pos = len(pos_idx)
    n_neg = len(neg_idx)
    target = min(n_neg, n_pos * 5)   # up to 5:1 ratio

    neg_sampled = resample(neg_idx, replace=False,
                           n_samples=target, random_state=42)
    keep_idx = pos_idx.tolist() + neg_sampled.tolist()

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X.loc[keep_idx], y.loc[keep_idx])
    return clf


# ─────────────────────────────────────────────────────────────────────────────
# Sentinel-Graph scorer (best-effort loader with graceful fallback)
# ─────────────────────────────────────────────────────────────────────────────

def load_sentinel_scorer():
    """
    Attempt to load the trained RiskScorer.  Returns (scorer, available).
    If the model artefacts are absent, returns (None, False) and the caller
    falls back to a heuristic approximation.
    """
    try:
        from scoring.risk_scorer import RiskScorer  # noqa: PLC0415
        scorer = RiskScorer()
        # Light smoke-test
        test_event = {
            "user_id": "TEST001",
            "resource_id": "PC0001",
            "timestamp": "2023-06-15T14:00:00",
            "device_id": "PC0001",
            "action": "Logon",
            "ip_lat": 40.4, "ip_lon": -79.9,
            "ip_city": "Pittsburgh",
            "device_fingerprint": "PC0001",
            "device_registered": True,
            "os": "Windows",
            "browser": "Chrome/120",
            "login_success": True,
        }
        result = scorer.score(test_event)
        # Expect a dict with at least a 'total' key
        if not isinstance(result, dict) or "total" not in result:
            raise ValueError("Unexpected RiskScorer output shape")
        return scorer, True
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] RiskScorer not available ({exc}); using heuristic proxy.")
        return None, False


def sentinel_heuristic_score(event: dict) -> float:
    """
    Heuristic proxy for Sentinel-Graph when the trained model is absent.
    Combines rule-based signals with a peer-deviation proxy (user vs. hour
    distribution deviation is simulated via a small logistic penalty).
    """
    base = rule_based_score(event)

    # Geo signal: Pittsburgh coordinates expected; any deviation adds risk
    lat_dev = abs(event.get("ip_lat", 40.4) - 40.4)
    lon_dev = abs(event.get("ip_lon", -79.9) - (-79.9))
    geo_penalty = min(20, (lat_dev + lon_dev) * 5)

    # Device signal
    device_penalty = 0 if event.get("device_registered", True) else 20

    # Composite (capped at 100)
    return float(min(100, base * 0.6 + geo_penalty + device_penalty))


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))

    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    fpr       = fp / max(fp + tn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-9)

    return {
        "precision":          round(precision * 100, 1),
        "recall_tpr":         round(recall    * 100, 1),
        "f1_score":           round(f1        * 100, 1),
        "false_positive_rate": round(fpr      * 100, 1),
        "true_positives":     tp,
        "false_positives":    fp,
        "false_negatives":    fn,
        "true_negatives":     tn,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print results table
# ─────────────────────────────────────────────────────────────────────────────

def print_results(dataset_info: dict, metrics: dict) -> None:
    W = 56  # inner width

    def row(label, sg, rb, rf):
        return (f"  {label:<18} {sg:>12}    {rb:>8}    {rf:>6}  ")

    print()
    print("╔" + "═" * W + "╗")
    print("║" + "   CERT Insider Threat Dataset r6.2 — Validation      " + "║")
    print("╠" + "═" * W + "╣")
    info = (
        f"  Dataset: {dataset_info['n_users']} users, "
        f"{dataset_info['n_events']} events, "
        f"{dataset_info['n_malicious_users']} malicious users"
    )
    print(f"║  {info:<{W - 2}}║")
    print("╠" + "═" * W + "╣")
    print(f"║  {'Metric':<18} {'Sentinel-Graph':>12}    {'Rule-Based':>8}    {'RF':>6}  ║")
    print("╠" + "═" * W + "╣")

    sg = metrics["sentinel_graph"]
    rb = metrics["rule_based"]
    rf = metrics["random_forest"]

    def fmt(v):
        return f"{v:.1f}%"

    print("║" + row("Precision",
                     fmt(sg["precision"]),
                     fmt(rb["precision"]),
                     fmt(rf["precision"])) + "║")
    print("║" + row("Recall (TPR)",
                     fmt(sg["recall_tpr"]),
                     fmt(rb["recall_tpr"]),
                     fmt(rf["recall_tpr"])) + "║")
    print("║" + row("F1 Score",
                     fmt(sg["f1_score"]),
                     fmt(rb["f1_score"]),
                     fmt(rf["f1_score"])) + "║")
    print("║" + row("False Pos Rate",
                     fmt(sg["false_positive_rate"]),
                     fmt(rb["false_positive_rate"]),
                     fmt(rf["false_positive_rate"])) + "║")
    print("╚" + "═" * W + "╝")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CERT data loader
# ─────────────────────────────────────────────────────────────────────────────

CERT_DOWNLOAD_INSTRUCTIONS = """
CERT Insider Threat Dataset r6.2 — Download Instructions
---------------------------------------------------------
1. Request access at: https://resources.sei.cmu.edu/library/asset-view.cfm?assetid=508099
2. Extract the archive and copy the following file into the project:
       sentinel-graph/data/cert/logon.csv
3. Expected columns: id, date, user, pc, activity, malicious

Run with --mock to use generated demo data instead.
"""


def load_cert_logon(cert_dir: Path) -> pd.DataFrame | None:
    """Return the CERT logon DataFrame, or None if the file is absent."""
    logon_path = cert_dir / "logon.csv"
    if not logon_path.exists():
        return None
    print(f"  Loading CERT logon data from {logon_path} …")
    df = pd.read_csv(logon_path)
    # Normalise column names (strip whitespace, lowercase)
    df.columns = [c.strip().lower() for c in df.columns]
    # Coerce 'malicious' to bool; handle string variants
    if "malicious" in df.columns:
        df["malicious"] = df["malicious"].astype(str).str.strip().str.lower().map(
            {"true": True, "1": True, "false": False, "0": False}
        ).fillna(False)
    else:
        df["malicious"] = False
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main validation routine
# ─────────────────────────────────────────────────────────────────────────────

def run_validation(threshold: int = 50, use_mock: bool = False) -> dict:
    print("\nSentinel-Graph — CERT Insider Threat Validation")
    print("=" * 55)

    # 1. Load or generate data ────────────────────────────────────────────
    cert_df = None
    if not use_mock:
        cert_df = load_cert_logon(CERT_DIR)

    if cert_df is None:
        if not use_mock:
            print(CERT_DOWNLOAD_INSTRUCTIONS)
            print("  Falling back to mock data for demonstration …\n")
        else:
            print("  Generating mock CERT data …")
        cert_df = generate_mock_cert_data()

    # 2. Build event list ─────────────────────────────────────────────────
    print(f"  Mapping {len(cert_df):,} rows to Sentinel-Graph event schema …")
    events = [map_cert_row_to_event(row) for _, row in cert_df.iterrows()]
    events_df = pd.DataFrame(events)
    events_df["timestamp"] = pd.to_datetime(events_df["timestamp"])

    # Ground truth: any event belonging to a malicious user is a positive
    malicious_users = set(cert_df.loc[cert_df["malicious"] == True, "user"])
    y_true = np.array([1 if e["user_id"] in malicious_users else 0
                       for e in events], dtype=int)

    dataset_info = {
        "n_users":          cert_df["user"].nunique(),
        "n_events":         len(cert_df),
        "n_malicious_users": len(malicious_users),
    }
    print(f"  {dataset_info['n_users']} users | "
          f"{dataset_info['n_events']:,} events | "
          f"{dataset_info['n_malicious_users']} malicious users")

    # 3. Rule-based baseline ──────────────────────────────────────────────
    print("\n  Running rule-based baseline …")
    rb_scores = np.array([rule_based_score(e) for e in events])
    rb_pred   = (rb_scores >= threshold).astype(int)
    rb_metrics = compute_metrics(y_true, rb_pred)

    # 4. Random Forest baseline ───────────────────────────────────────────
    print("  Training Random Forest baseline …")
    try:
        rf_clf = train_rf_baseline(events_df, pd.Series(y_true))
        X_rf   = _build_rf_features(events_df)
        rf_pred = rf_clf.predict(X_rf)
        rf_metrics = compute_metrics(y_true, rf_pred)
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] RF training failed ({exc}); using rule-based scores as proxy.")
        rf_pred = rb_pred.copy()
        rf_metrics = rb_metrics.copy()

    # 5. Sentinel-Graph scoring ───────────────────────────────────────────
    print("  Scoring events with Sentinel-Graph …")
    scorer, scorer_available = load_sentinel_scorer()

    sg_scores = []
    for event in events:
        if scorer_available:
            try:
                result = scorer.score(event)
                sg_scores.append(float(result["total"]))
            except Exception:  # noqa: BLE001
                sg_scores.append(sentinel_heuristic_score(event))
        else:
            sg_scores.append(sentinel_heuristic_score(event))

    sg_scores = np.array(sg_scores)
    sg_pred   = (sg_scores >= threshold).astype(int)
    sg_metrics = compute_metrics(y_true, sg_pred)

    # 6. Assemble results ─────────────────────────────────────────────────
    all_metrics = {
        "sentinel_graph": sg_metrics,
        "rule_based":     rb_metrics,
        "random_forest":  rf_metrics,
    }

    # 7. Pretty-print ─────────────────────────────────────────────────────
    print_results(dataset_info, all_metrics)

    # 8. Save JSON results ────────────────────────────────────────────────
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "dataset": dataset_info,
        "threshold": threshold,
        "sentinel_graph_model_loaded": scorer_available,
        "metrics": all_metrics,
        "per_event_sample": [
            {
                "user_id":            events[i]["user_id"],
                "timestamp":          str(events_df["timestamp"].iloc[i]),
                "ground_truth":       int(y_true[i]),
                "sentinel_score":     float(sg_scores[i]),
                "sentinel_flagged":   int(sg_pred[i]),
                "rule_based_score":   float(rb_scores[i]),
                "rule_based_flagged": int(rb_pred[i]),
            }
            for i in range(min(20, len(events)))
        ],
    }

    with open(OUTPUT_PATH, "w") as fh:
        json.dump(output, fh, indent=2)

    print(f"  Results saved to: {OUTPUT_PATH}")
    return output


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Validate Sentinel-Graph on the CERT Insider Threat Dataset r6.2"
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Generate synthetic CERT-like data instead of loading real files",
    )
    parser.add_argument(
        "--threshold", type=int, default=50,
        help="Risk score threshold for flagging events (default: 50)",
    )
    args = parser.parse_args()
    run_validation(threshold=args.threshold, use_mock=args.mock)


if __name__ == "__main__":
    main()
