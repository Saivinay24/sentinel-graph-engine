"""
Sentinel-Graph API — Model & Data Loader
Loads all trained models and data at application startup with graceful fallbacks.
Exposes a global `engine` object consumed by all API endpoints.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so scoring.* imports work from any cwd
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scoring.decision_engine import DecisionEngine  # noqa: E402
from scoring.device_trust import DeviceTrustScorer  # noqa: E402
from scoring.geo_velocity import GeoVelocityAnalyzer  # noqa: E402
from scoring.peer_deviation import PeerDeviationScorer  # noqa: E402
from scoring.risk_scorer import RiskScorer  # noqa: E402
from scoring.temporal_analyzer import TemporalAnalyzer  # noqa: E402

logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "generated"

# ---------------------------------------------------------------------------
# Community label mapping (community_N → human-readable label)
# The label is derived at load-time from the dominant department of each community.
# These are the canonical fallback names for mock/new communities.
# ---------------------------------------------------------------------------
COMMUNITY_LABEL_MAP: Dict[str, str] = {
    "community_0": "Engineering",
    "community_1": "HR Operations",
    "community_2": "Finance",
    "community_3": "IT Operations",
    "community_4": "Sales & Marketing",
    "community_5": "Legal & Compliance",
    "community_6": "Executive",
    "community_7": "Data & Analytics",
    "community_8": "DevSecOps",
    "community_9": "Product Management",
}

# Fallback: integer community labels
_DEPT_CYCLE = [
    "Engineering", "HR Operations", "Finance", "IT Operations",
    "Sales & Marketing", "Legal & Compliance", "Executive", "Data & Analytics",
    "DevSecOps", "Product Management",
]

# ---------------------------------------------------------------------------
# Engine dataclass
# ---------------------------------------------------------------------------

@dataclass
class SentinelEngine:
    """Container for all loaded models, scorers, and datasets."""

    # Core scorers
    risk_scorer: Optional[RiskScorer] = None
    decision_engine: Optional[DecisionEngine] = None
    temporal: Optional[TemporalAnalyzer] = None
    geo: Optional[GeoVelocityAnalyzer] = None
    device: Optional[DeviceTrustScorer] = None
    peer: Optional[PeerDeviationScorer] = None

    # Loaded data
    users_df: Optional[pd.DataFrame] = None
    events_df: Optional[pd.DataFrame] = None
    scored_events_df: Optional[pd.DataFrame] = None
    embeddings: Optional[np.ndarray] = None

    # Community metadata
    communities: Dict[str, str] = field(default_factory=dict)     # user_id → community_label
    centroids: Dict[str, np.ndarray] = field(default_factory=dict)  # community_label → centroid
    community_labels: Dict[str, str] = field(default_factory=dict)  # community_id → human label

    # Status flags
    model_loaded: bool = False
    startup_time: float = field(default_factory=time.time)

    # ---------------------------------------------------------------------------
    # Convenience accessors
    # ---------------------------------------------------------------------------

    @property
    def users_loaded(self) -> int:
        return len(self.users_df) if self.users_df is not None else 0

    def get_community_label(self, comm_id: str) -> str:
        """Resolve a community_id to its human-readable label."""
        return self.community_labels.get(comm_id, COMMUNITY_LABEL_MAP.get(comm_id, comm_id))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _community_label_from_id(comm_id: Any) -> str:
    """Convert a raw community ID (int or string) to a canonical string label."""
    if isinstance(comm_id, int):
        return f"community_{comm_id}"
    s = str(comm_id)
    if not s.startswith("community_"):
        return f"community_{s}"
    return s


def _derive_community_labels(
    communities: Dict[str, str],
    users_df: Optional[pd.DataFrame],
) -> Dict[str, str]:
    """
    Build human-readable community labels from dominant department per community.
    Falls back to COMMUNITY_LABEL_MAP if user data is unavailable.
    """
    labels: Dict[str, str] = {}

    if users_df is not None and "department" in users_df.columns and "user_id" in users_df.columns:
        dept_map: Dict[str, str] = dict(zip(users_df["user_id"], users_df["department"]))
        comm_dept_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for uid, comm_id in communities.items():
            dept = dept_map.get(uid, "Unknown")
            comm_dept_counts[comm_id][dept] += 1

        for comm_id, dept_counts in comm_dept_counts.items():
            dominant = max(dept_counts, key=dept_counts.get)
            # Try static map first (for consistency across runs)
            labels[comm_id] = COMMUNITY_LABEL_MAP.get(comm_id, dominant)
    else:
        for comm_id in set(communities.values()):
            labels[comm_id] = COMMUNITY_LABEL_MAP.get(comm_id, comm_id)

    return labels


def _make_mock_communities(n_users: int = 500, n_communities: int = 8) -> Dict[str, str]:
    """Generate deterministic mock community assignments."""
    rng = np.random.default_rng(42)
    return {
        f"U{i:03d}": f"community_{rng.integers(0, n_communities)}"
        for i in range(n_users)
    }


def _make_mock_centroids(communities: Dict[str, str], dim: int = 64) -> Dict[str, np.ndarray]:
    """Generate deterministic mock centroids for each community."""
    community_ids = list(set(communities.values()))
    centroids: Dict[str, np.ndarray] = {}
    for comm_id in community_ids:
        seed = int(comm_id.split("_")[-1]) if "_" in comm_id else hash(comm_id) % 10000
        rng = np.random.default_rng(seed)
        centroids[comm_id] = rng.normal(0, 1, dim).astype(np.float32)
    return centroids


def _make_mock_scored_events(n: int = 500) -> pd.DataFrame:
    """
    Generate mock scored events for demo mode (when scored_events.csv missing).
    Produces realistic-looking data for /live-events, /risk-history, and /metrics.
    """
    rng = np.random.default_rng(42)
    n_users = 50
    cities = ["New York", "London", "Bangalore", "Singapore", "Beijing", "São Paulo", "Mumbai"]
    resources = [
        "finance_db_prod", "hr_portal", "code_repo", "analytics_db",
        "email_server", "vpn_gateway", "admin_console", "legal_docs",
    ]
    decisions = ["ALLOW", "ALLOW", "ALLOW", "STEP_UP_MFA", "STEP_UP_MFA", "BLOCK_ALERT", "BLOCK_SOC"]
    attack_types = ["credential_stuffing", "lateral_movement", "data_exfiltration", "none"]

    rows = []
    base_ts = pd.Timestamp("2026-03-01T08:00:00Z")
    for i in range(n):
        decision = decisions[rng.integers(0, len(decisions))]
        score = {
            "ALLOW": rng.integers(0, 31),
            "STEP_UP_MFA": rng.integers(31, 61),
            "BLOCK_ALERT": rng.integers(61, 86),
            "BLOCK_SOC": rng.integers(86, 101),
        }[decision]
        is_attack = decision in ("BLOCK_ALERT", "BLOCK_SOC") and rng.random() < 0.6
        rows.append({
            "user_id": f"U{rng.integers(0, n_users):03d}",
            "resource_id": resources[rng.integers(0, len(resources))],
            "timestamp": (base_ts + pd.Timedelta(hours=int(i * 0.5))).isoformat(),
            "risk_score": float(score),
            "decision": decision,
            "attack_type": attack_types[rng.integers(0, 3)] if is_attack else "none",
            "ip_city": cities[rng.integers(0, len(cities))],
            "action": rng.choice(["READ", "WRITE", "DELETE", "ADMIN"]),
            "login_success": bool(rng.random() > 0.1),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main loader function
# ---------------------------------------------------------------------------

def load_engine() -> SentinelEngine:
    """
    Load all models, embeddings, and data files at startup.
    Applies graceful fallbacks for any missing artefacts.

    Returns:
        A fully-initialised SentinelEngine instance.
    """
    t0 = time.time()
    eng = SentinelEngine(startup_time=t0)

    logger.info("Sentinel-Graph: loading engine from %s", DATA_DIR)

    # ------------------------------------------------------------------
    # 1. Users dataset
    # ------------------------------------------------------------------
    users_path = DATA_DIR / "users.csv"
    if users_path.exists():
        try:
            eng.users_df = pd.read_csv(users_path)
            logger.info("  Loaded %d users from %s", len(eng.users_df), users_path)
        except Exception as exc:
            logger.warning("  Failed to load users.csv: %s — using empty frame", exc)
            eng.users_df = pd.DataFrame(columns=["user_id", "department", "role_id",
                                                   "office_lat", "office_lon", "office_city"])
    else:
        logger.warning("  users.csv not found at %s — using empty frame (run data pipeline first)", users_path)
        eng.users_df = pd.DataFrame(columns=["user_id", "department", "role_id",
                                               "office_lat", "office_lon", "office_city"])

    # ------------------------------------------------------------------
    # 2. Events dataset
    # ------------------------------------------------------------------
    events_path = DATA_DIR / "events_with_attacks.csv"
    if events_path.exists():
        try:
            # Load in chunks if very large (50K rows is fine for pandas)
            eng.events_df = pd.read_csv(events_path, low_memory=False)
            logger.info("  Loaded %d events from %s", len(eng.events_df), events_path)
        except Exception as exc:
            logger.warning("  Failed to load events_with_attacks.csv: %s — using empty frame", exc)
            eng.events_df = pd.DataFrame()
    else:
        logger.warning("  events_with_attacks.csv not found — scorers will use empty history")
        eng.events_df = pd.DataFrame()

    # ------------------------------------------------------------------
    # 3. Embeddings
    # ------------------------------------------------------------------
    embeddings_path = DATA_DIR / "embeddings.npy"
    if embeddings_path.exists():
        try:
            eng.embeddings = np.load(str(embeddings_path))
            logger.info("  Loaded embeddings: shape %s", eng.embeddings.shape)
        except Exception as exc:
            logger.warning("  Failed to load embeddings.npy: %s — using random embeddings", exc)
            eng.embeddings = None
    else:
        logger.warning("  embeddings.npy not found — using random embeddings for demo")
        eng.embeddings = None

    # ------------------------------------------------------------------
    # 4. Communities
    # ------------------------------------------------------------------
    communities_path = DATA_DIR / "communities.json"
    if communities_path.exists():
        try:
            with open(communities_path, "r") as f:
                raw = json.load(f)
            # Normalise to string community labels
            eng.communities = {
                str(uid): _community_label_from_id(comm)
                for uid, comm in raw.items()
            }
            logger.info("  Loaded %d community assignments", len(eng.communities))
        except Exception as exc:
            logger.warning("  Failed to load communities.json: %s — using mock communities", exc)
            eng.communities = _make_mock_communities()
    else:
        logger.warning("  communities.json not found — generating mock communities for demo")
        n_users = len(eng.users_df) if eng.users_df is not None and len(eng.users_df) > 0 else 500
        eng.communities = _make_mock_communities(n_users=n_users)

    # ------------------------------------------------------------------
    # 5. Generate centroids from embeddings + community assignments
    # ------------------------------------------------------------------
    dim = 64
    if eng.embeddings is not None:
        # Build user_id → row-index map from users.csv
        if eng.users_df is not None and len(eng.users_df) > 0:
            uid_to_idx = {uid: i for i, uid in enumerate(eng.users_df["user_id"].tolist())}
        else:
            uid_to_idx = {f"U{i:03d}": i for i in range(len(eng.embeddings))}

        dim = eng.embeddings.shape[1] if eng.embeddings.ndim == 2 else 64

        # Compute per-community centroid from member embeddings
        comm_embs: Dict[str, List[np.ndarray]] = defaultdict(list)
        for uid, comm_id in eng.communities.items():
            idx = uid_to_idx.get(uid)
            if idx is not None and idx < len(eng.embeddings):
                comm_embs[comm_id].append(eng.embeddings[idx])

        for comm_id, embs in comm_embs.items():
            if embs:
                eng.centroids[comm_id] = np.mean(np.stack(embs, axis=0), axis=0).astype(np.float32)
    else:
        # Generate random embeddings for demo mode
        logger.info("  Generating random embeddings for %d users (demo mode)", len(eng.communities))
        uid_list = sorted(eng.communities.keys())
        rng = np.random.default_rng(42)
        fake_embs = rng.normal(0, 1, (len(uid_list), dim)).astype(np.float32)
        uid_to_fake = {uid: i for i, uid in enumerate(uid_list)}

        comm_embs_fake: Dict[str, List[np.ndarray]] = defaultdict(list)
        for uid, comm_id in eng.communities.items():
            idx = uid_to_fake.get(uid)
            if idx is not None:
                comm_embs_fake[comm_id].append(fake_embs[idx])

        for comm_id, embs in comm_embs_fake.items():
            if embs:
                eng.centroids[comm_id] = np.mean(np.stack(embs, axis=0), axis=0).astype(np.float32)

        eng.embeddings = fake_embs  # Store for peer scorer

    # ------------------------------------------------------------------
    # 6. Derive human-readable community labels
    # ------------------------------------------------------------------
    eng.community_labels = _derive_community_labels(eng.communities, eng.users_df)

    # ------------------------------------------------------------------
    # 7. Scored events (for /live-events, /risk-history, /metrics)
    # ------------------------------------------------------------------
    scored_path = DATA_DIR / "scored_events.csv"
    if scored_path.exists():
        try:
            eng.scored_events_df = pd.read_csv(scored_path, low_memory=False)
            logger.info("  Loaded %d pre-scored events", len(eng.scored_events_df))
        except Exception as exc:
            logger.warning("  Failed to load scored_events.csv: %s — generating mock events", exc)
            eng.scored_events_df = _make_mock_scored_events()
    else:
        logger.warning("  scored_events.csv not found — generating mock events for demo")
        eng.scored_events_df = _make_mock_scored_events()

    # ------------------------------------------------------------------
    # 8. Initialise individual scorers
    # ------------------------------------------------------------------
    logger.info("  Initialising scorers...")

    eng.temporal = TemporalAnalyzer(business_hours_start=8, business_hours_end=18)
    eng.geo = GeoVelocityAnalyzer(max_speed_kmh=900)
    eng.device = DeviceTrustScorer(new_device_score=30, unregistered_score=20, jailbroken_score=40)
    eng.peer = PeerDeviationScorer(threshold_sigma=2.0)

    # Fit scorers on historical data if available
    if eng.events_df is not None and len(eng.events_df) > 0:
        try:
            eng.temporal.fit(eng.events_df)
        except Exception as exc:
            logger.warning("  TemporalAnalyzer.fit failed: %s", exc)

        try:
            eng.geo.fit(eng.events_df, users_df=eng.users_df)
        except Exception as exc:
            logger.warning("  GeoVelocityAnalyzer.fit failed: %s", exc)

        try:
            eng.device.fit(eng.events_df)
        except Exception as exc:
            logger.warning("  DeviceTrustScorer.fit failed: %s", exc)
    else:
        logger.warning("  No historical events — scorers will use default profiles")

    # Build user_id → embedding dict for peer scorer
    uid_emb_dict: Dict[str, np.ndarray] = {}
    if eng.embeddings is not None:
        uid_list_ordered = sorted(eng.communities.keys())
        for i, uid in enumerate(uid_list_ordered):
            if i < len(eng.embeddings):
                uid_emb_dict[uid] = eng.embeddings[i]

    try:
        eng.peer.fit(
            embeddings=uid_emb_dict,
            communities=eng.communities,
            centroids=eng.centroids,
        )
    except Exception as exc:
        logger.warning("  PeerDeviationScorer.fit failed: %s", exc)

    # ------------------------------------------------------------------
    # 9. RiskScorer + DecisionEngine
    # ------------------------------------------------------------------
    eng.risk_scorer = RiskScorer(
        temporal_analyzer=eng.temporal,
        geo_analyzer=eng.geo,
        device_scorer=eng.device,
        peer_scorer=eng.peer,
    )

    if eng.events_df is not None and len(eng.events_df) > 0:
        try:
            eng.risk_scorer.fit_entity_affinity(eng.events_df)
        except Exception as exc:
            logger.warning("  RiskScorer.fit_entity_affinity failed: %s", exc)

    eng.decision_engine = DecisionEngine()

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    eng.model_loaded = True
    elapsed = round(time.time() - t0, 2)
    logger.info(
        "Engine loaded in %.2fs: %d users, %d communities, model_loaded=%s",
        elapsed, eng.users_loaded, len(eng.centroids), eng.model_loaded,
    )
    return eng


# ---------------------------------------------------------------------------
# Module-level singleton — populated by lifespan handler in main.py
# ---------------------------------------------------------------------------
engine: Optional[SentinelEngine] = None


def get_engine() -> SentinelEngine:
    """Return the global engine, raising a RuntimeError if not yet initialised."""
    if engine is None:
        raise RuntimeError(
            "Sentinel engine has not been loaded. "
            "Ensure the FastAPI lifespan handler has run."
        )
    return engine
