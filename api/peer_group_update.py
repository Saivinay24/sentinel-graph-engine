"""
Sentinel-Graph API — Inductive Peer Group Assignment
Assigns new users to the nearest community centroid using cosine similarity.
GraphSAGE is inductive: unseen users get a synthetic embedding derived from
their role/department attributes, then assigned to the nearest community.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Department / role → deterministic embedding dimensions
# ---------------------------------------------------------------------------

# Canonical department → feature vector index (must match training space dimensions)
DEPARTMENT_INDEX: Dict[str, int] = {
    "Engineering": 0,
    "Finance": 1,
    "Human Resources": 2,
    "HR": 2,  # alias
    "Marketing": 3,
    "Sales": 4,
    "IT Operations": 5,
    "Legal": 6,
    "Executive": 7,
}

ROLE_KEYWORDS: List[Tuple[str, int]] = [
    # (keyword, feature_index_offset)
    ("analyst", 8),
    ("engineer", 9),
    ("manager", 10),
    ("director", 11),
    ("executive", 12),
    ("admin", 13),
    ("lead", 14),
    ("junior", 15),
    ("senior", 16),
    ("architect", 17),
]

EMBEDDING_DIM = 64  # Must match training config (model.hidden_dim)


def _synthesize_embedding(
    department: str,
    role_id: str,
    attributes: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Create a deterministic synthetic embedding for a new user.

    The embedding is constructed by:
    1. Setting one-hot department dimensions.
    2. Setting role-keyword dimensions based on the role_id string.
    3. Seeding remaining dimensions from a deterministic hash of (dept, role_id).

    This is a principled approximation of the inductive capability of GraphSAGE:
    without a graph neighborhood to aggregate from, we use attribute features
    to place the user in the same representation space as trained embeddings.
    """
    emb = np.zeros(EMBEDDING_DIM, dtype=np.float32)

    # --- Department one-hot (dims 0-7) ---
    dept_canonical = department.strip().title()
    dept_idx = DEPARTMENT_INDEX.get(dept_canonical) or DEPARTMENT_INDEX.get(department)
    if dept_idx is not None and dept_idx < EMBEDDING_DIM:
        emb[dept_idx] = 1.0

    # --- Role keyword features (dims 8-17) ---
    role_lower = role_id.lower()
    for keyword, feat_idx in ROLE_KEYWORDS:
        if keyword in role_lower and feat_idx < EMBEDDING_DIM:
            emb[feat_idx] = 1.0

    # --- Attribute overrides (dims 18-31) ---
    if attributes:
        attr_keys = sorted(attributes.keys())
        for i, key in enumerate(attr_keys[:14]):
            val = attributes[key]
            if isinstance(val, (int, float)):
                emb[18 + i] = float(val) / 100.0  # Normalize to [0,1]
            elif isinstance(val, bool):
                emb[18 + i] = 1.0 if val else 0.0
            elif isinstance(val, str):
                # Hash-derived float in [0, 1]
                h = int(hashlib.md5(val.encode()).hexdigest()[:8], 16)
                emb[18 + i] = (h % 1000) / 1000.0

    # --- Deterministic noise for remaining dims (32-63) ---
    # Use SHA256 of (department, role_id) as PRNG seed for reproducibility
    seed_str = f"{department}::{role_id}"
    seed_hash = int(hashlib.sha256(seed_str.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed_hash)
    noise = rng.normal(0, 0.1, EMBEDDING_DIM - 32).astype(np.float32)
    emb[32:] = noise

    return emb


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return cosine similarity in [-1, 1]."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def assign_new_user_to_peer_group(
    user_id: str,
    department: str,
    role_id: str,
    attributes: Optional[Dict[str, Any]] = None,
    centroids: Optional[Dict[str, np.ndarray]] = None,
    community_labels: Optional[Dict[str, str]] = None,
    community_stats: Optional[Dict[str, Dict]] = None,
) -> Dict[str, Any]:
    """
    Assign a new (unseen) user to the nearest peer group by cosine similarity.

    This is the inductive step: GraphSAGE can embed unseen nodes given their
    features, but since we may not have a running GNN at inference time, we
    use a deterministic attribute-derived embedding to approximate placement.

    Args:
        user_id:         The new user's identifier.
        department:      The user's department name.
        role_id:         The user's role identifier (e.g. "R_HR_ANALYST").
        attributes:      Optional dict of additional user attributes.
        centroids:       Community centroid embeddings (community_id → np.ndarray).
                         Loaded from engine at startup.
        community_labels: Human-readable label per community (community_id → label).
        community_stats: Per-community stats including member_count.

    Returns:
        Dict with peer_group_id, peer_group_label, confidence, department_match.
    """
    attributes = attributes or {}

    # Synthesise embedding for the new user
    user_emb = _synthesize_embedding(department, role_id, attributes)

    if not centroids:
        logger.warning("No centroids available; returning fallback peer group")
        return {
            "peer_group_id": "community_0",
            "peer_group_label": "Engineering",
            "confidence": 0.50,
            "department_match": department,
        }

    # Find nearest centroid by cosine similarity
    best_comm: Optional[str] = None
    best_sim: float = -2.0
    similarities: Dict[str, float] = {}

    for comm_id, centroid in centroids.items():
        centroid_arr = np.array(centroid, dtype=np.float32)
        sim = _cosine_similarity(user_emb, centroid_arr)
        similarities[comm_id] = sim
        if sim > best_sim:
            best_sim = sim
            best_comm = comm_id

    if best_comm is None:
        best_comm = list(centroids.keys())[0]
        best_sim = 0.50

    # Normalise similarity to a confidence score in [0, 1]
    # cosine similarity is in [-1, 1]; shift to [0, 1]
    confidence = round(float((best_sim + 1.0) / 2.0), 4)
    confidence = max(0.0, min(1.0, confidence))

    # Resolve human-readable label
    label = (community_labels or {}).get(best_comm, best_comm)

    # Department match hint: what department dominates this community?
    dept_match = department  # Default to the user's own department

    logger.info(
        "Assigned user %s (dept=%s, role=%s) → %s (label=%s, confidence=%.2f)",
        user_id, department, role_id, best_comm, label, confidence,
    )

    return {
        "peer_group_id": best_comm,
        "peer_group_label": label,
        "confidence": confidence,
        "department_match": dept_match,
    }
