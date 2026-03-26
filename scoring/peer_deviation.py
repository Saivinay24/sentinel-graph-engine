"""
Sentinel-Graph Engine — Peer Group Deviation Scorer
Measures how far a user deviates from their functional peer group.
Weight: 25% of composite score. THIS IS THE #1 DIFFERENTIATOR.
"""

import numpy as np
from collections import defaultdict


class PeerDeviationScorer:
    """
    Scores peer group deviation using cosine distance from community centroid.
    
    Key innovation: Instead of comparing users to their OWN history,
    we compare them to their FUNCTIONAL PEER GROUP, detecting:
    - HR users accessing financial resources (even if they have permission)
    - New employees accessing team resources (peers do it, so it's normal)
    - Admins slowly expanding access (peers don't, so it's flagged)
    """

    def __init__(self, threshold_sigma=2.0):
        self.threshold_sigma = threshold_sigma
        self.communities = {}       # user_id → community_label
        self.centroids = {}         # community_label → centroid embedding
        self.user_embeddings = {}   # user_id → embedding
        self.community_stats = {}   # community_label → {mean_dist, std_dist}

    def fit(self, embeddings, communities, centroids, node_id_map=None, G=None):
        """
        Initialize peer deviation scoring.
        
        Args:
            embeddings: numpy array or dict of embeddings
            communities: dict user_id → community_label
            centroids: dict community_label → centroid embedding
            node_id_map: optional mapping node_id → idx
            G: optional NetworkX graph
        """
        self.communities = communities
        self.centroids = centroids

        # Store user embeddings
        if isinstance(embeddings, dict):
            self.user_embeddings = {
                uid: embeddings[uid] for uid in communities.keys()
                if uid in embeddings
            }
        elif node_id_map is not None:
            self.user_embeddings = {}
            for uid in communities.keys():
                if uid in node_id_map:
                    self.user_embeddings[uid] = embeddings[node_id_map[uid]]

        # Compute per-community distance statistics
        for comm_label, centroid in centroids.items():
            members = [uid for uid, c in communities.items() if c == comm_label]
            distances = []
            for uid in members:
                if uid in self.user_embeddings:
                    d = self._cosine_distance(self.user_embeddings[uid], centroid)
                    distances.append(d)

            if distances:
                self.community_stats[comm_label] = {
                    "mean_dist": np.mean(distances),
                    "std_dist": np.std(distances) if len(distances) > 1 else 0.1,
                    "max_dist": np.max(distances),
                    "member_count": len(members),
                }

        print(f"   ✅ Peer deviation scorer: {len(self.communities)} users, "
              f"{len(self.centroids)} peer groups")

    def _cosine_distance(self, a, b):
        """Compute cosine distance (1 - cosine_similarity)."""
        a = np.array(a, dtype=np.float64)
        b = np.array(b, dtype=np.float64)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 1.0
        return 1.0 - np.dot(a, b) / (norm_a * norm_b)

    def score(self, user_id, event=None):
        """
        Score peer group deviation for a user (0-100).
        
        High score = user is behaving very differently from their peer group.
        """
        if user_id not in self.communities:
            return 30.0  # Moderate score for unknown users

        comm = self.communities[user_id]
        centroid = self.centroids.get(comm)
        user_emb = self.user_embeddings.get(user_id)

        if centroid is None or user_emb is None:
            return 25.0

        # Cosine distance from centroid
        distance = self._cosine_distance(user_emb, centroid)

        # Z-score within community
        stats = self.community_stats.get(comm, {"mean_dist": 0.5, "std_dist": 0.1})
        z_score = (distance - stats["mean_dist"]) / max(stats["std_dist"], 0.01)

        # Map to 0-100
        if z_score <= 0:
            score = 0
        elif z_score <= self.threshold_sigma:
            # Linear scale from 0 to 40
            score = (z_score / self.threshold_sigma) * 40
        else:
            # Above threshold: rapid escalation 40-100
            excess = z_score - self.threshold_sigma
            score = 40 + min(60, excess * 30)

        return round(min(100, max(0, score)), 1)

    def get_explanation(self, user_id, G=None):
        """Generate explanation of peer deviation."""
        if user_id not in self.communities:
            return "User not assigned to any peer group"

        comm = self.communities[user_id]
        centroid = self.centroids.get(comm)
        user_emb = self.user_embeddings.get(user_id)

        if centroid is None or user_emb is None:
            return "Insufficient data for peer comparison"

        distance = self._cosine_distance(user_emb, centroid)
        stats = self.community_stats.get(comm, {"mean_dist": 0.5, "std_dist": 0.1})
        z_score = (distance - stats["mean_dist"]) / max(stats["std_dist"], 0.01)

        # Get community info
        comm_size = stats.get("member_count", 0)
        dept_info = ""
        if G:
            members = [uid for uid, c in self.communities.items() if c == comm]
            depts = [G.nodes[uid].get("department", "?") for uid in members if uid in G.nodes]
            if depts:
                dept_info = f" (dominant dept: {max(set(depts), key=depts.count)})"

        if z_score > self.threshold_sigma:
            return (
                f"⚠️ Significant deviation from Peer Group {comm}{dept_info}: "
                f"{z_score:.1f}σ from centroid (threshold: {self.threshold_sigma}σ). "
                f"Peer group has {comm_size} members."
            )
        elif z_score > 1.0:
            return (
                f"Mild deviation from Peer Group {comm}{dept_info}: "
                f"{z_score:.1f}σ from centroid."
            )
        else:
            return f"Normal behavior within Peer Group {comm}{dept_info}."

    def get_nearest_other_community(self, user_id):
        """Find which other community the user is drifting towards."""
        user_emb = self.user_embeddings.get(user_id)
        user_comm = self.communities.get(user_id)
        if user_emb is None or user_comm is None:
            return None

        min_dist = float('inf')
        nearest = None
        for comm, centroid in self.centroids.items():
            if comm == user_comm:
                continue
            d = self._cosine_distance(user_emb, centroid)
            if d < min_dist:
                min_dist = d
                nearest = comm

        return nearest
