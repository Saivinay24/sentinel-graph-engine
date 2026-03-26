"""
Sentinel-Graph Engine — Anomaly Detector
Isolation Forest on node embeddings for secondary anomaly scoring.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


def train_anomaly_detector(embeddings, node_id_map, G, contamination=0.05, method="isolation_forest"):
    """
    Train an anomaly detector on user embeddings.
    
    Args:
        embeddings: numpy array or dict of embeddings
        node_id_map: mapping node_id → idx
        G: NetworkX graph
        contamination: expected proportion of outliers
        method: 'isolation_forest' or 'lof'
    
    Returns:
        anomaly_scores: dict mapping user_id → anomaly score (0-1, higher = more anomalous)
        model: fitted anomaly detection model
    """
    # Extract user embeddings
    user_nodes = [n for n in G.nodes() if G.nodes[n].get("node_type") == "user"]

    if isinstance(embeddings, dict):
        user_embs = {uid: embeddings[uid] for uid in user_nodes if uid in embeddings}
    else:
        user_embs = {}
        for uid in user_nodes:
            if uid in node_id_map:
                user_embs[uid] = embeddings[node_id_map[uid]]

    user_ids = list(user_embs.keys())
    emb_matrix = np.array([user_embs[uid] for uid in user_ids])

    print(f"🔄 Training {method} anomaly detector on {len(user_ids)} users...")

    if method == "isolation_forest":
        model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100,
        )
        model.fit(emb_matrix)
        # Anomaly scores: lower = more anomalous, normalize to 0-1
        raw_scores = model.decision_function(emb_matrix)
        # Invert and normalize: more anomalous → higher score
        normalized = 1 - (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-8)
    else:
        model = LocalOutlierFactor(
            contamination=contamination,
            n_neighbors=20,
            novelty=False,
        )
        model.fit_predict(emb_matrix)
        raw_scores = model.negative_outlier_factor_
        normalized = 1 - (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-8)

    anomaly_scores = {uid: float(normalized[i]) for i, uid in enumerate(user_ids)}

    n_anomalous = sum(1 for s in normalized if s > 0.7)
    print(f"   ✅ {n_anomalous} users flagged as potentially anomalous (score > 0.7)")

    return anomaly_scores, model
