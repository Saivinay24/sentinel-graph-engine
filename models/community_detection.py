"""
Sentinel-Graph Engine — Community Detection
Louvain community detection on GraphSAGE embeddings to discover functional peer groups.
"""

from pathlib import Path

import numpy as np
import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def _load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def detect_communities_louvain(embeddings, node_id_map, G, config=None):
    """
    Run Louvain community detection on node embeddings.
    
    Strategy: Build a similarity graph from embeddings, then run Louvain.
    Only operates on user nodes.
    
    Args:
        embeddings: dict or numpy array of node embeddings
        node_id_map: dict mapping node_id → index (or index → node_id)
        G: NetworkX graph (for node type info)
        config: optional config dict
    
    Returns:
        communities: dict mapping user_id → community_label
        centroids: dict mapping community_label → centroid embedding
        community_stats: summary statistics
    """
    import networkx as nx
    try:
        import community as community_louvain
    except ImportError:
        print("⚠️  python-louvain not installed. Using fallback k-means clustering.")
        return _kmeans_fallback(embeddings, node_id_map, G, config)

    if config is None:
        config = _load_config()

    resolution = config["community"]["resolution"]

    print("🔄 Running Louvain community detection...")

    # Extract user embeddings only
    user_nodes = [n for n in G.nodes() if G.nodes[n].get("node_type") == "user"]

    if isinstance(embeddings, dict):
        user_embeddings = {uid: embeddings[uid] for uid in user_nodes if uid in embeddings}
    else:
        # embeddings is numpy array, node_id_map maps node_id → idx
        if isinstance(list(node_id_map.keys())[0], str):
            user_embeddings = {
                uid: embeddings[node_id_map[uid]]
                for uid in user_nodes
                if uid in node_id_map
            }
        else:
            # inv_map: idx → node_id
            inv_map = {v: k for k, v in node_id_map.items()} if not isinstance(list(node_id_map.keys())[0], int) else node_id_map
            user_embeddings = {}
            for uid in user_nodes:
                if uid in node_id_map:
                    user_embeddings[uid] = embeddings[node_id_map[uid]]
                else:
                    for idx, nid in inv_map.items():
                        if nid == uid:
                            user_embeddings[uid] = embeddings[idx]
                            break

    if len(user_embeddings) == 0:
        print("   ⚠️ No user embeddings found.")
        return {}, {}, {}

    user_ids = list(user_embeddings.keys())
    emb_matrix = np.array([user_embeddings[uid] for uid in user_ids])

    # Build similarity graph from embeddings (cosine similarity)
    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(emb_matrix)

    # Threshold to create sparse graph (keep top connections)
    threshold = np.percentile(sim_matrix, 70)  # Keep top 30% similarities
    sim_graph = nx.Graph()

    for i, uid_i in enumerate(user_ids):
        sim_graph.add_node(uid_i)
        for j, uid_j in enumerate(user_ids):
            if i < j and sim_matrix[i, j] > threshold:
                sim_graph.add_edge(uid_i, uid_j, weight=float(sim_matrix[i, j]))

    # Run Louvain
    communities = community_louvain.best_partition(
        sim_graph, resolution=resolution, random_state=42
    )

    # Compute centroids for each community
    community_members = {}
    for uid, comm in communities.items():
        if comm not in community_members:
            community_members[comm] = []
        community_members[comm].append(uid)

    centroids = {}
    for comm, members in community_members.items():
        member_embs = np.array([user_embeddings[uid] for uid in members])
        centroids[comm] = member_embs.mean(axis=0)

    # Stats
    community_stats = {
        "num_communities": len(set(communities.values())),
        "community_sizes": {
            comm: len(members) for comm, members in community_members.items()
        },
        "avg_community_size": np.mean([len(m) for m in community_members.values()]),
    }

    print(f"   ✅ Found {community_stats['num_communities']} peer groups")
    for comm, size in community_stats["community_sizes"].items():
        members = community_members[comm]
        depts = [G.nodes[uid].get("department", "?") for uid in members if uid in G.nodes]
        top_dept = max(set(depts), key=depts.count) if depts else "Unknown"
        print(f"      Peer Group {comm}: {size} users (dominant: {top_dept})")

    return communities, centroids, community_stats


def _kmeans_fallback(embeddings, node_id_map, G, config=None):
    """Fallback: use K-Means clustering on embeddings."""
    from sklearn.cluster import KMeans

    if config is None:
        config = _load_config()

    n_clusters = len(config["data"]["departments"])
    user_nodes = [n for n in G.nodes() if G.nodes[n].get("node_type") == "user"]

    if isinstance(embeddings, dict):
        user_embeddings = {uid: embeddings[uid] for uid in user_nodes if uid in embeddings}
    else:
        user_embeddings = {}
        for uid in user_nodes:
            if uid in node_id_map:
                user_embeddings[uid] = embeddings[node_id_map[uid]]

    user_ids = list(user_embeddings.keys())
    emb_matrix = np.array([user_embeddings[uid] for uid in user_ids])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(emb_matrix)

    communities = {uid: int(labels[i]) for i, uid in enumerate(user_ids)}
    centroids = {i: kmeans.cluster_centers_[i] for i in range(n_clusters)}

    community_sizes = {}
    for label in labels:
        community_sizes[int(label)] = community_sizes.get(int(label), 0) + 1

    stats = {
        "num_communities": n_clusters,
        "community_sizes": community_sizes,
        "avg_community_size": len(user_ids) / n_clusters,
    }

    print(f"   ✅ K-Means fallback: {n_clusters} clusters")
    return communities, centroids, stats
