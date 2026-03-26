"""
Sentinel-Graph Engine — Heterogeneous Identity Graph Builder
Constructs the knowledge graph from synthetic data using NetworkX.
"""

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import yaml

from graph.graph_schema import (
    NodeType, EdgeType,
    DEPARTMENT_ENCODING, SENIORITY_ENCODING,
    SENSITIVITY_ENCODING, ACTION_ENCODING,
)
from graph.graph_utils import compute_edge_weights

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def _load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def build_identity_graph(data, events_df=None):
    """
    Build the heterogeneous identity knowledge graph.
    
    Args:
        data: dict with DataFrames: users, roles, resources, permissions, user_roles
        events_df: optional events DataFrame for behavioral edges
    
    Returns:
        G: NetworkX DiGraph with typed nodes and edges
        node_features: dict mapping node_id → feature vector (numpy array)
    """
    config = _load_config()
    G = nx.DiGraph()
    node_features = {}

    users_df = data["users"]
    roles_df = data["roles"]
    resources_df = data["resources"]
    permissions_df = data["permissions"]
    user_roles_df = data["user_roles"]

    print("🔄 Building identity knowledge graph...")

    # --- User Nodes ---
    for _, user in users_df.iterrows():
        uid = user["user_id"]
        G.add_node(uid, node_type=NodeType.USER.value, **user.to_dict())
        # Feature vector: [dept_enc, seniority_enc, lat, lon, is_active]
        node_features[uid] = np.array([
            DEPARTMENT_ENCODING.get(user.get("department", ""), 0),
            SENIORITY_ENCODING.get(user.get("seniority", ""), 0),
            float(user.get("office_lat", 0)),
            float(user.get("office_lon", 0)),
            1.0 if user.get("is_active", True) else 0.0,
        ], dtype=np.float32)

    print(f"   ✅ {len(users_df)} user nodes")

    # --- Role Nodes ---
    for _, role in roles_df.iterrows():
        rid = role["role_id"]
        G.add_node(rid, node_type=NodeType.ROLE.value, **role.to_dict())
        node_features[rid] = np.array([
            DEPARTMENT_ENCODING.get(role.get("department", ""), 0),
            0.0, 0.0, 0.0, 1.0,
        ], dtype=np.float32)

    print(f"   ✅ {len(roles_df)} role nodes")

    # --- Resource Nodes ---
    for _, res in resources_df.iterrows():
        resid = res["resource_id"]
        G.add_node(resid, node_type=NodeType.RESOURCE.value, **res.to_dict())
        node_features[resid] = np.array([
            SENSITIVITY_ENCODING.get(res.get("sensitivity", "Low"), 0),
            0.0, 0.0, 0.0, 1.0,
        ], dtype=np.float32)

    print(f"   ✅ {len(resources_df)} resource nodes")

    # --- Permission Nodes + Edges ---
    for _, perm in permissions_df.iterrows():
        pid = perm["permission_id"]
        G.add_node(pid, node_type=NodeType.PERMISSION.value, **perm.to_dict())
        node_features[pid] = np.array([
            ACTION_ENCODING.get(perm.get("action", "read"), 0),
            0.0, 0.0, 0.0, 1.0,
        ], dtype=np.float32)

        # Role → Permission edge
        G.add_edge(perm["role_id"], pid, edge_type=EdgeType.ROLE_GRANTS_PERMISSION.value)
        # Permission → Resource edge
        G.add_edge(pid, perm["resource_id"], edge_type=EdgeType.PERMISSION_ACCESSES_RESOURCE.value)

    print(f"   ✅ {len(permissions_df)} permission nodes")

    # --- User → Role edges ---
    for _, ur in user_roles_df.iterrows():
        G.add_edge(ur["user_id"], ur["role_id"], edge_type=EdgeType.USER_HAS_ROLE.value)

    print(f"   ✅ {len(user_roles_df)} user→role edges")

    # --- Behavioral edges: User → Resource (from event data) ---
    if events_df is not None:
        edge_weights = compute_edge_weights(
            events_df,
            method=config["graph"]["edge_weight_method"],
            decay=config["graph"]["recency_decay"],
        )
        for _, ew in edge_weights.iterrows():
            uid = ew["user_id"]
            rid = ew["resource_id"]
            if uid in G.nodes and rid in G.nodes:
                G.add_edge(
                    uid, rid,
                    edge_type=EdgeType.USER_ACCESSED_RESOURCE.value,
                    weight=ew["weight"],
                    access_count=ew["access_count"],
                )
        print(f"   ✅ {len(edge_weights)} behavioral edges")

    total_nodes = G.number_of_nodes()
    total_edges = G.number_of_edges()
    print(f"\n   📊 Graph: {total_nodes} nodes, {total_edges} edges")

    return G, node_features


def graph_to_pyg_data(G, node_features):
    """
    Convert NetworkX graph to PyTorch Geometric Data object.
    
    Args:
        G: NetworkX DiGraph
        node_features: dict mapping node_id → feature vector
    
    Returns:
        PyG Data object, node_id_map
    """
    try:
        import torch
        from torch_geometric.data import Data
    except ImportError:
        print("⚠️  PyTorch Geometric not installed. Using fallback mode.")
        return None, None

    # Create node ID mapping
    node_ids = list(G.nodes())
    node_id_map = {nid: i for i, nid in enumerate(node_ids)}

    # Build feature matrix
    feat_dim = 5  # All feature vectors have dimension 5
    x = np.zeros((len(node_ids), feat_dim), dtype=np.float32)
    for nid, idx in node_id_map.items():
        if nid in node_features:
            x[idx] = node_features[nid]

    # Build edge index
    edges_src = []
    edges_dst = []
    for u, v in G.edges():
        if u in node_id_map and v in node_id_map:
            edges_src.append(node_id_map[u])
            edges_dst.append(node_id_map[v])
            # Add reverse edge for undirected message passing
            edges_src.append(node_id_map[v])
            edges_dst.append(node_id_map[u])

    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    x_tensor = torch.tensor(x, dtype=torch.float)

    # Node type labels
    node_type_map = {}
    for nid in node_ids:
        nt = G.nodes[nid].get("node_type", "unknown")
        node_type_map[node_id_map[nid]] = nt

    data = Data(x=x_tensor, edge_index=edge_index)
    data.num_nodes = len(node_ids)

    print(f"   📊 PyG Data: {data.num_nodes} nodes, {data.num_edges} edges, features={feat_dim}")

    return data, node_id_map


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from data.generate_synthetic_data import generate_all
    from data.inject_attacks import inject_attacks

    data = generate_all()
    events = inject_attacks(data["events"], data["users"], data["resources"])

    G, node_features = build_identity_graph(data, events)

    # Try PyG conversion
    pyg_data, node_map = graph_to_pyg_data(G, node_features)
    print("\n✅ Graph construction complete!")
