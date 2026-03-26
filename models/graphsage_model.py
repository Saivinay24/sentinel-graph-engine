"""
Sentinel-Graph Engine — GraphSAGE Model
2-layer GraphSAGE with mean aggregation for identity embedding learning.
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import SAGEConv
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch/PyG not available. Using embedding fallback.")


if TORCH_AVAILABLE:
    class GraphSAGEModel(nn.Module):
        """
        2-layer GraphSAGE for identity node embedding.
        
        Architecture:
            Input → SAGEConv(in, hidden) → ReLU → Dropout → SAGEConv(hidden, hidden) → Embeddings
        """

        def __init__(self, in_channels, hidden_channels=64, out_channels=64, dropout=0.3):
            super().__init__()
            self.conv1 = SAGEConv(in_channels, hidden_channels, aggr='mean')
            self.conv2 = SAGEConv(hidden_channels, out_channels, aggr='mean')
            self.dropout = dropout

        def forward(self, x, edge_index):
            """Forward pass: generate node embeddings."""
            h = self.conv1(x, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.conv2(h, edge_index)
            return h

        def encode(self, x, edge_index):
            """Generate embeddings (alias for forward)."""
            return self.forward(x, edge_index)

    class LinkPredictor(nn.Module):
        """Predict whether an edge should exist between two nodes."""

        def __init__(self, in_channels):
            super().__init__()
            self.lin1 = nn.Linear(in_channels * 2, in_channels)
            self.lin2 = nn.Linear(in_channels, 1)

        def forward(self, z_src, z_dst):
            """Predict link probability."""
            z = torch.cat([z_src, z_dst], dim=-1)
            z = F.relu(self.lin1(z))
            return torch.sigmoid(self.lin2(z)).squeeze()


def generate_embeddings_fallback(node_features, G):
    """
    Fallback: generate embeddings using graph structure features when PyTorch is unavailable.
    Uses degree centrality, betweenness centrality, and node features.
    """
    import networkx as nx
    from sklearn.decomposition import PCA

    node_ids = list(G.nodes())

    # Compute graph metrics
    try:
        degree_cent = nx.degree_centrality(G)
    except:
        degree_cent = {n: 0.0 for n in node_ids}

    try:
        # Use approximate betweenness for speed
        between_cent = nx.betweenness_centrality(G, k=min(100, len(node_ids)))
    except:
        between_cent = {n: 0.0 for n in node_ids}

    try:
        pagerank = nx.pagerank(G, max_iter=50)
    except:
        pagerank = {n: 1.0/len(node_ids) for n in node_ids}

    # Combine features
    embeddings = {}
    for nid in node_ids:
        base_feat = node_features.get(nid, np.zeros(5))
        graph_feat = np.array([
            degree_cent.get(nid, 0),
            between_cent.get(nid, 0),
            pagerank.get(nid, 0),
        ])
        embeddings[nid] = np.concatenate([base_feat, graph_feat]).astype(np.float32)

    # Project to 64 dimensions using PCA if we have enough nodes
    all_embs = np.array([embeddings[nid] for nid in node_ids])
    if all_embs.shape[0] > 64 and all_embs.shape[1] < 64:
        # Pad with random projections
        rng = np.random.default_rng(42)
        proj = rng.standard_normal((all_embs.shape[1], 64)).astype(np.float32)
        all_embs_64 = all_embs @ proj
        for i, nid in enumerate(node_ids):
            embeddings[nid] = all_embs_64[i]
    elif all_embs.shape[1] > 64:
        pca = PCA(n_components=64)
        all_embs_64 = pca.fit_transform(all_embs)
        for i, nid in enumerate(node_ids):
            embeddings[nid] = all_embs_64[i].astype(np.float32)

    return embeddings
