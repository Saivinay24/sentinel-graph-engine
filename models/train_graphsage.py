"""
Sentinel-Graph Engine — GraphSAGE Training Loop
Link prediction training with neighbor sampling.
"""

import sys
import time
import json
from pathlib import Path

import numpy as np
import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def _load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def train_graphsage(pyg_data, config=None):
    """
    Train GraphSAGE model on link prediction task.
    
    Args:
        pyg_data: PyTorch Geometric Data object
        config: optional config dict
    
    Returns:
        model: trained GraphSAGE model
        embeddings: node embeddings as numpy array
        metrics: training metrics dict
    """
    try:
        import torch
        import torch.nn.functional as F
        from torch_geometric.loader import LinkNeighborLoader
        from models.graphsage_model import GraphSAGEModel, LinkPredictor
    except ImportError:
        print("⚠️  PyTorch/PyG not available. Cannot train GraphSAGE.")
        return None, None, None

    if config is None:
        config = _load_config()

    model_cfg = config["model"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔄 Training GraphSAGE on {device}...")

    # Model init
    in_channels = pyg_data.x.shape[1]
    hidden = model_cfg["hidden_dim"]
    model = GraphSAGEModel(in_channels, hidden, hidden, model_cfg["dropout"]).to(device)
    predictor = LinkPredictor(hidden).to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(predictor.parameters()),
        lr=model_cfg["learning_rate"],
    )

    epochs = model_cfg["epochs"]
    pyg_data = pyg_data.to(device)

    # Training with edge-level batching
    num_edges = pyg_data.edge_index.shape[1]
    edge_perm = torch.randperm(num_edges)

    metrics = {"losses": [], "epoch_times": []}

    for epoch in range(epochs):
        model.train()
        predictor.train()
        t0 = time.time()

        # Forward
        z = model(pyg_data.x, pyg_data.edge_index)

        # Positive edges
        pos_edge_index = pyg_data.edge_index
        pos_src = z[pos_edge_index[0]]
        pos_dst = z[pos_edge_index[1]]
        pos_pred = predictor(pos_src, pos_dst)
        pos_label = torch.ones(pos_pred.shape[0], device=device)

        # Negative sampling
        num_neg = pos_edge_index.shape[1]
        neg_src_idx = torch.randint(0, pyg_data.num_nodes, (num_neg,), device=device)
        neg_dst_idx = torch.randint(0, pyg_data.num_nodes, (num_neg,), device=device)
        neg_src = z[neg_src_idx]
        neg_dst = z[neg_dst_idx]
        neg_pred = predictor(neg_src, neg_dst)
        neg_label = torch.zeros(neg_pred.shape[0], device=device)

        # Loss
        pred = torch.cat([pos_pred, neg_pred])
        labels = torch.cat([pos_label, neg_label])
        loss = F.binary_cross_entropy(pred, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        elapsed = time.time() - t0
        metrics["losses"].append(loss.item())
        metrics["epoch_times"].append(elapsed)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"   Epoch {epoch+1:3d}/{epochs} | Loss: {loss.item():.4f} | Time: {elapsed:.2f}s")

    # Generate final embeddings
    model.eval()
    with torch.no_grad():
        embeddings = model(pyg_data.x, pyg_data.edge_index).cpu().numpy()

    print(f"   ✅ Training complete. Final loss: {metrics['losses'][-1]:.4f}")
    print(f"   ✅ Embeddings shape: {embeddings.shape}")

    return model, embeddings, metrics


def save_embeddings(embeddings, node_id_map, output_path):
    """Save embeddings to disk."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, embeddings)
    # Save node map
    map_path = path.with_suffix(".json")
    # Invert map: idx → node_id
    inv_map = {v: k for k, v in node_id_map.items()}
    with open(map_path, "w") as f:
        json.dump(inv_map, f)
    print(f"   💾 Embeddings saved to {path}")


def load_embeddings(path):
    """Load embeddings from disk."""
    embeddings = np.load(path)
    map_path = Path(path).with_suffix(".json")
    with open(map_path) as f:
        inv_map = json.load(f)
    # Convert string keys back to int
    inv_map = {int(k): v for k, v in inv_map.items()}
    return embeddings, inv_map


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from data.generate_synthetic_data import generate_all
    from data.inject_attacks import inject_attacks
    from graph.build_graph import build_identity_graph, graph_to_pyg_data

    # Generate data
    data = generate_all()
    events = inject_attacks(data["events"], data["users"], data["resources"])
    G, node_features = build_identity_graph(data, events)
    pyg_data, node_id_map = graph_to_pyg_data(G, node_features)

    if pyg_data is not None:
        config = _load_config()
        # Quick test with fewer epochs
        if "--test" in sys.argv:
            config["model"]["epochs"] = 5
        model, embeddings, metrics = train_graphsage(pyg_data, config)

        if embeddings is not None:
            out_path = Path(__file__).resolve().parent.parent / "data" / "generated" / "embeddings.npy"
            save_embeddings(embeddings, node_id_map, out_path)
    else:
        print("⚠️  Using fallback embeddings (no PyG available)")
        from models.graphsage_model import generate_embeddings_fallback
        embeddings = generate_embeddings_fallback(node_features, G)
        print(f"   ✅ Fallback embeddings generated for {len(embeddings)} nodes")
