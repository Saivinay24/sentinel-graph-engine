#!/usr/bin/env python3
"""
Sentinel-Graph Engine — Master Pipeline
One-command execution: Generate Data → Build Graph → Train Model → Run Scoring → Launch Dashboard

Usage:
    python run_pipeline.py              # Full pipeline + launch dashboard
    python run_pipeline.py --no-dash    # Pipeline only, no dashboard
    python run_pipeline.py --test       # Quick test mode (fewer epochs)
"""

import sys
import json
import argparse
import pickle
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import yaml


def load_config():
    with open(PROJECT_ROOT / "config.yaml", "r") as f:
        return yaml.safe_load(f)


def run_pipeline(test_mode=False):
    """Execute the full Sentinel-Graph pipeline."""

    config = load_config()
    output_dir = PROJECT_ROOT / "data" / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ===================================================================
    # PHASE 1: Data Generation
    # ===================================================================
    print("\n" + "=" * 60)
    print("  PHASE 1: Synthetic Data Generation")
    print("=" * 60)

    from data.generate_synthetic_data import generate_all
    from data.inject_attacks import inject_attacks
    from data.anonymizer import anonymize_dataset

    data = generate_all(output_dir=output_dir)
    events_with_attacks = inject_attacks(data["events"], data["users"], data["resources"])
    events_with_attacks.to_csv(output_dir / "events_with_attacks.csv", index=False)

    # Anonymize
    anon_data = anonymize_dataset(data)

    # ===================================================================
    # PHASE 2: Graph Construction
    # ===================================================================
    print("\n" + "=" * 60)
    print("  PHASE 2: Identity Knowledge Graph Construction")
    print("=" * 60)

    from graph.build_graph import build_identity_graph, graph_to_pyg_data

    G, node_features = build_identity_graph(data, events_with_attacks)

    # Save graph
    import pickle
    with open(output_dir / "identity_graph.pkl", "wb") as f:
        pickle.dump(G, f)

    # ===================================================================
    # PHASE 3: GNN Training & Community Detection
    # ===================================================================
    print("\n" + "=" * 60)
    print("  PHASE 3: GraphSAGE Training & Community Detection")
    print("=" * 60)

    pyg_data, node_id_map = graph_to_pyg_data(G, node_features)

    embeddings = None
    model = None

    if pyg_data is not None:
        from models.train_graphsage import train_graphsage
        if test_mode:
            config["model"]["epochs"] = 10
        model, embeddings_arr, train_metrics = train_graphsage(pyg_data, config)

        if embeddings_arr is not None:
            embeddings = embeddings_arr
            np.save(output_dir / "embeddings.npy", embeddings)
            # Save node_id_map
            inv_map = {v: k for k, v in node_id_map.items()}
            with open(output_dir / "node_id_map.json", "w") as f:
                json.dump({str(k): v for k, v in inv_map.items()}, f)
    
    if embeddings is None:
        print("   ⚠️ Using fallback embeddings...")
        from models.graphsage_model import generate_embeddings_fallback
        embeddings_dict = generate_embeddings_fallback(node_features, G)

        # Convert dict to array format compatible with node_id_map
        node_ids = list(G.nodes())
        node_id_map = {nid: i for i, nid in enumerate(node_ids)}
        embeddings = np.zeros((len(node_ids), 64), dtype=np.float32)
        for nid, emb in embeddings_dict.items():
            if nid in node_id_map:
                emb_arr = np.array(emb, dtype=np.float32)
                if len(emb_arr) >= 64:
                    embeddings[node_id_map[nid]] = emb_arr[:64]
                else:
                    embeddings[node_id_map[nid], :len(emb_arr)] = emb_arr

        np.save(output_dir / "embeddings.npy", embeddings)
        inv_map = {v: k for k, v in node_id_map.items()}
        with open(output_dir / "node_id_map.json", "w") as f:
            json.dump({str(k): v for k, v in inv_map.items()}, f)

    # Community Detection
    from models.community_detection import detect_communities_louvain
    communities, centroids, community_stats = detect_communities_louvain(
        embeddings, node_id_map, G, config
    )

    # Save communities
    with open(output_dir / "communities.json", "w") as f:
        json.dump(communities, f)
    with open(output_dir / "community_stats.json", "w") as f:
        json.dump(community_stats, f, default=str)

    # Anomaly Detection
    from models.anomaly_detector import train_anomaly_detector
    anomaly_scores, anomaly_model = train_anomaly_detector(
        embeddings, node_id_map, G
    )
    with open(output_dir / "anomaly_scores.json", "w") as f:
        json.dump(anomaly_scores, f)

    # ===================================================================
    # PHASE 4: Scoring Engine Setup
    # ===================================================================
    print("\n" + "=" * 60)
    print("  PHASE 4: Composite Risk Scoring Engine")
    print("=" * 60)

    from scoring.temporal_analyzer import TemporalAnalyzer
    from scoring.geo_velocity import GeoVelocityAnalyzer
    from scoring.device_trust import DeviceTrustScorer
    from scoring.peer_deviation import PeerDeviationScorer
    from scoring.risk_scorer import RiskScorer
    from scoring.decision_engine import DecisionEngine

    # Initialize all scorers
    print("🔄 Initializing scoring components...")

    temporal = TemporalAnalyzer(
        business_hours_start=config["scoring"]["temporal"]["business_hours_start"],
        business_hours_end=config["scoring"]["temporal"]["business_hours_end"],
    )
    temporal.fit(events_with_attacks[events_with_attacks["is_attack"] == False])

    geo = GeoVelocityAnalyzer(
        max_speed_kmh=config["scoring"]["geo_velocity"]["max_speed_kmh"]
    )
    geo.fit(events_with_attacks[events_with_attacks["is_attack"] == False], data["users"])

    device = DeviceTrustScorer(
        new_device_score=config["scoring"]["device"]["new_device_score"],
        unregistered_score=config["scoring"]["device"]["unregistered_score"],
        jailbroken_score=config["scoring"]["device"]["jailbroken_score"],
    )
    device.fit(events_with_attacks[events_with_attacks["is_attack"] == False], data["devices"])

    peer = PeerDeviationScorer(
        threshold_sigma=config["community"]["deviation_threshold_sigma"]
    )
    peer.fit(embeddings, communities, centroids, node_id_map, G)

    scorer = RiskScorer(
        temporal_analyzer=temporal,
        geo_analyzer=geo,
        device_scorer=device,
        peer_scorer=peer,
        config=config,
    )
    scorer.fit_entity_affinity(events_with_attacks[events_with_attacks["is_attack"] == False])

    decision = DecisionEngine(config=config)

    # ===================================================================
    # PHASE 5: Score All Events & Evaluate
    # ===================================================================
    print("\n" + "=" * 60)
    print("  PHASE 5: Scoring & Evaluation")
    print("=" * 60)

    # Score a sample of events for the dashboard
    sample_size = min(5000, len(events_with_attacks))
    sample_events = events_with_attacks.sample(n=sample_size, random_state=42)

    print(f"🔄 Scoring {sample_size} events...")
    scored_results = []
    for _, evt in sample_events.iterrows():
        result = scorer.score(evt.to_dict())
        dec = decision.decide(result)
        scored_results.append({
            "event_id": evt["event_id"],
            "timestamp": evt["timestamp"],
            "user_id": evt["user_id"],
            "resource_id": evt["resource_id"],
            "ip_city": evt.get("ip_city", ""),
            "is_attack": evt["is_attack"],
            "attack_type": evt.get("attack_type", "none"),
            "risk_score": result["total"],
            "temporal_score": result["scores"]["temporal"],
            "geo_score": result["scores"]["geo_velocity"],
            "device_score": result["scores"]["device_trust"],
            "entity_score": result["scores"]["entity_affinity"],
            "peer_score": result["scores"]["peer_alignment"],
            "decision": dec["decision"],
            "severity": dec["severity"],
            "color": dec["color"],
        })

    scored_df = pd.DataFrame(scored_results)
    scored_df.to_csv(output_dir / "scored_events.csv", index=False)
    print(f"   ✅ Scored {len(scored_df)} events → saved to scored_events.csv")

    # Print summary stats
    print("\n📊 Decision Distribution:")
    for decision_type, count in scored_df["decision"].value_counts().items():
        print(f"   {decision_type}: {count} ({count/len(scored_df)*100:.1f}%)")

    print("\n📊 Attack Detection:")
    attacks = scored_df[scored_df["is_attack"] == True]
    detected = attacks[attacks["risk_score"] > 50]
    if len(attacks) > 0:
        print(f"   Total attacks: {len(attacks)}")
        print(f"   Detected (score > 50): {len(detected)} ({len(detected)/len(attacks)*100:.1f}%)")
    
    # Comparison evaluation
    print("\n🔄 Running comparison evaluation...")
    from evaluation.compare_approaches import evaluate_approaches
    comparison, capabilities, detailed = evaluate_approaches(
        sample_events, scorer, decision
    )
    comparison.to_csv(output_dir / "comparison_metrics.csv")
    capabilities.to_csv(output_dir / "capabilities_comparison.csv", index=False)
    detailed.to_csv(output_dir / "detailed_comparison.csv", index=False)

    print("\n📊 Approach Comparison:")
    print(comparison.to_string())

    # Save privacy report
    privacy_report = anon_data.get("privacy_report", {})
    with open(output_dir / "privacy_report.json", "w") as f:
        json.dump(privacy_report, f, indent=2)

    print("\n" + "=" * 60)
    print("  ✅ Pipeline Complete!")
    print("=" * 60)
    print(f"\n  Output directory: {output_dir}")
    print(f"  Files generated:")
    for f in sorted(output_dir.iterdir()):
        if f.is_file():
            size_kb = f.stat().st_size / 1024
            print(f"    📄 {f.name} ({size_kb:.1f} KB)")

    return {
        "data": data,
        "events": events_with_attacks,
        "graph": G,
        "embeddings": embeddings,
        "node_id_map": node_id_map,
        "communities": communities,
        "centroids": centroids,
        "scored_results": scored_df,
        "comparison": comparison,
        "privacy_report": privacy_report,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentinel-Graph Engine Pipeline")
    parser.add_argument("--no-dash", action="store_true", help="Skip dashboard launch")
    parser.add_argument("--test", action="store_true", help="Quick test mode")
    args = parser.parse_args()

    results = run_pipeline(test_mode=args.test)

    if not args.no_dash:
        print("\n🚀 Launching Sentinel-Graph Engine...")
        print("   Landing Page → http://localhost:8000")
        print("   Dashboard   → http://localhost:8000/static/dashboard.html")
        print("   API Docs    → http://localhost:8000/docs")
        import subprocess
        subprocess.run([
            sys.executable, "-m", "uvicorn", "api.main:app",
            "--host", "0.0.0.0", "--port", "8000",
        ])

