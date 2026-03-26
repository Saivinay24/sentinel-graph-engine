# Sentinel-Graph Engine — Comprehensive Project Report
## PS 11: AI-Driven Identity Risk Scoring for Adaptive Access Control
### Team Velosta | Atos Srijan 2026

---

## 1. Executive Summary

Sentinel-Graph Engine is a **graph-native identity intelligence system** that dynamically evaluates identity risk using Graph Neural Networks, peer group deviation scoring, and privacy-preserving data pipelines. Unlike traditional rule-based IAM systems, Sentinel-Graph creates a **heterogeneous knowledge graph** of the entire identity ecosystem (users, roles, permissions, resources) and uses **GraphSAGE inductive learning** to detect behavioral anomalies — including authorized-but-anomalous access that traditional systems completely miss.

**Key Innovation:** Peer Group Deviation Scoring — comparing users to functional peers identified via Louvain community detection on graph embeddings, rather than just their own history.

---

## 2. Problem Analysis

### 2.1 The Challenge
- Static conditional access policies fail to adapt to changing behavior and evolving threats
- 68% of data breaches involve compromised credentials (Verizon DBIR 2025)
- Traditional IAM systems generate excessive false positives, leading to "alert fatigue"
- New users have no baseline — existing systems cannot score them accurately

### 2.2 Limitations of Current Approaches
| Approach | Limitation |
|----------|-----------|
| Rule-Based | Cannot detect novel attacks; high false positive rate; no adaptation |
| Anomaly Detection (flat) | Cannot model relationships; no peer comparison; black box |
| RBAC/ABAC | Static policies; no behavioral analysis; no risk scores |

---

## 3. Architecture

### 3.1 Five-Layer Architecture

**Layer 1 — Data + Privacy:** Simulated IAM event logs with k-anonymity (k=5) and differential privacy (ε=1.0). GDPR/HIPAA/SOX compliant by design.

**Layer 2 — Graph Construction:** Heterogeneous identity graph mapping User→Role→Permission→Resource relationships plus behavioral User→Resource access patterns.

**Layer 3 — Intelligence:** GraphSAGE inductive embeddings with Louvain community detection for peer group discovery and Isolation Forest for secondary anomaly scoring.

**Layer 4 — Scoring Engine:** Five weighted signals (temporal 20%, geo-velocity 25%, device trust 15%, entity affinity 15%, peer alignment 25%) producing a composite risk score (0-100) mapped to adaptive decisions.

**Layer 5 — Dashboard:** Real-time static HTML SPA dashboard served by FastAPI with 10 views: Live Feed, User Risk, Peer Map, Geo Anomaly, Compliance, Metrics, Threats, Assets, Logs, and Reports.

### 3.2 Technology Stack
- **Python 3.10+** — Core language
- **PyTorch Geometric** — GraphSAGE implementation
- **NetworkX** — Graph construction and analysis
- **scikit-learn** — Isolation Forest, PCA, K-Means
- **FastAPI + Uvicorn** — REST API + static file serving
- **Static HTML SPA** — Dashboard with Tailwind CSS, Canvas, SVG
- **UMAP** — Dimensionality reduction for visualization

---

## 4. Key Innovations

### 4.1 Peer Group Deviation Scoring (Primary Differentiator)
Traditional UEBA systems compare a user to their own history. Our approach:
1. Generate node embeddings using GraphSAGE on the identity graph
2. Apply Louvain community detection to discover functional peer groups
3. For each user, compute cosine distance from their peer group centroid
4. Flag users whose behavior deviates >2σ from their peer group

**This catches scenarios that no other approach can:**
- HR user accessing financial databases (has permission, but no HR peer does this)
- New employee accessing team resources (no history, but all peers do this — allow)
- Admin slowly expanding access scope (gradual change, but diverging from admin peers)

### 4.2 GraphSAGE Inductive Learning
Unlike transductive methods (DeepWalk, Node2Vec), GraphSAGE can generate embeddings for **unseen nodes**, meaning:
- New employees get scored from Day 1 based on their graph context
- Temporary contractors can be assessed without historical baselines

### 4.3 Privacy-Preserving Pipeline
- **k-Anonymity (k=5):** Quasi-identifier generalization ensures no individual can be re-identified
- **ε-Differential Privacy (ε=1.0):** Laplace noise injection provides mathematical privacy guarantees
- **Data Masking:** All PII pseudonymized throughout the pipeline

---

## 5. Attack Detection Capabilities

| Attack Type | Description | Sentinel-Graph Signal |
|-------------|-------------|----------------------|
| Impossible Travel | Login from distant cities in impossibly short time | Geo-velocity (primary) |
| Privilege Escalation | Junior user accessing critical cross-dept resources | Entity affinity + Peer deviation |
| Credential Stuffing | Many failed logins → success from new device | Device trust + Temporal |
| Insider Lateral Movement | HR user browsing financial DBs on weekends | Peer deviation (primary) |
| Account Takeover | New device + new location + off-hours | All signals contribute |

---

## 6. Business Impact for Atos

### 6.1 Direct Alignment
Identity & Access Management is a core Atos service line. This solution:
- Addresses the #1 enterprise security challenge (credential-based attacks)
- Provides a path to differentiate Atos IAM offerings with AI/graph intelligence
- Enables privacy-compliant deployment across GDPR/HIPAA regulated clients

### 6.2 Quantifiable Benefits
- **1.3% false positive rate** vs 5% for ML-only and 20% for rule-based → 4x reduction in false alerts
- **85.5% of attacks hard-blocked**, 97% flagged or escalated → fewer security gaps
- **Day-1 user scoring** via inductive GNN → no cold start problem
- **Privacy by design** → deployable in regulated industries without PII exposure

---

## 7. Evaluation Results

### 7.1 Three-Way Comparison

| Metric | Rule-Based | ML-Only (Isolation Forest) | Sentinel-Graph |
|--------|-----------|---------------------------|----------------|
| Recall (Detection Rate) | 13.2% | 84.2% | **85.5%** |
| Precision | 100% | 20.7% | **51.2%** |
| F1 Score | 23.3% | 33.2% | **64.0%** |
| False Positive Rate | 0.0% | 5.0% | **1.3%** |
| AUC-ROC | N/A | ~0.97 | **0.993** |
| New User Scoring | ❌ | ❌ | ✅ |
| Insider Detection | ❌ | ❌ | ✅ |
| Privacy Compliant | ❌ | ❌ | ✅ |
| Explainable | Partially | ❌ | ✅ |

> **Note on precision vs recall tradeoff:** Rule-based precision is 100% because its hard rules never fire incorrectly — but it only catches 13.2% of attacks. ML-only's precision collapses to 20.7% because it generates too many false positives (5% FPR). Sentinel-Graph's 51.2% precision is intentional: we optimize for recall (catching attacks) while keeping FPR low at 1.3%. In security, a missed attack costs orders of magnitude more than a false block.

---

## 8. Future Roadmap

1. **Temporal Graph Networks (TGN)** — Model time-evolving behavior
2. **Federated Learning** — Train across organizational boundaries without sharing data
3. **Integration with Okta/Azure AD** — Real-world IAM log ingestion
4. **Reinforcement Learning** — Auto-tune scoring weights based on analyst feedback
5. **SOC Integration** — SIEM connector for automated incident response

---

*Team Velosta — Atos Srijan 2026*
