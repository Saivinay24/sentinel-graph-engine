"""
Sentinel-Graph API — FastAPI Application
Identity Threat Detection & Response (ITDR) backend.

Endpoints:
  POST /score-event             — Score a single access event
  GET  /health                  — Service health & readiness check
  GET  /user/{user_id}/risk-history — Last 30 decisions for a user
  GET  /peer-groups             — All discovered peer groups
  GET  /metrics                 — Precision/Recall/F1/FPR against attack labels
  POST /batch-score             — Score multiple events
  POST /users/{user_id}/assign-peer-group — Assign new user to nearest peer group
  GET  /live-events             — Last 50 events for live feed
"""

from __future__ import annotations

import logging
import sys
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import api.loader as _loader_module
from api.explanation import generate_explanation
from api.loader import SentinelEngine, get_engine, load_engine
from api.models import (
    AssignPeerGroupRequest,
    AssignPeerGroupResponse,
    BatchScoreRequest,
    BatchScoreResponse,
    ComplianceInfo,
    HealthResponse,
    LiveEventEntry,
    LiveEventsResponse,
    MetricsResponse,
    PeerGroupDetail,
    PeerGroupsResponse,
    PeerGroupSummary,
    RiskHistoryEntry,
    RiskHistoryResponse,
    ScoreEventRequest,
    ScoreEventResponse,
    SignalsResponse,
)
from api.peer_group_update import assign_new_user_to_peer_group

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lifespan: load engine at startup, release at shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Sentinel-Graph API starting up — loading engine...")
    _loader_module.engine = load_engine()
    logger.info("Engine ready. model_loaded=%s, users=%d",
                _loader_module.engine.model_loaded, _loader_module.engine.users_loaded)
    yield
    logger.info("Sentinel-Graph API shutting down.")
    _loader_module.engine = None


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Sentinel-Graph ITDR API",
    description=(
        "AI-driven Identity Threat Detection & Response (ITDR) backend. "
        "Scores access events in real-time using GraphSAGE peer group embeddings, "
        "geo-velocity, temporal, device trust, and entity affinity signals."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# CORS — allow all origins for demo/dashboard access
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Middleware: request ID + latency headers
# ---------------------------------------------------------------------------

@app.middleware("http")
async def add_request_metadata(request: Request, call_next):
    request_id = str(uuid.uuid4())
    t_start = time.perf_counter()

    response: Response = await call_next(request)

    elapsed_ms = int((time.perf_counter() - t_start) * 1000)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time-Ms"] = str(elapsed_ms)
    return response


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize_signal(raw_score: float) -> float:
    """Convert a 0-100 scorer output to a 0-1 normalised signal value."""
    return round(min(1.0, max(0.0, raw_score / 100.0)), 4)


def _compute_confidence(risk_score: float, signals: Dict[str, float]) -> float:
    """
    Heuristic confidence estimate.
    Higher confidence when multiple strong signals agree.
    """
    elevated = [v for v in signals.values() if v >= 0.4]
    n_elevated = len(elevated)
    base_conf = 0.55 + 0.08 * n_elevated
    # Scale up slightly for extreme scores
    if risk_score >= 85 or risk_score <= 15:
        base_conf = min(0.98, base_conf + 0.08)
    return round(min(0.99, max(0.50, base_conf)), 4)


def _get_peer_info(eng: SentinelEngine, user_id: str, peer_score_raw: float) -> PeerGroupSummary:
    """Resolve peer group metadata for a user."""
    comm_id = eng.communities.get(user_id)
    if comm_id is None:
        return PeerGroupSummary(
            id="unknown",
            label="Unknown",
            size=0,
            deviation_percentile=0.0,
        )

    label = eng.get_community_label(comm_id)
    stats = eng.peer.community_stats.get(comm_id, {}) if eng.peer else {}
    size = int(stats.get("member_count", 0))

    # Compute percentile within community: map peer_score (0-100) to percentile
    # Higher peer score → higher percentile (more deviant)
    deviation_percentile = round(min(99.9, max(0.1, peer_score_raw)), 1)

    return PeerGroupSummary(
        id=comm_id,
        label=label,
        size=size,
        deviation_percentile=deviation_percentile,
    )


def _build_audit_id(user_id: str, timestamp: str) -> str:
    """Generate a deterministic audit log ID."""
    ts_compact = timestamp.replace("-", "").replace(":", "").replace("T", "_").split("+")[0].split("Z")[0]
    return f"audit_{ts_compact}_{user_id}"


def _decision_key_to_str(decision_key: str) -> str:
    """Return the display name for a decision key."""
    return {
        "ALLOW": "ALLOW",
        "STEP_UP_MFA": "STEP_UP_MFA",
        "BLOCK_ALERT": "BLOCK_ALERT",
        "BLOCK_SOC": "BLOCK_SOC",
    }.get(decision_key, decision_key)


def _severity_from_decision(decision_key: str) -> str:
    return {
        "ALLOW": "low",
        "STEP_UP_MFA": "medium",
        "BLOCK_ALERT": "high",
        "BLOCK_SOC": "critical",
    }.get(decision_key, "unknown")


def _score_single_event(
    req: ScoreEventRequest,
    eng: SentinelEngine,
) -> ScoreEventResponse:
    """Core scoring logic, shared by /score-event and /batch-score."""
    t0 = time.perf_counter()

    event_dict = req.to_event_dict()

    # Run risk scorer
    risk_result = eng.risk_scorer.score(event_dict)
    decision_result = eng.decision_engine.decide(risk_result)

    risk_score = float(risk_result["total"])
    scores_raw = risk_result.get("scores", {})
    decision_key = decision_result["decision_key"]

    # Normalise signals to 0-1
    signals_norm = {
        "peer_deviation":  _normalize_signal(scores_raw.get("peer_alignment", 0.0)),
        "geo_velocity":    _normalize_signal(scores_raw.get("geo_velocity", 0.0)),
        "device_trust":    _normalize_signal(scores_raw.get("device_trust", 0.0)),
        "temporal":        _normalize_signal(scores_raw.get("temporal", 0.0)),
        "entity_affinity": _normalize_signal(scores_raw.get("entity_affinity", 0.0)),
    }

    confidence = _compute_confidence(risk_score, signals_norm)

    # Peer group info
    peer_info = _get_peer_info(eng, req.user_id, scores_raw.get("peer_alignment", 0.0))
    peer_group_dict = {
        "label": peer_info.label,
        "size": peer_info.size,
    }

    # Natural language explanation
    explanation = generate_explanation(
        user_id=req.user_id,
        resource_id=req.resource_id,
        signals=signals_norm,
        decision_key=decision_key,
        peer_group=peer_group_dict,
        event_dict=event_dict,
    )

    # Compliance metadata
    ts_str = req.timestamp.isoformat()
    audit_id = _build_audit_id(req.user_id, ts_str)
    compliance = ComplianceInfo(
        nis2_article_21=True,
        audit_id=audit_id,
        gdpr_pseudonymized=True,
    )

    latency_ms = int((time.perf_counter() - t0) * 1000)

    return ScoreEventResponse(
        risk_score=risk_score,
        decision=_decision_key_to_str(decision_key),
        confidence=confidence,
        signals=SignalsResponse(
            peer_deviation=signals_norm["peer_deviation"],
            geo_velocity=signals_norm["geo_velocity"],
            device_trust=signals_norm["device_trust"],
            temporal=signals_norm["temporal"],
            entity_affinity=signals_norm["entity_affinity"],
        ),
        explanation=explanation,
        peer_group=peer_info,
        compliance=compliance,
        latency_ms=latency_ms,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health() -> HealthResponse:
    """Service health and readiness check."""
    try:
        eng = get_engine()
        uptime = round(time.time() - eng.startup_time, 1)
        return HealthResponse(
            status="ok",
            model_loaded=eng.model_loaded,
            users_loaded=eng.users_loaded,
            version="1.0.0",
            uptime_seconds=uptime,
        )
    except RuntimeError:
        return HealthResponse(
            status="loading",
            model_loaded=False,
            users_loaded=0,
            version="1.0.0",
        )


@app.post("/score-event", response_model=ScoreEventResponse, tags=["Scoring"])
async def score_event(req: ScoreEventRequest) -> ScoreEventResponse:
    """
    Score a single access event and return a risk decision with full signal breakdown.
    """
    eng = get_engine()
    if not eng.model_loaded:
        raise HTTPException(status_code=503, detail="Model not yet loaded. Retry in a moment.")

    try:
        return _score_single_event(req, eng)
    except Exception as exc:
        logger.exception("Error scoring event for user %s: %s", req.user_id, exc)
        raise HTTPException(status_code=500, detail=f"Scoring error: {exc}") from exc


@app.post("/batch-score", response_model=BatchScoreResponse, tags=["Scoring"])
async def batch_score(req: BatchScoreRequest) -> BatchScoreResponse:
    """Score multiple events in a single request (max 500)."""
    eng = get_engine()
    if not eng.model_loaded:
        raise HTTPException(status_code=503, detail="Model not yet loaded.")

    t0 = time.perf_counter()
    results: List[ScoreEventResponse] = []
    errors: List[str] = []

    for i, event_req in enumerate(req.events):
        try:
            result = _score_single_event(event_req, eng)
            results.append(result)
        except Exception as exc:
            logger.warning("Batch score error at index %d: %s", i, exc)
            errors.append(f"Event {i}: {exc}")

    if errors and not results:
        raise HTTPException(status_code=500, detail="; ".join(errors))

    total_latency = int((time.perf_counter() - t0) * 1000)
    return BatchScoreResponse(
        results=results,
        total=len(results),
        latency_ms=total_latency,
    )


@app.get("/user/{user_id}/risk-history", response_model=RiskHistoryResponse, tags=["Users"])
async def risk_history(user_id: str) -> RiskHistoryResponse:
    """Return the last 30 scored decisions for a specific user."""
    eng = get_engine()

    df = eng.scored_events_df
    if df is None or len(df) == 0:
        return RiskHistoryResponse(user_id=user_id, history=[], count=0)

    user_events = df[df["user_id"] == user_id].copy()
    if "timestamp" in user_events.columns:
        user_events = user_events.sort_values("timestamp", ascending=False)

    user_events = user_events.head(30)

    history: List[RiskHistoryEntry] = []
    for _, row in user_events.iterrows():
        history.append(RiskHistoryEntry(
            timestamp=str(row.get("timestamp", "")),
            resource_id=str(row.get("resource_id", "")),
            risk_score=float(row.get("risk_score", 0.0)),
            decision=str(row.get("decision", "UNKNOWN")),
            attack_type=str(row.get("attack_type", "none")) if pd.notna(row.get("attack_type")) else None,
            ip_city=str(row.get("ip_city", "")) if pd.notna(row.get("ip_city")) else None,
            action=str(row.get("action", "")) if pd.notna(row.get("action")) else None,
        ))

    return RiskHistoryResponse(user_id=user_id, history=history, count=len(history))


@app.get("/peer-groups", response_model=PeerGroupsResponse, tags=["Peer Groups"])
async def peer_groups() -> PeerGroupsResponse:
    """Return all discovered peer groups with sizes and dominant departments."""
    eng = get_engine()

    if not eng.communities:
        return PeerGroupsResponse(peer_groups=[], total=0)

    # Aggregate community membership
    comm_members: Dict[str, List[str]] = defaultdict(list)
    for uid, comm_id in eng.communities.items():
        comm_members[comm_id].append(uid)

    # Optionally compute avg risk score from scored events
    comm_avg_risk: Dict[str, Optional[float]] = {}
    if eng.scored_events_df is not None and len(eng.scored_events_df) > 0 and "risk_score" in eng.scored_events_df.columns:
        for comm_id, members in comm_members.items():
            member_events = eng.scored_events_df[eng.scored_events_df["user_id"].isin(members)]
            if len(member_events) > 0:
                comm_avg_risk[comm_id] = round(float(member_events["risk_score"].mean()), 2)
            else:
                comm_avg_risk[comm_id] = None

    # Dominant department per community
    dept_map: Dict[str, str] = {}
    if eng.users_df is not None and "department" in eng.users_df.columns:
        uid_dept = dict(zip(eng.users_df["user_id"], eng.users_df["department"]))
        for comm_id, members in comm_members.items():
            depts = [uid_dept.get(uid, "Unknown") for uid in members]
            dept_counts: Dict[str, int] = defaultdict(int)
            for d in depts:
                dept_counts[d] += 1
            dept_map[comm_id] = max(dept_counts, key=dept_counts.get)
    else:
        for comm_id in comm_members:
            dept_map[comm_id] = eng.get_community_label(comm_id)

    groups: List[PeerGroupDetail] = []
    for comm_id in sorted(comm_members.keys()):
        members = comm_members[comm_id]
        groups.append(PeerGroupDetail(
            id=comm_id,
            label=eng.get_community_label(comm_id),
            size=len(members),
            dominant_department=dept_map.get(comm_id, "Unknown"),
            avg_risk_score=comm_avg_risk.get(comm_id),
            member_sample=members,  # All member user IDs for full graph mapping
        ))

    return PeerGroupsResponse(peer_groups=groups, total=len(groups))


@app.get("/metrics", response_model=MetricsResponse, tags=["Evaluation"])
async def metrics() -> MetricsResponse:
    """
    Compute precision, recall, F1, and FPR against ground-truth attack labels
    in scored_events.csv. Treats BLOCK_ALERT and BLOCK_SOC as positive predictions.
    """
    eng = get_engine()

    df = eng.scored_events_df
    if df is None or len(df) == 0:
        raise HTTPException(status_code=503, detail="No scored events available for metrics computation.")

    required_cols = {"decision", "attack_type"}
    if not required_cols.issubset(df.columns):
        raise HTTPException(
            status_code=503,
            detail=f"scored_events.csv missing columns: {required_cols - set(df.columns)}",
        )

    # Ground truth: is_attack = attack_type not in {"none", "normal", NaN}
    def _is_attack(at: Any) -> bool:
        if pd.isna(at):
            return False
        return str(at).strip().lower() not in ("none", "normal", "")

    # Prediction: positive = any BLOCK decision (handles both key and display formats)
    _block_markers = {"block_alert", "block_soc", "block + admin alert", "block + soc + session terminate"}
    def _is_flagged(d: Any) -> bool:
        return str(d).strip().lower() in _block_markers

    y_true = df["attack_type"].apply(_is_attack).astype(int)
    y_pred = df["decision"].apply(_is_flagged).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # Optional AUC-ROC if risk_score column present
    auc_roc: Optional[float] = None
    if "risk_score" in df.columns:
        try:
            from sklearn.metrics import roc_auc_score  # type: ignore
            auc_roc = round(float(roc_auc_score(y_true, df["risk_score"].fillna(0))), 4)
        except Exception:
            pass

    return MetricsResponse(
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1_score=round(f1, 4),
        false_positive_rate=round(fpr, 4),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        true_negatives=tn,
        total_events=int(len(df)),
        attack_events=int(y_true.sum()),
        auc_roc=auc_roc,
    )


@app.post(
    "/users/{user_id}/assign-peer-group",
    response_model=AssignPeerGroupResponse,
    tags=["Users"],
)
async def assign_peer_group(
    user_id: str,
    req: AssignPeerGroupRequest,
) -> AssignPeerGroupResponse:
    """
    Inductively assign a new (unseen) user to the nearest peer group.
    Uses cosine similarity between a synthesised attribute-derived embedding
    and the stored community centroids.
    """
    eng = get_engine()

    result = assign_new_user_to_peer_group(
        user_id=user_id,
        department=req.department,
        role_id=req.role_id,
        attributes=req.attributes,
        centroids=eng.centroids,
        community_labels=eng.community_labels,
        community_stats=eng.peer.community_stats if eng.peer else None,
    )

    return AssignPeerGroupResponse(
        peer_group_id=result["peer_group_id"],
        peer_group_label=result["peer_group_label"],
        confidence=result["confidence"],
        department_match=result.get("department_match"),
    )


@app.get("/live-events", response_model=LiveEventsResponse, tags=["Live Feed"])
async def live_events(limit: int = 50) -> LiveEventsResponse:
    """Return scored events for the live feed, sorted by timestamp descending.
    Use limit=0 to return ALL events (for peer map / analytics).
    For limit>0, the feed rotates through events (~1 row/second) to simulate streaming."""
    eng = get_engine()

    df = eng.scored_events_df
    if df is None or len(df) == 0:
        return LiveEventsResponse(events=[], count=0)

    if limit == 0:
        # Analytics / peer map: full dataset sorted by timestamp
        df_out = df.sort_values("timestamp", ascending=False) if "timestamp" in df.columns else df.iloc[::-1]
    else:
        # Live feed: rotate through events using time-based offset so the feed
        # visually advances ~1 new event per second on each 3-second poll.
        n = len(df)
        offset = int(time.time()) % n
        indices = [(offset + i) % n for i in range(min(limit, n))]
        df_out = df.iloc[indices]

    events: List[LiveEventEntry] = []
    for i, row in df_out.iterrows():
        decision_key = str(row.get("decision", "ALLOW")).strip().upper()
        events.append(LiveEventEntry(
            event_id=f"evt_{i}",
            user_id=str(row.get("user_id", "")),
            resource_id=str(row.get("resource_id", "")),
            timestamp=str(row.get("timestamp", "")),
            risk_score=float(row.get("risk_score", 0.0)),
            decision=decision_key,
            attack_type=str(row.get("attack_type", "none")) if pd.notna(row.get("attack_type")) else None,
            ip_city=str(row.get("ip_city", "")) if pd.notna(row.get("ip_city")) else None,
            action=str(row.get("action", "")) if pd.notna(row.get("action")) else None,
            severity=_severity_from_decision(decision_key),
        ))

    return LiveEventsResponse(events=events, count=len(events))


# ---------------------------------------------------------------------------
# Global exception handler — return JSON for all unhandled errors
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred.", "error": str(exc)},
    )


# ---------------------------------------------------------------------------
# Root redirect → landing page
# ---------------------------------------------------------------------------

@app.get("/", tags=["Root"], include_in_schema=False)
async def root():
    """Redirect to the landing page."""
    return RedirectResponse(url="/static/index.html")


# ---------------------------------------------------------------------------
# Static file serving — dashboard + landing page
# ---------------------------------------------------------------------------
_static_dir = PROJECT_ROOT / "static"
if _static_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


# ---------------------------------------------------------------------------
# Entry point for direct execution: uvicorn api.main:app
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
