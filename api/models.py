"""
Sentinel-Graph API — Pydantic Request/Response Schemas
All input validation and output serialization models for the FastAPI backend.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class ScoreEventRequest(BaseModel):
    """Single event scoring request."""

    user_id: str = Field(..., description="Unique user identifier (e.g. U123)")
    resource_id: str = Field(..., description="Target resource identifier")
    timestamp: datetime = Field(..., description="Event timestamp in ISO-8601 format")
    source_ip: str = Field(..., description="Source IP address")
    device_id: str = Field(..., description="Device identifier")
    action: str = Field(default="READ", description="Action performed (READ/WRITE/DELETE/ADMIN)")
    session_id: Optional[str] = Field(default=None, description="Session identifier")

    # Geo fields
    ip_lat: Optional[float] = Field(default=None, description="IP geolocation latitude")
    ip_lon: Optional[float] = Field(default=None, description="IP geolocation longitude")
    ip_city: Optional[str] = Field(default=None, description="IP geolocation city name")

    # Device fields
    device_fingerprint: Optional[str] = Field(default=None, description="Browser/device fingerprint hash")
    device_registered: Optional[bool] = Field(default=True, description="Whether the device is org-registered")
    os: Optional[str] = Field(default=None, description="Operating system string")
    browser: Optional[str] = Field(default=None, description="Browser string (name/version)")

    # Optional extras
    login_success: Optional[bool] = Field(default=True, description="Whether the login succeeded")

    @field_validator("timestamp", mode="before")
    @classmethod
    def parse_timestamp(cls, v):
        if isinstance(v, str):
            return datetime.fromisoformat(v.replace("Z", "+00:00"))
        return v

    def to_event_dict(self) -> Dict[str, Any]:
        """Convert to the flat dict format expected by scorers."""
        return {
            "user_id": self.user_id,
            "resource_id": self.resource_id,
            "timestamp": self.timestamp.isoformat(),
            "source_ip": self.source_ip,
            "device_id": self.device_id,
            "action": self.action,
            "session_id": self.session_id,
            "ip_lat": self.ip_lat if self.ip_lat is not None else 0.0,
            "ip_lon": self.ip_lon if self.ip_lon is not None else 0.0,
            "ip_city": self.ip_city or "Unknown",
            "device_fingerprint": self.device_fingerprint or self.device_id,
            "device_registered": self.device_registered if self.device_registered is not None else True,
            "os": self.os or "",
            "browser": self.browser or "",
            "login_success": self.login_success if self.login_success is not None else True,
        }


class BatchScoreRequest(BaseModel):
    """Batch event scoring request."""

    events: List[ScoreEventRequest] = Field(
        ..., min_length=1, max_length=500, description="List of events to score (max 500)"
    )


class AssignPeerGroupRequest(BaseModel):
    """Request to assign a new user to the nearest peer group."""

    role_id: str = Field(..., description="User's role identifier (e.g. R_HR_ANALYST)")
    department: str = Field(..., description="User's department name")
    attributes: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional user attributes for embedding synthesis",
    )


# ---------------------------------------------------------------------------
# Response Sub-models
# ---------------------------------------------------------------------------

class SignalsResponse(BaseModel):
    peer_deviation: float = Field(..., description="Peer group deviation score (0-1)")
    geo_velocity: float = Field(..., description="Geo-velocity anomaly score (0-1)")
    device_trust: float = Field(..., description="Device trust penalty score (0-1)")
    temporal: float = Field(..., description="Temporal anomaly score (0-1)")
    entity_affinity: float = Field(..., description="Entity affinity anomaly score (0-1)")


class PeerGroupSummary(BaseModel):
    id: str = Field(..., description="Internal community identifier")
    label: str = Field(..., description="Human-readable peer group label")
    size: int = Field(..., description="Number of users in this peer group")
    deviation_percentile: float = Field(..., description="User's deviation percentile within the group (0-100)")


class ComplianceInfo(BaseModel):
    nis2_article_21: bool = Field(default=True, description="NIS2 Article 21 compliance flag")
    audit_id: str = Field(..., description="Unique audit log identifier for this decision")
    gdpr_pseudonymized: bool = Field(default=True, description="GDPR pseudonymisation applied")


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class ScoreEventResponse(BaseModel):
    risk_score: float = Field(..., description="Composite risk score (0-100)")
    decision: str = Field(..., description="Access control decision string")
    confidence: float = Field(..., description="Model confidence (0-1)")
    signals: SignalsResponse
    explanation: str = Field(..., description="Human-readable NL explanation")
    peer_group: PeerGroupSummary
    compliance: ComplianceInfo
    latency_ms: int = Field(..., description="End-to-end scoring latency in milliseconds")


class BatchScoreResponse(BaseModel):
    results: List[ScoreEventResponse]
    total: int = Field(..., description="Number of events scored")
    latency_ms: int = Field(..., description="Total batch latency in milliseconds")


class HealthResponse(BaseModel):
    status: str = Field(default="ok")
    model_loaded: bool
    users_loaded: int
    version: str = Field(default="1.0.0")
    uptime_seconds: Optional[float] = None


class RiskHistoryEntry(BaseModel):
    timestamp: str
    resource_id: str
    risk_score: float
    decision: str
    attack_type: Optional[str] = None
    ip_city: Optional[str] = None
    action: Optional[str] = None


class RiskHistoryResponse(BaseModel):
    user_id: str
    history: List[RiskHistoryEntry]
    count: int


class PeerGroupDetail(BaseModel):
    id: str
    label: str
    size: int
    dominant_department: str
    avg_risk_score: Optional[float] = None
    member_sample: List[str] = Field(default_factory=list)


class PeerGroupsResponse(BaseModel):
    peer_groups: List[PeerGroupDetail]
    total: int


class MetricsResponse(BaseModel):
    precision: float
    recall: float
    f1_score: float
    false_positive_rate: float
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    total_events: int
    attack_events: int
    auc_roc: Optional[float] = None


class LiveEventEntry(BaseModel):
    event_id: Optional[str] = None
    user_id: str
    resource_id: str
    timestamp: str
    risk_score: float
    decision: str
    attack_type: Optional[str] = None
    ip_city: Optional[str] = None
    action: Optional[str] = None
    severity: Optional[str] = None


class LiveEventsResponse(BaseModel):
    events: List[LiveEventEntry]
    count: int


class AssignPeerGroupResponse(BaseModel):
    peer_group_id: str
    peer_group_label: str
    confidence: float
    department_match: Optional[str] = None
