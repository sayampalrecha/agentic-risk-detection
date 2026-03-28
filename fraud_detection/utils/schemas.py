"""
fraud_detection/utils/schemas.py
Shared Pydantic models — the common language between all agents and the API.
"""

from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
import uuid


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class FraudDecision(str, Enum):
    APPROVE = "approve"
    REVIEW = "review"         # Human review queue
    BLOCK = "block"
    STEP_UP = "step_up"       # Trigger additional auth


class Transaction(BaseModel):
    """A single payment transaction flowing through the system."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime
    user_id: str
    merchant_id: str
    merchant_category: str
    amount: float
    currency: str = "USD"
    country: str
    city: str
    ip_address: str
    device_id: str
    is_online: bool
    card_present: bool

    # Ground truth (only in training data, never sent to agents in prod)
    is_fraud: Optional[bool] = None


class AgentSignal(BaseModel):
    """What a specialist agent returns to the decision agent."""
    agent_name: str
    risk_score: float = Field(ge=0.0, le=1.0)
    risk_level: RiskLevel
    flags: list[str] = Field(default_factory=list)
    reasoning: str
    latency_ms: float


class FraudAlert(BaseModel):
    """Final decision output from the decision agent."""
    transaction_id: str
    decision: FraudDecision
    risk_score: float = Field(ge=0.0, le=1.0)
    risk_level: RiskLevel
    explanation: str                   # Human-readable reason (LLM-generated)
    agent_signals: list[AgentSignal]
    processing_time_ms: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)