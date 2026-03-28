"""
fraud_detection/api/app.py

FastAPI application exposing the fraud detection pipeline over HTTP.

Endpoints:
  POST /evaluate        — score a single transaction
  POST /evaluate/batch  — score a list of transactions
  GET  /health          — liveness check
  GET  /stats           — live decision counters
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

load_dotenv()

from fraud_detection.agents.pipeline import FraudPipeline
from fraud_detection.utils.schemas import FraudAlert, FraudDecision, Transaction

# ── In-memory stats counter ───────────────────────────────────────────────────

class StatsCounter:
    def __init__(self):
        self.total = 0
        self.decisions: dict[str, int] = defaultdict(int)
        self.total_latency_ms = 0.0
        self.started_at = datetime.utcnow()

    def record(self, alert: FraudAlert):
        self.total += 1
        self.decisions[alert.decision.value] += 1
        self.total_latency_ms += alert.processing_time_ms

    @property
    def avg_latency_ms(self) -> float:
        return round(self.total_latency_ms / max(self.total, 1), 2)

    def to_dict(self) -> dict:
        return {
            "total_evaluated": self.total,
            "decisions": dict(self.decisions),
            "avg_latency_ms": self.avg_latency_ms,
            "fraud_rate": round(
                self.decisions.get("block", 0) / max(self.total, 1), 4
            ),
            "uptime_seconds": int(
                (datetime.utcnow() - self.started_at).total_seconds()
            ),
        }


stats = StatsCounter()
pipeline: FraudPipeline | None = None


# ── Lifespan (loads models once at startup) ───────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    print("Loading fraud detection pipeline...")
    pipeline = FraudPipeline()
    print("Pipeline ready.")
    yield
    pipeline.shutdown()
    print("Pipeline shut down.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Fraud Detection API",
    description="Agentic fraud detection — multi-agent pipeline with LLM explanations.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / response schemas ────────────────────────────────────────────────

class EvaluateRequest(BaseModel):
    """
    Submit a transaction for fraud evaluation.
    All fields match the Transaction schema — id and timestamp are optional
    (auto-generated if omitted).
    """
    user_id: str
    merchant_id: str
    merchant_category: str
    amount: float
    currency: str = "USD"
    country: str
    city: str = "Unknown"
    ip_address: str = "0.0.0.0"
    device_id: str = "unknown"
    is_online: bool = True
    card_present: bool = False

    def to_transaction(self) -> Transaction:
        return Transaction(
            timestamp=datetime.utcnow(),
            **self.model_dump(),
        )


class EvaluateResponse(BaseModel):
    transaction_id: str
    decision: str
    risk_score: float
    risk_level: str
    explanation: str
    flags: list[str]
    processing_time_ms: float
    timestamp: str

    @classmethod
    def from_alert(cls, alert: FraudAlert) -> EvaluateResponse:
        all_flags = [f for s in alert.agent_signals for f in s.flags]
        return cls(
            transaction_id=alert.transaction_id,
            decision=alert.decision.value,
            risk_score=alert.risk_score,
            risk_level=alert.risk_level.value,
            explanation=alert.explanation,
            flags=all_flags,
            processing_time_ms=alert.processing_time_ms,
            timestamp=alert.timestamp.isoformat(),
        )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "pipeline_loaded": pipeline is not None,
        "timestamp": datetime.utcnow().isoformat(),
    }
@app.get("/dashboard")
async def dashboard():
    return FileResponse("fraud_detection/api/dashboard.html")

@app.get("/stats")
async def get_stats():
    return stats.to_dict()


@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(request: EvaluateRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready")

    txn = request.to_transaction()
    alert = pipeline.evaluate(txn)
    stats.record(alert)
    return EvaluateResponse.from_alert(alert)


@app.post("/evaluate/batch", response_model=list[EvaluateResponse])
async def evaluate_batch(requests: list[EvaluateRequest]):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready")

    if len(requests) > 100:
        raise HTTPException(status_code=400, detail="Max batch size is 100")

    results = []
    for req in requests:
        txn = req.to_transaction()
        alert = pipeline.evaluate(txn)
        stats.record(alert)
        results.append(EvaluateResponse.from_alert(alert))
    return results


# ── Error handler ─────────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "type": type(exc).__name__},
    )