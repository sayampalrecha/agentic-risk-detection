"""
fraud_detection/agents/anomaly_agent.py

Uses the trained ML models to score a transaction.
Combines XGBoost (supervised) and Isolation Forest (unsupervised)
into a single blended risk score.
"""

from __future__ import annotations

import pandas as pd

from fraud_detection.agents.base import BaseAgent, score_to_risk_level
from fraud_detection.models.features import build_features
from fraud_detection.models.trainer import FraudModels
from fraud_detection.utils.schemas import AgentSignal, Transaction


class AnomalyAgent(BaseAgent):
    name = "anomaly_agent"

    def __init__(self):
        self.models = FraudModels.load()

    def analyze(self, transaction: Transaction) -> AgentSignal:
        # Build a single-row DataFrame from the transaction
        row = pd.DataFrame([transaction.model_dump()])
        row["timestamp"] = pd.to_datetime(row["timestamp"])

        # Feature engineering (same pipeline as training)
        feats = build_features(row)

        # Get scores from both models
        xgb_score = float(self.models.predict_xgb(feats)[0])
        iso_score = float(self.models.predict_iso(feats)[0])

        # Blend: XGBoost gets more weight (supervised signal is stronger)
        blended_score = round(0.65 * xgb_score + 0.35 * iso_score, 4)

        flags = []
        if xgb_score > 0.6:
            flags.append(f"XGBoost high fraud probability ({xgb_score:.2f})")
        if iso_score > 0.7:
            flags.append(f"Isolation Forest anomaly score high ({iso_score:.2f})")
        if blended_score > 0.5:
            flags.append(f"Blended score exceeds threshold ({blended_score:.2f})")

        reasoning = (
            f"XGBoost score: {xgb_score:.3f}, "
            f"Isolation Forest score: {iso_score:.3f}, "
            f"Blended: {blended_score:.3f}"
        )

        return AgentSignal(
            agent_name=self.name,
            risk_score=blended_score,
            risk_level=score_to_risk_level(blended_score),
            flags=flags,
            reasoning=reasoning,
            latency_ms=0.0,
        )