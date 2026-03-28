"""
fraud_detection/agents/base.py
Abstract base class that every specialist agent implements.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

from fraud_detection.utils.schemas import AgentSignal, RiskLevel, Transaction


def score_to_risk_level(score: float) -> RiskLevel:
    if score < 0.35:
        return RiskLevel.LOW
    elif score < 0.70:
        return RiskLevel.MEDIUM
    return RiskLevel.HIGH


class BaseAgent(ABC):
    """
    Every agent must implement `analyze()`.
    It receives a Transaction and returns an AgentSignal.
    """

    name: str = "base_agent"

    def run(self, transaction: Transaction) -> AgentSignal:
        """Wraps analyze() with timing."""
        start = time.perf_counter()
        signal = self.analyze(transaction)
        signal.latency_ms = round((time.perf_counter() - start) * 1000, 2)
        return signal

    @abstractmethod
    def analyze(self, transaction: Transaction) -> AgentSignal:
        """Core analysis logic — implement in each subclass."""
        ...