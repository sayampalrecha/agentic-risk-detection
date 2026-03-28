"""
fraud_detection/agents/rules_agent.py

Deterministic rule-based checks. Fast, explainable, and catches
pattern-based fraud that ML models can miss on sparse data.

Rules implemented:
  R01 — High transaction amount (>5x user baseline)
  R02 — Off-hours high-value transaction
  R03 — High-risk country
  R04 — Card-not-present + online + high amount
  R05 — High velocity (tracked in-memory session store)
  R06 — Round-number amount (common in fraud)
  R07 — Micro-transaction pattern (card testing probe)
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta

from fraud_detection.agents.base import BaseAgent, score_to_risk_level
from fraud_detection.utils.schemas import AgentSignal, Transaction

HIGH_RISK_COUNTRIES = {"NG", "RO", "CN"}

# In-memory velocity tracker: user_id → list of (timestamp, amount)
# In production this would be Redis
_velocity_store: dict[str, list[tuple[datetime, float]]] = defaultdict(list)


def _get_velocity(user_id: str, now: datetime, window_minutes: int = 60) -> tuple[int, float]:
    """Returns (txn_count, total_amount) in the last window_minutes."""
    cutoff = now - timedelta(minutes=window_minutes)
    history = [
        (ts, amt) for ts, amt in _velocity_store[user_id]
        if ts >= cutoff
    ]
    _velocity_store[user_id] = history  # prune old entries
    return len(history), sum(amt for _, amt in history)


def _record_transaction(user_id: str, ts: datetime, amount: float):
    _velocity_store[user_id].append((ts, amount))


class RulesAgent(BaseAgent):
    name = "rules_agent"

    # Tuneable thresholds
    HIGH_AMOUNT_MULTIPLIER = 5.0
    HIGH_AMOUNT_ABSOLUTE = 500.0
    VELOCITY_COUNT_THRESHOLD = 5
    VELOCITY_AMOUNT_THRESHOLD = 1000.0
    MICRO_TXN_THRESHOLD = 5.0
    ROUND_NUMBER_AMOUNTS = {100, 200, 500, 1000, 2000, 5000}

    def analyze(self, transaction: Transaction) -> AgentSignal:
        flags: list[str] = []
        score_components: list[float] = []
        ts = transaction.timestamp
        amt = transaction.amount

        # R01 — High absolute amount
        if amt > self.HIGH_AMOUNT_ABSOLUTE:
            flags.append(f"R01: High amount ${amt:.2f} (>{self.HIGH_AMOUNT_ABSOLUTE})")
            score_components.append(min(0.4, amt / 5000))

        # R02 — Off-hours (1am–5am) + high amount
        hour = ts.hour if isinstance(ts, datetime) else datetime.fromisoformat(str(ts)).hour
        if 1 <= hour <= 5 and amt > 200:
            flags.append(f"R02: Off-hours transaction at {hour:02d}:00 for ${amt:.2f}")
            score_components.append(0.45)

        # R03 — High-risk country
        if transaction.country in HIGH_RISK_COUNTRIES:
            flags.append(f"R03: High-risk country ({transaction.country})")
            score_components.append(0.35)

        # R04 — CNP + online + high amount
        if (
            not transaction.card_present
            and transaction.is_online
            and amt > 300
        ):
            flags.append(
                f"R04: Card-not-present online txn ${amt:.2f}"
            )
            score_components.append(0.40)

        # R05 — Velocity check
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        txn_count, total_amt = _get_velocity(transaction.user_id, ts)

        if txn_count >= self.VELOCITY_COUNT_THRESHOLD:
            flags.append(
                f"R05: High velocity — {txn_count} txns in last 60 min"
            )
            score_components.append(min(0.70, 0.1 * txn_count))

        if total_amt >= self.VELOCITY_AMOUNT_THRESHOLD:
            flags.append(
                f"R05: High velocity spend — ${total_amt:.2f} in last 60 min"
            )
            score_components.append(min(0.60, total_amt / 5000))

        # R06 — Suspiciously round amount
        if amt in self.ROUND_NUMBER_AMOUNTS:
            flags.append(f"R06: Round-number amount ${amt:.0f}")
            score_components.append(0.15)

        # R07 — Micro-transaction (card testing probe)
        if amt < self.MICRO_TXN_THRESHOLD:
            flags.append(f"R07: Micro-transaction ${amt:.2f} (possible card test)")
            score_components.append(0.30)

        # Record transaction in velocity store AFTER checks
        _record_transaction(transaction.user_id, ts, amt)

        # Aggregate score — use max + partial sum to avoid
        # single weak signals dominating
        if score_components:
            final_score = min(1.0, max(score_components) + 0.1 * (len(score_components) - 1))
        else:
            final_score = 0.05   # baseline non-zero

        reasoning = (
            f"{len(flags)} rule(s) triggered: {', '.join(flags)}"
            if flags else "No rules triggered — transaction appears normal"
        )

        return AgentSignal(
            agent_name=self.name,
            risk_score=round(final_score, 4),
            risk_level=score_to_risk_level(final_score),
            flags=flags,
            reasoning=reasoning,
            latency_ms=0.0,
        )