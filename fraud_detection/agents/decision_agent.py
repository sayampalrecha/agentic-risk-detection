"""
fraud_detection/agents/decision_agent.py

Aggregates all specialist agent signals and makes a final decision.
Supported LLM providers: groq | anthropic | openai | mock
Set LLM_PROVIDER and the matching API key in your .env file.
"""

from __future__ import annotations

import os
from datetime import datetime

from fraud_detection.config import config
from fraud_detection.utils.schemas import (
    AgentSignal,
    FraudAlert,
    FraudDecision,
    RiskLevel,
    Transaction,
)


def _weighted_score(signals: list[AgentSignal]) -> float:
    weights = {
        "anomaly_agent": 0.45,
        "rules_agent":   0.55,
    }
    total_weight = 0.0
    weighted_sum = 0.0
    for sig in signals:
        w = weights.get(sig.agent_name, 0.5)
        weighted_sum += sig.risk_score * w
        total_weight += w
    return round(weighted_sum / max(total_weight, 1e-9), 4)


def _score_to_decision(score: float, signals: list[AgentSignal]) -> FraudDecision:
    cfg = config.agent
    any_high = any(s.risk_level == RiskLevel.HIGH for s in signals)

    if score >= cfg.high_risk_threshold or any_high:
        return FraudDecision.BLOCK

    if score >= cfg.low_risk_threshold:
        is_online = any(
            "online" in f.lower() or "cnp" in f.lower()
            for s in signals for f in s.flags
        )
        return FraudDecision.STEP_UP if is_online else FraudDecision.REVIEW

    return FraudDecision.APPROVE


def _build_prompt(transaction: Transaction, signals: list[AgentSignal], score: float) -> str:
    all_flags = [f for s in signals for f in s.flags]
    flags_text = "\n".join(f"  - {f}" for f in all_flags) if all_flags else "  - None"
    return f"""You are a fraud analyst AI. Analyze this transaction and provide a concise explanation of the fraud risk.

TRANSACTION:
  Amount:   ${transaction.amount:.2f} {transaction.currency}
  User:     {transaction.user_id}
  Merchant: {transaction.merchant_id} ({transaction.merchant_category})
  Country:  {transaction.country}
  Channel:  {"Online" if transaction.is_online else "In-store"}, card {"present" if transaction.card_present else "NOT present"}
  Time:     {transaction.timestamp}

AGENT SIGNALS:
{chr(10).join(f"  {s.agent_name}: score={s.risk_score:.3f} ({s.risk_level.value}) — {s.reasoning}" for s in signals)}

FLAGS RAISED:
{flags_text}

AGGREGATE RISK SCORE: {score:.3f}

Provide a 2-3 sentence explanation of WHY this transaction is or isn't suspicious. Be specific. Plain prose, no bullet points."""


def _explain_groq(prompt: str) -> str:
    try:
        from groq import Groq
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=config.agent.llm_max_tokens,
            temperature=config.agent.llm_temperature,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Groq unavailable: {e}]"


def _explain_anthropic(prompt: str) -> str:
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        message = client.messages.create(
            model=config.agent.llm_model,
            max_tokens=config.agent.llm_max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text.strip()
    except Exception as e:
        return f"[Anthropic unavailable: {e}]"


def _explain_openai(prompt: str) -> str:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=config.agent.llm_max_tokens,
            temperature=config.agent.llm_temperature,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[OpenAI unavailable: {e}]"


def _explain_mock(transaction, signals, score, decision) -> str:
    all_flags = [f for s in signals for f in s.flags]
    if not all_flags:
        return (
            f"Transaction of ${transaction.amount:.2f} at {transaction.merchant_category} "
            f"shows no significant risk signals. Aggregate score {score:.2f}. "
            f"Decision: {decision.value.upper()}."
        )
    flag_summary = "; ".join(all_flags[:3])
    return (
        f"Transaction flagged with risk score {score:.2f}. "
        f"Key signals: {flag_summary}. "
        f"Decision: {decision.value.upper()} based on {len(all_flags)} active rule(s)."
    )


def _generate_explanation(transaction, signals, score, decision) -> str:
    provider = os.environ.get("LLM_PROVIDER", config.agent.llm_provider).lower()
    if provider == "groq":
        return _explain_groq(_build_prompt(transaction, signals, score))
    elif provider == "anthropic":
        return _explain_anthropic(_build_prompt(transaction, signals, score))
    elif provider == "openai":
        return _explain_openai(_build_prompt(transaction, signals, score))
    else:
        return _explain_mock(transaction, signals, score, decision)


class DecisionAgent:
    name = "decision_agent"

    def decide(self, transaction, signals, processing_start) -> FraudAlert:
        import time

        score = _weighted_score(signals)
        decision = _score_to_decision(score, signals)

        if score >= config.agent.high_risk_threshold:
            risk_level = RiskLevel.HIGH
        elif score >= config.agent.low_risk_threshold:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW

        explanation = _generate_explanation(transaction, signals, score, decision)
        processing_ms = round((time.perf_counter() - processing_start) * 1000, 2)

        return FraudAlert(
            transaction_id=transaction.id,
            decision=decision,
            risk_score=score,
            risk_level=risk_level,
            explanation=explanation,
            agent_signals=signals,
            processing_time_ms=processing_ms,
            timestamp=datetime.utcnow(),
        )