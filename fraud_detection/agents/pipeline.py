"""
fraud_detection/agents/pipeline.py

The orchestrator. Runs specialist agents concurrently, then passes
all signals to the decision agent for a final verdict.

Usage:
    pipeline = FraudPipeline()
    alert = pipeline.evaluate(transaction)
    print(alert.decision, alert.explanation)
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from fraud_detection.agents.anomaly_agent import AnomalyAgent
from fraud_detection.agents.base import BaseAgent
from fraud_detection.agents.decision_agent import DecisionAgent
from fraud_detection.utils.schemas import AgentSignal, FraudAlert, FraudDecision, Transaction

from dotenv import load_dotenv
load_dotenv()

console = Console()

DECISION_COLORS = {
    FraudDecision.APPROVE:  "green",
    FraudDecision.REVIEW:   "yellow",
    FraudDecision.STEP_UP:  "cyan",
    FraudDecision.BLOCK:    "red",
}


class FraudPipeline:
    """
    Instantiate once, call evaluate() for each transaction.
    Agents are loaded once at init (models stay in memory).
    """

    def __init__(self):
        # Import rules agent here to avoid circular imports
        from fraud_detection.agents.rules_agent import RulesAgent

        self.specialist_agents: list[BaseAgent] = [
            AnomalyAgent(),
            RulesAgent(),
        ]
        self.decision_agent = DecisionAgent()
        self._executor = ThreadPoolExecutor(max_workers=len(self.specialist_agents))

    def evaluate(self, transaction: Transaction) -> FraudAlert:
        """
        Run all specialist agents concurrently, then decide.
        Thread-safe — can be called from multiple threads.
        """
        start = time.perf_counter()

        # Run specialist agents in parallel
        futures = {
            self._executor.submit(agent.run, transaction): agent.name
            for agent in self.specialist_agents
        }

        signals: list[AgentSignal] = []
        for future in as_completed(futures):
            try:
                signals.append(future.result(timeout=5.0))
            except Exception as e:
                agent_name = futures[future]
                console.print(f"[red]Agent {agent_name} failed: {e}[/red]")

        # Decision agent synthesizes all signals
        alert = self.decision_agent.decide(transaction, signals, start)
        return alert

    def evaluate_batch(self, transactions: list[Transaction]) -> list[FraudAlert]:
        """Evaluate a list of transactions, printing a summary."""
        alerts = []
        for txn in transactions:
            alerts.append(self.evaluate(txn))
        return alerts

    def shutdown(self):
        self._executor.shutdown(wait=False)


# ── Pretty-print helpers ──────────────────────────────────────────────────────

def print_alert(alert: FraudAlert, transaction: Transaction | None = None):
    """Rich-formatted output for a single fraud alert."""
    color = DECISION_COLORS.get(alert.decision, "white")
    decision_str = f"[bold {color}]{alert.decision.value.upper()}[/bold {color}]"

    # Agent signals table
    table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    table.add_column("Agent", style="dim", width=20)
    table.add_column("Score", justify="right", width=8)
    table.add_column("Risk", width=8)
    table.add_column("Flags")

    for sig in alert.agent_signals:
        risk_color = {"low": "green", "medium": "yellow", "high": "red"}.get(
            sig.risk_level.value, "white"
        )
        flags_str = " | ".join(sig.flags[:2]) if sig.flags else "—"
        table.add_row(
            sig.agent_name,
            f"{sig.risk_score:.3f}",
            f"[{risk_color}]{sig.risk_level.value}[/{risk_color}]",
            flags_str,
        )

    from io import StringIO
    from rich.console import Console as _C
    _buf = StringIO()
    _c = _C(file=_buf, highlight=False)
    _c.print(table)
    table_str = _buf.getvalue()

    content = (
        f"Decision: {decision_str}   "
        f"Risk score: [bold]{alert.risk_score:.3f}[/bold]   "
        f"Time: {alert.processing_time_ms:.0f}ms\n\n"
        + table_str
        + f"\n[italic]{alert.explanation}[/italic]"
    )

    title = f"Transaction {alert.transaction_id[:8]}…"
    if transaction:
        title += f"  ${transaction.amount:.2f} @ {transaction.merchant_category}"

    console.print(Panel(content, title=title, border_style=color))


# ── CLI demo ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pandas as pd
    from fraud_detection.data.simulator import TransactionSimulator

    console.print("\n[bold]Fraud Detection Pipeline — Demo[/bold]\n")

    # Load a handful of transactions (mix of fraud and normal)
    sim = TransactionSimulator()
    df = sim.generate_dataframe()

    fraud_samples   = df[df["is_fraud"] == True].head(4)
    normal_samples  = df[df["is_fraud"] == False].head(4)
    sample_df = pd.concat([fraud_samples, normal_samples]).sample(
        frac=1, random_state=42
    )

    pipeline = FraudPipeline()

    for _, row in sample_df.iterrows():
        txn_data = {k: row[k] for k in Transaction.model_fields if k in row}
        txn_data.pop("is_fraud", None)
        txn = Transaction(**txn_data)

        alert = pipeline.evaluate(txn)
        print_alert(alert, txn)

        ground_truth = "FRAUD" if row["is_fraud"] else "NORMAL"
        correct = (
            (alert.decision == FraudDecision.BLOCK and row["is_fraud"]) or
            (alert.decision == FraudDecision.APPROVE and not row["is_fraud"])
        )
        status = "[green]CORRECT[/green]" if correct else "[red]WRONG[/red]"
        console.print(f"  Ground truth: {ground_truth}  →  {status}\n")

    pipeline.shutdown()