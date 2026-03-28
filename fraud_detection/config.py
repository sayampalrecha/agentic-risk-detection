"""
fraud_detection/config.py
Central configuration — edit these values to tune the system.
"""

from dataclasses import dataclass, field
from pathlib import Path

ROOT_DIR = Path(__file__).parent


@dataclass
class DataConfig:
    """Transaction simulation parameters."""
    num_users: int = 500
    num_merchants: int = 200
    num_transactions: int = 10_000
    fraud_rate: float = 0.025          # 2.5% base fraud rate
    seed: int = 42

    # Transaction amount distributions (USD)
    normal_amount_mean: float = 85.0
    normal_amount_std: float = 120.0
    fraud_amount_mean: float = 340.0
    fraud_amount_std: float = 280.0

    # Velocity thresholds used by rules agent
    velocity_window_minutes: int = 60
    velocity_max_normal: int = 5        # txns in window before flagging
    geo_distance_threshold_km: float = 500.0


@dataclass
class ModelConfig:
    """ML model training parameters."""
    test_size: float = 0.2
    cv_folds: int = 5
    xgb_n_estimators: int = 300
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.05
    anomaly_contamination: float = 0.03
    model_dir: Path = ROOT_DIR / "models" / "artifacts"


@dataclass
class AgentConfig:
    """Agentic layer parameters."""
    # LLM provider: "anthropic" | "openai" | "mock"
    llm_provider: str = "mock"
    llm_model: str = "claude-sonnet-4-20250514"
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.1       # Low temp for consistent decisions

    # Decision thresholds
    low_risk_threshold: float = 0.3
    high_risk_threshold: float = 0.7

    # Timeouts
    agent_timeout_seconds: float = 5.0


@dataclass
class APIConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True


@dataclass
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    api: APIConfig = field(default_factory=APIConfig)


# Singleton — import this throughout the project
config = AppConfig()