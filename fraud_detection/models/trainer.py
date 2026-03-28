"""
fraud_detection/models/trainer.py

Trains two complementary models:
  1. XGBoostClassifier   — supervised, uses labeled fraud data
  2. IsolationForest     — unsupervised anomaly detector (no labels needed)

Both are saved to models/artifacts/ and loaded by the anomaly agent at runtime.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from fraud_detection.config import config
from fraud_detection.models.features import build_features

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"


def _ensure_artifacts_dir():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def _print_section(title: str):
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print(f"{'─' * 50}")


# ── XGBoost classifier ────────────────────────────────────────────────────────

def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> xgb.XGBClassifier:
    _print_section("Training XGBoost Classifier")

    cfg = config.model
    fraud_count = y_train.sum()
    normal_count = len(y_train) - fraud_count
    scale_pos_weight = normal_count / max(fraud_count, 1)
    print(f"  Class balance — normal: {normal_count:,}  fraud: {fraud_count:,}")
    print(f"  scale_pos_weight: {scale_pos_weight:.1f}")

    model = xgb.XGBClassifier(
        n_estimators=cfg.xgb_n_estimators,
        max_depth=cfg.xgb_max_depth,
        learning_rate=cfg.xgb_learning_rate,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="aucpr",
        random_state=cfg.xgb_n_estimators,
        n_jobs=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # ── Evaluation ────────────────────────────────────────────────────
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc  = average_precision_score(y_test, y_prob)

    print(f"\n  ROC-AUC : {roc_auc:.4f}")
    print(f"  PR-AUC  : {pr_auc:.4f}  (key metric for imbalanced data)")
    print(f"\n{classification_report(y_test, y_pred, target_names=['normal','fraud'])}")

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"  Confusion matrix:")
    print(f"    True Negatives  (correct normal): {tn:>5}")
    print(f"    False Positives (false alarms)  : {fp:>5}")
    print(f"    False Negatives (missed fraud)  : {fn:>5}")
    print(f"    True Positives  (caught fraud)  : {tp:>5}")

    # Feature importance top-10
    importances = pd.Series(
        model.feature_importances_, index=X_train.columns
    ).sort_values(ascending=False)
    print(f"\n  Top 10 features:")
    for feat, score in importances.head(10).items():
        bar = "█" * int(score * 200)
        print(f"    {feat:35s} {score:.4f}  {bar}")

    return model


# ── Isolation Forest ──────────────────────────────────────────────────────────

def train_isolation_forest(
    X_train: pd.DataFrame,
    scaler: StandardScaler,
) -> IsolationForest:
    _print_section("Training Isolation Forest (unsupervised)")

    cfg = config.model
    X_scaled = scaler.transform(X_train)

    model = IsolationForest(
        n_estimators=200,
        contamination=cfg.anomaly_contamination,
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_scaled)
    print(f"  Trained on {len(X_train):,} samples  "
          f"(contamination={cfg.anomaly_contamination})")

    # Anomaly score distribution
    scores = model.decision_function(X_scaled)
    print(f"  Anomaly score range: [{scores.min():.3f}, {scores.max():.3f}]")
    print(f"  Mean: {scores.mean():.3f}  Std: {scores.std():.3f}")

    return model


# ── Main training pipeline ────────────────────────────────────────────────────

def train(df: pd.DataFrame | None = None) -> dict:
    """
    Full training pipeline.
    Returns paths to saved artifacts.
    """
    _ensure_artifacts_dir()

    # ── Load data ─────────────────────────────────────────────────────
    _print_section("Loading & preparing data")
    if df is None:
        data_path = Path(__file__).parent.parent / "data" / "transactions.csv"
        if not data_path.exists():
            print("  transactions.csv not found — generating now...")
            from fraud_detection.data.simulator import TransactionSimulator
            sim = TransactionSimulator()
            df = sim.generate_dataframe()
            df.to_csv(data_path, index=False)
        else:
            df = pd.read_csv(data_path)
            print(f"  Loaded {len(df):,} transactions from {data_path}")

    print(f"  Fraud rate: {df['is_fraud'].mean():.2%}")

    # ── Feature engineering ───────────────────────────────────────────
    _print_section("Feature engineering")
    X = build_features(df)
    y = df["is_fraud"].astype(int)
    print(f"  Feature matrix: {X.shape[0]:,} rows × {X.shape[1]} features")

    # ── Train / test split (time-based — no data leakage) ─────────────
    df_ts = pd.to_datetime(df["timestamp"])
    split_date = df_ts.quantile(0.8)
    train_mask = df_ts <= split_date
    test_mask  = df_ts >  split_date

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    print(f"  Train: {len(X_train):,} rows  |  Test: {len(X_test):,} rows")
    print(f"  (Split at {split_date.date()} — time-based, no leakage)")

    # ── Scaler (fit on train only) ─────────────────────────────────────
    scaler = StandardScaler()
    scaler.fit(X_train)

    # ── Train models ──────────────────────────────────────────────────
    xgb_model = train_xgboost(X_train, y_train, X_test, y_test)
    iso_model  = train_isolation_forest(X_train, scaler)

    # ── Save artifacts ────────────────────────────────────────────────
    _print_section("Saving artifacts")

    artifacts = {
        "xgboost":          ARTIFACTS_DIR / "xgboost_model.pkl",
        "isolation_forest": ARTIFACTS_DIR / "isolation_forest.pkl",
        "scaler":           ARTIFACTS_DIR / "scaler.pkl",
        "feature_names":    ARTIFACTS_DIR / "feature_names.json",
    }

    with open(artifacts["xgboost"], "wb") as f:
        pickle.dump(xgb_model, f)
    with open(artifacts["isolation_forest"], "wb") as f:
        pickle.dump(iso_model, f)
    with open(artifacts["scaler"], "wb") as f:
        pickle.dump(scaler, f)
    with open(artifacts["feature_names"], "w") as f:
        json.dump(X_train.columns.tolist(), f)

    for name, path in artifacts.items():
        size_kb = Path(path).stat().st_size / 1024
        print(f"  {name:20s} → {path.name}  ({size_kb:.1f} KB)")

    print(f"\n  All artifacts saved to: {ARTIFACTS_DIR}")
    return {k: str(v) for k, v in artifacts.items()}


# ── Model loader (used by agents at runtime) ──────────────────────────────────

class FraudModels:
    """
    Loads trained models from disk. Singleton pattern — call FraudModels.load().

    Usage:
        models = FraudModels.load()
        xgb_score = models.predict_xgb(feature_vector)
        iso_score = models.predict_iso(feature_vector)
    """
    _instance: FraudModels | None = None

    def __init__(self, xgb_model, iso_model, scaler, feature_names):
        self.xgb_model     = xgb_model
        self.iso_model     = iso_model
        self.scaler        = scaler
        self.feature_names = feature_names

    @classmethod
    def load(cls) -> FraudModels:
        if cls._instance is not None:
            return cls._instance

        required = [
            ARTIFACTS_DIR / "xgboost_model.pkl",
            ARTIFACTS_DIR / "isolation_forest.pkl",
            ARTIFACTS_DIR / "scaler.pkl",
            ARTIFACTS_DIR / "feature_names.json",
        ]
        missing = [p for p in required if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"Model artifacts missing: {missing}\n"
                "Run: python -m fraud_detection.models.trainer"
            )

        with open(ARTIFACTS_DIR / "xgboost_model.pkl", "rb") as f:
            xgb_model = pickle.load(f)
        with open(ARTIFACTS_DIR / "isolation_forest.pkl", "rb") as f:
            iso_model = pickle.load(f)
        with open(ARTIFACTS_DIR / "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        with open(ARTIFACTS_DIR / "feature_names.json") as f:
            feature_names = json.load(f)

        cls._instance = cls(xgb_model, iso_model, scaler, feature_names)
        return cls._instance

    def predict_xgb(self, X: pd.DataFrame) -> np.ndarray:
        """Returns fraud probability [0, 1] for each row."""
        X = X[self.feature_names]  # ensure column order
        return self.xgb_model.predict_proba(X)[:, 1]

    def predict_iso(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns anomaly score mapped to [0, 1].
        Higher = more anomalous.
        """
        X = X[self.feature_names]
        X_scaled = self.scaler.transform(X)
        raw = self.iso_model.decision_function(X_scaled)
        # decision_function returns negative scores for anomalies
        # Map to [0,1]: more negative = higher anomaly score
        score = 1 - (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
        return score.clip(0, 1)


if __name__ == "__main__":
    results = train()
    print("\nTraining complete. Artifacts:")
    for k, v in results.items():
        print(f"  {k}: {v}")