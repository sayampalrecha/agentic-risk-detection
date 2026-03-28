"""
fraud_detection/models/features.py

Transforms raw transaction DataFrames into ML-ready feature matrices.

Features built:
  - Amount features      : log amount, z-score vs user baseline
  - Velocity features    : txn count / amount sum in rolling windows
  - Time features        : hour, day-of-week, is_off_hours, is_weekend
  - Geo features         : country risk score, is_foreign
  - Merchant features    : category risk, is_high_risk_category
  - Device/channel       : is_online, card_present mismatch flag
  - User baseline        : deviation from personal average
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# High-risk country codes (empirical — tune to your data)
HIGH_RISK_COUNTRIES = {"NG", "RO", "CN"}

# Merchant categories ranked by historical fraud rate (0=low, 1=high)
CATEGORY_RISK = {
    "grocery": 0.1,
    "restaurant": 0.15,
    "pharmacy": 0.1,
    "subscription": 0.2,
    "gas_station": 0.25,
    "retail": 0.3,
    "entertainment": 0.3,
    "electronics": 0.7,
    "atm": 0.65,
    "travel": 0.5,
}

FEATURE_COLS: list[str] = []   # populated by build_features()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main entry point.
    Takes the raw simulator DataFrame and returns a feature DataFrame
    with the same index. All columns are numeric.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    feats = pd.DataFrame(index=df.index)

    # ── Amount features ────────────────────────────────────────────────
    feats["amount"] = df["amount"]
    feats["log_amount"] = np.log1p(df["amount"])

    # Per-user mean and std (computed over entire dataset as proxy baseline)
    user_stats = df.groupby("user_id")["amount"].agg(["mean", "std"]).rename(
        columns={"mean": "user_mean_amount", "std": "user_std_amount"}
    )
    user_stats["user_std_amount"] = user_stats["user_std_amount"].fillna(1.0).clip(lower=1.0)
    df = df.join(user_stats, on="user_id")

    feats["amount_zscore"] = (
        (df["amount"] - df["user_mean_amount"]) / df["user_std_amount"]
    ).clip(-5, 5)
    feats["amount_vs_user_mean_ratio"] = (
        df["amount"] / df["user_mean_amount"].clip(lower=0.01)
    ).clip(0, 50)

    # ── Time features ──────────────────────────────────────────────────
    feats["hour"] = df["timestamp"].dt.hour
    feats["day_of_week"] = df["timestamp"].dt.dayofweek   # 0=Mon, 6=Sun
    feats["is_weekend"] = (feats["day_of_week"] >= 5).astype(int)
    feats["is_off_hours"] = ((feats["hour"] >= 1) & (feats["hour"] <= 5)).astype(int)
    feats["is_business_hours"] = ((feats["hour"] >= 9) & (feats["hour"] <= 17)).astype(int)

    # ── Velocity features (rolling 1-hour window per user) ────────────
    # We approximate this using rank-based rolling since true time-aware
    # per-user rolling needs a more complex approach on static data.
    df["_ts_epoch"] = df["timestamp"].astype(np.int64) // 10**9

    velocity_txn_count = []
    velocity_amount_sum = []
    velocity_unique_merchants = []

    for _, group in df.groupby("user_id"):
        group = group.sort_values("_ts_epoch")
        ts = group["_ts_epoch"].values
        amounts = group["amount"].values
        merchants = group["merchant_id"].values
        window = 3600   # 1 hour in seconds

        for i in range(len(group)):
            mask = (ts[i] - ts) <= window
            mask[i] = False  # exclude self
            velocity_txn_count.append(mask.sum())
            velocity_amount_sum.append(amounts[mask].sum())
            velocity_unique_merchants.append(len(set(merchants[mask])))

    # Align back to original index order
    velocity_df = pd.DataFrame({
        "velocity_txn_count": velocity_txn_count,
        "velocity_amount_sum": velocity_amount_sum,
        "velocity_unique_merchants": velocity_unique_merchants,
    }, index=df.index)  # groupby preserves original index within groups

    feats["velocity_txn_count"] = velocity_df["velocity_txn_count"]
    feats["velocity_amount_sum"] = velocity_df["velocity_amount_sum"]
    feats["velocity_unique_merchants"] = velocity_df["velocity_unique_merchants"]
    feats["is_high_velocity"] = (feats["velocity_txn_count"] >= 5).astype(int)

    # ── Geo features ───────────────────────────────────────────────────
    feats["is_high_risk_country"] = df["country"].isin(HIGH_RISK_COUNTRIES).astype(int)

    # Encode country as ordinal (simple but effective)
    le_country = LabelEncoder()
    feats["country_encoded"] = le_country.fit_transform(df["country"].fillna("US"))

    # ── Merchant / category features ──────────────────────────────────
    feats["category_risk_score"] = df["merchant_category"].map(CATEGORY_RISK).fillna(0.3)

    le_cat = LabelEncoder()
    feats["category_encoded"] = le_cat.fit_transform(
        df["merchant_category"].fillna("retail")
    )

    # ── Channel features ───────────────────────────────────────────────
    feats["is_online"] = df["is_online"].astype(int)
    feats["card_present"] = df["card_present"].astype(int)
    # Card-not-present + online = higher risk
    feats["cnp_online_flag"] = (
        (df["is_online"]) & (~df["card_present"])
    ).astype(int)

    # ── Composite risk signals ─────────────────────────────────────────
    feats["high_amount_off_hours"] = (
        (feats["is_off_hours"] == 1) & (feats["amount_zscore"] > 2)
    ).astype(int)
    feats["high_risk_cnp"] = (
        (feats["is_high_risk_country"] == 1) & (feats["cnp_online_flag"] == 1)
    ).astype(int)
    feats["velocity_amount_spike"] = (
        (feats["velocity_amount_sum"] > 500) & (feats["velocity_txn_count"] >= 3)
    ).astype(int)

    global FEATURE_COLS
    FEATURE_COLS = feats.columns.tolist()

    return feats


def get_feature_names() -> list[str]:
    return FEATURE_COLS if FEATURE_COLS else []


if __name__ == "__main__":
    from fraud_detection.data.simulator import TransactionSimulator
    sim = TransactionSimulator()
    df = sim.generate_dataframe()
    feats = build_features(df)
    print(f"Feature matrix shape: {feats.shape}")
    print(f"\nFeatures ({len(feats.columns)}):")
    for col in feats.columns:
        print(f"  {col:40s}  mean={feats[col].mean():.3f}  std={feats[col].std():.3f}")