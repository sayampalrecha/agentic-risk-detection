"""
fraud_detection/data/simulator.py

Generates realistic synthetic transactions with injected fraud patterns.
Fraud patterns modelled:
  1. Velocity attack      — many small txns in a short window
  2. Geo impossibility    — transaction from a distant location too quickly
  3. High-value anomaly   — single large txn outside user's normal range
  4. Card testing         — rapid micro-transactions before a large fraud txn
  5. Off-hours activity   — legitimate user pattern violated (3-5am spike)
"""

from __future__ import annotations

import random
import math
from datetime import datetime, timedelta
from typing import Generator

import numpy as np
import pandas as pd
from faker import Faker

from fraud_detection.config import config
from fraud_detection.utils.schemas import Transaction

fake = Faker()
Faker.seed(config.data.seed)
np.random.seed(config.data.seed)
random.seed(config.data.seed)


# ── Static lookup tables ──────────────────────────────────────────────────────

MERCHANT_CATEGORIES = [
    "grocery", "restaurant", "gas_station", "retail", "pharmacy",
    "electronics", "travel", "entertainment", "subscription", "atm",
]

# Rough (lat, lon) by country code — used for geo impossibility checks
COUNTRY_COORDS = {
    "US": (37.09, -95.71),  "GB": (55.37, -3.43),  "DE": (51.16, 10.45),
    "FR": (46.22, 2.21),    "CA": (56.13, -106.34), "AU": (-25.27, 133.77),
    "BR": (-14.23, -51.92), "NG": (9.08, 8.67),     "RO": (45.94, 24.96),
    "CN": (35.86, 104.19),
}

# High-risk countries (used by rules agent later)
HIGH_RISK_COUNTRIES = {"NG", "RO"}


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points in km."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ── User & merchant generators ────────────────────────────────────────────────

def _build_user_profiles(n: int) -> dict:
    """Create stable per-user behavioural baselines."""
    profiles = {}
    countries = list(COUNTRY_COORDS.keys())
    for i in range(n):
        uid = f"user_{i:04d}"
        home_country = random.choices(
            countries, weights=[50, 10, 8, 7, 8, 5, 4, 2, 3, 3], k=1
        )[0]
        profiles[uid] = {
            "home_country": home_country,
            "avg_txn_amount": abs(np.random.normal(80, 50)),
            "active_hours": random.choice([(7, 22), (6, 23), (9, 21)]),
            "preferred_categories": random.sample(MERCHANT_CATEGORIES, k=4),
            "device_id": fake.uuid4(),
        }
    return profiles


def _build_merchant_list(n: int) -> list[dict]:
    merchants = []
    for i in range(n):
        cat = random.choice(MERCHANT_CATEGORIES)
        country = random.choice(list(COUNTRY_COORDS.keys()))
        merchants.append({
            "id": f"merch_{i:04d}",
            "category": cat,
            "country": country,
            "city": fake.city(),
            "high_risk": country in HIGH_RISK_COUNTRIES,
        })
    return merchants


# ── Fraud pattern injectors ───────────────────────────────────────────────────

def _inject_velocity_attack(
    base: dict, user_id: str, merchant: dict, base_time: datetime
) -> list[dict]:
    """8–15 rapid micro-transactions → 1 large fraud transaction."""
    txns = []
    for j in range(random.randint(8, 15)):
        t = base.copy()
        t.update({
            "user_id": user_id,
            "merchant_id": merchant["id"],
            "merchant_category": merchant["category"],
            "amount": round(random.uniform(0.5, 4.99), 2),
            "timestamp": base_time + timedelta(seconds=j * 20),
            "is_fraud": False,          # card-testing phase — not yet fraud
            "fraud_pattern": "velocity_probe",
        })
        txns.append(t)
    # The actual fraud hit
    fraud = base.copy()
    fraud.update({
        "user_id": user_id,
        "merchant_id": merchant["id"],
        "merchant_category": "electronics",
        "amount": round(random.uniform(800, 3000), 2),
        "timestamp": base_time + timedelta(minutes=5),
        "is_fraud": True,
        "fraud_pattern": "velocity_attack",
    })
    txns.append(fraud)
    return txns


def _inject_geo_impossible(
    base: dict, user_id: str, profile: dict, merchants: list[dict], base_time: datetime
) -> list[dict]:
    """Two transactions in geographically impossible locations within 2 hours."""
    # Pick a foreign merchant far from home
    home = COUNTRY_COORDS[profile["home_country"]]
    foreign_merchant = random.choice(
        [m for m in merchants if m["country"] != profile["home_country"]]
    )
    foreign_coords = COUNTRY_COORDS[foreign_merchant["country"]]
    dist = _haversine_km(*home, *foreign_coords)

    if dist < 500:      # not far enough — skip pattern
        return []

    t1 = base.copy()
    t1.update({
        "user_id": user_id, "merchant_id": f"merch_{random.randint(0,199):04d}",
        "merchant_category": "restaurant",
        "amount": round(random.uniform(20, 80), 2),
        "country": profile["home_country"],
        "timestamp": base_time, "is_fraud": False,
        "fraud_pattern": "geo_anchor",
    })
    t2 = base.copy()
    t2.update({
        "user_id": user_id, "merchant_id": foreign_merchant["id"],
        "merchant_category": foreign_merchant["category"],
        "amount": round(random.uniform(200, 1500), 2),
        "country": foreign_merchant["country"],
        "city": foreign_merchant["city"],
        "timestamp": base_time + timedelta(hours=1, minutes=random.randint(10, 50)),
        "is_fraud": True,
        "fraud_pattern": "geo_impossible",
    })
    return [t1, t2]


# ── Main simulator ─────────────────────────────────────────────────────────────

class TransactionSimulator:
    """
    Generates a dataset of transactions with realistic fraud patterns.

    Usage:
        sim = TransactionSimulator()
        df = sim.generate_dataframe()          # → pandas DataFrame
        txns = list(sim.generate_stream())     # → iterator of Transaction objects
    """

    def __init__(self):
        self.cfg = config.data
        self.users = _build_user_profiles(self.cfg.num_users)
        self.merchants = _build_merchant_list(self.cfg.num_merchants)
        self._rows: list[dict] = []

    # ── Core generation ───────────────────────────────────────────────────

    def _base_txn(self, user_id: str, profile: dict, merchant: dict, ts: datetime) -> dict:
        return {
            "id": fake.uuid4(),
            "timestamp": ts,
            "user_id": user_id,
            "merchant_id": merchant["id"],
            "merchant_category": merchant["category"],
            "amount": max(0.01, abs(np.random.normal(
                profile["avg_txn_amount"], profile["avg_txn_amount"] * 0.6
            ))),
            "currency": "USD",
            "country": profile["home_country"],
            "city": fake.city(),
            "ip_address": fake.ipv4(),
            "device_id": profile["device_id"],
            "is_online": random.random() < 0.45,
            "card_present": random.random() < 0.6,
            "is_fraud": False,
            "fraud_pattern": None,
        }

    def _simulate_normal_transactions(self, n: int) -> list[dict]:
        rows = []
        start = datetime(2024, 1, 1)
        end = datetime(2024, 12, 31)
        user_ids = list(self.users.keys())

        for _ in range(n):
            uid = random.choice(user_ids)
            profile = self.users[uid]
            merch = random.choice(self.merchants)
            # Bias timestamps toward active hours
            h_start, h_end = profile["active_hours"]
            rand_days = random.randint(0, (end - start).days)
            rand_hour = random.randint(h_start, h_end)
            ts = start + timedelta(days=rand_days, hours=rand_hour,
                                   minutes=random.randint(0, 59))
            rows.append(self._base_txn(uid, profile, merch, ts))
        return rows

    def _simulate_fraud_transactions(self) -> list[dict]:
        rows = []
        user_ids = list(self.users.keys())
        start = datetime(2024, 1, 1)

        # Pattern 1 — velocity attacks (40% of fraud events)
        for _ in range(int(self.cfg.num_transactions * self.cfg.fraud_rate * 0.4)):
            uid = random.choice(user_ids)
            merch = random.choice(self.merchants)
            base_time = start + timedelta(days=random.randint(0, 364),
                                          hours=random.randint(0, 23))
            rows.extend(_inject_velocity_attack(
                self._base_txn(uid, self.users[uid], merch, base_time),
                uid, merch, base_time,
            ))

        # Pattern 2 — geo impossibility (35%)
        for _ in range(int(self.cfg.num_transactions * self.cfg.fraud_rate * 0.35)):
            uid = random.choice(user_ids)
            profile = self.users[uid]
            base_time = start + timedelta(days=random.randint(0, 364),
                                          hours=random.randint(0, 23))
            base = self._base_txn(uid, profile, random.choice(self.merchants), base_time)
            rows.extend(_inject_geo_impossible(base, uid, profile, self.merchants, base_time))

        # Pattern 3 — standalone high-value anomalies (25%)
        for _ in range(int(self.cfg.num_transactions * self.cfg.fraud_rate * 0.25)):
            uid = random.choice(user_ids)
            profile = self.users[uid]
            merch = random.choice(self.merchants)
            base_time = start + timedelta(days=random.randint(0, 364),
                                          hours=random.randint(1, 4))   # off-hours
            row = self._base_txn(uid, profile, merch, base_time)
            row.update({
                "amount": round(abs(np.random.normal(
                    self.cfg.fraud_amount_mean, self.cfg.fraud_amount_std
                )), 2),
                "is_online": True,
                "card_present": False,
                "is_fraud": True,
                "fraud_pattern": "high_value_anomaly",
            })
            rows.append(row)

        return rows

    # ── Public API ─────────────────────────────────────────────────────────

    def generate_dataframe(self) -> pd.DataFrame:
        """Return a full labeled DataFrame ready for feature engineering."""
        normal = self._simulate_normal_transactions(
            int(self.cfg.num_transactions * (1 - self.cfg.fraud_rate))
        )
        fraud = self._simulate_fraud_transactions()
        df = pd.DataFrame(normal + fraud)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sample(frac=1, random_state=self.cfg.seed).reset_index(drop=True)
        df["amount"] = df["amount"].round(2)
        print(f"[simulator] Generated {len(df):,} transactions  "
              f"({df['is_fraud'].sum():,} fraud, "
              f"{df['is_fraud'].mean():.1%} rate)")
        return df

    def generate_stream(self) -> Generator[Transaction, None, None]:
        """Yield Transaction objects one at a time (for real-time simulation)."""
        df = self.generate_dataframe()
        for _, row in df.iterrows():
            yield Transaction(**{
                k: row[k] for k in Transaction.model_fields
                if k in row and k != "is_fraud"
            })


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sim = TransactionSimulator()
    df = sim.generate_dataframe()

    # Save to CSV for notebooks / inspection
    out_path = "data/data.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
    print("\nSample fraud transactions:")
    print(df[df["is_fraud"]].head(5)[
        ["user_id", "amount", "country", "fraud_pattern", "timestamp"]
    ].to_string(index=False))
    print("\nFraud pattern breakdown:")
    print(df[df["is_fraud"]]["fraud_pattern"].value_counts().to_string())