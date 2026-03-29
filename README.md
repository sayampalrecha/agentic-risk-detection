# Agentic Fraud Detection System

A production-grade multi-agent fraud detection pipeline that scores payment transactions in real-time using parallel ML agents, deterministic rules, and an LLM decision layer that generates natural language explanations.

**[Live Demo](https://agentic-risk-detection-production.up.railway.app/dashboard)** · **[API Docs](https://agentic-risk-detection-production.up.railway.app/docs)**

---

## Demo

Open the live dashboard and try it yourself:

- **+ Submit Transaction** — opens a drawer where you fill in transaction details and see a real decision + Groq/Llama explanation instantly
- **Quick presets** — one-click scenarios: Normal purchase, Velocity attack, Geo impossible, Off-hours anomaly
- **Random** — fires a random transaction from the simulator
- **Auto Stream** — streams transactions continuously every 2.2 seconds
- **Transaction detail** — click any row to see full agent signal breakdown, risk score bar, all flags, LLM explanation, and decision reasoning

---

## How it works

```
Transaction  →  Anomaly Agent  ─┐
                                 ├→  Decision Agent (Llama 3.3 70B)  →  APPROVE / BLOCK / REVIEW / STEP-UP
             →  Rules Agent   ─┘
```

Two specialist agents run in parallel on every transaction. Their signals are aggregated by a weighted decision agent powered by Groq/Llama 3.3 70B, which produces a risk score and a plain-English explanation of why the transaction was flagged.

### Decision thresholds

| Score | Decision | Action |
|---|---|---|
| < 0.30 | APPROVE | Transaction clears |
| 0.30 – 0.50 | REVIEW | Routed to human analyst |
| 0.50 – 0.70 | STEP-UP | Additional authentication required |
| > 0.70 | BLOCK | Transaction rejected |

---

## Fraud patterns detected

| Pattern | Description |
|---|---|
| Velocity attack | 8–15 micro-transactions followed by a large fraud hit |
| Geo impossibility | Same user transacts in two countries within 2 hours |
| High-value anomaly | Large transaction, off-hours, card-not-present |
| Card testing | Rapid micro-transactions probing a stolen card |
| High-risk country | Transaction from a known high-fraud-rate region |
| Off-hours spike | High-value transaction between 1am–5am |
| Round-number amount | Suspiciously round amounts ($500, $1000, $2000) |

---

## Tech stack

| Layer | Technologies |
|---|---|
| ML Models | XGBoost, Isolation Forest, scikit-learn |
| Feature Engineering | Pandas, NumPy — velocity windows, z-scores, geo risk, time-of-day |
| Agentic Layer | Custom multi-agent framework, ThreadPoolExecutor (parallel execution) |
| LLM | Groq API — Llama 3.3 70B |
| API | FastAPI, Uvicorn, Pydantic |
| Frontend | Vanilla JS — real-time feed, input drawer, agent signal breakdown |
| Deployment | Railway, CI/CD via GitHub |

---

## Project structure

```
fraud_detection/
├── config.py                 ← all tuneable parameters
├── data/
│   └── simulator.py          ← synthetic transaction generator (3 fraud patterns)
├── models/
│   ├── features.py           ← 23-feature engineering pipeline
│   ├── trainer.py            ← XGBoost + Isolation Forest training
│   └── artifacts/            ← saved model files (generated)
├── agents/
│   ├── base.py               ← BaseAgent abstract class
│   ├── anomaly_agent.py      ← ML-based scoring (XGBoost + IF blended)
│   ├── rules_agent.py        ← 7 deterministic rules with velocity tracking
│   ├── decision_agent.py     ← LLM aggregation + explanation generation
│   └── pipeline.py           ← parallel orchestrator
└── api/
    ├── app.py                ← FastAPI REST endpoints
    └── dashboard.html        ← live monitoring dashboard with input form
```

---

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `GET` | `/stats` | Live decision counters + avg latency |
| `POST` | `/evaluate` | Score a single transaction |
| `POST` | `/evaluate/batch` | Score up to 100 transactions |
| `GET` | `/dashboard` | Live monitoring UI with manual input |

### Example request

```bash
curl -X POST https://agentic-risk-detection-production.up.railway.app/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_0001",
    "merchant_id": "merch_0042",
    "merchant_category": "electronics",
    "amount": 1899.99,
    "country": "NG",
    "city": "Lagos",
    "is_online": true,
    "card_present": false
  }'
```

### Example response

```json
{
  "transaction_id": "6be81a79-...",
  "decision": "block",
  "risk_score": 0.57,
  "risk_level": "medium",
  "explanation": "This transaction raises significant fraud concerns — high-value electronics purchase from a high-risk country at 3am with no card present strongly matches known fraud patterns.",
  "flags": [
    "R01: High amount $1899.99 (>500.0)",
    "R02: Off-hours transaction at 03:00",
    "R03: High-risk country (NG)",
    "R04: Card-not-present online txn"
  ],
  "processing_time_ms": 157.76
}
```

---

## Local setup

```bash
# Clone and set up environment
git clone https://github.com/sayampalrecha/agentic-risk-detection.git
cd agentic-risk-detection
python -m venv fraud && source fraud/bin/activate
pip install -r requirements.txt && pip install -e .

# Configure
cp .env.example .env
# Add your GROQ_API_KEY to .env (get free key at console.groq.com)

# Generate data and train models
python -m fraud_detection.data.simulator
python -m fraud_detection.models.trainer

# Run the pipeline demo
python -m fraud_detection.agents.pipeline

# Start the API
uvicorn fraud_detection.api.app:app --reload
# Open http://localhost:8000/dashboard
```

---

## LLM configuration

Set `LLM_PROVIDER` in `.env` to switch providers — the system is fully swappable:

| Provider | Value | Key |
|---|---|---|
| Groq (Llama 3.3 70B) | `groq` | `GROQ_API_KEY` |
| Anthropic (Claude) | `anthropic` | `ANTHROPIC_API_KEY` |
| OpenAI | `openai` | `OPENAI_API_KEY` |
| Rule-based fallback | `mock` | none |

---

## Tuning

All thresholds live in `config.py` — no code changes needed:

```python
low_risk_threshold: float = 0.3   # below → APPROVE
high_risk_threshold: float = 0.7  # above → BLOCK
fraud_rate: float = 0.025         # training data fraud rate
velocity_window_minutes: int = 60 # rolling window for velocity checks
```

---

Built by [Sayam Palrecha](https://linkedin.com/in/sayampalrecha/)