# Agentic Fraud Detection System

A production-grade multi-agent fraud detection pipeline that scores payment transactions in real-time using parallel ML agents, deterministic rules, and an LLM decision layer that generates natural language explanations.

**[Live Demo](https://agentic-risk-detection-production.up.railway.app/dashboard)** · **[API Docs](https://agentic-risk-detection-production.up.railway.app/docs)**

---

## How it works

```
Transaction  →  Anomaly Agent  ─┐
                                 ├→  Decision Agent (Llama 3.3 70B)  →  APPROVE / BLOCK / REVIEW
             →  Rules Agent   ─┘
```

Two specialist agents run in parallel on every transaction. Their signals are aggregated by a weighted decision agent powered by Groq/Llama 3.3 70B, which produces a risk score and a plain-English explanation of why the transaction was flagged.

---

## Fraud patterns detected

| Pattern | Description |
|---|---|
| Velocity attack | 8–15 micro-transactions followed by a large fraud hit |
| Geo impossibility | Same user transacts in two countries within 2 hours |
| High-value anomaly | Large transaction, off-hours, card-not-present |
| Card testing | Rapid micro-transactions probing a stolen card |
| High-risk country | Transaction from a known high-fraud-rate region |

---

## Tech stack

| Layer | Technologies |
|---|---|
| ML Models | XGBoost, Isolation Forest, scikit-learn |
| Feature Engineering | Pandas, NumPy — velocity windows, z-scores, geo risk |
| Agentic Layer | Custom multi-agent framework, ThreadPoolExecutor |
| LLM | Groq API — Llama 3.3 70B |
| API | FastAPI, Uvicorn, Pydantic |
| Frontend | Vanilla JS dashboard — real-time feed + agent signal breakdown |
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
    └── dashboard.html        ← live monitoring dashboard
```

---

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `GET` | `/stats` | Live decision counters + avg latency |
| `POST` | `/evaluate` | Score a single transaction |
| `POST` | `/evaluate/batch` | Score up to 100 transactions |
| `GET` | `/dashboard` | Live monitoring UI |

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
  "explanation": "This transaction raises significant fraud concerns...",
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
# Add your GROQ_API_KEY to .env

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

Set `LLM_PROVIDER` in `.env` to switch providers:

| Provider | Value | Key needed |
|---|---|---|
| Groq (Llama 3.3 70B) | `groq` | `GROQ_API_KEY` |
| Anthropic (Claude) | `anthropic` | `ANTHROPIC_API_KEY` |
| OpenAI | `openai` | `OPENAI_API_KEY` |
| No LLM (rule-based) | `mock` | none |

---

Built by [Sayam Palrecha](https://linkedin.com/in/sayampalrecha/)