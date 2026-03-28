# Agentic-Risk-Detection

# Architecture

<img width="759" height="579" alt="image" src="https://github.com/user-attachments/assets/efe387a6-ffde-432c-b644-7eb940f6e698" />

# Tech Stack
1. Data & Streaming — Apache Kafka for real-time event ingestion, Apache Flink or Spark Streaming for stateful stream processing, and Feast or Tecton as your feature store to serve pre-computed features at low latency. <br><br>
2. ML Models (inside agents) — XGBoost/LightGBM for tabular anomaly scoring, PyTorch Graph Neural Networks (GraphSAGE or GAT) for the graph agent, and scikit-learn or Isolation Forest for unsupervised anomaly baselines. These run as microservices behind the agents.<br><br>
3. The Agentic Layer — This is where Claude or GPT-4 class models live. You use the LLM as the orchestrator and decision agent. Each specialist agent gets a set of tools (function calls) like query_feature_store, run_ml_model, lookup_graph_neighbors, check_rule_engine. The orchestrator decides which agents to invoke, collects results, and reasons over them. Frameworks: LangGraph, CrewAI, or a custom implementation using the Anthropic or OpenAI API directly. <br><br>
4. Memory & State — Redis for short-term session state (is this the 5th failed attempt in 2 minutes?), Pinecone or Weaviate as a vector store for semantic similarity (does this transaction look like known fraud patterns?), and a time-series DB like TimescaleDB for behavioral history.<br><br>
5. Graph Database — Neo4j or Amazon Neptune to store entity relationships — devices, accounts, IPs, merchants — and detect ring fraud, money mule networks, and account takeover chains.<br><br>
6. Action Layer — gRPC or REST calls to your core banking/payment system to block, flag, or trigger step-up authentication. PagerDuty or Slack webhooks to alert analysts.<br><br>
7. Observability — Prometheus + Grafana for latency and throughput, MLflow for model drift monitoring, and a custom audit log (append-only) for regulatory compliance. Every agent decision should be stored with its full reasoning chain.<br>

<img width="750" height="347" alt="image" src="https://github.com/user-attachments/assets/a01cb97b-4d57-4512-a3a0-576eedc94da1" />
