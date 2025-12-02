# AgentOps OSS

Open-source, Kubernetes-native observability platform for AI agents.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         AgentOps OSS Stack                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────────────────┐ │
│  │  Your Agent  │     │  Collector   │     │      Storage Layer       │ │
│  │              │     │  (FastAPI)   │     │                          │ │
│  │  ┌────────┐  │     │              │     │  ┌────────┐ ┌─────────┐  │ │
│  │  │AgentOps│──┼────▶│  /v1/traces  │────▶│  │SQLite/ │ │ QDrant  │  │ │
│  │  │  SDK   │  │     │  /v1/spans   │     │  │ClickHs │ │(vectors)│  │ │
│  │  └────────┘  │     │              │     │  └────────┘ └─────────┘  │ │
│  └──────────────┘     └──────┬───────┘     └──────────────────────────┘ │
│                              │                          │               │
│                              │         ┌────────────────┘               │
│                              ▼         ▼                                │
│                       ┌─────────────────────┐                           │
│                       │     Web UI          │                           │
│                       │  • Trace Explorer   │                           │
│                       │  • Semantic Search  │                           │
│                       │  • RCA Dashboard    │                           │
│                       └─────────────────────┘                           │
└─────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Local Development (Docker Compose)

```bash
# Clone and start
git clone https://github.com/yourorg/agentops-oss.git
cd agentops-oss
docker-compose up -d

# Install SDK
pip install -e ./sdk/python

# Run example
python examples/basic_agent.py

# Open UI
open http://localhost:8501
```

### Kubernetes Deployment

```bash
helm install agentops ./deploy/helm/agentops \
  --namespace agentops \
  --create-namespace
```

## SDK Usage

```python
from agentops import AgentTracer, trace_agent, trace_tool

tracer = AgentTracer(
    service_name="my-agent",
    collector_url="http://localhost:8000"
)

@trace_agent(tracer)
async def my_agent(query: str):
    # Your agent logic
    result = await think_and_act(query)
    return result

@trace_tool(tracer)
async def search_database(query: str):
    # Tool implementation
    return results
```

## Components

| Component | Description | Port |
|-----------|-------------|------|
| SDK | Python instrumentation library | - |
| Collector | FastAPI service for ingestion | 8000 |
| UI | Streamlit dashboard | 8501 |
| Storage | SQLite (dev) / ClickHouse (prod) | - |

## Features

### MVP (v0.1)
- [x] Python SDK with decorators
- [x] Trace and span collection
- [x] Basic trace visualization
- [x] Docker Compose deployment

### Roadmap (v0.2+)
- [ ] Semantic search over failures
- [ ] Auto-RCA with LLM
- [ ] Kubernetes operator
- [ ] OpenTelemetry export
- [ ] Multi-tenancy

## License

Apache 2.0
