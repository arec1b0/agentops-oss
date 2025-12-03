# AgentOps OSS

Open-source, Kubernetes-native observability platform for AI agents.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         AgentOps OSS Stack                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────────────────┐ │
│  │  Your Agent  │     │  Collector   │     │      ClickHouse          │ │
│  │              │     │  (FastAPI)   │     │   (Columnar Storage)     │ │
│  │  ┌────────┐  │     │              │     │                          │ │
│  │  │AgentOps│──┼────▶│  /v1/traces  │────▶│  • MergeTree engine      │ │
│  │  │  SDK   │  │     │  /v1/spans   │     │  • Time partitioning     │ │
│  │  └────────┘  │     │              │     │  • Bloom filter search   │ │
│  └──────────────┘     └──────┬───────┘     │  • Native percentiles    │ │
│                              │             └──────────────────────────┘ │
│                              │                          │               │
│                              │         ┌────────────────┘               │
│                              ▼         ▼                                │
│                       ┌─────────────────────┐                           │
│                       │     Web UI          │                           │
│                       │  • Trace Explorer   │                           │
│                       │  • Text Search      │                           │
│                       │  • Analytics        │                           │
│                       └─────────────────────┘                           │
└─────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Docker Compose (Recommended)

```bash
git clone https://github.com/arec1b0/agentops-oss.git
cd agentops-oss
docker-compose up -d

# Services:
# - ClickHouse: localhost:8123 (HTTP), localhost:9000 (Native)
# - Collector:  localhost:8000 (API), localhost:8000/docs (Swagger)
# - UI:         localhost:8501
```

### Install SDK

```bash
pip install -e ./sdk/python
```

### Run Example

```bash
python examples/basic_agent.py
open http://localhost:8501
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
| ClickHouse | Columnar DB for traces/spans | 8123 (HTTP), 9000 (Native) |
| Collector | FastAPI ingestion service | 8000 |
| UI | Streamlit dashboard | 8501 |
| SDK | Python instrumentation library | - |

## Why ClickHouse?

- **Columnar storage**: 10-100x faster analytical queries vs row-based DBs
- **MergeTree engine**: Optimized for time-series append workloads
- **Native percentiles**: `quantile(0.95)(duration_ms)` without pre-aggregation
- **Bloom filter indexes**: Fast text search over span content
- **Compression**: 5-10x storage reduction for trace data

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
