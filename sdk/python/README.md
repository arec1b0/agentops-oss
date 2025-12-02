# AgentOps SDK

Open-source observability SDK for AI agents.

## Installation

```bash
pip install agentops-sdk
```

## Quick Start

```python
from agentops import AgentTracer, trace_agent, trace_tool

# Initialize tracer
tracer = AgentTracer(
    service_name="my-agent",
    collector_url="http://localhost:8000"
)

# Instrument your agent
@trace_agent(tracer)
async def my_agent(query: str):
    result = await process(query)
    return result

# Instrument tools
@trace_tool(tracer)
def search_database(query: str):
    return db.search(query)
```

## Features

- Automatic span context propagation
- Async-safe with contextvars
- Framework integrations (LangChain, OpenAI, Anthropic)
- Multiple exporters (HTTP, Console, File)
- Agent-specific semantic conventions

## License

Apache 2.0
