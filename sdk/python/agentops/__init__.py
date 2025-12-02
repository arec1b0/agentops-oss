"""
AgentOps SDK

Open-source observability for AI agents.

Usage:
    from agentops import AgentTracer, trace_agent, trace_tool
    
    # Initialize tracer
    tracer = AgentTracer(
        service_name="my-agent",
        collector_url="http://localhost:8000"
    )
    
    # Use decorators
    @trace_agent(tracer)
    async def my_agent(query: str):
        result = await process(query)
        return result
    
    @trace_tool(tracer)
    def search_database(query: str):
        return db.search(query)
    
    # Or use context manager
    with tracer.start_span("custom_operation") as span:
        span.input = "some input"
        # do work
        span.output = "result"
"""

__version__ = "0.1.0"

# Core tracer
from .tracer import (
    AgentTracer,
    get_tracer,
    set_tracer,
)

# Models
from .models import (
    AgentSpan,
    Trace,
    SpanKind,
    SpanStatus,
    ToolCall,
    LLMCall,
    TokenUsage,
)

# Decorators
from .decorators import (
    trace_agent,
    trace_tool,
    trace_llm,
    trace_thought,
    trace_decision,
)

# Exporters
from .exporters import (
    SpanExporter,
    HTTPExporter,
    ConsoleExporter,
    FileExporter,
    CompositeExporter,
)

# Integrations
from .integrations import (
    LangChainInstrumentation,
    OpenAIInstrumentation,
    AnthropicInstrumentation,
)

__all__ = [
    # Version
    "__version__",
    # Core
    "AgentTracer",
    "get_tracer",
    "set_tracer",
    # Models
    "AgentSpan",
    "Trace",
    "SpanKind",
    "SpanStatus",
    "ToolCall",
    "LLMCall",
    "TokenUsage",
    # Decorators
    "trace_agent",
    "trace_tool",
    "trace_llm",
    "trace_thought",
    "trace_decision",
    # Exporters
    "SpanExporter",
    "HTTPExporter",
    "ConsoleExporter",
    "FileExporter",
    "CompositeExporter",
    # Integrations
    "LangChainInstrumentation",
    "OpenAIInstrumentation",
    "AnthropicInstrumentation",
]
