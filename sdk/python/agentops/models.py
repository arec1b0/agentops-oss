"""
AgentOps SDK - Core Models

Defines the semantic conventions for agent observability.
These models extend OpenTelemetry concepts with agent-specific attributes.
"""

from __future__ import annotations

import uuid
import time
from enum import Enum
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime


class SpanKind(str, Enum):
    """Types of spans in agent execution."""
    AGENT = "agent"           # Top-level agent invocation
    THOUGHT = "thought"       # Reasoning/CoT step
    TOOL_CALL = "tool_call"   # External tool invocation
    LLM = "llm"               # LLM API call
    DECISION = "decision"     # Branch point in execution
    ACTION = "action"         # External action (API call, etc.)
    RETRIEVAL = "retrieval"   # RAG retrieval step


class SpanStatus(str, Enum):
    """Execution status of a span."""
    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


@dataclass
class TokenUsage:
    """Token consumption for LLM calls."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    @property
    def cost_estimate(self) -> float:
        """Rough cost estimate (GPT-4 pricing as baseline)."""
        return (self.prompt_tokens * 0.00003) + (self.completion_tokens * 0.00006)


@dataclass
class ToolCall:
    """Represents a tool invocation within an agent."""
    name: str
    arguments: Dict[str, Any]
    result: Optional[Any] = None
    error: Optional[str] = None
    duration_ms: float = 0


@dataclass  
class LLMCall:
    """Represents an LLM API call."""
    model: str
    provider: str  # openai, anthropic, etc.
    messages: List[Dict[str, str]]
    response: Optional[str] = None
    token_usage: Optional[TokenUsage] = None
    temperature: float = 1.0
    error: Optional[str] = None


@dataclass
class AgentSpan:
    """
    Core span model for agent observability.
    
    Extends standard tracing spans with agent-specific semantics:
    - Reasoning traces (CoT, scratchpad)
    - Confidence scores
    - Alternative paths considered
    - Tool calls with full context
    """
    
    # Identity
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    parent_span_id: Optional[str] = None
    
    # Metadata
    name: str = ""
    kind: SpanKind = SpanKind.AGENT
    status: SpanStatus = SpanStatus.UNSET
    
    # Timing
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    
    # Service context
    service_name: str = ""
    service_version: str = ""
    
    # Agent-specific attributes
    input: Optional[str] = None
    output: Optional[str] = None
    reasoning: Optional[str] = None  # CoT, scratchpad
    confidence: Optional[float] = None  # 0.0 - 1.0
    alternatives: List[str] = field(default_factory=list)  # Rejected paths
    
    # Nested data
    tool_calls: List[ToolCall] = field(default_factory=list)
    llm_calls: List[LLMCall] = field(default_factory=list)
    
    # Context window tracking
    context_tokens_used: int = 0
    context_tokens_limit: int = 0
    
    # Error handling
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    stack_trace: Optional[str] = None
    
    # Custom attributes
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    # Events within span
    events: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def duration_ms(self) -> float:
        """Calculate span duration in milliseconds."""
        if self.end_time is None:
            return (time.time() - self.start_time) * 1000
        return (self.end_time - self.start_time) * 1000
    
    @property
    def context_utilization(self) -> float:
        """Percentage of context window used."""
        if self.context_tokens_limit == 0:
            return 0.0
        return self.context_tokens_used / self.context_tokens_limit
    
    @property
    def total_tokens(self) -> int:
        """Sum of all token usage in this span."""
        return sum(
            (llm.token_usage.total_tokens if llm.token_usage else 0)
            for llm in self.llm_calls
        )
    
    @property
    def total_cost(self) -> float:
        """Estimated total cost of LLM calls."""
        return sum(
            (llm.token_usage.cost_estimate if llm.token_usage else 0)
            for llm in self.llm_calls
        )
    
    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Add a timestamped event to the span."""
        self.events.append({
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {}
        })
    
    def set_error(self, error: Exception):
        """Record an error in the span."""
        self.status = SpanStatus.ERROR
        self.error_type = type(error).__name__
        self.error_message = str(error)
        import traceback
        self.stack_trace = traceback.format_exc()
    
    def end(self, status: SpanStatus = SpanStatus.OK):
        """Mark span as complete."""
        self.end_time = time.time()
        if self.status == SpanStatus.UNSET:
            self.status = status
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize span to dictionary for export."""
        data = asdict(self)
        # Convert enums to strings
        data["kind"] = self.kind.value
        data["status"] = self.status.value
        # Convert nested dataclasses
        data["tool_calls"] = [asdict(tc) for tc in self.tool_calls]
        data["llm_calls"] = [
            {**asdict(llm), "token_usage": asdict(llm.token_usage) if llm.token_usage else None}
            for llm in self.llm_calls
        ]
        return data


@dataclass
class Trace:
    """
    A complete trace representing one agent execution.
    Contains all spans in the execution tree.
    """
    trace_id: str
    service_name: str
    start_time: float
    end_time: Optional[float] = None
    root_span_id: Optional[str] = None
    spans: List[AgentSpan] = field(default_factory=list)
    
    # Aggregated metrics
    total_duration_ms: float = 0
    total_tokens: int = 0
    total_cost: float = 0
    total_tool_calls: int = 0
    total_llm_calls: int = 0
    
    # Status
    status: SpanStatus = SpanStatus.UNSET
    error_count: int = 0
    
    # Metadata
    tags: Dict[str, str] = field(default_factory=dict)
    
    def compute_metrics(self):
        """Aggregate metrics from all spans."""
        self.total_tokens = sum(s.total_tokens for s in self.spans)
        self.total_cost = sum(s.total_cost for s in self.spans)
        self.total_tool_calls = sum(len(s.tool_calls) for s in self.spans)
        self.total_llm_calls = sum(len(s.llm_calls) for s in self.spans)
        self.error_count = sum(1 for s in self.spans if s.status == SpanStatus.ERROR)
        
        if self.spans:
            self.start_time = min(s.start_time for s in self.spans)
            end_times = [s.end_time for s in self.spans if s.end_time]
            if end_times:
                self.end_time = max(end_times)
                self.total_duration_ms = (self.end_time - self.start_time) * 1000
        
        # Set trace status based on spans
        if self.error_count > 0:
            self.status = SpanStatus.ERROR
        elif all(s.status == SpanStatus.OK for s in self.spans):
            self.status = SpanStatus.OK
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize trace for export."""
        return {
            "trace_id": self.trace_id,
            "service_name": self.service_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "root_span_id": self.root_span_id,
            "spans": [s.to_dict() for s in self.spans],
            "total_duration_ms": self.total_duration_ms,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "total_tool_calls": self.total_tool_calls,
            "total_llm_calls": self.total_llm_calls,
            "status": self.status.value,
            "error_count": self.error_count,
            "tags": self.tags
        }
