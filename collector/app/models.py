"""
AgentOps Collector - API Models

Pydantic models for the collector API.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum

from pydantic import BaseModel, Field


class SpanKind(str, Enum):
    AGENT = "agent"
    THOUGHT = "thought"
    TOOL_CALL = "tool_call"
    LLM = "llm"
    DECISION = "decision"
    ACTION = "action"
    RETRIEVAL = "retrieval"


class SpanStatus(str, Enum):
    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ToolCallModel(BaseModel):
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    result: Optional[Any] = None
    error: Optional[str] = None
    duration_ms: float = 0


class LLMCallModel(BaseModel):
    model: str
    provider: str
    messages: List[Dict[str, str]] = Field(default_factory=list)
    response: Optional[str] = None
    token_usage: Optional[TokenUsage] = None
    temperature: float = 1.0
    error: Optional[str] = None


class SpanEvent(BaseModel):
    name: str
    timestamp: float
    attributes: Dict[str, Any] = Field(default_factory=dict)


class SpanCreate(BaseModel):
    """Model for creating a new span."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    
    name: str
    kind: SpanKind = SpanKind.AGENT
    status: SpanStatus = SpanStatus.UNSET
    
    start_time: float
    end_time: Optional[float] = None
    
    service_name: str = ""
    service_version: str = ""
    
    input: Optional[str] = None
    output: Optional[str] = None
    reasoning: Optional[str] = None
    confidence: Optional[float] = None
    alternatives: List[str] = Field(default_factory=list)
    
    tool_calls: List[ToolCallModel] = Field(default_factory=list)
    llm_calls: List[LLMCallModel] = Field(default_factory=list)
    
    context_tokens_used: int = 0
    context_tokens_limit: int = 0
    
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    stack_trace: Optional[str] = None
    
    attributes: Dict[str, Any] = Field(default_factory=dict)
    events: List[SpanEvent] = Field(default_factory=list)


class SpanBatchCreate(BaseModel):
    """Model for batch span ingestion."""
    spans: List[SpanCreate]


class TraceCreate(BaseModel):
    """Model for creating a complete trace."""
    trace_id: str
    service_name: str
    start_time: float
    end_time: Optional[float] = None
    root_span_id: Optional[str] = None
    spans: List[SpanCreate] = Field(default_factory=list)
    
    total_duration_ms: float = 0
    total_tokens: int = 0
    total_cost: float = 0
    total_tool_calls: int = 0
    total_llm_calls: int = 0
    
    status: SpanStatus = SpanStatus.UNSET
    error_count: int = 0
    tags: Dict[str, str] = Field(default_factory=dict)


class SpanResponse(BaseModel):
    """Response model for span queries."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    name: str
    kind: str
    status: str
    start_time: datetime
    end_time: Optional[datetime]
    duration_ms: float
    service_name: str
    input: Optional[str]
    output: Optional[str]
    reasoning: Optional[str]
    error_message: Optional[str]
    tool_calls_count: int
    llm_calls_count: int
    total_tokens: int


class TraceResponse(BaseModel):
    """Response model for trace queries."""
    trace_id: str
    service_name: str
    start_time: datetime
    end_time: Optional[datetime]
    duration_ms: float
    span_count: int
    status: str
    error_count: int
    total_tokens: int
    total_cost: float
    root_span_name: Optional[str]


class TraceDetailResponse(BaseModel):
    """Detailed trace with all spans."""
    trace_id: str
    service_name: str
    start_time: datetime
    end_time: Optional[datetime]
    duration_ms: float
    status: str
    error_count: int
    total_tokens: int
    total_cost: float
    tags: Dict[str, str]
    spans: List[SpanResponse]


class TracesListResponse(BaseModel):
    """Paginated list of traces."""
    traces: List[TraceResponse]
    total: int
    page: int
    page_size: int
    has_more: bool


class SearchQuery(BaseModel):
    """Semantic search query."""
    query: str
    service_name: Optional[str] = None
    status: Optional[SpanStatus] = None
    min_duration_ms: Optional[float] = None
    max_duration_ms: Optional[float] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    limit: int = Field(default=20, le=100)


class SearchResult(BaseModel):
    """Search result item."""
    trace_id: str
    span_id: str
    span_name: str
    relevance_score: float
    snippet: str
    timestamp: datetime


class SearchResponse(BaseModel):
    """Search response."""
    results: List[SearchResult]
    total: int
    query: str


class ServiceStats(BaseModel):
    """Statistics for a service."""
    service_name: str
    total_traces: int
    total_spans: int
    error_rate: float
    avg_duration_ms: float
    total_tokens: int
    total_cost: float
    p50_duration_ms: float
    p95_duration_ms: float
    p99_duration_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    uptime_seconds: float
    traces_count: int
    spans_count: int
