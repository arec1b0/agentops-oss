"""
AgentOps Collector - FastAPI service for trace ingestion.

Production-ready with:
- API key authentication
- Rate limiting
- CORS configuration
- Health checks
"""

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .security import (
    APIKey,
    RateLimitMiddleware,
    RequestIDMiddleware,
    api_key_manager,
    get_cors_config,
    require_auth,
)
from .storage import Storage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Storage Initialization
# =============================================================================

storage: Optional[Storage] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler for startup/shutdown."""
    global storage
    
    # Initialize storage
    logger.info("Initializing ClickHouse storage...")
    storage = Storage(
        host=os.getenv("CLICKHOUSE_HOST", "localhost"),
        port=int(os.getenv("CLICKHOUSE_PORT", "8123")),
        database=os.getenv("CLICKHOUSE_DB", "agentops"),
        username=os.getenv("CLICKHOUSE_USER", "default"),
        password=os.getenv("CLICKHOUSE_PASSWORD", ""),
    )
    
    # Log security status
    if api_key_manager.has_keys():
        logger.info("API key authentication ENABLED")
    else:
        logger.warning("API key authentication DISABLED (no keys configured)")
    
    logger.info("Collector ready")
    
    yield
    
    # Cleanup
    if storage:
        storage.close()
    logger.info("Collector shutdown complete")


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="AgentOps Collector",
    description="Trace ingestion service for AI agent observability",
    version="0.2.0",
    lifespan=lifespan,
    docs_url="/docs" if os.getenv("AGENTOPS_ENABLE_DOCS", "true").lower() == "true" else None,
    redoc_url="/redoc" if os.getenv("AGENTOPS_ENABLE_DOCS", "true").lower() == "true" else None,
)

# Middleware (order matters: last added = first executed)
# 1. Request ID (first to execute)
app.add_middleware(RequestIDMiddleware)

# 2. Rate limiting
app.add_middleware(
    RateLimitMiddleware,
    default_limit=int(os.getenv("AGENTOPS_DEFAULT_RATE_LIMIT", "100")),
)

# 3. CORS (last to execute, first in chain)
app.add_middleware(CORSMiddleware, **get_cors_config())


# =============================================================================
# Models
# =============================================================================

class SpanCreate(BaseModel):
    """Span creation model."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    name: str
    kind: str = "agent"
    status: str = "ok"
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    input: Optional[str] = None
    output: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    events: list[dict[str, Any]] = Field(default_factory=list)
    
    # LLM-specific fields
    model: Optional[str] = None
    provider: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cost: Optional[float] = None
    
    # Agent-specific fields
    reasoning: Optional[str] = None
    tools_called: list[str] = Field(default_factory=list)
    
    # Error tracking
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None


class SpanBatchCreate(BaseModel):
    """Batch span creation model."""
    spans: list[SpanCreate]
    service_name: str = "default"
    service_version: Optional[str] = None


class TraceResponse(BaseModel):
    """Trace response model."""
    trace_id: str
    service_name: str
    status: str
    start_time: datetime
    end_time: Optional[datetime]
    duration_ms: Optional[float]
    total_tokens: Optional[int]
    total_cost: Optional[float]
    spans: list[dict[str, Any]]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    storage: str
    auth_enabled: bool
    timestamp: datetime


class StatsResponse(BaseModel):
    """Service stats response."""
    service_name: str
    trace_count: int
    error_count: int
    error_rate: float
    avg_duration_ms: float
    p50_duration_ms: float
    p95_duration_ms: float
    p99_duration_ms: float
    total_tokens: int
    total_cost: float


# =============================================================================
# Health Endpoints (no auth required)
# =============================================================================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
@app.get("/healthz", response_model=HealthResponse, tags=["Health"], include_in_schema=False)
async def health_check():
    """Health check endpoint for load balancers and orchestrators."""
    storage_status = "connected"
    try:
        if storage:
            # Quick connectivity check
            storage.client.command("SELECT 1")
    except Exception as e:
        storage_status = f"error: {str(e)}"
    
    return HealthResponse(
        status="healthy" if storage_status == "connected" else "degraded",
        version="0.2.0",
        storage=storage_status,
        auth_enabled=api_key_manager.has_keys(),
        timestamp=datetime.utcnow(),
    )


@app.get("/ready", tags=["Health"], include_in_schema=False)
async def readiness_check():
    """Readiness check for Kubernetes."""
    if not storage:
        raise HTTPException(status_code=503, detail="Storage not initialized")
    
    try:
        storage.client.command("SELECT 1")
        return {"status": "ready"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Storage unavailable: {e}")


# =============================================================================
# Ingestion Endpoints (auth required)
# =============================================================================

@app.post(
    "/v1/ingest",
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Ingestion"],
    summary="Ingest a batch of spans",
)
async def ingest_spans(
    batch: SpanBatchCreate,
    api_key: APIKey = Depends(require_auth("ingest")),
):
    """
    Ingest a batch of spans.
    
    Requires API key with 'ingest' scope.
    """
    if not storage:
        raise HTTPException(status_code=503, detail="Storage not initialized")
    
    try:
        spans_data = []
        for span in batch.spans:
            span_dict = span.model_dump()
            span_dict["service_name"] = batch.service_name
            span_dict["service_version"] = batch.service_version
            spans_data.append(span_dict)
        
        storage.insert_spans(spans_data)
        
        logger.info(
            f"Ingested {len(spans_data)} spans for service={batch.service_name} "
            f"(key={api_key.name})"
        )
        
        return {
            "status": "accepted",
            "spans_received": len(spans_data),
            "service_name": batch.service_name,
        }
    
    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")


# Legacy endpoint for backward compatibility
@app.post("/ingest", include_in_schema=False)
async def ingest_spans_legacy(
    batch: SpanBatchCreate,
    api_key: APIKey = Depends(require_auth("ingest")),
):
    """Legacy ingestion endpoint (deprecated, use /v1/ingest)."""
    return await ingest_spans(batch, api_key)


# =============================================================================
# Query Endpoints (auth required)
# =============================================================================

@app.get(
    "/v1/traces",
    tags=["Traces"],
    summary="List traces",
)
async def list_traces(
    service_name: Optional[str] = Query(None, description="Filter by service name"),
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    api_key: APIKey = Depends(require_auth("read")),
):
    """
    List traces with optional filtering.
    
    Requires API key with 'read' scope.
    """
    if not storage:
        raise HTTPException(status_code=503, detail="Storage not initialized")
    
    try:
        traces = storage.list_traces(
            service_name=service_name,
            status=status,
            limit=limit,
            offset=offset,
        )
        return {"traces": traces, "count": len(traces)}
    except Exception as e:
        logger.error(f"List traces error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/v1/traces/{trace_id}",
    response_model=TraceResponse,
    tags=["Traces"],
    summary="Get trace by ID",
)
async def get_trace(
    trace_id: str,
    api_key: APIKey = Depends(require_auth("read")),
):
    """
    Get a single trace with all spans.
    
    Requires API key with 'read' scope.
    """
    if not storage:
        raise HTTPException(status_code=503, detail="Storage not initialized")
    
    try:
        trace = storage.get_trace(trace_id)
        if not trace:
            raise HTTPException(status_code=404, detail="Trace not found")
        return trace
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get trace error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/v1/search",
    tags=["Search"],
    summary="Search spans by text",
)
async def search_spans(
    q: str = Query(..., min_length=1, description="Search query"),
    service_name: Optional[str] = Query(None, description="Filter by service"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum results"),
    api_key: APIKey = Depends(require_auth("read")),
):
    """
    Full-text search across span content.
    
    Searches in: name, input, output, reasoning, error_message.
    Requires API key with 'read' scope.
    """
    if not storage:
        raise HTTPException(status_code=503, detail="Storage not initialized")
    
    try:
        results = storage.search_spans(
            query=q,
            service_name=service_name,
            limit=limit,
        )
        return {"results": results, "count": len(results), "query": q}
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/v1/stats",
    response_model=StatsResponse,
    tags=["Analytics"],
    summary="Get service statistics",
)
async def get_stats(
    service_name: str = Query(..., description="Service name"),
    hours: int = Query(24, ge=1, le=720, description="Time window in hours"),
    api_key: APIKey = Depends(require_auth("read")),
):
    """
    Get aggregated statistics for a service.
    
    Includes: trace count, error rate, latency percentiles, token usage.
    Requires API key with 'read' scope.
    """
    if not storage:
        raise HTTPException(status_code=503, detail="Storage not initialized")
    
    try:
        stats = storage.get_service_stats(service_name=service_name, hours=hours)
        return stats
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Admin Endpoints
# =============================================================================

@app.get(
    "/v1/admin/keys",
    tags=["Admin"],
    summary="List API key metadata (admin only)",
)
async def list_api_keys(
    api_key: APIKey = Depends(require_auth("admin")),
):
    """
    List configured API keys (hashes only, not actual keys).
    
    Requires API key with 'admin' scope.
    """
    keys = []
    for key_hash, key_info in api_key_manager._keys.items():
        keys.append({
            "name": key_info.name,
            "key_prefix": key_hash[:8] + "...",
            "scopes": list(key_info.scopes),
            "rate_limit": key_info.rate_limit,
            "created_at": datetime.fromtimestamp(key_info.created_at).isoformat(),
        })
    return {"keys": keys}


# =============================================================================
# Error Handlers
# =============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "detail": "An unexpected error occurred",
            "request_id": getattr(request.state, "request_id", None),
        },
    )


from starlette.responses import JSONResponse