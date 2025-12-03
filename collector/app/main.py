"""
AgentOps Collector - FastAPI Application

REST API for span and trace ingestion and querying.
"""

from __future__ import annotations

import os
import time
import logging
from datetime import datetime
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .models import (
    SpanCreate, SpanBatchCreate, TraceCreate,
    TraceDetailResponse, TracesListResponse,
    SearchQuery, SearchResponse, SearchResult,
    ServiceStats, HealthResponse, SpanStatus
)
from .storage import Storage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global state
START_TIME = time.time()
storage: Optional[Storage] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global storage
    
    # Startup - Initialize ClickHouse connection
    storage = Storage(
        host=os.getenv("CLICKHOUSE_HOST", "localhost"),
        port=int(os.getenv("CLICKHOUSE_PORT", "8123")),
        database=os.getenv("CLICKHOUSE_DB", "agentops"),
        username=os.getenv("CLICKHOUSE_USER", "default"),
        password=os.getenv("CLICKHOUSE_PASSWORD", ""),
    )
    logger.info(f"ClickHouse storage initialized: {storage.host}:{storage.port}")
    
    yield
    
    # Shutdown
    if storage:
        storage.close()
    logger.info("Collector shutting down")


# Create FastAPI app
app = FastAPI(
    title="AgentOps Collector",
    description="Open-source observability collector for AI agents",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Health & Info Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    traces_count, spans_count = storage.get_counts()
    
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        uptime_seconds=time.time() - START_TIME,
        traces_count=traces_count,
        spans_count=spans_count,
    )


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint with API info."""
    return {
        "name": "AgentOps Collector",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
    }


# ============================================================================
# Ingestion Endpoints
# ============================================================================

@app.post("/v1/spans", tags=["Ingestion"])
async def ingest_spans(batch: SpanBatchCreate, background_tasks: BackgroundTasks):
    """
    Ingest a batch of spans.
    
    This endpoint accepts spans from the AgentOps SDK.
    Spans are processed asynchronously for better throughput.
    """
    if not batch.spans:
        return {"status": "ok", "inserted": 0}
    
    # Insert spans
    inserted = storage.insert_spans(batch.spans)
    
    logger.info(f"Ingested {inserted}/{len(batch.spans)} spans")
    
    return {
        "status": "ok",
        "inserted": inserted,
        "total": len(batch.spans),
    }


@app.post("/v1/traces", tags=["Ingestion"])
async def ingest_trace(trace: TraceCreate):
    """
    Ingest a complete trace with all spans.
    
    Use this endpoint when sending a complete trace at once,
    typically at the end of an agent execution.
    """
    success = storage.insert_trace(trace)
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to insert trace")
    
    logger.info(f"Ingested trace {trace.trace_id} with {len(trace.spans)} spans")
    
    return {
        "status": "ok",
        "trace_id": trace.trace_id,
        "spans_count": len(trace.spans),
    }


# ============================================================================
# Query Endpoints
# ============================================================================

@app.get("/v1/traces", response_model=TracesListResponse, tags=["Query"])
async def list_traces(
    service: Optional[str] = Query(None, description="Filter by service name"),
    status: Optional[str] = Query(None, description="Filter by status (ok, error, unset)"),
    start_time: Optional[float] = Query(None, description="Filter by start time (unix timestamp)"),
    end_time: Optional[float] = Query(None, description="Filter by end time (unix timestamp)"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Results per page"),
):
    """
    List traces with optional filtering and pagination.
    
    Returns a paginated list of trace summaries ordered by start time (descending).
    """
    return storage.list_traces(
        service_name=service,
        status=status,
        start_time=start_time,
        end_time=end_time,
        page=page,
        page_size=page_size,
    )


@app.get("/v1/traces/{trace_id}", response_model=TraceDetailResponse, tags=["Query"])
async def get_trace(trace_id: str):
    """
    Get a specific trace with all its spans.
    
    Returns the complete trace including all spans in execution order.
    """
    trace = storage.get_trace(trace_id)
    
    if not trace:
        raise HTTPException(status_code=404, detail=f"Trace {trace_id} not found")
    
    return trace


@app.post("/v1/search", response_model=SearchResponse, tags=["Query"])
async def search_spans(query: SearchQuery):
    """
    Semantic search over spans.
    
    Searches across span names, inputs, outputs, reasoning, and error messages
    using full-text search.
    """
    results = storage.search_spans(query.query, limit=query.limit)
    
    return SearchResponse(
        results=[
            SearchResult(
                trace_id=r["trace_id"],
                span_id=r["span_id"],
                span_name=r["span_name"],
                relevance_score=r["relevance_score"],
                snippet=r["snippet"],
                timestamp=r["timestamp"],
            )
            for r in results
        ],
        total=len(results),
        query=query.query,
    )


@app.get("/v1/search", response_model=SearchResponse, tags=["Query"])
async def search_spans_get(
    q: str = Query(..., description="Search query"),
    limit: int = Query(20, ge=1, le=100, description="Max results"),
):
    """Search spans using GET request."""
    results = storage.search_spans(q, limit=limit)
    
    return SearchResponse(
        results=[
            SearchResult(
                trace_id=r["trace_id"],
                span_id=r["span_id"],
                span_name=r["span_name"],
                relevance_score=r["relevance_score"],
                snippet=r["snippet"],
                timestamp=r["timestamp"],
            )
            for r in results
        ],
        total=len(results),
        query=q,
    )


# ============================================================================
# Analytics Endpoints
# ============================================================================

@app.get("/v1/services", tags=["Analytics"])
async def list_services():
    """List all services that have sent traces."""
    services = storage.get_services()
    return {"services": services}


@app.get("/v1/services/{service_name}/stats", response_model=ServiceStats, tags=["Analytics"])
async def get_service_stats(service_name: str):
    """
    Get aggregated statistics for a service.
    
    Returns metrics including:
    - Total traces and spans
    - Error rate
    - Duration percentiles (p50, p95, p99)
    - Token usage and cost
    """
    stats = storage.get_service_stats(service_name)
    
    if not stats:
        raise HTTPException(status_code=404, detail=f"Service {service_name} not found")
    
    return stats


# ============================================================================
# Error Handling
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# ============================================================================
# Development Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
