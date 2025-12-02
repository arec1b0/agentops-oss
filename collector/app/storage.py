"""
AgentOps Collector - Storage Layer

SQLite-based storage for development/small deployments.
Replace with ClickHouse for production scale.
"""

from __future__ import annotations

import json
import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from contextlib import contextmanager

from .models import (
    SpanCreate, TraceCreate, SpanResponse, TraceResponse,
    TraceDetailResponse, TracesListResponse, ServiceStats,
    SpanStatus
)

logger = logging.getLogger(__name__)


class Storage:
    """
    SQLite storage backend.
    
    Schema optimized for:
    - Fast trace lookups by ID
    - Time-range queries
    - Service filtering
    - Error filtering
    """
    
    def __init__(self, db_path: str = "agentops.db"):
        self.db_path = Path(db_path)
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.executescript("""
                -- Traces table
                CREATE TABLE IF NOT EXISTS traces (
                    trace_id TEXT PRIMARY KEY,
                    service_name TEXT NOT NULL,
                    start_time REAL NOT NULL,
                    end_time REAL,
                    duration_ms REAL DEFAULT 0,
                    root_span_id TEXT,
                    status TEXT DEFAULT 'unset',
                    error_count INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    total_cost REAL DEFAULT 0,
                    total_tool_calls INTEGER DEFAULT 0,
                    total_llm_calls INTEGER DEFAULT 0,
                    tags TEXT DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                -- Spans table
                CREATE TABLE IF NOT EXISTS spans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trace_id TEXT NOT NULL,
                    span_id TEXT NOT NULL,
                    parent_span_id TEXT,
                    name TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    status TEXT DEFAULT 'unset',
                    start_time REAL NOT NULL,
                    end_time REAL,
                    duration_ms REAL DEFAULT 0,
                    service_name TEXT,
                    service_version TEXT,
                    input TEXT,
                    output TEXT,
                    reasoning TEXT,
                    confidence REAL,
                    alternatives TEXT DEFAULT '[]',
                    tool_calls TEXT DEFAULT '[]',
                    llm_calls TEXT DEFAULT '[]',
                    context_tokens_used INTEGER DEFAULT 0,
                    context_tokens_limit INTEGER DEFAULT 0,
                    error_message TEXT,
                    error_type TEXT,
                    stack_trace TEXT,
                    attributes TEXT DEFAULT '{}',
                    events TEXT DEFAULT '[]',
                    total_tokens INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(trace_id, span_id)
                );
                
                -- Indexes for common queries
                CREATE INDEX IF NOT EXISTS idx_traces_service ON traces(service_name);
                CREATE INDEX IF NOT EXISTS idx_traces_start_time ON traces(start_time);
                CREATE INDEX IF NOT EXISTS idx_traces_status ON traces(status);
                
                CREATE INDEX IF NOT EXISTS idx_spans_trace_id ON spans(trace_id);
                CREATE INDEX IF NOT EXISTS idx_spans_name ON spans(name);
                CREATE INDEX IF NOT EXISTS idx_spans_kind ON spans(kind);
                CREATE INDEX IF NOT EXISTS idx_spans_status ON spans(status);
                CREATE INDEX IF NOT EXISTS idx_spans_start_time ON spans(start_time);
                
                -- Full-text search for semantic queries
                CREATE VIRTUAL TABLE IF NOT EXISTS spans_fts USING fts5(
                    trace_id,
                    span_id,
                    name,
                    input,
                    output,
                    reasoning,
                    error_message,
                    content='spans',
                    content_rowid='id'
                );
                
                -- Triggers to keep FTS in sync
                CREATE TRIGGER IF NOT EXISTS spans_ai AFTER INSERT ON spans BEGIN
                    INSERT INTO spans_fts(rowid, trace_id, span_id, name, input, output, reasoning, error_message)
                    VALUES (NEW.id, NEW.trace_id, NEW.span_id, NEW.name, NEW.input, NEW.output, NEW.reasoning, NEW.error_message);
                END;
                
                CREATE TRIGGER IF NOT EXISTS spans_ad AFTER DELETE ON spans BEGIN
                    INSERT INTO spans_fts(spans_fts, rowid, trace_id, span_id, name, input, output, reasoning, error_message)
                    VALUES('delete', OLD.id, OLD.trace_id, OLD.span_id, OLD.name, OLD.input, OLD.output, OLD.reasoning, OLD.error_message);
                END;
            """)
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with row factory."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def insert_span(self, span: SpanCreate) -> bool:
        """Insert a single span."""
        try:
            # Calculate derived fields
            duration_ms = 0
            if span.end_time and span.start_time:
                duration_ms = (span.end_time - span.start_time) * 1000
            
            total_tokens = sum(
                (llm.token_usage.total_tokens if llm.token_usage else 0)
                for llm in span.llm_calls
            )
            
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO spans (
                        trace_id, span_id, parent_span_id, name, kind, status,
                        start_time, end_time, duration_ms, service_name, service_version,
                        input, output, reasoning, confidence, alternatives,
                        tool_calls, llm_calls, context_tokens_used, context_tokens_limit,
                        error_message, error_type, stack_trace, attributes, events, total_tokens
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    span.trace_id, span.span_id, span.parent_span_id,
                    span.name, span.kind.value, span.status.value,
                    span.start_time, span.end_time, duration_ms,
                    span.service_name, span.service_version,
                    span.input, span.output, span.reasoning, span.confidence,
                    json.dumps(span.alternatives),
                    json.dumps([tc.model_dump() for tc in span.tool_calls]),
                    json.dumps([llm.model_dump() for llm in span.llm_calls]),
                    span.context_tokens_used, span.context_tokens_limit,
                    span.error_message, span.error_type, span.stack_trace,
                    json.dumps(span.attributes),
                    json.dumps([e.model_dump() for e in span.events]),
                    total_tokens,
                ))
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to insert span: {e}")
            return False
    
    def insert_spans(self, spans: List[SpanCreate]) -> int:
        """Insert multiple spans. Returns count of inserted."""
        inserted = 0
        for span in spans:
            if self.insert_span(span):
                inserted += 1
        return inserted
    
    def insert_trace(self, trace: TraceCreate) -> bool:
        """Insert a complete trace with all spans."""
        try:
            with self._get_connection() as conn:
                # Insert trace metadata
                conn.execute("""
                    INSERT OR REPLACE INTO traces (
                        trace_id, service_name, start_time, end_time, duration_ms,
                        root_span_id, status, error_count, total_tokens, total_cost,
                        total_tool_calls, total_llm_calls, tags
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trace.trace_id, trace.service_name,
                    trace.start_time, trace.end_time, trace.total_duration_ms,
                    trace.root_span_id, trace.status.value, trace.error_count,
                    trace.total_tokens, trace.total_cost,
                    trace.total_tool_calls, trace.total_llm_calls,
                    json.dumps(trace.tags),
                ))
                conn.commit()
            
            # Insert all spans
            for span in trace.spans:
                self.insert_span(span)
            
            return True
        except Exception as e:
            logger.error(f"Failed to insert trace: {e}")
            return False
    
    def get_trace(self, trace_id: str) -> Optional[TraceDetailResponse]:
        """Get a trace with all its spans."""
        with self._get_connection() as conn:
            # Get trace metadata
            trace_row = conn.execute(
                "SELECT * FROM traces WHERE trace_id = ?",
                (trace_id,)
            ).fetchone()
            
            if not trace_row:
                return None
            
            # Get all spans for this trace
            span_rows = conn.execute(
                "SELECT * FROM spans WHERE trace_id = ? ORDER BY start_time",
                (trace_id,)
            ).fetchall()
            
            spans = []
            for row in span_rows:
                spans.append(SpanResponse(
                    trace_id=row["trace_id"],
                    span_id=row["span_id"],
                    parent_span_id=row["parent_span_id"],
                    name=row["name"],
                    kind=row["kind"],
                    status=row["status"],
                    start_time=datetime.fromtimestamp(row["start_time"]),
                    end_time=datetime.fromtimestamp(row["end_time"]) if row["end_time"] else None,
                    duration_ms=row["duration_ms"],
                    service_name=row["service_name"] or "",
                    input=row["input"],
                    output=row["output"],
                    reasoning=row["reasoning"],
                    error_message=row["error_message"],
                    tool_calls_count=len(json.loads(row["tool_calls"] or "[]")),
                    llm_calls_count=len(json.loads(row["llm_calls"] or "[]")),
                    total_tokens=row["total_tokens"],
                ))
            
            return TraceDetailResponse(
                trace_id=trace_row["trace_id"],
                service_name=trace_row["service_name"],
                start_time=datetime.fromtimestamp(trace_row["start_time"]),
                end_time=datetime.fromtimestamp(trace_row["end_time"]) if trace_row["end_time"] else None,
                duration_ms=trace_row["duration_ms"],
                status=trace_row["status"],
                error_count=trace_row["error_count"],
                total_tokens=trace_row["total_tokens"],
                total_cost=trace_row["total_cost"],
                tags=json.loads(trace_row["tags"] or "{}"),
                spans=spans,
            )
    
    def list_traces(
        self,
        service_name: Optional[str] = None,
        status: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> TracesListResponse:
        """List traces with filtering and pagination."""
        conditions = []
        params = []
        
        if service_name:
            conditions.append("service_name = ?")
            params.append(service_name)
        
        if status:
            conditions.append("status = ?")
            params.append(status)
        
        if start_time:
            conditions.append("start_time >= ?")
            params.append(start_time)
        
        if end_time:
            conditions.append("start_time <= ?")
            params.append(end_time)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        with self._get_connection() as conn:
            # Get total count
            total = conn.execute(
                f"SELECT COUNT(*) FROM traces WHERE {where_clause}",
                params
            ).fetchone()[0]
            
            # Get page
            offset = (page - 1) * page_size
            rows = conn.execute(
                f"""
                SELECT t.*, COUNT(s.id) as span_count,
                       (SELECT name FROM spans WHERE trace_id = t.trace_id AND parent_span_id IS NULL LIMIT 1) as root_span_name
                FROM traces t
                LEFT JOIN spans s ON t.trace_id = s.trace_id
                WHERE {where_clause}
                GROUP BY t.trace_id
                ORDER BY t.start_time DESC
                LIMIT ? OFFSET ?
                """,
                params + [page_size, offset]
            ).fetchall()
            
            traces = []
            for row in rows:
                traces.append(TraceResponse(
                    trace_id=row["trace_id"],
                    service_name=row["service_name"],
                    start_time=datetime.fromtimestamp(row["start_time"]),
                    end_time=datetime.fromtimestamp(row["end_time"]) if row["end_time"] else None,
                    duration_ms=row["duration_ms"],
                    span_count=row["span_count"],
                    status=row["status"],
                    error_count=row["error_count"],
                    total_tokens=row["total_tokens"],
                    total_cost=row["total_cost"],
                    root_span_name=row["root_span_name"],
                ))
            
            return TracesListResponse(
                traces=traces,
                total=total,
                page=page,
                page_size=page_size,
                has_more=(offset + page_size) < total,
            )
    
    def search_spans(
        self,
        query: str,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Full-text search over spans."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT 
                    s.trace_id,
                    s.span_id,
                    s.name,
                    s.start_time,
                    snippet(spans_fts, 3, '<mark>', '</mark>', '...', 50) as snippet,
                    bm25(spans_fts) as score
                FROM spans_fts
                JOIN spans s ON spans_fts.rowid = s.id
                WHERE spans_fts MATCH ?
                ORDER BY score
                LIMIT ?
                """,
                (query, limit)
            ).fetchall()
            
            return [
                {
                    "trace_id": row["trace_id"],
                    "span_id": row["span_id"],
                    "span_name": row["name"],
                    "timestamp": datetime.fromtimestamp(row["start_time"]),
                    "snippet": row["snippet"],
                    "relevance_score": abs(row["score"]),
                }
                for row in rows
            ]
    
    def get_service_stats(self, service_name: str) -> Optional[ServiceStats]:
        """Get aggregated stats for a service."""
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT 
                    service_name,
                    COUNT(*) as total_traces,
                    SUM(error_count > 0) as error_traces,
                    AVG(duration_ms) as avg_duration,
                    SUM(total_tokens) as total_tokens,
                    SUM(total_cost) as total_cost
                FROM traces
                WHERE service_name = ?
                GROUP BY service_name
                """,
                (service_name,)
            ).fetchone()
            
            if not row:
                return None
            
            # Get span count
            span_count = conn.execute(
                "SELECT COUNT(*) FROM spans WHERE service_name = ?",
                (service_name,)
            ).fetchone()[0]
            
            # Get percentiles
            durations = conn.execute(
                "SELECT duration_ms FROM traces WHERE service_name = ? ORDER BY duration_ms",
                (service_name,)
            ).fetchall()
            
            durations = [d[0] for d in durations]
            p50 = durations[len(durations) // 2] if durations else 0
            p95 = durations[int(len(durations) * 0.95)] if durations else 0
            p99 = durations[int(len(durations) * 0.99)] if durations else 0
            
            error_rate = row["error_traces"] / row["total_traces"] if row["total_traces"] > 0 else 0
            
            return ServiceStats(
                service_name=service_name,
                total_traces=row["total_traces"],
                total_spans=span_count,
                error_rate=error_rate,
                avg_duration_ms=row["avg_duration"] or 0,
                total_tokens=row["total_tokens"] or 0,
                total_cost=row["total_cost"] or 0,
                p50_duration_ms=p50,
                p95_duration_ms=p95,
                p99_duration_ms=p99,
            )
    
    def get_counts(self) -> Tuple[int, int]:
        """Get total trace and span counts."""
        with self._get_connection() as conn:
            traces = conn.execute("SELECT COUNT(*) FROM traces").fetchone()[0]
            spans = conn.execute("SELECT COUNT(*) FROM spans").fetchone()[0]
            return traces, spans
    
    def get_services(self) -> List[str]:
        """Get list of all services."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT DISTINCT service_name FROM traces ORDER BY service_name"
            ).fetchall()
            return [row[0] for row in rows]
