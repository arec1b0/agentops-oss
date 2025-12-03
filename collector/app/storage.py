"""
AgentOps Collector - ClickHouse Storage Layer

Columnar storage optimized for observability workloads:
- Fast analytical queries over time-series data
- Efficient compression for traces/spans
- Native percentile calculations
- Full-text search with tokenbf_v1 index
"""

from __future__ import annotations

import os
import json
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple

import clickhouse_connect
from clickhouse_connect.driver.client import Client

from .models import (
    SpanCreate, TraceCreate, SpanResponse, TraceResponse,
    TraceDetailResponse, TracesListResponse, ServiceStats,
    SpanStatus
)

logger = logging.getLogger(__name__)


class Storage:
    """
    ClickHouse storage backend.
    
    Optimizations:
    - MergeTree engine with time-based partitioning
    - ORDER BY (service_name, start_time) for fast filtering
    - Bloom filter index for text search
    - LowCardinality for enums (status, kind)
    - Materialized views for aggregations
    """
    
    def __init__(
        self,
        host: str = None,
        port: int = None,
        database: str = "agentops",
        username: str = None,
        password: str = None,
    ):
        self.host = host or os.getenv("CLICKHOUSE_HOST", "localhost")
        self.port = port or int(os.getenv("CLICKHOUSE_PORT", "8123"))
        self.database = database
        self.username = username or os.getenv("CLICKHOUSE_USER", "default")
        self.password = password or os.getenv("CLICKHOUSE_PASSWORD", "")
        
        self._client: Optional[Client] = None
        self._init_db()
    
    @property
    def client(self) -> Client:
        """Lazy client initialization with reconnection."""
        if self._client is None:
            self._client = clickhouse_connect.get_client(
                host=self.host,
                port=self.port,
                username=self.username,
                password=self.password,
                database=self.database,
            )
        return self._client
    
    def _init_db(self):
        """Initialize ClickHouse schema."""
        # Create database
        admin_client = clickhouse_connect.get_client(
            host=self.host,
            port=self.port,
            username=self.username,
            password=self.password,
        )
        admin_client.command(f"CREATE DATABASE IF NOT EXISTS {self.database}")
        admin_client.close()
        
        # Create tables
        self.client.command("""
            CREATE TABLE IF NOT EXISTS traces (
                trace_id String,
                service_name LowCardinality(String),
                start_time DateTime64(3),
                end_time Nullable(DateTime64(3)),
                duration_ms Float64,
                root_span_id Nullable(String),
                status LowCardinality(String) DEFAULT 'unset',
                error_count UInt32 DEFAULT 0,
                total_tokens UInt64 DEFAULT 0,
                total_cost Float64 DEFAULT 0,
                total_tool_calls UInt32 DEFAULT 0,
                total_llm_calls UInt32 DEFAULT 0,
                tags Map(String, String),
                created_at DateTime64(3) DEFAULT now64(3)
            )
            ENGINE = ReplacingMergeTree(created_at)
            PARTITION BY toYYYYMM(start_time)
            ORDER BY (service_name, start_time, trace_id)
            SETTINGS index_granularity = 8192
        """)
        
        self.client.command("""
            CREATE TABLE IF NOT EXISTS spans (
                trace_id String,
                span_id String,
                parent_span_id Nullable(String),
                name String,
                kind LowCardinality(String),
                status LowCardinality(String) DEFAULT 'unset',
                start_time DateTime64(3),
                end_time Nullable(DateTime64(3)),
                duration_ms Float64 DEFAULT 0,
                service_name LowCardinality(String),
                service_version String DEFAULT '',
                
                -- Agent-specific fields
                input String DEFAULT '',
                output String DEFAULT '',
                reasoning String DEFAULT '',
                confidence Nullable(Float32),
                alternatives Array(String),
                
                -- Nested JSON (stored as String for flexibility)
                tool_calls String DEFAULT '[]',
                llm_calls String DEFAULT '[]',
                
                -- Context tracking
                context_tokens_used UInt32 DEFAULT 0,
                context_tokens_limit UInt32 DEFAULT 0,
                
                -- Error handling
                error_message Nullable(String),
                error_type Nullable(String),
                stack_trace Nullable(String),
                
                -- Flexible attributes
                attributes String DEFAULT '{}',
                events String DEFAULT '[]',
                
                -- Derived
                total_tokens UInt64 DEFAULT 0,
                created_at DateTime64(3) DEFAULT now64(3),
                
                -- Text search index
                INDEX idx_input input TYPE tokenbf_v1(32768, 3, 0) GRANULARITY 4,
                INDEX idx_output output TYPE tokenbf_v1(32768, 3, 0) GRANULARITY 4,
                INDEX idx_reasoning reasoning TYPE tokenbf_v1(32768, 3, 0) GRANULARITY 4,
                INDEX idx_error error_message TYPE tokenbf_v1(32768, 3, 0) GRANULARITY 4,
                INDEX idx_name name TYPE tokenbf_v1(32768, 3, 0) GRANULARITY 4
            )
            ENGINE = ReplacingMergeTree(created_at)
            PARTITION BY toYYYYMM(start_time)
            ORDER BY (service_name, trace_id, start_time, span_id)
            SETTINGS index_granularity = 8192
        """)
        
        # Materialized view for service stats (pre-aggregated)
        self.client.command("""
            CREATE MATERIALIZED VIEW IF NOT EXISTS mv_service_stats
            ENGINE = SummingMergeTree()
            ORDER BY (service_name, hour)
            AS SELECT
                service_name,
                toStartOfHour(start_time) AS hour,
                count() AS trace_count,
                countIf(status = 'error') AS error_count,
                sum(duration_ms) AS total_duration_ms,
                sum(total_tokens) AS total_tokens,
                sum(total_cost) AS total_cost
            FROM traces
            GROUP BY service_name, hour
        """)
        
        logger.info(f"ClickHouse schema initialized: {self.database}")
    
    def insert_span(self, span: SpanCreate) -> bool:
        """Insert a single span."""
        try:
            duration_ms = 0.0
            if span.end_time and span.start_time:
                duration_ms = (span.end_time - span.start_time) * 1000
            
            total_tokens = sum(
                (llm.token_usage.total_tokens if llm.token_usage else 0)
                for llm in span.llm_calls
            )
            
            self.client.insert("spans", [[
                span.trace_id,
                span.span_id,
                span.parent_span_id,
                span.name,
                span.kind.value,
                span.status.value,
                datetime.fromtimestamp(span.start_time),
                datetime.fromtimestamp(span.end_time) if span.end_time else None,
                duration_ms,
                span.service_name,
                span.service_version,
                span.input or "",
                span.output or "",
                span.reasoning or "",
                span.confidence,
                span.alternatives,
                json.dumps([tc.model_dump() for tc in span.tool_calls]),
                json.dumps([llm.model_dump() for llm in span.llm_calls]),
                span.context_tokens_used,
                span.context_tokens_limit,
                span.error_message,
                span.error_type,
                span.stack_trace,
                json.dumps(span.attributes),
                json.dumps([e.model_dump() for e in span.events]),
                total_tokens,
                datetime.now(),
            ]], column_names=[
                "trace_id", "span_id", "parent_span_id", "name", "kind", "status",
                "start_time", "end_time", "duration_ms", "service_name", "service_version",
                "input", "output", "reasoning", "confidence", "alternatives",
                "tool_calls", "llm_calls", "context_tokens_used", "context_tokens_limit",
                "error_message", "error_type", "stack_trace", "attributes", "events",
                "total_tokens", "created_at"
            ])
            return True
        except Exception as e:
            logger.error(f"Failed to insert span: {e}")
            return False
    
    def insert_spans(self, spans: List[SpanCreate]) -> int:
        """Batch insert spans."""
        if not spans:
            return 0
        
        try:
            rows = []
            for span in spans:
                duration_ms = 0.0
                if span.end_time and span.start_time:
                    duration_ms = (span.end_time - span.start_time) * 1000
                
                total_tokens = sum(
                    (llm.token_usage.total_tokens if llm.token_usage else 0)
                    for llm in span.llm_calls
                )
                
                rows.append([
                    span.trace_id, span.span_id, span.parent_span_id, span.name,
                    span.kind.value, span.status.value,
                    datetime.fromtimestamp(span.start_time),
                    datetime.fromtimestamp(span.end_time) if span.end_time else None,
                    duration_ms, span.service_name, span.service_version,
                    span.input or "", span.output or "", span.reasoning or "",
                    span.confidence, span.alternatives,
                    json.dumps([tc.model_dump() for tc in span.tool_calls]),
                    json.dumps([llm.model_dump() for llm in span.llm_calls]),
                    span.context_tokens_used, span.context_tokens_limit,
                    span.error_message, span.error_type, span.stack_trace,
                    json.dumps(span.attributes), json.dumps([e.model_dump() for e in span.events]),
                    total_tokens, datetime.now(),
                ])
            
            self.client.insert("spans", rows, column_names=[
                "trace_id", "span_id", "parent_span_id", "name", "kind", "status",
                "start_time", "end_time", "duration_ms", "service_name", "service_version",
                "input", "output", "reasoning", "confidence", "alternatives",
                "tool_calls", "llm_calls", "context_tokens_used", "context_tokens_limit",
                "error_message", "error_type", "stack_trace", "attributes", "events",
                "total_tokens", "created_at"
            ])
            return len(spans)
        except Exception as e:
            logger.error(f"Failed to batch insert spans: {e}")
            return 0
    
    def insert_trace(self, trace: TraceCreate) -> bool:
        """Insert a complete trace with all spans."""
        try:
            self.client.insert("traces", [[
                trace.trace_id,
                trace.service_name,
                datetime.fromtimestamp(trace.start_time),
                datetime.fromtimestamp(trace.end_time) if trace.end_time else None,
                trace.total_duration_ms,
                trace.root_span_id,
                trace.status.value,
                trace.error_count,
                trace.total_tokens,
                trace.total_cost,
                trace.total_tool_calls,
                trace.total_llm_calls,
                trace.tags,
                datetime.now(),
            ]], column_names=[
                "trace_id", "service_name", "start_time", "end_time", "duration_ms",
                "root_span_id", "status", "error_count", "total_tokens", "total_cost",
                "total_tool_calls", "total_llm_calls", "tags", "created_at"
            ])
            
            # Insert spans in batch
            if trace.spans:
                self.insert_spans(trace.spans)
            
            return True
        except Exception as e:
            logger.error(f"Failed to insert trace: {e}")
            return False
    
    def get_trace(self, trace_id: str) -> Optional[TraceDetailResponse]:
        """Get a trace with all its spans."""
        # Get trace metadata
        trace_result = self.client.query(
            "SELECT * FROM traces FINAL WHERE trace_id = {trace_id:String}",
            parameters={"trace_id": trace_id}
        )
        
        if not trace_result.result_rows:
            return None
        
        trace_row = dict(zip(trace_result.column_names, trace_result.result_rows[0]))
        
        # Get spans
        spans_result = self.client.query(
            """
            SELECT * FROM spans FINAL 
            WHERE trace_id = {trace_id:String}
            ORDER BY start_time
            """,
            parameters={"trace_id": trace_id}
        )
        
        spans = []
        for row in spans_result.result_rows:
            span_data = dict(zip(spans_result.column_names, row))
            spans.append(SpanResponse(
                trace_id=span_data["trace_id"],
                span_id=span_data["span_id"],
                parent_span_id=span_data["parent_span_id"],
                name=span_data["name"],
                kind=span_data["kind"],
                status=span_data["status"],
                start_time=span_data["start_time"],
                end_time=span_data["end_time"],
                duration_ms=span_data["duration_ms"],
                service_name=span_data["service_name"] or "",
                input=span_data["input"],
                output=span_data["output"],
                reasoning=span_data["reasoning"],
                error_message=span_data["error_message"],
                tool_calls_count=len(json.loads(span_data["tool_calls"] or "[]")),
                llm_calls_count=len(json.loads(span_data["llm_calls"] or "[]")),
                total_tokens=span_data["total_tokens"],
            ))
        
        return TraceDetailResponse(
            trace_id=trace_row["trace_id"],
            service_name=trace_row["service_name"],
            start_time=trace_row["start_time"],
            end_time=trace_row["end_time"],
            duration_ms=trace_row["duration_ms"],
            status=trace_row["status"],
            error_count=trace_row["error_count"],
            total_tokens=trace_row["total_tokens"],
            total_cost=trace_row["total_cost"],
            tags=dict(trace_row["tags"]) if trace_row["tags"] else {},
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
        conditions = ["1=1"]
        params = {}
        
        if service_name:
            conditions.append("service_name = {service_name:String}")
            params["service_name"] = service_name
        
        if status:
            conditions.append("status = {status:String}")
            params["status"] = status
        
        if start_time:
            conditions.append("start_time >= {start_time:DateTime64(3)}")
            params["start_time"] = datetime.fromtimestamp(start_time)
        
        if end_time:
            conditions.append("start_time <= {end_time:DateTime64(3)}")
            params["end_time"] = datetime.fromtimestamp(end_time)
        
        where_clause = " AND ".join(conditions)
        offset = (page - 1) * page_size
        
        # Get total count
        count_result = self.client.query(
            f"SELECT count() FROM traces FINAL WHERE {where_clause}",
            parameters=params
        )
        total = count_result.result_rows[0][0]
        
        # Get traces with span count
        params["limit"] = page_size
        params["offset"] = offset
        
        result = self.client.query(
            f"""
            SELECT 
                t.*,
                (SELECT count() FROM spans WHERE trace_id = t.trace_id) AS span_count,
                (SELECT name FROM spans WHERE trace_id = t.trace_id AND parent_span_id IS NULL LIMIT 1) AS root_span_name
            FROM traces t FINAL
            WHERE {where_clause}
            ORDER BY start_time DESC
            LIMIT {{limit:UInt32}} OFFSET {{offset:UInt32}}
            """,
            parameters=params
        )
        
        traces = []
        for row in result.result_rows:
            data = dict(zip(result.column_names, row))
            traces.append(TraceResponse(
                trace_id=data["trace_id"],
                service_name=data["service_name"],
                start_time=data["start_time"],
                end_time=data["end_time"],
                duration_ms=data["duration_ms"],
                span_count=data["span_count"],
                status=data["status"],
                error_count=data["error_count"],
                total_tokens=data["total_tokens"],
                total_cost=data["total_cost"],
                root_span_name=data["root_span_name"],
            ))
        
        return TracesListResponse(
            traces=traces,
            total=total,
            page=page,
            page_size=page_size,
            has_more=(offset + page_size) < total,
        )
    
    def search_spans(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Full-text search over spans using bloom filter index."""
        # Split query into terms for OR matching
        terms = query.lower().split()
        
        if not terms:
            return []
        
        # Build LIKE conditions for each searchable field
        like_conditions = []
        for term in terms:
            like_conditions.append(f"(lower(name) LIKE '%{term}%' OR lower(input) LIKE '%{term}%' OR lower(output) LIKE '%{term}%' OR lower(reasoning) LIKE '%{term}%' OR lower(error_message) LIKE '%{term}%')")
        
        where_clause = " OR ".join(like_conditions)
        
        result = self.client.query(
            f"""
            SELECT 
                trace_id,
                span_id,
                name,
                start_time,
                multiIf(
                    error_message != '', substring(error_message, 1, 200),
                    reasoning != '', substring(reasoning, 1, 200),
                    output != '', substring(output, 1, 200),
                    substring(input, 1, 200)
                ) AS snippet
            FROM spans FINAL
            WHERE {where_clause}
            ORDER BY start_time DESC
            LIMIT {{limit:UInt32}}
            """,
            parameters={"limit": limit}
        )
        
        return [
            {
                "trace_id": row[0],
                "span_id": row[1],
                "span_name": row[2],
                "timestamp": row[3],
                "snippet": row[4],
                "relevance_score": 1.0,  # ClickHouse doesn't have BM25, using simple match
            }
            for row in result.result_rows
        ]
    
    def get_service_stats(self, service_name: str) -> Optional[ServiceStats]:
        """Get aggregated stats for a service with native percentiles."""
        result = self.client.query(
            """
            SELECT
                service_name,
                count() AS total_traces,
                countIf(status = 'error') AS error_traces,
                avg(duration_ms) AS avg_duration,
                sum(total_tokens) AS total_tokens,
                sum(total_cost) AS total_cost,
                quantile(0.5)(duration_ms) AS p50,
                quantile(0.95)(duration_ms) AS p95,
                quantile(0.99)(duration_ms) AS p99
            FROM traces FINAL
            WHERE service_name = {service_name:String}
            GROUP BY service_name
            """,
            parameters={"service_name": service_name}
        )
        
        if not result.result_rows:
            return None
        
        row = dict(zip(result.column_names, result.result_rows[0]))
        
        # Get span count
        span_result = self.client.query(
            "SELECT count() FROM spans FINAL WHERE service_name = {svc:String}",
            parameters={"svc": service_name}
        )
        span_count = span_result.result_rows[0][0]
        
        error_rate = row["error_traces"] / row["total_traces"] if row["total_traces"] > 0 else 0
        
        return ServiceStats(
            service_name=service_name,
            total_traces=row["total_traces"],
            total_spans=span_count,
            error_rate=error_rate,
            avg_duration_ms=row["avg_duration"] or 0,
            total_tokens=row["total_tokens"] or 0,
            total_cost=row["total_cost"] or 0,
            p50_duration_ms=row["p50"] or 0,
            p95_duration_ms=row["p95"] or 0,
            p99_duration_ms=row["p99"] or 0,
        )
    
    def get_counts(self) -> Tuple[int, int]:
        """Get total trace and span counts."""
        traces = self.client.query("SELECT count() FROM traces FINAL").result_rows[0][0]
        spans = self.client.query("SELECT count() FROM spans FINAL").result_rows[0][0]
        return traces, spans
    
    def get_services(self) -> List[str]:
        """Get list of all services."""
        result = self.client.query(
            "SELECT DISTINCT service_name FROM traces FINAL ORDER BY service_name"
        )
        return [row[0] for row in result.result_rows]
    
    def close(self):
        """Close connection."""
        if self._client:
            self._client.close()
            self._client = None
