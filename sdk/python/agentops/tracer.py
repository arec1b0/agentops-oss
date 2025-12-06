"""
AgentOps SDK - Tracer

Main interface for instrumenting AI agents.
Thread-safe, async-compatible tracer with automatic context propagation.
"""

from __future__ import annotations

import asyncio
import os
import threading
import logging
import atexit
from typing import Optional, Dict, Any, List, Callable
from contextlib import contextmanager, asynccontextmanager
from contextvars import ContextVar
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor

from .models import AgentSpan, Trace, SpanKind, SpanStatus, ToolCall, LLMCall, TokenUsage
from .exporters import SpanExporter, HTTPExporter, ConsoleExporter

logger = logging.getLogger(__name__)

# Context variables for trace propagation
_current_span: ContextVar[Optional[AgentSpan]] = ContextVar("current_span", default=None)
_current_trace_id: ContextVar[Optional[str]] = ContextVar("current_trace_id", default=None)


class AgentTracer:
    """
    Main tracer class for agent observability.
    
    Features:
    - Automatic span context propagation
    - Async-safe with contextvars
    - Batched export with configurable flush
    - Multiple exporter support
    - API key authentication
    
    Usage:
        tracer = AgentTracer(
            service_name="my-agent",
            collector_url="http://localhost:8000",
            api_key="sk-your-api-key",  # Optional, required if auth enabled
        )
        
        with tracer.start_span("process_query", kind=SpanKind.AGENT) as span:
            span.input = query
            result = process(query)
            span.output = result
    """
    
    def __init__(
        self,
        service_name: str,
        service_version: str = "0.0.1",
        collector_url: Optional[str] = None,
        api_key: Optional[str] = None,
        exporters: Optional[List[SpanExporter]] = None,
        batch_size: int = 100,
        flush_interval: float = 5.0,
        enabled: bool = True,
        console_output: bool = False,
    ):
        self.service_name = service_name
        self.service_version = service_version
        self.enabled = enabled
        
        # Get API key from parameter or environment
        self._api_key = api_key or os.getenv("AGENTOPS_API_KEY")
        
        # Initialize exporters
        self._exporters: List[SpanExporter] = exporters or []
        
        if collector_url:
            self._exporters.append(HTTPExporter(
                collector_url=collector_url,
                api_key=self._api_key,
                service_name=service_name,
                service_version=service_version,
            ))
        
        if console_output:
            self._exporters.append(ConsoleExporter())
        
        # Batching configuration
        self._batch_size = batch_size
        self._flush_interval = flush_interval
        self._span_queue: Queue[AgentSpan] = Queue()
        self._traces: Dict[str, Trace] = {}
        self._lock = threading.Lock()
        
        # Background export thread
        self._shutdown = threading.Event()
        self._export_thread = threading.Thread(target=self._export_loop, daemon=True)
        self._export_thread.start()
        
        # Register shutdown hook
        atexit.register(self.shutdown)
        
        auth_status = "enabled" if self._api_key else "disabled"
        logger.info(f"AgentTracer initialized for service: {service_name} (auth: {auth_status})")
    
    @contextmanager
    def start_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.AGENT,
        attributes: Optional[Dict[str, Any]] = None,
    ):
        """
        Start a new span with automatic context propagation.
        
        Args:
            name: Span name (e.g., "process_query", "search_tool")
            kind: Type of span (agent, tool_call, llm, etc.)
            attributes: Additional attributes to attach
            
        Yields:
            AgentSpan: The active span
        """
        if not self.enabled:
            yield AgentSpan(name=name, kind=kind)
            return
        
        # Get parent context
        parent_span = _current_span.get()
        trace_id = _current_trace_id.get()
        
        # Create new span
        span = AgentSpan(
            name=name,
            kind=kind,
            service_name=self.service_name,
            service_version=self.service_version,
            attributes=attributes or {},
        )
        
        # Set trace context
        if trace_id:
            span.trace_id = trace_id
        else:
            trace_id = span.trace_id
            _current_trace_id.set(trace_id)
            # Create new trace
            with self._lock:
                self._traces[trace_id] = Trace(
                    trace_id=trace_id,
                    service_name=self.service_name,
                    start_time=span.start_time,
                    root_span_id=span.span_id,
                )
        
        # Set parent relationship
        if parent_span:
            span.parent_span_id = parent_span.span_id
        
        # Set as current span
        token = _current_span.set(span)
        
        try:
            yield span
            if span.status == SpanStatus.UNSET:
                span.status = SpanStatus.OK
        except Exception as e:
            span.set_error(e)
            raise
        finally:
            span.end()
            _current_span.reset(token)
            
            # Add to trace and queue for export
            with self._lock:
                if trace_id in self._traces:
                    self._traces[trace_id].spans.append(span)
            
            self._span_queue.put(span)
            
            # Check if trace is complete (no parent = root span ending)
            if span.parent_span_id is None:
                self._finalize_trace(trace_id)
    
    @asynccontextmanager
    async def start_span_async(
        self,
        name: str,
        kind: SpanKind = SpanKind.AGENT,
        attributes: Optional[Dict[str, Any]] = None,
    ):
        """Async version of start_span."""
        with self.start_span(name, kind, attributes) as span:
            yield span
    
    def current_span(self) -> Optional[AgentSpan]:
        """Get the currently active span."""
        return _current_span.get()
    
    def current_trace_id(self) -> Optional[str]:
        """Get the current trace ID."""
        return _current_trace_id.get()
    
    def record_llm_call(
        self,
        model: str,
        provider: str,
        messages: List[Dict[str, str]],
        response: Optional[str] = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        temperature: float = 1.0,
        error: Optional[str] = None,
    ):
        """Record an LLM API call on the current span."""
        span = self.current_span()
        if not span:
            logger.warning("No active span to record LLM call")
            return
        
        llm_call = LLMCall(
            model=model,
            provider=provider,
            messages=messages,
            response=response,
            token_usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            temperature=temperature,
            error=error,
        )
        span.llm_calls.append(llm_call)
    
    def record_tool_call(
        self,
        name: str,
        arguments: Dict[str, Any],
        result: Optional[Any] = None,
        error: Optional[str] = None,
        duration_ms: float = 0,
    ):
        """Record a tool invocation on the current span."""
        span = self.current_span()
        if not span:
            logger.warning("No active span to record tool call")
            return
        
        tool_call = ToolCall(
            name=name,
            arguments=arguments,
            result=result,
            error=error,
            duration_ms=duration_ms,
        )
        span.tool_calls.append(tool_call)
    
    def set_reasoning(self, reasoning: str):
        """Set the reasoning/CoT for the current span."""
        span = self.current_span()
        if span:
            span.reasoning = reasoning
    
    def set_confidence(self, confidence: float):
        """Set confidence score (0.0-1.0) for current span."""
        span = self.current_span()
        if span:
            span.confidence = max(0.0, min(1.0, confidence))
    
    def add_alternative(self, alternative: str):
        """Add a rejected alternative path to current span."""
        span = self.current_span()
        if span:
            span.alternatives.append(alternative)
    
    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Add an event to the current span."""
        span = self.current_span()
        if span:
            span.add_event(name, attributes)
    
    def _finalize_trace(self, trace_id: str):
        """Finalize and export a completed trace."""
        with self._lock:
            trace = self._traces.pop(trace_id, None)
        
        if trace:
            trace.compute_metrics()
            # Export trace
            for exporter in self._exporters:
                try:
                    exporter.export_trace(trace)
                except Exception as e:
                    logger.error(f"Failed to export trace: {e}")
        
        # Clear context
        _current_trace_id.set(None)
    
    def _export_loop(self):
        """Background thread for batched span export."""
        batch: List[AgentSpan] = []
        
        while not self._shutdown.is_set():
            try:
                # Collect spans with timeout
                try:
                    span = self._span_queue.get(timeout=self._flush_interval)
                    batch.append(span)
                except Empty:
                    pass
                
                # Flush if batch is full or timeout reached
                if len(batch) >= self._batch_size or (batch and self._span_queue.empty()):
                    self._flush_batch(batch)
                    batch = []
                    
            except Exception as e:
                logger.error(f"Error in export loop: {e}")
        
        # Final flush on shutdown
        while not self._span_queue.empty():
            try:
                batch.append(self._span_queue.get_nowait())
            except Empty:
                break
        
        if batch:
            self._flush_batch(batch)
    
    def _flush_batch(self, batch: List[AgentSpan]):
        """Export a batch of spans."""
        if not batch:
            return
        
        for exporter in self._exporters:
            try:
                exporter.export_spans(batch)
            except Exception as e:
                logger.error(f"Failed to export spans: {e}")
    
    def flush(self):
        """Force flush all pending spans."""
        batch = []
        while not self._span_queue.empty():
            try:
                batch.append(self._span_queue.get_nowait())
            except Empty:
                break
        
        if batch:
            self._flush_batch(batch)
    
    def shutdown(self):
        """Graceful shutdown of the tracer."""
        logger.info("Shutting down AgentTracer...")
        self._shutdown.set()
        self.flush()
        
        for exporter in self._exporters:
            try:
                exporter.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down exporter: {e}")


# Global tracer instance (optional singleton pattern)
_global_tracer: Optional[AgentTracer] = None


def get_tracer() -> Optional[AgentTracer]:
    """Get the global tracer instance."""
    return _global_tracer


def set_tracer(tracer: AgentTracer):
    """Set the global tracer instance."""
    global _global_tracer
    _global_tracer = tracer