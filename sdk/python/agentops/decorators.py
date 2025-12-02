"""
AgentOps SDK - Decorators

Convenient decorators for instrumenting agents and tools.
"""

from __future__ import annotations

import functools
import inspect
import time
import logging
from typing import Callable, Optional, Any, TypeVar, ParamSpec

from .models import SpanKind, SpanStatus
from .tracer import AgentTracer, get_tracer

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


def trace_agent(
    tracer: Optional[AgentTracer] = None,
    name: Optional[str] = None,
    capture_input: bool = True,
    capture_output: bool = True,
):
    """
    Decorator to trace an agent function.
    
    Args:
        tracer: AgentTracer instance (uses global if not provided)
        name: Span name (defaults to function name)
        capture_input: Whether to capture function arguments
        capture_output: Whether to capture return value
    
    Usage:
        @trace_agent()
        async def my_agent(query: str):
            return process(query)
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        span_name = name or func.__name__
        
        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            _tracer = tracer or get_tracer()
            if not _tracer:
                return await func(*args, **kwargs)
            
            async with _tracer.start_span_async(span_name, kind=SpanKind.AGENT) as span:
                if capture_input:
                    span.input = _format_args(func, args, kwargs)
                
                result = await func(*args, **kwargs)
                
                if capture_output:
                    span.output = _format_output(result)
                
                return result
        
        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            _tracer = tracer or get_tracer()
            if not _tracer:
                return func(*args, **kwargs)
            
            with _tracer.start_span(span_name, kind=SpanKind.AGENT) as span:
                if capture_input:
                    span.input = _format_args(func, args, kwargs)
                
                result = func(*args, **kwargs)
                
                if capture_output:
                    span.output = _format_output(result)
                
                return result
        
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def trace_tool(
    tracer: Optional[AgentTracer] = None,
    name: Optional[str] = None,
    capture_input: bool = True,
    capture_output: bool = True,
):
    """
    Decorator to trace a tool function.
    
    Usage:
        @trace_tool()
        def search_database(query: str) -> List[Dict]:
            return db.search(query)
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        span_name = name or func.__name__
        
        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            _tracer = tracer or get_tracer()
            if not _tracer:
                return await func(*args, **kwargs)
            
            start_time = time.time()
            error_msg = None
            result = None
            
            async with _tracer.start_span_async(span_name, kind=SpanKind.TOOL_CALL) as span:
                if capture_input:
                    span.input = _format_args(func, args, kwargs)
                
                try:
                    result = await func(*args, **kwargs)
                    if capture_output:
                        span.output = _format_output(result)
                except Exception as e:
                    error_msg = str(e)
                    raise
                finally:
                    # Also record as tool call on parent span
                    duration_ms = (time.time() - start_time) * 1000
                    _tracer.record_tool_call(
                        name=span_name,
                        arguments=_args_to_dict(func, args, kwargs),
                        result=result if capture_output else None,
                        error=error_msg,
                        duration_ms=duration_ms,
                    )
                
                return result
        
        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            _tracer = tracer or get_tracer()
            if not _tracer:
                return func(*args, **kwargs)
            
            start_time = time.time()
            error_msg = None
            result = None
            
            with _tracer.start_span(span_name, kind=SpanKind.TOOL_CALL) as span:
                if capture_input:
                    span.input = _format_args(func, args, kwargs)
                
                try:
                    result = func(*args, **kwargs)
                    if capture_output:
                        span.output = _format_output(result)
                except Exception as e:
                    error_msg = str(e)
                    raise
                finally:
                    duration_ms = (time.time() - start_time) * 1000
                    _tracer.record_tool_call(
                        name=span_name,
                        arguments=_args_to_dict(func, args, kwargs),
                        result=result if capture_output else None,
                        error=error_msg,
                        duration_ms=duration_ms,
                    )
                
                return result
        
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def trace_llm(
    tracer: Optional[AgentTracer] = None,
    model: str = "unknown",
    provider: str = "unknown",
    capture_messages: bool = True,
    capture_response: bool = True,
):
    """
    Decorator to trace an LLM call.
    
    Usage:
        @trace_llm(model="gpt-4", provider="openai")
        async def call_openai(messages: List[Dict]) -> str:
            response = await client.chat.completions.create(...)
            return response.choices[0].message.content
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        span_name = f"llm_{provider}_{model}"
        
        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            _tracer = tracer or get_tracer()
            if not _tracer:
                return await func(*args, **kwargs)
            
            async with _tracer.start_span_async(span_name, kind=SpanKind.LLM) as span:
                messages = kwargs.get("messages", args[0] if args else [])
                
                result = await func(*args, **kwargs)
                
                # Record LLM call
                _tracer.record_llm_call(
                    model=model,
                    provider=provider,
                    messages=messages if capture_messages else [],
                    response=str(result) if capture_response else None,
                    # Token counts would need to be extracted from response
                )
                
                if capture_response:
                    span.output = _format_output(result)
                
                return result
        
        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            _tracer = tracer or get_tracer()
            if not _tracer:
                return func(*args, **kwargs)
            
            with _tracer.start_span(span_name, kind=SpanKind.LLM) as span:
                messages = kwargs.get("messages", args[0] if args else [])
                
                result = func(*args, **kwargs)
                
                _tracer.record_llm_call(
                    model=model,
                    provider=provider,
                    messages=messages if capture_messages else [],
                    response=str(result) if capture_response else None,
                )
                
                if capture_response:
                    span.output = _format_output(result)
                
                return result
        
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def trace_thought(
    tracer: Optional[AgentTracer] = None,
    name: str = "reasoning",
):
    """
    Decorator to trace a reasoning/thinking step.
    
    Usage:
        @trace_thought()
        def analyze_query(query: str) -> str:
            # Chain of thought reasoning
            return reasoning
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            _tracer = tracer or get_tracer()
            if not _tracer:
                return await func(*args, **kwargs)
            
            async with _tracer.start_span_async(name, kind=SpanKind.THOUGHT) as span:
                span.input = _format_args(func, args, kwargs)
                result = await func(*args, **kwargs)
                span.reasoning = _format_output(result)
                span.output = span.reasoning
                return result
        
        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            _tracer = tracer or get_tracer()
            if not _tracer:
                return func(*args, **kwargs)
            
            with _tracer.start_span(name, kind=SpanKind.THOUGHT) as span:
                span.input = _format_args(func, args, kwargs)
                result = func(*args, **kwargs)
                span.reasoning = _format_output(result)
                span.output = span.reasoning
                return result
        
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def trace_decision(
    tracer: Optional[AgentTracer] = None,
    name: str = "decision",
):
    """
    Decorator to trace a decision point.
    
    Usage:
        @trace_decision()
        def choose_action(context: Dict) -> str:
            # Decision logic
            return chosen_action
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            _tracer = tracer or get_tracer()
            if not _tracer:
                return await func(*args, **kwargs)
            
            async with _tracer.start_span_async(name, kind=SpanKind.DECISION) as span:
                span.input = _format_args(func, args, kwargs)
                result = await func(*args, **kwargs)
                span.output = _format_output(result)
                return result
        
        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            _tracer = tracer or get_tracer()
            if not _tracer:
                return func(*args, **kwargs)
            
            with _tracer.start_span(name, kind=SpanKind.DECISION) as span:
                span.input = _format_args(func, args, kwargs)
                result = func(*args, **kwargs)
                span.output = _format_output(result)
                return result
        
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


# Helper functions

def _format_args(func: Callable, args: tuple, kwargs: dict) -> str:
    """Format function arguments for logging."""
    try:
        sig = inspect.signature(func)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        
        parts = []
        for name, value in bound.arguments.items():
            formatted = _truncate(str(value), 500)
            parts.append(f"{name}={formatted}")
        
        return ", ".join(parts)
    except Exception:
        return f"args={_truncate(str(args), 500)}, kwargs={_truncate(str(kwargs), 500)}"


def _args_to_dict(func: Callable, args: tuple, kwargs: dict) -> dict:
    """Convert function arguments to dictionary."""
    try:
        sig = inspect.signature(func)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        return {k: _serialize_value(v) for k, v in bound.arguments.items()}
    except Exception:
        return {"args": list(args), "kwargs": kwargs}


def _format_output(value: Any) -> str:
    """Format output value for logging."""
    return _truncate(str(value), 1000)


def _truncate(s: str, max_len: int) -> str:
    """Truncate string to max length."""
    if len(s) <= max_len:
        return s
    return s[:max_len - 3] + "..."


def _serialize_value(value: Any) -> Any:
    """Serialize value for JSON storage."""
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, (list, tuple)):
        return [_serialize_value(v) for v in value[:10]]  # Limit list length
    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in list(value.items())[:10]}
    return str(value)[:500]
