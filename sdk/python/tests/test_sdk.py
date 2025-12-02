"""
AgentOps SDK - Unit Tests
"""

import pytest
import time
import asyncio
from unittest.mock import Mock, patch, MagicMock

from agentops import (
    AgentTracer,
    AgentSpan,
    Trace,
    SpanKind,
    SpanStatus,
    TokenUsage,
    ToolCall,
    LLMCall,
    trace_agent,
    trace_tool,
    trace_thought,
    set_tracer,
    get_tracer,
)
from agentops.exporters import ConsoleExporter, FileExporter


# =============================================================================
# Model Tests
# =============================================================================

class TestTokenUsage:
    def test_cost_estimate(self):
        usage = TokenUsage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)
        # GPT-4 pricing: $0.03/1K prompt, $0.06/1K completion
        expected = (1000 * 0.00003) + (500 * 0.00006)
        assert usage.cost_estimate == pytest.approx(expected)
    
    def test_zero_tokens(self):
        usage = TokenUsage()
        assert usage.cost_estimate == 0.0


class TestAgentSpan:
    def test_create_span(self):
        span = AgentSpan(name="test_span", kind=SpanKind.AGENT)
        
        assert span.name == "test_span"
        assert span.kind == SpanKind.AGENT
        assert span.status == SpanStatus.UNSET
        assert span.trace_id is not None
        assert span.span_id is not None
    
    def test_span_duration(self):
        span = AgentSpan(name="test")
        span.start_time = time.time()
        time.sleep(0.01)
        span.end_time = time.time()
        
        assert span.duration_ms >= 10
    
    def test_set_error(self):
        span = AgentSpan(name="test")
        
        try:
            raise ValueError("Test error")
        except Exception as e:
            span.set_error(e)
        
        assert span.status == SpanStatus.ERROR
        assert span.error_type == "ValueError"
        assert span.error_message == "Test error"
        assert span.stack_trace is not None
    
    def test_add_event(self):
        span = AgentSpan(name="test")
        span.add_event("cache_hit", {"key": "abc"})
        
        assert len(span.events) == 1
        assert span.events[0]["name"] == "cache_hit"
        assert span.events[0]["attributes"]["key"] == "abc"
    
    def test_total_tokens(self):
        span = AgentSpan(name="test")
        span.llm_calls = [
            LLMCall(
                model="gpt-4",
                provider="openai",
                messages=[],
                token_usage=TokenUsage(total_tokens=100),
            ),
            LLMCall(
                model="gpt-4",
                provider="openai",
                messages=[],
                token_usage=TokenUsage(total_tokens=200),
            ),
        ]
        
        assert span.total_tokens == 300
    
    def test_to_dict(self):
        span = AgentSpan(
            name="test",
            kind=SpanKind.TOOL_CALL,
            input="test input",
            output="test output",
        )
        span.end()
        
        data = span.to_dict()
        
        assert data["name"] == "test"
        assert data["kind"] == "tool_call"
        assert data["status"] == "ok"
        assert data["input"] == "test input"
        assert data["output"] == "test output"


class TestTrace:
    def test_compute_metrics(self):
        trace = Trace(
            trace_id="test-trace",
            service_name="test-service",
            start_time=time.time(),
        )
        
        # Add spans
        span1 = AgentSpan(name="span1", trace_id="test-trace")
        span1.llm_calls = [
            LLMCall(model="gpt-4", provider="openai", messages=[], 
                   token_usage=TokenUsage(total_tokens=100))
        ]
        span1.tool_calls = [ToolCall(name="search", arguments={})]
        span1.end()
        
        span2 = AgentSpan(name="span2", trace_id="test-trace")
        span2.status = SpanStatus.ERROR
        span2.end()
        
        trace.spans = [span1, span2]
        trace.compute_metrics()
        
        assert trace.total_tokens == 100
        assert trace.total_tool_calls == 1
        assert trace.total_llm_calls == 1
        assert trace.error_count == 1
        assert trace.status == SpanStatus.ERROR


# =============================================================================
# Tracer Tests
# =============================================================================

class TestAgentTracer:
    def test_create_tracer(self):
        tracer = AgentTracer(
            service_name="test-service",
            enabled=True,
        )
        
        assert tracer.service_name == "test-service"
        assert tracer.enabled is True
        
        tracer.shutdown()
    
    def test_start_span_context_manager(self):
        tracer = AgentTracer(service_name="test", enabled=True)
        
        with tracer.start_span("test_span", kind=SpanKind.AGENT) as span:
            span.input = "test input"
            span.output = "test output"
        
        assert span.status == SpanStatus.OK
        assert span.end_time is not None
        
        tracer.shutdown()
    
    def test_nested_spans(self):
        tracer = AgentTracer(service_name="test", enabled=True)
        
        with tracer.start_span("parent", kind=SpanKind.AGENT) as parent:
            parent_id = parent.span_id
            
            with tracer.start_span("child", kind=SpanKind.TOOL_CALL) as child:
                assert child.parent_span_id == parent_id
                assert child.trace_id == parent.trace_id
        
        tracer.shutdown()
    
    def test_span_error_handling(self):
        tracer = AgentTracer(service_name="test", enabled=True)
        
        with pytest.raises(ValueError):
            with tracer.start_span("failing_span") as span:
                raise ValueError("Test error")
        
        assert span.status == SpanStatus.ERROR
        assert span.error_type == "ValueError"
        
        tracer.shutdown()
    
    def test_record_llm_call(self):
        tracer = AgentTracer(service_name="test", enabled=True)
        
        with tracer.start_span("test") as span:
            tracer.record_llm_call(
                model="gpt-4",
                provider="openai",
                messages=[{"role": "user", "content": "Hello"}],
                response="Hi!",
                prompt_tokens=10,
                completion_tokens=5,
            )
        
        assert len(span.llm_calls) == 1
        assert span.llm_calls[0].model == "gpt-4"
        assert span.llm_calls[0].token_usage.total_tokens == 15
        
        tracer.shutdown()
    
    def test_record_tool_call(self):
        tracer = AgentTracer(service_name="test", enabled=True)
        
        with tracer.start_span("test") as span:
            tracer.record_tool_call(
                name="search",
                arguments={"query": "test"},
                result=["item1", "item2"],
                duration_ms=100,
            )
        
        assert len(span.tool_calls) == 1
        assert span.tool_calls[0].name == "search"
        assert span.tool_calls[0].result == ["item1", "item2"]
        
        tracer.shutdown()
    
    def test_set_reasoning(self):
        tracer = AgentTracer(service_name="test", enabled=True)
        
        with tracer.start_span("test") as span:
            tracer.set_reasoning("Step 1: Think. Step 2: Act.")
        
        assert span.reasoning == "Step 1: Think. Step 2: Act."
        
        tracer.shutdown()
    
    def test_set_confidence(self):
        tracer = AgentTracer(service_name="test", enabled=True)
        
        with tracer.start_span("test") as span:
            tracer.set_confidence(0.85)
        
        assert span.confidence == 0.85
        
        tracer.shutdown()
    
    def test_confidence_clamping(self):
        tracer = AgentTracer(service_name="test", enabled=True)
        
        with tracer.start_span("test") as span:
            tracer.set_confidence(1.5)  # Should be clamped to 1.0
        
        assert span.confidence == 1.0
        
        with tracer.start_span("test2") as span2:
            tracer.set_confidence(-0.5)  # Should be clamped to 0.0
        
        assert span2.confidence == 0.0
        
        tracer.shutdown()
    
    def test_disabled_tracer(self):
        tracer = AgentTracer(service_name="test", enabled=False)
        
        with tracer.start_span("test") as span:
            span.input = "test"
        
        # Should not raise, span is a no-op
        assert span.name == "test"
        
        tracer.shutdown()
    
    def test_global_tracer(self):
        tracer = AgentTracer(service_name="global-test", enabled=True)
        set_tracer(tracer)
        
        assert get_tracer() is tracer
        
        tracer.shutdown()


# =============================================================================
# Decorator Tests
# =============================================================================

class TestDecorators:
    def test_trace_agent_sync(self):
        tracer = AgentTracer(service_name="test", enabled=True)
        set_tracer(tracer)
        
        @trace_agent(tracer)
        def my_agent(query: str) -> str:
            return f"Response to: {query}"
        
        result = my_agent("Hello")
        
        assert result == "Response to: Hello"
        
        tracer.shutdown()
    
    @pytest.mark.asyncio
    async def test_trace_agent_async(self):
        tracer = AgentTracer(service_name="test", enabled=True)
        set_tracer(tracer)
        
        @trace_agent(tracer)
        async def my_async_agent(query: str) -> str:
            await asyncio.sleep(0.01)
            return f"Async response to: {query}"
        
        result = await my_async_agent("Hello")
        
        assert result == "Async response to: Hello"
        
        tracer.shutdown()
    
    def test_trace_tool(self):
        tracer = AgentTracer(service_name="test", enabled=True)
        
        @trace_tool(tracer)
        def search(query: str) -> list:
            return [f"Result for {query}"]
        
        with tracer.start_span("parent") as parent_span:
            results = search("test")
        
        assert results == ["Result for test"]
        # Tool call should be recorded on parent span
        assert len(parent_span.tool_calls) == 1
        
        tracer.shutdown()
    
    def test_trace_thought(self):
        tracer = AgentTracer(service_name="test", enabled=True)
        
        @trace_thought(tracer)
        def analyze(data: str) -> str:
            return f"Analysis: {data} is interesting"
        
        result = analyze("test data")
        
        assert "Analysis:" in result
        
        tracer.shutdown()
    
    def test_decorator_error_handling(self):
        tracer = AgentTracer(service_name="test", enabled=True)
        
        @trace_agent(tracer)
        def failing_agent():
            raise RuntimeError("Agent failed")
        
        with pytest.raises(RuntimeError):
            failing_agent()
        
        tracer.shutdown()


# =============================================================================
# Exporter Tests
# =============================================================================

class TestExporters:
    def test_console_exporter(self, capsys):
        exporter = ConsoleExporter(colors=False)
        
        span = AgentSpan(name="test_span", kind=SpanKind.AGENT)
        span.input = "test input"
        span.output = "test output"
        span.end()
        
        exporter.export_spans([span])
        
        captured = capsys.readouterr()
        assert "test_span" in captured.out
    
    def test_file_exporter(self, tmp_path):
        filepath = tmp_path / "traces.jsonl"
        exporter = FileExporter(str(filepath))
        
        span = AgentSpan(name="test_span")
        span.end()
        
        exporter.export_spans([span])
        
        # Check file was written
        assert filepath.exists()
        content = filepath.read_text()
        assert "test_span" in content


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    def test_full_agent_flow(self):
        """Test a complete agent execution flow."""
        tracer = AgentTracer(
            service_name="integration-test",
            enabled=True,
            console_output=False,
        )
        set_tracer(tracer)
        
        @trace_tool(tracer)
        def search_tool(query: str) -> list:
            return [{"id": 1, "title": f"Result: {query}"}]
        
        @trace_thought(tracer)
        def analyze_query(query: str) -> str:
            return f"Query '{query}' requires database search"
        
        @trace_agent(tracer)
        def test_agent(query: str) -> str:
            # Reasoning
            reasoning = analyze_query(query)
            tracer.set_reasoning(reasoning)
            
            # Tool use
            results = search_tool(query)
            tracer.add_event("search_complete", {"count": len(results)})
            
            # LLM call
            tracer.record_llm_call(
                model="gpt-4",
                provider="openai",
                messages=[{"role": "user", "content": query}],
                response="Based on search...",
                prompt_tokens=50,
                completion_tokens=20,
            )
            
            # Confidence
            tracer.set_confidence(0.9)
            
            return f"Found {len(results)} results for: {query}"
        
        # Run agent
        result = test_agent("machine learning")
        
        assert "Found 1 results" in result
        
        # Flush and shutdown
        tracer.flush()
        tracer.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
