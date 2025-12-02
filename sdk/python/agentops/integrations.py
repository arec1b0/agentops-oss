"""
AgentOps SDK - Framework Integrations

Auto-instrumentation for popular LLM/Agent frameworks.
"""

from __future__ import annotations

import logging
from typing import Optional, Any, Dict, List, Callable
from functools import wraps

from .tracer import AgentTracer, get_tracer
from .models import SpanKind, TokenUsage

logger = logging.getLogger(__name__)


class LangChainInstrumentation:
    """
    Auto-instrumentation for LangChain.
    
    Patches LangChain components to automatically trace:
    - LLM calls
    - Chain executions
    - Tool invocations
    - Agent runs
    
    Usage:
        from agentops.integrations import LangChainInstrumentation
        
        tracer = AgentTracer(service_name="my-agent", collector_url="...")
        LangChainInstrumentation(tracer).instrument()
    """
    
    def __init__(self, tracer: Optional[AgentTracer] = None):
        self.tracer = tracer or get_tracer()
        self._original_methods: Dict[str, Callable] = {}
    
    def instrument(self):
        """Apply instrumentation patches."""
        self._patch_llms()
        self._patch_chains()
        self._patch_tools()
        self._patch_agents()
        logger.info("LangChain instrumentation applied")
    
    def uninstrument(self):
        """Remove instrumentation patches."""
        for key, method in self._original_methods.items():
            module_name, class_name, method_name = key.rsplit(".", 2)
            try:
                import importlib
                module = importlib.import_module(module_name)
                cls = getattr(module, class_name)
                setattr(cls, method_name, method)
            except Exception as e:
                logger.warning(f"Failed to restore {key}: {e}")
        
        self._original_methods.clear()
        logger.info("LangChain instrumentation removed")
    
    def _patch_llms(self):
        """Patch LLM classes."""
        try:
            from langchain_core.language_models.llms import BaseLLM
            self._patch_method(
                BaseLLM, "_generate",
                self._wrap_llm_generate,
                "langchain_core.language_models.llms.BaseLLM._generate"
            )
        except ImportError:
            pass
        
        try:
            from langchain_core.language_models.chat_models import BaseChatModel
            self._patch_method(
                BaseChatModel, "_generate",
                self._wrap_chat_generate,
                "langchain_core.language_models.chat_models.BaseChatModel._generate"
            )
        except ImportError:
            pass
    
    def _patch_chains(self):
        """Patch Chain classes."""
        try:
            from langchain_core.runnables import Runnable
            self._patch_method(
                Runnable, "invoke",
                self._wrap_chain_invoke,
                "langchain_core.runnables.Runnable.invoke"
            )
        except ImportError:
            pass
    
    def _patch_tools(self):
        """Patch Tool classes."""
        try:
            from langchain_core.tools import BaseTool
            self._patch_method(
                BaseTool, "_run",
                self._wrap_tool_run,
                "langchain_core.tools.BaseTool._run"
            )
        except ImportError:
            pass
    
    def _patch_agents(self):
        """Patch Agent classes."""
        try:
            from langchain.agents import AgentExecutor
            self._patch_method(
                AgentExecutor, "invoke",
                self._wrap_agent_invoke,
                "langchain.agents.AgentExecutor.invoke"
            )
        except ImportError:
            pass
    
    def _patch_method(self, cls: type, method_name: str, wrapper: Callable, key: str):
        """Apply a wrapper to a class method."""
        original = getattr(cls, method_name, None)
        if original is None:
            return
        
        self._original_methods[key] = original
        
        @wraps(original)
        def patched(self_obj, *args, **kwargs):
            return wrapper(original, self_obj, *args, **kwargs)
        
        setattr(cls, method_name, patched)
    
    def _wrap_llm_generate(self, original: Callable, llm_self: Any, *args, **kwargs):
        """Wrap LLM generate method."""
        if not self.tracer:
            return original(llm_self, *args, **kwargs)
        
        model_name = getattr(llm_self, "model_name", "unknown")
        
        with self.tracer.start_span(f"llm_{model_name}", kind=SpanKind.LLM) as span:
            span.attributes["model"] = model_name
            span.attributes["provider"] = type(llm_self).__name__
            
            if args:
                span.input = str(args[0])[:1000]
            
            result = original(llm_self, *args, **kwargs)
            
            if hasattr(result, "generations") and result.generations:
                span.output = str(result.generations[0][0].text)[:1000]
            
            # Extract token usage if available
            if hasattr(result, "llm_output") and result.llm_output:
                usage = result.llm_output.get("token_usage", {})
                self.tracer.record_llm_call(
                    model=model_name,
                    provider=type(llm_self).__name__,
                    messages=[{"content": str(args[0]) if args else ""}],
                    response=span.output,
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                )
            
            return result
    
    def _wrap_chat_generate(self, original: Callable, chat_self: Any, *args, **kwargs):
        """Wrap ChatModel generate method."""
        if not self.tracer:
            return original(chat_self, *args, **kwargs)
        
        model_name = getattr(chat_self, "model_name", "unknown")
        
        with self.tracer.start_span(f"chat_{model_name}", kind=SpanKind.LLM) as span:
            span.attributes["model"] = model_name
            span.attributes["provider"] = type(chat_self).__name__
            
            # Capture messages
            if args:
                messages = args[0]
                span.input = str(messages)[:1000]
            
            result = original(chat_self, *args, **kwargs)
            
            if hasattr(result, "generations") and result.generations:
                gen = result.generations[0][0]
                if hasattr(gen, "message"):
                    span.output = str(gen.message.content)[:1000]
            
            return result
    
    def _wrap_chain_invoke(self, original: Callable, chain_self: Any, *args, **kwargs):
        """Wrap chain invoke method."""
        if not self.tracer:
            return original(chain_self, *args, **kwargs)
        
        chain_name = getattr(chain_self, "name", type(chain_self).__name__)
        
        with self.tracer.start_span(f"chain_{chain_name}", kind=SpanKind.AGENT) as span:
            span.attributes["chain_type"] = type(chain_self).__name__
            
            if args:
                span.input = str(args[0])[:1000]
            
            result = original(chain_self, *args, **kwargs)
            span.output = str(result)[:1000]
            
            return result
    
    def _wrap_tool_run(self, original: Callable, tool_self: Any, *args, **kwargs):
        """Wrap tool run method."""
        if not self.tracer:
            return original(tool_self, *args, **kwargs)
        
        tool_name = getattr(tool_self, "name", type(tool_self).__name__)
        
        with self.tracer.start_span(f"tool_{tool_name}", kind=SpanKind.TOOL_CALL) as span:
            span.attributes["tool_name"] = tool_name
            span.attributes["tool_description"] = getattr(tool_self, "description", "")[:200]
            
            if args:
                span.input = str(args[0])[:1000]
            
            result = original(tool_self, *args, **kwargs)
            span.output = str(result)[:1000]
            
            # Record tool call
            self.tracer.record_tool_call(
                name=tool_name,
                arguments={"input": args[0] if args else kwargs},
                result=result,
            )
            
            return result
    
    def _wrap_agent_invoke(self, original: Callable, agent_self: Any, *args, **kwargs):
        """Wrap agent executor invoke method."""
        if not self.tracer:
            return original(agent_self, *args, **kwargs)
        
        with self.tracer.start_span("agent_executor", kind=SpanKind.AGENT) as span:
            if args:
                span.input = str(args[0])[:1000]
            
            result = original(agent_self, *args, **kwargs)
            
            if isinstance(result, dict):
                span.output = str(result.get("output", result))[:1000]
            else:
                span.output = str(result)[:1000]
            
            return result


class OpenAIInstrumentation:
    """
    Auto-instrumentation for OpenAI Python client.
    
    Usage:
        from agentops.integrations import OpenAIInstrumentation
        
        tracer = AgentTracer(service_name="my-agent", collector_url="...")
        OpenAIInstrumentation(tracer).instrument()
    """
    
    def __init__(self, tracer: Optional[AgentTracer] = None):
        self.tracer = tracer or get_tracer()
        self._original_methods: Dict[str, Callable] = {}
    
    def instrument(self):
        """Apply instrumentation patches."""
        self._patch_completions()
        self._patch_chat_completions()
        logger.info("OpenAI instrumentation applied")
    
    def uninstrument(self):
        """Remove instrumentation patches."""
        for key, method in self._original_methods.items():
            parts = key.rsplit(".", 1)
            if len(parts) == 2:
                module_path, method_name = parts
                try:
                    import importlib
                    parts = module_path.rsplit(".", 1)
                    module = importlib.import_module(parts[0])
                    cls = getattr(module, parts[1])
                    setattr(cls, method_name, method)
                except Exception as e:
                    logger.warning(f"Failed to restore {key}: {e}")
        
        self._original_methods.clear()
        logger.info("OpenAI instrumentation removed")
    
    def _patch_completions(self):
        """Patch completions endpoint."""
        try:
            from openai.resources import Completions
            
            original = Completions.create
            self._original_methods["openai.resources.Completions.create"] = original
            
            @wraps(original)
            def patched(self_obj, *args, **kwargs):
                return self._wrap_completion(original, self_obj, *args, **kwargs)
            
            Completions.create = patched
        except ImportError:
            pass
    
    def _patch_chat_completions(self):
        """Patch chat completions endpoint."""
        try:
            from openai.resources.chat import Completions as ChatCompletions
            
            original = ChatCompletions.create
            self._original_methods["openai.resources.chat.Completions.create"] = original
            
            @wraps(original)
            def patched(self_obj, *args, **kwargs):
                return self._wrap_chat_completion(original, self_obj, *args, **kwargs)
            
            ChatCompletions.create = patched
        except ImportError:
            pass
    
    def _wrap_completion(self, original: Callable, client_self: Any, *args, **kwargs):
        """Wrap completions.create."""
        if not self.tracer:
            return original(client_self, *args, **kwargs)
        
        model = kwargs.get("model", "unknown")
        
        with self.tracer.start_span(f"openai_completion_{model}", kind=SpanKind.LLM) as span:
            span.attributes["model"] = model
            span.attributes["provider"] = "openai"
            
            prompt = kwargs.get("prompt", "")
            span.input = str(prompt)[:1000]
            
            result = original(client_self, *args, **kwargs)
            
            # Extract response
            if hasattr(result, "choices") and result.choices:
                span.output = result.choices[0].text[:1000]
            
            # Record LLM call with usage
            usage = getattr(result, "usage", None)
            self.tracer.record_llm_call(
                model=model,
                provider="openai",
                messages=[{"role": "user", "content": str(prompt)}],
                response=span.output,
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                temperature=kwargs.get("temperature", 1.0),
            )
            
            return result
    
    def _wrap_chat_completion(self, original: Callable, client_self: Any, *args, **kwargs):
        """Wrap chat.completions.create."""
        if not self.tracer:
            return original(client_self, *args, **kwargs)
        
        model = kwargs.get("model", "unknown")
        
        with self.tracer.start_span(f"openai_chat_{model}", kind=SpanKind.LLM) as span:
            span.attributes["model"] = model
            span.attributes["provider"] = "openai"
            
            messages = kwargs.get("messages", [])
            span.input = str(messages)[:1000]
            
            result = original(client_self, *args, **kwargs)
            
            # Extract response
            if hasattr(result, "choices") and result.choices:
                message = result.choices[0].message
                span.output = str(message.content)[:1000] if message.content else ""
                
                # Check for tool calls
                if hasattr(message, "tool_calls") and message.tool_calls:
                    for tc in message.tool_calls:
                        self.tracer.record_tool_call(
                            name=tc.function.name,
                            arguments=tc.function.arguments,
                        )
            
            # Record LLM call with usage
            usage = getattr(result, "usage", None)
            self.tracer.record_llm_call(
                model=model,
                provider="openai",
                messages=messages,
                response=span.output,
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                temperature=kwargs.get("temperature", 1.0),
            )
            
            return result


class AnthropicInstrumentation:
    """
    Auto-instrumentation for Anthropic Python client.
    
    Usage:
        from agentops.integrations import AnthropicInstrumentation
        
        tracer = AgentTracer(service_name="my-agent", collector_url="...")
        AnthropicInstrumentation(tracer).instrument()
    """
    
    def __init__(self, tracer: Optional[AgentTracer] = None):
        self.tracer = tracer or get_tracer()
        self._original_methods: Dict[str, Callable] = {}
    
    def instrument(self):
        """Apply instrumentation patches."""
        try:
            from anthropic.resources import Messages
            
            original = Messages.create
            self._original_methods["anthropic.resources.Messages.create"] = original
            
            @wraps(original)
            def patched(self_obj, *args, **kwargs):
                return self._wrap_messages_create(original, self_obj, *args, **kwargs)
            
            Messages.create = patched
            logger.info("Anthropic instrumentation applied")
        except ImportError:
            logger.warning("Anthropic package not found, skipping instrumentation")
    
    def uninstrument(self):
        """Remove instrumentation patches."""
        for key, method in self._original_methods.items():
            try:
                from anthropic.resources import Messages
                Messages.create = method
            except Exception as e:
                logger.warning(f"Failed to restore {key}: {e}")
        
        self._original_methods.clear()
        logger.info("Anthropic instrumentation removed")
    
    def _wrap_messages_create(self, original: Callable, client_self: Any, *args, **kwargs):
        """Wrap messages.create."""
        if not self.tracer:
            return original(client_self, *args, **kwargs)
        
        model = kwargs.get("model", "unknown")
        
        with self.tracer.start_span(f"anthropic_{model}", kind=SpanKind.LLM) as span:
            span.attributes["model"] = model
            span.attributes["provider"] = "anthropic"
            
            messages = kwargs.get("messages", [])
            span.input = str(messages)[:1000]
            
            result = original(client_self, *args, **kwargs)
            
            # Extract response
            if hasattr(result, "content") and result.content:
                text_blocks = [b.text for b in result.content if hasattr(b, "text")]
                span.output = " ".join(text_blocks)[:1000]
            
            # Record LLM call with usage
            usage = getattr(result, "usage", None)
            self.tracer.record_llm_call(
                model=model,
                provider="anthropic",
                messages=messages,
                response=span.output,
                prompt_tokens=usage.input_tokens if usage else 0,
                completion_tokens=usage.output_tokens if usage else 0,
            )
            
            return result
