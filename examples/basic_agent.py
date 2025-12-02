"""
Example: Basic Agent with AgentOps Instrumentation

This example demonstrates how to instrument a simple AI agent
using the AgentOps SDK.

Run:
    pip install -e ./sdk/python
    python examples/basic_agent.py
"""

import asyncio
import random
import time
from typing import List, Dict, Any

# Import AgentOps SDK
from agentops import (
    AgentTracer,
    trace_agent,
    trace_tool,
    trace_thought,
    trace_decision,
    SpanKind,
    set_tracer,
)


# Initialize tracer
tracer = AgentTracer(
    service_name="example-agent",
    collector_url="http://localhost:8000",
    console_output=True,  # Also print to console for debugging
)
set_tracer(tracer)


# ============================================================================
# Tools
# ============================================================================

@trace_tool(tracer)
def search_database(query: str) -> List[Dict[str, Any]]:
    """Simulate database search."""
    time.sleep(0.1)  # Simulate latency
    
    # Mock results
    results = [
        {"id": 1, "title": f"Result 1 for '{query}'", "score": 0.95},
        {"id": 2, "title": f"Result 2 for '{query}'", "score": 0.87},
        {"id": 3, "title": f"Result 3 for '{query}'", "score": 0.72},
    ]
    
    return results


@trace_tool(tracer)
def fetch_document(doc_id: int) -> str:
    """Simulate document retrieval."""
    time.sleep(0.05)
    
    return f"Document content for ID {doc_id}. Lorem ipsum dolor sit amet..."


@trace_tool(tracer)
def call_api(endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Simulate external API call."""
    time.sleep(0.2)
    
    # Randomly fail sometimes
    if random.random() < 0.1:
        raise Exception(f"API timeout: {endpoint}")
    
    return {"status": "success", "data": {"endpoint": endpoint, "processed": True}}


# ============================================================================
# Agent Components
# ============================================================================

@trace_thought(tracer)
def analyze_query(query: str) -> str:
    """Analyze user query and plan approach."""
    time.sleep(0.05)
    
    reasoning = f"""
    Query Analysis:
    - Input: "{query}"
    - Type: Information retrieval
    - Complexity: Medium
    - Required tools: search_database, fetch_document
    - Approach: Search first, then retrieve top document
    """
    
    return reasoning.strip()


@trace_decision(tracer)
def choose_action(context: Dict[str, Any]) -> str:
    """Decide which action to take based on context."""
    time.sleep(0.02)
    
    if not context.get("search_results"):
        return "search"
    elif not context.get("document_content"):
        return "fetch"
    else:
        return "respond"


# ============================================================================
# Main Agent
# ============================================================================

@trace_agent(tracer)
def simple_agent(query: str) -> str:
    """
    Simple agent that:
    1. Analyzes the query
    2. Searches database
    3. Fetches relevant document
    4. Generates response
    """
    context = {"query": query}
    
    # Step 1: Analyze query
    reasoning = analyze_query(query)
    tracer.set_reasoning(reasoning)
    
    # Step 2: Decide action and execute
    action = choose_action(context)
    
    if action == "search":
        # Search database
        results = search_database(query)
        context["search_results"] = results
        tracer.add_event("search_completed", {"results_count": len(results)})
        
        # Record alternatives considered
        tracer.add_alternative("Could use web search instead")
        tracer.add_alternative("Could use vector similarity search")
    
    # Step 3: Fetch top document
    action = choose_action(context)
    
    if action == "fetch" and context.get("search_results"):
        top_result = context["search_results"][0]
        content = fetch_document(top_result["id"])
        context["document_content"] = content
        tracer.add_event("document_fetched", {"doc_id": top_result["id"]})
    
    # Step 4: Generate response
    response = f"Based on my search for '{query}', I found relevant information. "
    response += f"The top result was: {context.get('search_results', [{}])[0].get('title', 'N/A')}. "
    response += "Here's a summary of the content..."
    
    # Set confidence
    tracer.set_confidence(0.85)
    
    # Simulate LLM call for response generation
    with tracer.start_span("generate_response", kind=SpanKind.LLM) as span:
        span.input = f"Generate response for: {query}"
        time.sleep(0.1)  # Simulate LLM latency
        span.output = response
        
        # Record LLM metrics
        tracer.record_llm_call(
            model="gpt-4",
            provider="openai",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Answer based on: {context}"}
            ],
            response=response,
            prompt_tokens=150,
            completion_tokens=50,
            temperature=0.7,
        )
    
    return response


@trace_agent(tracer)
async def async_agent(query: str) -> str:
    """Async version of the agent."""
    
    # Simulate async operations
    with tracer.start_span("async_search", kind=SpanKind.TOOL_CALL) as span:
        span.input = query
        await asyncio.sleep(0.1)
        span.output = "Async search results"
    
    return f"Async response for: {query}"


# ============================================================================
# Demo
# ============================================================================

def run_demo():
    """Run demonstration of the agent."""
    print("\n" + "=" * 60)
    print("AgentOps SDK Demo")
    print("=" * 60 + "\n")
    
    # Run several queries
    queries = [
        "What is machine learning?",
        "How do neural networks work?",
        "Explain transformers architecture",
    ]
    
    for query in queries:
        print(f"\n{'─' * 40}")
        print(f"Query: {query}")
        print('─' * 40)
        
        try:
            response = simple_agent(query)
            print(f"Response: {response[:100]}...")
        except Exception as e:
            print(f"Error: {e}")
    
    # Run async agent
    print(f"\n{'─' * 40}")
    print("Running async agent...")
    print('─' * 40)
    
    async def run_async():
        response = await async_agent("Async query test")
        print(f"Async Response: {response}")
    
    asyncio.run(run_async())
    
    # Flush tracer
    tracer.flush()
    
    print("\n" + "=" * 60)
    print("Demo complete! Check the UI at http://localhost:8501")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_demo()
