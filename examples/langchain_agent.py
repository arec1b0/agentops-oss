"""
Example: LangChain Agent with AgentOps Auto-Instrumentation

This example shows how to use automatic instrumentation
with LangChain agents.

Prerequisites:
    pip install langchain langchain-openai
    export OPENAI_API_KEY=your-key

Run:
    python examples/langchain_agent.py
"""

import os
from typing import List

# Check for API key
if not os.getenv("OPENAI_API_KEY"):
    print("Warning: OPENAI_API_KEY not set. Using mock mode.")
    MOCK_MODE = True
else:
    MOCK_MODE = False

# Import AgentOps
from agentops import AgentTracer, set_tracer
from agentops.integrations import LangChainInstrumentation, OpenAIInstrumentation

# Initialize tracer
tracer = AgentTracer(
    service_name="langchain-agent",
    collector_url="http://localhost:8000",
    console_output=True,
)
set_tracer(tracer)

# Apply auto-instrumentation
langchain_instr = LangChainInstrumentation(tracer)
langchain_instr.instrument()

openai_instr = OpenAIInstrumentation(tracer)
openai_instr.instrument()

print("✓ AgentOps instrumentation applied")


def run_langchain_example():
    """Run LangChain example with tracing."""
    
    if MOCK_MODE:
        print("\n[Mock Mode] Simulating LangChain agent...")
        
        # Simulate traced execution
        with tracer.start_span("langchain_agent") as span:
            span.input = "What is the capital of France?"
            
            # Simulate chain execution
            with tracer.start_span("chain_execution") as chain_span:
                chain_span.input = "Query processing"
                
                # Simulate LLM call
                tracer.record_llm_call(
                    model="gpt-4",
                    provider="openai",
                    messages=[{"role": "user", "content": "What is the capital of France?"}],
                    response="The capital of France is Paris.",
                    prompt_tokens=20,
                    completion_tokens=10,
                )
                
                chain_span.output = "Paris"
            
            span.output = "The capital of France is Paris."
        
        print("Response: The capital of France is Paris.")
        tracer.flush()
        return
    
    # Real LangChain execution
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        
        # Create simple chain
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Answer concisely."),
            ("user", "{question}")
        ])
        chain = prompt | llm | StrOutputParser()
        
        # Run chain (automatically traced)
        questions = [
            "What is the capital of France?",
            "Who wrote Romeo and Juliet?",
            "What is 2 + 2?",
        ]
        
        for question in questions:
            print(f"\nQuestion: {question}")
            response = chain.invoke({"question": question})
            print(f"Response: {response}")
        
    except ImportError as e:
        print(f"LangChain not installed: {e}")
        print("Install with: pip install langchain langchain-openai")
    except Exception as e:
        print(f"Error: {e}")
    
    # Cleanup
    tracer.flush()


def run_langchain_agent_example():
    """Run LangChain agent with tools."""
    
    if MOCK_MODE:
        print("\n[Mock Mode] Simulating LangChain agent with tools...")
        
        with tracer.start_span("agent_executor") as span:
            span.input = "Search for recent AI news"
            
            # Simulate tool use
            tracer.record_tool_call(
                name="search",
                arguments={"query": "recent AI news"},
                result="Found 5 articles about AI...",
                duration_ms=150,
            )
            
            # Simulate LLM reasoning
            tracer.record_llm_call(
                model="gpt-4",
                provider="openai",
                messages=[{"role": "user", "content": "Summarize AI news"}],
                response="Here are the key AI developments...",
                prompt_tokens=100,
                completion_tokens=200,
            )
            
            span.output = "Based on recent articles, here are key AI developments..."
        
        print("Agent completed with mock data.")
        tracer.flush()
        return
    
    try:
        from langchain_openai import ChatOpenAI
        from langchain.agents import AgentExecutor, create_openai_functions_agent
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        from langchain_core.tools import tool
        
        @tool
        def search(query: str) -> str:
            """Search for information."""
            return f"Search results for '{query}': Found relevant information."
        
        @tool
        def calculator(expression: str) -> str:
            """Calculate mathematical expression."""
            try:
                result = eval(expression)
                return f"Result: {result}"
            except Exception as e:
                return f"Error: {e}"
        
        # Create agent
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        tools = [search, calculator]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant with access to tools."),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        agent = create_openai_functions_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        
        # Run agent (automatically traced)
        result = agent_executor.invoke({
            "input": "What is 25 * 4 + 10?"
        })
        
        print(f"\nAgent Result: {result['output']}")
        
    except ImportError as e:
        print(f"Required packages not installed: {e}")
    except Exception as e:
        print(f"Error: {e}")
    
    tracer.flush()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("LangChain + AgentOps Integration Demo")
    print("=" * 60)
    
    print("\n--- Simple Chain Example ---")
    run_langchain_example()
    
    print("\n--- Agent with Tools Example ---")
    run_langchain_agent_example()
    
    print("\n" + "=" * 60)
    print("Demo complete! Check traces at http://localhost:8501")
    print("=" * 60)
