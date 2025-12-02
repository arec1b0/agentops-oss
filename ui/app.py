"""
AgentOps UI - Streamlit Dashboard

Interactive dashboard for exploring agent traces and spans.
"""

import os
import json
import streamlit as st
import httpx
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

# Configuration
COLLECTOR_URL = os.getenv("COLLECTOR_URL", "http://localhost:8000")

st.set_page_config(
    page_title="AgentOps",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .trace-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .span-ok { border-left: 4px solid #28a745; }
    .span-error { border-left: 4px solid #dc3545; }
    .span-unset { border-left: 4px solid #ffc107; }
    .metric-card {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# API Client
# ============================================================================

class APIClient:
    """Client for AgentOps Collector API."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=30.0)
    
    def health(self) -> Dict[str, Any]:
        """Check collector health."""
        try:
            response = self.client.get(f"{self.base_url}/health")
            return response.json()
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    def list_traces(
        self,
        service: Optional[str] = None,
        status: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> Dict[str, Any]:
        """List traces with filtering."""
        params = {"page": page, "page_size": page_size}
        if service:
            params["service"] = service
        if status:
            params["status"] = status
        
        response = self.client.get(f"{self.base_url}/v1/traces", params=params)
        return response.json()
    
    def get_trace(self, trace_id: str) -> Dict[str, Any]:
        """Get trace details."""
        response = self.client.get(f"{self.base_url}/v1/traces/{trace_id}")
        return response.json()
    
    def search(self, query: str, limit: int = 20) -> Dict[str, Any]:
        """Search spans."""
        response = self.client.get(
            f"{self.base_url}/v1/search",
            params={"q": query, "limit": limit}
        )
        return response.json()
    
    def get_services(self) -> List[str]:
        """Get list of services."""
        response = self.client.get(f"{self.base_url}/v1/services")
        return response.json().get("services", [])
    
    def get_service_stats(self, service: str) -> Dict[str, Any]:
        """Get service statistics."""
        response = self.client.get(f"{self.base_url}/v1/services/{service}/stats")
        return response.json()


@st.cache_resource
def get_client():
    """Get cached API client."""
    return APIClient(COLLECTOR_URL)


# ============================================================================
# UI Components
# ============================================================================

def render_sidebar():
    """Render sidebar with navigation and filters."""
    with st.sidebar:
        st.title("🤖 AgentOps")
        st.caption("Open-source Agent Observability")
        
        st.divider()
        
        # Navigation
        page = st.radio(
            "Navigation",
            ["📊 Dashboard", "🔍 Traces", "🔎 Search", "📈 Analytics"],
            label_visibility="collapsed",
        )
        
        st.divider()
        
        # Health status
        client = get_client()
        health = client.health()
        
        if health.get("status") == "healthy":
            st.success("Collector: Connected")
            st.caption(f"Traces: {health.get('traces_count', 0):,}")
            st.caption(f"Spans: {health.get('spans_count', 0):,}")
        else:
            st.error("Collector: Disconnected")
            st.caption(health.get("error", "Unknown error"))
        
        return page


def render_dashboard():
    """Render main dashboard."""
    st.title("📊 Dashboard")
    
    client = get_client()
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    
    health = client.health()
    
    with col1:
        st.metric("Total Traces", f"{health.get('traces_count', 0):,}")
    
    with col2:
        st.metric("Total Spans", f"{health.get('spans_count', 0):,}")
    
    with col3:
        uptime = health.get('uptime_seconds', 0)
        hours = int(uptime // 3600)
        minutes = int((uptime % 3600) // 60)
        st.metric("Uptime", f"{hours}h {minutes}m")
    
    with col4:
        st.metric("Version", health.get('version', 'N/A'))
    
    st.divider()
    
    # Recent traces
    st.subheader("Recent Traces")
    
    try:
        traces_data = client.list_traces(page_size=10)
        traces = traces_data.get("traces", [])
        
        if not traces:
            st.info("No traces yet. Start sending data from your agent!")
        else:
            for trace in traces:
                render_trace_card(trace)
    except Exception as e:
        st.error(f"Failed to load traces: {e}")


def render_trace_card(trace: Dict[str, Any]):
    """Render a trace summary card."""
    status_class = f"span-{trace.get('status', 'unset')}"
    status_emoji = {"ok": "✅", "error": "❌", "unset": "⏳"}.get(trace.get("status"), "❓")
    
    with st.container():
        col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
        
        with col1:
            name = trace.get("root_span_name", "Unknown")
            st.markdown(f"**{status_emoji} {name}**")
            st.caption(f"Service: {trace.get('service_name', 'N/A')}")
        
        with col2:
            duration = trace.get("duration_ms", 0)
            st.metric("Duration", f"{duration:.0f}ms", label_visibility="collapsed")
        
        with col3:
            tokens = trace.get("total_tokens", 0)
            cost = trace.get("total_cost", 0)
            st.caption(f"🎯 {tokens:,} tokens")
            st.caption(f"💰 ${cost:.4f}")
        
        with col4:
            if st.button("View", key=f"view_{trace['trace_id']}"):
                st.session_state["selected_trace"] = trace["trace_id"]
                st.rerun()
        
        st.divider()


def render_traces_page():
    """Render traces list page."""
    st.title("🔍 Traces")
    
    client = get_client()
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        services = ["All"] + client.get_services()
        selected_service = st.selectbox("Service", services)
    
    with col2:
        status_filter = st.selectbox("Status", ["All", "ok", "error", "unset"])
    
    with col3:
        page_size = st.selectbox("Per page", [10, 20, 50, 100])
    
    # Get traces
    service = None if selected_service == "All" else selected_service
    status = None if status_filter == "All" else status_filter
    
    # Check if a trace is selected
    if "selected_trace" in st.session_state:
        render_trace_detail(st.session_state["selected_trace"])
        if st.button("← Back to list"):
            del st.session_state["selected_trace"]
            st.rerun()
        return
    
    # Pagination
    if "traces_page" not in st.session_state:
        st.session_state["traces_page"] = 1
    
    try:
        traces_data = client.list_traces(
            service=service,
            status=status,
            page=st.session_state["traces_page"],
            page_size=page_size,
        )
        
        traces = traces_data.get("traces", [])
        total = traces_data.get("total", 0)
        has_more = traces_data.get("has_more", False)
        
        st.caption(f"Showing {len(traces)} of {total} traces")
        
        if not traces:
            st.info("No traces found matching filters.")
        else:
            for trace in traces:
                render_trace_card(trace)
        
        # Pagination controls
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            if st.session_state["traces_page"] > 1:
                if st.button("← Previous"):
                    st.session_state["traces_page"] -= 1
                    st.rerun()
        
        with col2:
            st.caption(f"Page {st.session_state['traces_page']}")
        
        with col3:
            if has_more:
                if st.button("Next →"):
                    st.session_state["traces_page"] += 1
                    st.rerun()
    
    except Exception as e:
        st.error(f"Failed to load traces: {e}")


def render_trace_detail(trace_id: str):
    """Render detailed trace view."""
    client = get_client()
    
    try:
        trace = client.get_trace(trace_id)
        
        st.subheader(f"Trace: {trace_id[:8]}...")
        
        # Trace summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Duration", f"{trace.get('duration_ms', 0):.0f}ms")
        
        with col2:
            st.metric("Spans", len(trace.get("spans", [])))
        
        with col3:
            st.metric("Tokens", f"{trace.get('total_tokens', 0):,}")
        
        with col4:
            st.metric("Cost", f"${trace.get('total_cost', 0):.4f}")
        
        st.divider()
        
        # Spans visualization
        st.subheader("Spans")
        
        spans = trace.get("spans", [])
        
        # Build span tree
        span_map = {s["span_id"]: s for s in spans}
        root_spans = [s for s in spans if not s.get("parent_span_id")]
        
        def render_span(span: Dict[str, Any], indent: int = 0):
            """Recursively render span and children."""
            status_emoji = {"ok": "✅", "error": "❌", "unset": "⏳"}.get(span.get("status"), "❓")
            kind_emoji = {
                "agent": "🤖", "thought": "💭", "tool_call": "🔧",
                "llm": "🧠", "decision": "🔀", "action": "⚡", "retrieval": "📚"
            }.get(span.get("kind"), "📦")
            
            prefix = "│   " * indent
            
            with st.expander(
                f"{prefix}{kind_emoji} {span.get('name')} {status_emoji} ({span.get('duration_ms', 0):.0f}ms)",
                expanded=indent == 0
            ):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.caption(f"**Kind:** {span.get('kind')}")
                    st.caption(f"**Status:** {span.get('status')}")
                    st.caption(f"**Span ID:** {span.get('span_id')}")
                
                with col2:
                    st.caption(f"**Tokens:** {span.get('total_tokens', 0)}")
                    st.caption(f"**Tools:** {span.get('tool_calls_count', 0)}")
                    st.caption(f"**LLM Calls:** {span.get('llm_calls_count', 0)}")
                
                if span.get("input"):
                    st.markdown("**Input:**")
                    st.code(span["input"][:500] + ("..." if len(span.get("input", "")) > 500 else ""))
                
                if span.get("output"):
                    st.markdown("**Output:**")
                    st.code(span["output"][:500] + ("..." if len(span.get("output", "")) > 500 else ""))
                
                if span.get("reasoning"):
                    st.markdown("**Reasoning:**")
                    st.info(span["reasoning"][:500])
                
                if span.get("error_message"):
                    st.markdown("**Error:**")
                    st.error(span["error_message"])
            
            # Render children
            children = [s for s in spans if s.get("parent_span_id") == span["span_id"]]
            for child in sorted(children, key=lambda x: x.get("start_time", 0)):
                render_span(child, indent + 1)
        
        for root in sorted(root_spans, key=lambda x: x.get("start_time", 0)):
            render_span(root)
    
    except Exception as e:
        st.error(f"Failed to load trace: {e}")


def render_search_page():
    """Render search page."""
    st.title("🔎 Search")
    
    client = get_client()
    
    # Search input
    query = st.text_input(
        "Search spans",
        placeholder="Enter search query (e.g., 'error timeout', 'tool search')",
    )
    
    if query:
        try:
            results = client.search(query, limit=50)
            
            st.caption(f"Found {results.get('total', 0)} results")
            
            if not results.get("results"):
                st.info("No results found.")
            else:
                for result in results["results"]:
                    with st.container():
                        col1, col2, col3 = st.columns([3, 1, 1])
                        
                        with col1:
                            st.markdown(f"**{result.get('span_name', 'Unknown')}**")
                            # Display snippet with highlights
                            snippet = result.get("snippet", "").replace("<mark>", "**").replace("</mark>", "**")
                            st.markdown(snippet)
                        
                        with col2:
                            score = result.get("relevance_score", 0)
                            st.caption(f"Score: {score:.2f}")
                        
                        with col3:
                            if st.button("View", key=f"search_{result['span_id']}"):
                                st.session_state["selected_trace"] = result["trace_id"]
                                st.rerun()
                        
                        st.divider()
        
        except Exception as e:
            st.error(f"Search failed: {e}")
    else:
        st.info("Enter a search query to find relevant spans.")
        st.caption("Examples:")
        st.caption("- `error` - Find spans with errors")
        st.caption("- `tool search` - Find tool calls")
        st.caption("- `timeout` - Find timeout-related issues")


def render_analytics_page():
    """Render analytics page."""
    st.title("📈 Analytics")
    
    client = get_client()
    
    services = client.get_services()
    
    if not services:
        st.info("No services found. Start sending traces to see analytics.")
        return
    
    selected_service = st.selectbox("Select Service", services)
    
    if selected_service:
        try:
            stats = client.get_service_stats(selected_service)
            
            st.subheader(f"Statistics for {selected_service}")
            
            # Metrics row 1
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Traces", f"{stats.get('total_traces', 0):,}")
            
            with col2:
                st.metric("Total Spans", f"{stats.get('total_spans', 0):,}")
            
            with col3:
                error_rate = stats.get("error_rate", 0) * 100
                st.metric("Error Rate", f"{error_rate:.1f}%")
            
            with col4:
                st.metric("Avg Duration", f"{stats.get('avg_duration_ms', 0):.0f}ms")
            
            st.divider()
            
            # Metrics row 2
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Tokens", f"{stats.get('total_tokens', 0):,}")
            
            with col2:
                st.metric("Total Cost", f"${stats.get('total_cost', 0):.2f}")
            
            with col3:
                st.metric("P50 Duration", f"{stats.get('p50_duration_ms', 0):.0f}ms")
            
            with col4:
                st.metric("P95 Duration", f"{stats.get('p95_duration_ms', 0):.0f}ms")
            
            st.divider()
            
            # Duration percentiles chart
            st.subheader("Duration Percentiles")
            
            percentiles_data = pd.DataFrame({
                "Percentile": ["P50", "P95", "P99"],
                "Duration (ms)": [
                    stats.get("p50_duration_ms", 0),
                    stats.get("p95_duration_ms", 0),
                    stats.get("p99_duration_ms", 0),
                ]
            })
            
            st.bar_chart(percentiles_data.set_index("Percentile"))
        
        except Exception as e:
            st.error(f"Failed to load stats: {e}")


# ============================================================================
# Main
# ============================================================================

def main():
    """Main application entry point."""
    page = render_sidebar()
    
    if page == "📊 Dashboard":
        render_dashboard()
    elif page == "🔍 Traces":
        render_traces_page()
    elif page == "🔎 Search":
        render_search_page()
    elif page == "📈 Analytics":
        render_analytics_page()


if __name__ == "__main__":
    main()
