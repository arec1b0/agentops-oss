"""
AgentOps Web UI - Streamlit Dashboard

Updated with API key authentication support.
"""

import os
from datetime import datetime, timedelta
from typing import Any, Optional, Dict

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# =============================================================================
# Configuration
# =============================================================================

COLLECTOR_URL = os.getenv("COLLECTOR_URL", "http://localhost:8000")
API_KEY = os.getenv("AGENTOPS_API_KEY", "")


def get_headers() -> Dict[str, str]:
    """Get request headers with API key if configured."""
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["X-API-Key"] = API_KEY
    return headers


def api_get(endpoint: str, params: Optional[dict] = None) -> Optional[dict]:
    """Make authenticated GET request to collector."""
    try:
        response = requests.get(
            f"{COLLECTOR_URL}{endpoint}",
            params=params,
            headers=get_headers(),
            timeout=30,
        )

        if response.status_code == 401:
            st.error("Authentication required. Set AGENTOPS_API_KEY environment variable.")
            return None

        if response.status_code == 403:
            st.error("Access denied. Check API key permissions.")
            return None

        if response.status_code == 429:
            st.warning("Rate limited. Please wait and try again.")
            return None

        response.raise_for_status()
        return response.json()

    except requests.exceptions.ConnectionError:
        st.error(f"Cannot connect to collector at {COLLECTOR_URL}")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="AgentOps",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# Sidebar
# =============================================================================

st.sidebar.title("🔍 AgentOps")
st.sidebar.markdown("---")

# Auth status indicator
if API_KEY:
    st.sidebar.success("🔐 Authenticated")
else:
    st.sidebar.warning("⚠️ No API key configured")

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Traces", "Search", "Settings"],
)

st.sidebar.markdown("---")
st.sidebar.caption(f"Collector: {COLLECTOR_URL}")


# =============================================================================
# Dashboard Page
# =============================================================================

if page == "Dashboard":
    st.title("📊 Dashboard")

    # Service selector
    col1, col2 = st.columns([2, 1])
    with col1:
        service_name = st.text_input("Service Name", value="default")
    with col2:
        hours = st.selectbox("Time Window", [1, 6, 12, 24, 48, 168], index=3)

    if st.button("Load Stats", type="primary"):
        stats = api_get("/v1/stats", {"service_name": service_name, "hours": hours})

        if stats:
            # Metrics row
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Traces", stats.get("trace_count", 0))
            with col2:
                st.metric(
                    "Error Rate",
                    f"{stats.get('error_rate', 0):.1%}",
                    delta=None,
                    delta_color="inverse",
                )
            with col3:
                st.metric("P95 Latency", f"{stats.get('p95_duration_ms', 0):.0f}ms")
            with col4:
                st.metric("Total Cost", f"${stats.get('total_cost', 0):.4f}")

            st.markdown("---")

            # Latency distribution
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Latency Percentiles")
                latency_data = {
                    "Percentile": ["P50", "P95", "P99"],
                    "Duration (ms)": [
                        stats.get("p50_duration_ms", 0),
                        stats.get("p95_duration_ms", 0),
                        stats.get("p99_duration_ms", 0),
                    ],
                }
                fig = px.bar(
                    latency_data,
                    x="Percentile",
                    y="Duration (ms)",
                    color="Percentile",
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("Summary")
                st.write(f"**Service:** {stats.get('service_name', '')}")
                st.write(f"**Traces:** {stats.get('trace_count', 0)}")
                st.write(f"**Errors:** {stats.get('error_count', 0)}")
                st.write(f"**Avg Duration:** {stats.get('avg_duration_ms', 0):.0f}ms")
                st.write(f"**Total Tokens:** {stats.get('total_tokens', 0):,}")


# =============================================================================
# Traces Page
# =============================================================================

elif page == "Traces":
    st.title("📋 Traces")

    col1, col2, col3 = st.columns(3)
    with col1:
        service_filter = st.text_input("Service", value="")
    with col2:
        status_filter = st.selectbox("Status", ["", "ok", "error"])
    with col3:
        limit = st.number_input("Limit", min_value=10, max_value=1000, value=100)

    if st.button("Load Traces", type="primary"):
        params = {"limit": limit}
        if service_filter:
            params["service_name"] = service_filter
        if status_filter:
            params["status"] = status_filter

        data = api_get("/v1/traces", params)

        if data and data.get("traces"):
            traces = data["traces"]

            # Display as table
            df = pd.DataFrame(traces)
            if not df.empty:
                display_cols = [
                    "trace_id", "service_name", "status",
                    "duration_ms", "total_tokens", "start_time"
                ]
                display_cols = [c for c in display_cols if c in df.columns]
                st.dataframe(df[display_cols], use_container_width=True)

                # Trace detail viewer
                st.markdown("---")
                selected_trace = st.selectbox(
                    "Select trace for details",
                    options=df["trace_id"].tolist(),
                )

                if selected_trace and st.button("View Trace"):
                    trace_data = api_get(f"/v1/traces/{selected_trace}")
                    if trace_data:
                        st.json(trace_data)
        else:
            st.info("No traces found")


# =============================================================================
# Search Page
# =============================================================================

elif page == "Search":
    st.title("🔎 Search")

    query = st.text_input("Search query", placeholder="Search in spans...")

    col1, col2 = st.columns(2)
    with col1:
        service_filter = st.text_input("Service (optional)", value="")
    with col2:
        limit = st.number_input("Max results", min_value=10, max_value=500, value=100)

    if query and st.button("Search", type="primary"):
        params = {"q": query, "limit": limit}
        if service_filter:
            params["service_name"] = service_filter

        data = api_get("/v1/search", params)

        if data and data.get("results"):
            results = data["results"]
            st.success(f"Found {len(results)} results")

            for result in results[:20]:  # Show first 20
                with st.expander(f"🔹 {result.get('name', 'Span')} - {result.get('trace_id', '')[:8]}"):
                    st.write(f"**Kind:** {result.get('kind', '')}")
                    st.write(f"**Status:** {result.get('status', '')}")
                    if result.get("input"):
                        st.write("**Input:**")
                        st.code(str(result["input"])[:500])
                    if result.get("output"):
                        st.write("**Output:**")
                        st.code(str(result["output"])[:500])
        else:
            st.info("No results found")


# =============================================================================
# Settings Page
# =============================================================================

elif page == "Settings":
    st.title("⚙️ Settings")

    st.subheader("Connection")
    st.write(f"**Collector URL:** `{COLLECTOR_URL}`")
    st.write(f"**API Key:** `{'*' * 8 + API_KEY[-4:] if API_KEY else 'Not configured'}`")

    st.markdown("---")

    st.subheader("Health Check")
    if st.button("Check Collector Health"):
        try:
            response = requests.get(
                f"{COLLECTOR_URL}/health",
                headers=get_headers(),
                timeout=5,
            )
            if response.status_code == 200:
                health = response.json()
                st.success(f"Status: {health.get('status', 'unknown')}")
                st.json(health)
            else:
                st.error(f"Health check failed: {response.status_code}")
        except Exception as e:
            st.error(f"Connection failed: {e}")

    st.markdown("---")

    st.subheader("Configuration")
    st.markdown("""
    **Environment Variables:**
```bash
    # Collector URL
    export COLLECTOR_URL=http://localhost:8000

    # API Key for authentication
    export AGENTOPS_API_KEY=sk-your-api-key
```
    """)