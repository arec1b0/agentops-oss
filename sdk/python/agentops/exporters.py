"""
AgentOps SDK - Exporters

Exporters for sending spans and traces to various backends.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class SpanExporter(ABC):
    """Abstract base class for span exporters."""
    
    @abstractmethod
    def export_spans(self, spans: List[Any]) -> bool:
        """Export a batch of spans. Returns True on success."""
        pass
    
    @abstractmethod
    def export_trace(self, trace: Any) -> bool:
        """Export a complete trace. Returns True on success."""
        pass
    
    def shutdown(self):
        """Cleanup resources."""
        pass


class HTTPExporter(SpanExporter):
    """
    Export spans and traces to HTTP collector endpoint.
    
    Endpoints:
    - POST /v1/ingest - Batch span ingestion
    - POST /v1/traces - Complete trace ingestion
    
    Authentication:
    - Set api_key to enable X-API-Key header authentication
    """
    
    def __init__(
        self,
        collector_url: str,
        timeout: float = 10.0,
        headers: Optional[Dict[str, str]] = None,
        api_key: Optional[str] = None,
        retry_count: int = 3,
        retry_delay: float = 1.0,
        service_name: str = "default",
        service_version: Optional[str] = None,
    ):
        self.collector_url = collector_url.rstrip("/")
        self.timeout = timeout
        self.headers = headers or {}
        self.retry_count = retry_count
        self.retry_delay = retry_delay
        self.service_name = service_name
        self.service_version = service_version
        
        # Set API key header (X-API-Key for AgentOps collector)
        if api_key:
            self.headers["X-API-Key"] = api_key
        
        self.headers.setdefault("Content-Type", "application/json")
        self.headers.setdefault("User-Agent", "agentops-sdk/0.1.0")
        
        # Lazy import to avoid startup cost
        self._session = None
        self._lock = threading.Lock()
    
    def _get_session(self):
        """Lazy initialization of HTTP session."""
        if self._session is None:
            with self._lock:
                if self._session is None:
                    try:
                        import httpx
                        self._session = httpx.Client(
                            timeout=self.timeout,
                            headers=self.headers,
                        )
                    except ImportError:
                        import urllib.request
                        self._session = "urllib"  # Fallback marker
        return self._session
    
    def _post(self, endpoint: str, data: Dict[str, Any]) -> bool:
        """Send POST request to collector with retry."""
        url = f"{self.collector_url}{endpoint}"
        payload = json.dumps(data, default=str)
        
        session = self._get_session()
        
        for attempt in range(self.retry_count):
            try:
                if session == "urllib":
                    # Fallback to urllib
                    import urllib.request
                    req = urllib.request.Request(
                        url,
                        data=payload.encode("utf-8"),
                        headers=self.headers,
                        method="POST",
                    )
                    with urllib.request.urlopen(req, timeout=self.timeout) as response:
                        return response.status in (200, 201, 202)
                else:
                    # Use httpx
                    response = session.post(url, content=payload)
                    
                    if response.status_code in (200, 201, 202):
                        return True
                    
                    if response.status_code == 429:
                        # Rate limited
                        retry_after = int(response.headers.get("Retry-After", "60"))
                        logger.warning(f"Rate limited, retry after {retry_after}s")
                        time.sleep(min(retry_after, 60))
                        continue
                    
                    if response.status_code in (401, 403):
                        logger.error(
                            f"Authentication failed: {response.status_code}. "
                            "Check api_key parameter or AGENTOPS_API_KEY env var."
                        )
                        return False  # Don't retry auth errors
                    
                    logger.warning(f"Collector returned {response.status_code}: {response.text}")
                    
            except Exception as e:
                logger.warning(f"Export attempt {attempt + 1} failed: {e}")
            
            # Exponential backoff
            if attempt < self.retry_count - 1:
                time.sleep(self.retry_delay * (2 ** attempt))
        
        logger.error(f"Failed to export to {url} after {self.retry_count} retries")
        return False
    
    def export_spans(self, spans: List[Any]) -> bool:
        """Export batch of spans."""
        data = {
            "spans": [s.to_dict() if hasattr(s, "to_dict") else s for s in spans],
            "service_name": self.service_name,
            "service_version": self.service_version,
        }
        return self._post("/v1/ingest", data)
    
    def export_trace(self, trace: Any) -> bool:
        """Export complete trace."""
        data = trace.to_dict() if hasattr(trace, "to_dict") else trace
        return self._post("/v1/traces", data)
    
    def shutdown(self):
        """Close HTTP session."""
        if self._session and self._session != "urllib":
            self._session.close()


class ConsoleExporter(SpanExporter):
    """
    Export spans to console for debugging.
    Useful during development.
    """
    
    def __init__(self, pretty: bool = True, colors: bool = True):
        self.pretty = pretty
        self.colors = colors
    
    def _format_span(self, span: Any) -> str:
        """Format span for console output."""
        data = span.to_dict() if hasattr(span, "to_dict") else span
        
        if self.colors:
            # ANSI color codes
            status_colors = {
                "ok": "\033[32m",      # Green
                "error": "\033[31m",   # Red
                "unset": "\033[33m",   # Yellow
            }
            reset = "\033[0m"
            bold = "\033[1m"
            dim = "\033[2m"
            
            status = data.get("status", "unset")
            color = status_colors.get(status, "")
            
            output = [
                f"{bold}[{data.get('kind', 'span').upper()}]{reset} {data.get('name', 'unnamed')}",
                f"  {dim}trace_id:{reset} {data.get('trace_id', 'N/A')[:8]}...",
                f"  {dim}span_id:{reset} {data.get('span_id', 'N/A')}",
                f"  {dim}status:{reset} {color}{status}{reset}",
                f"  {dim}duration:{reset} {data.get('duration_ms', 0):.2f}ms",
            ]
            
            if data.get("input"):
                output.append(f"  {dim}input:{reset} {str(data['input'])[:100]}...")
            if data.get("output"):
                output.append(f"  {dim}output:{reset} {str(data['output'])[:100]}...")
            if data.get("error_message"):
                output.append(f"  {dim}error:{reset} {color}{data['error_message']}{reset}")
            if data.get("tool_calls"):
                output.append(f"  {dim}tools:{reset} {len(data['tool_calls'])} calls")
            if data.get("llm_calls"):
                output.append(f"  {dim}llm_calls:{reset} {len(data['llm_calls'])} calls")
            
            return "\n".join(output)
        else:
            if self.pretty:
                return json.dumps(data, indent=2, default=str)
            return json.dumps(data, default=str)
    
    def export_spans(self, spans: List[Any]) -> bool:
        """Print spans to console."""
        print("\n" + "=" * 60)
        print(f"[AgentOps] Exporting {len(spans)} spans")
        print("=" * 60)
        
        for span in spans:
            print(self._format_span(span))
            print("-" * 40)
        
        return True
    
    def export_trace(self, trace: Any) -> bool:
        """Print trace summary to console."""
        data = trace.to_dict() if hasattr(trace, "to_dict") else trace
        
        print("\n" + "=" * 60)
        print("[AgentOps] TRACE COMPLETE")
        print("=" * 60)
        print(f"  trace_id: {data.get('trace_id', 'N/A')}")
        print(f"  service: {data.get('service_name', 'N/A')}")
        print(f"  duration: {data.get('total_duration_ms', 0):.2f}ms")
        print(f"  spans: {len(data.get('spans', []))}")
        print(f"  tokens: {data.get('total_tokens', 0)}")
        print(f"  cost: ${data.get('total_cost', 0):.4f}")
        print(f"  status: {data.get('status', 'unknown')}")
        print(f"  errors: {data.get('error_count', 0)}")
        print("=" * 60 + "\n")
        
        return True


class FileExporter(SpanExporter):
    """
    Export spans to JSONL file for offline analysis.
    """
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self._lock = threading.Lock()
    
    def export_spans(self, spans: List[Any]) -> bool:
        """Append spans to file."""
        try:
            with self._lock:
                with open(self.filepath, "a") as f:
                    for span in spans:
                        data = span.to_dict() if hasattr(span, "to_dict") else span
                        f.write(json.dumps(data, default=str) + "\n")
            return True
        except Exception as e:
            logger.error(f"Failed to write to file: {e}")
            return False
    
    def export_trace(self, trace: Any) -> bool:
        """Append trace to file."""
        try:
            with self._lock:
                with open(self.filepath, "a") as f:
                    data = trace.to_dict() if hasattr(trace, "to_dict") else trace
                    data["_type"] = "trace"
                    f.write(json.dumps(data, default=str) + "\n")
            return True
        except Exception as e:
            logger.error(f"Failed to write trace to file: {e}")
            return False


class CompositeExporter(SpanExporter):
    """
    Combine multiple exporters.
    """
    
    def __init__(self, exporters: List[SpanExporter]):
        self.exporters = exporters
    
    def export_spans(self, spans: List[Any]) -> bool:
        """Export to all configured exporters."""
        results = []
        for exporter in self.exporters:
            try:
                results.append(exporter.export_spans(spans))
            except Exception as e:
                logger.error(f"Exporter {type(exporter).__name__} failed: {e}")
                results.append(False)
        return all(results)
    
    def export_trace(self, trace: Any) -> bool:
        """Export trace to all exporters."""
        results = []
        for exporter in self.exporters:
            try:
                results.append(exporter.export_trace(trace))
            except Exception as e:
                logger.error(f"Exporter {type(exporter).__name__} failed: {e}")
                results.append(False)
        return all(results)
    
    def shutdown(self):
        """Shutdown all exporters."""
        for exporter in self.exporters:
            try:
                exporter.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down {type(exporter).__name__}: {e}")