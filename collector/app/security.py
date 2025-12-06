"""
Security module: API key authentication, rate limiting, CORS configuration.

Production-ready security layer for AgentOps Collector.
"""

import hashlib
import hmac
import os
import time
from dataclasses import dataclass
from functools import wraps
from typing import Optional

from fastapi import HTTPException, Request, status
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


# =============================================================================
# API Key Authentication
# =============================================================================

@dataclass
class APIKey:
    """API key metadata."""
    key_hash: str
    name: str
    created_at: float
    rate_limit: int  # requests per minute, 0 = unlimited
    scopes: set[str]  # "ingest", "read", "admin"


class APIKeyManager:
    """
    API key management with secure hash comparison.
    
    Keys are stored as SHA-256 hashes for security.
    Supports multiple keys with different scopes and rate limits.
    """
    
    def __init__(self):
        self._keys: dict[str, APIKey] = {}  # key_hash -> APIKey
        self._load_keys_from_env()
    
    def _hash_key(self, key: str) -> str:
        """SHA-256 hash of API key."""
        return hashlib.sha256(key.encode()).hexdigest()
    
    def _load_keys_from_env(self):
        """
        Load API keys from environment variables.
        
        Format: AGENTOPS_API_KEY_<NAME>=<key>:<scopes>:<rate_limit>
        Example: AGENTOPS_API_KEY_PROD=sk-abc123:ingest,read:1000
        
        Special keys:
        - AGENTOPS_ADMIN_KEY: full access admin key
        - AGENTOPS_INGEST_KEY: simple ingest-only key (legacy)
        """
        # Admin key (full access)
        admin_key = os.getenv("AGENTOPS_ADMIN_KEY")
        if admin_key:
            self.add_key(
                key=admin_key,
                name="admin",
                scopes={"ingest", "read", "admin"},
                rate_limit=0,  # unlimited
            )
        
        # Simple ingest key (backward compatibility)
        ingest_key = os.getenv("AGENTOPS_INGEST_KEY")
        if ingest_key:
            self.add_key(
                key=ingest_key,
                name="ingest",
                scopes={"ingest"},
                rate_limit=int(os.getenv("AGENTOPS_INGEST_RATE_LIMIT", "1000")),
            )
        
        # Named keys: AGENTOPS_API_KEY_<NAME>=<key>:<scopes>:<rate_limit>
        for env_name, env_value in os.environ.items():
            if env_name.startswith("AGENTOPS_API_KEY_"):
                name = env_name.replace("AGENTOPS_API_KEY_", "").lower()
                parts = env_value.split(":")
                if len(parts) >= 1:
                    key = parts[0]
                    scopes = set(parts[1].split(",")) if len(parts) > 1 else {"ingest"}
                    rate_limit = int(parts[2]) if len(parts) > 2 else 1000
                    self.add_key(key=key, name=name, scopes=scopes, rate_limit=rate_limit)
    
    def add_key(
        self,
        key: str,
        name: str,
        scopes: set[str],
        rate_limit: int = 1000,
    ):
        """Add API key to manager."""
        key_hash = self._hash_key(key)
        self._keys[key_hash] = APIKey(
            key_hash=key_hash,
            name=name,
            created_at=time.time(),
            rate_limit=rate_limit,
            scopes=scopes,
        )
    
    def validate_key(self, key: str, required_scope: str = "ingest") -> Optional[APIKey]:
        """
        Validate API key and check scope.
        
        Returns APIKey if valid, None otherwise.
        Uses constant-time comparison to prevent timing attacks.
        """
        if not key:
            return None
        
        key_hash = self._hash_key(key)
        
        # Constant-time lookup (prevent timing attacks)
        api_key = None
        for stored_hash, stored_key in self._keys.items():
            if hmac.compare_digest(stored_hash, key_hash):
                api_key = stored_key
                break
        
        if api_key is None:
            return None
        
        # Check scope
        if required_scope not in api_key.scopes and "admin" not in api_key.scopes:
            return None
        
        return api_key
    
    def has_keys(self) -> bool:
        """Check if any API keys are configured."""
        return len(self._keys) > 0


# Global API key manager
api_key_manager = APIKeyManager()

# FastAPI security scheme
api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
    description="API key for authentication",
)


async def get_api_key(
    request: Request,
    api_key: Optional[str] = None,
) -> Optional[APIKey]:
    """
    Extract and validate API key from request.
    
    Checks in order:
    1. X-API-Key header
    2. Authorization: Bearer <key>
    3. ?api_key= query parameter (not recommended, for debugging only)
    """
    # Try X-API-Key header first
    key = request.headers.get("X-API-Key")
    
    # Try Authorization: Bearer
    if not key:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            key = auth_header[7:]
    
    # Try query parameter (last resort)
    if not key:
        key = request.query_params.get("api_key")
    
    return key


def require_auth(scope: str = "ingest"):
    """
    Dependency for requiring API key authentication.
    
    Usage:
        @app.post("/ingest")
        async def ingest(api_key: APIKey = Depends(require_auth("ingest"))):
            ...
    """
    async def dependency(request: Request) -> APIKey:
        # Skip auth if no keys configured (development mode)
        if not api_key_manager.has_keys():
            return APIKey(
                key_hash="dev",
                name="development",
                created_at=time.time(),
                rate_limit=0,
                scopes={"ingest", "read", "admin"},
            )
        
        key = await get_api_key(request)
        if not key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing API key. Provide X-API-Key header.",
                headers={"WWW-Authenticate": "ApiKey"},
            )
        
        api_key = api_key_manager.validate_key(key, scope)
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Invalid API key or insufficient permissions for scope: {scope}",
            )
        
        # Store for rate limiting
        request.state.api_key = api_key
        return api_key
    
    return dependency


# =============================================================================
# Rate Limiting
# =============================================================================

class RateLimitState:
    """In-memory rate limit tracking per API key."""
    
    def __init__(self):
        # key_hash -> (window_start, request_count)
        self._windows: dict[str, tuple[float, int]] = {}
        self._window_size = 60  # 1 minute window
    
    def check_and_increment(self, key_hash: str, limit: int) -> tuple[bool, int, int]:
        """
        Check rate limit and increment counter.
        
        Returns: (allowed, remaining, reset_seconds)
        """
        if limit <= 0:  # unlimited
            return True, -1, 0
        
        now = time.time()
        window_start = now - (now % self._window_size)
        
        if key_hash in self._windows:
            stored_window, count = self._windows[key_hash]
            if stored_window == window_start:
                # Same window, check limit
                if count >= limit:
                    reset = int(self._window_size - (now % self._window_size))
                    return False, 0, reset
                self._windows[key_hash] = (window_start, count + 1)
                return True, limit - count - 1, int(self._window_size - (now % self._window_size))
        
        # New window
        self._windows[key_hash] = (window_start, 1)
        return True, limit - 1, int(self._window_size - (now % self._window_size))
    
    def cleanup(self):
        """Remove expired windows (call periodically)."""
        now = time.time()
        window_start = now - (now % self._window_size)
        expired = [k for k, (w, _) in self._windows.items() if w < window_start]
        for k in expired:
            del self._windows[k]


# Global rate limiter
rate_limiter = RateLimitState()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware using sliding window.
    
    Limits are per API key. Anonymous requests use IP-based limiting.
    """
    
    def __init__(self, app, default_limit: int = 100):
        super().__init__(app)
        self.default_limit = default_limit
        self._ip_limits: dict[str, tuple[float, int]] = {}
    
    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health checks
        if request.url.path in ("/health", "/healthz", "/ready"):
            return await call_next(request)
        
        # Get rate limit key and limit
        api_key: Optional[APIKey] = getattr(request.state, "api_key", None)
        
        if api_key:
            key_hash = api_key.key_hash
            limit = api_key.rate_limit
        else:
            # IP-based rate limiting for unauthenticated requests
            client_ip = request.client.host if request.client else "unknown"
            key_hash = f"ip:{client_ip}"
            limit = self.default_limit
        
        # Check rate limit
        allowed, remaining, reset = rate_limiter.check_and_increment(key_hash, limit)
        
        if not allowed:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "rate_limit_exceeded",
                    "detail": f"Rate limit exceeded. Retry after {reset} seconds.",
                    "retry_after": reset,
                },
                headers={
                    "Retry-After": str(reset),
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(time.time()) + reset),
                },
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        if limit > 0:
            response.headers["X-RateLimit-Limit"] = str(limit)
            response.headers["X-RateLimit-Remaining"] = str(max(0, remaining))
            response.headers["X-RateLimit-Reset"] = str(int(time.time()) + reset)
        
        return response


# =============================================================================
# CORS Configuration
# =============================================================================

def get_cors_origins() -> list[str]:
    """
    Get allowed CORS origins from environment.
    
    AGENTOPS_CORS_ORIGINS: comma-separated list of allowed origins
    Default: localhost only (secure default)
    """
    origins_env = os.getenv("AGENTOPS_CORS_ORIGINS", "")
    
    if origins_env == "*":
        # Explicit wildcard (not recommended for production)
        return ["*"]
    
    if origins_env:
        return [o.strip() for o in origins_env.split(",") if o.strip()]
    
    # Secure defaults: localhost only
    return [
        "http://localhost:3000",
        "http://localhost:8501",  # Streamlit UI
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8501",
    ]


def get_cors_config() -> dict:
    """Get CORS middleware configuration."""
    origins = get_cors_origins()
    
    return {
        "allow_origins": origins,
        "allow_credentials": True,
        "allow_methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": [
            "Authorization",
            "X-API-Key",
            "Content-Type",
            "X-Request-ID",
            "X-Trace-ID",
        ],
        "expose_headers": [
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "X-RateLimit-Reset",
            "X-Request-ID",
        ],
        "max_age": 600,  # Cache preflight for 10 minutes
    }


# =============================================================================
# Request ID Middleware
# =============================================================================

class RequestIDMiddleware(BaseHTTPMiddleware):
    """Add unique request ID to each request for tracing."""
    
    async def dispatch(self, request: Request, call_next):
        import uuid
        
        # Use provided ID or generate new one
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        
        return response