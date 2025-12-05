"""
Custom Middleware
Request logging, rate limiting, and security middleware
"""

import time
import uuid
from typing import Dict, Any
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse
from starlette.status import HTTP_429_TOO_MANY_REQUESTS
import structlog

from app.core.security import RateLimiter, get_client_ip, sanitize_log_data
from app.core.config import get_settings

logger = structlog.get_logger(__name__)

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for comprehensive request/response logging
    Includes performance metrics and security monitoring
    """

    async def dispatch(self, request: Request, call_next):
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        # Get client info
        client_ip = get_client_ip(request)
        request.state.client_ip = client_ip

        # Log request start
        start_time = time.time()

        # Sanitize headers for logging
        safe_headers = sanitize_log_data(dict(request.headers))

        logger.info(
            "Request started",
            request_id=request_id,
            method=request.method,
            url=str(request.url),
            client_ip=client_ip,
            user_agent=request.headers.get("user-agent", ""),
            headers=safe_headers,
        )

        try:
            # Process request
            response = await call_next(request)

            # Calculate processing time
            process_time = time.time() - start_time

            # Log response
            logger.info(
                "Request completed",
                request_id=request_id,
                method=request.method,
                url=str(request.url),
                status_code=response.status_code,
                process_time=process_time,
                client_ip=client_ip,
            )

            # Add custom headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = f"{process_time:.3f}"

            return response

        except Exception as exc:
            process_time = time.time() - start_time

            logger.error(
                "Request failed",
                request_id=request_id,
                method=request.method,
                url=str(request.url),
                error=str(exc),
                process_time=process_time,
                client_ip=client_ip,
                exc_info=True,
            )

            # Return error response
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "request_id": request_id,
                    "timestamp": time.time(),
                },
                headers={"X-Request-ID": request_id},
            )

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware with configurable limits per endpoint
    """

    def __init__(self, app):
        super().__init__(app)
        settings = get_settings()

        # Configure rate limiters for different endpoint types
        self.rate_limiters = {
            # Global rate limiting
            "global": RateLimiter(
                max_requests=settings.RATE_LIMIT_REQUESTS,
                window_seconds=settings.RATE_LIMIT_WINDOW_SECONDS
            ),
            # Stricter limits for resource-intensive endpoints
            "upload": RateLimiter(max_requests=10, window_seconds=3600),  # 10 uploads per hour
            "processing": RateLimiter(max_requests=5, window_seconds=3600),  # 5 processing jobs per hour
            "auth": RateLimiter(max_requests=20, window_seconds=900),  # 20 auth attempts per 15 min
        }

    def get_rate_limit_key(self, request: Request) -> str:
        """Generate rate limit key based on user or IP"""
        # Try to get user ID from request state (set by auth middleware)
        user_id = getattr(request.state, 'user_id', None)
        if user_id:
            return f"user:{user_id}"

        # Fallback to client IP
        client_ip = get_client_ip(request)
        return f"ip:{client_ip}"

    def get_endpoint_type(self, request: Request) -> str:
        """Determine endpoint type for specific rate limiting"""
        path = request.url.path.lower()

        if "/auth/" in path:
            return "auth"
        elif "/upload" in path:
            return "upload"
        elif "/process" in path:
            return "processing"
        else:
            return "global"

    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health checks and metrics
        if request.url.path in ["/health", "/health/", "/metrics", "/metrics/"]:
            return await call_next(request)

        # Get rate limiting parameters
        rate_limit_key = self.get_rate_limit_key(request)
        endpoint_type = self.get_endpoint_type(request)
        rate_limiter = self.rate_limiters.get(endpoint_type, self.rate_limiters["global"])

        # Check rate limit
        if not rate_limiter.is_allowed(rate_limit_key):
            remaining = rate_limiter.get_remaining(rate_limit_key)

            logger.warning(
                "Rate limit exceeded",
                key=rate_limit_key,
                endpoint_type=endpoint_type,
                path=request.url.path,
                remaining=remaining,
            )

            return JSONResponse(
                status_code=HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "limit": rate_limiter.max_requests,
                    "window_seconds": rate_limiter.window_seconds,
                    "remaining": remaining,
                    "retry_after": rate_limiter.window_seconds,
                },
                headers={
                    "X-RateLimit-Limit": str(rate_limiter.max_requests),
                    "X-RateLimit-Remaining": str(remaining),
                    "X-RateLimit-Window": str(rate_limiter.window_seconds),
                    "Retry-After": str(rate_limiter.window_seconds),
                },
            )

        # Process request
        response = await call_next(request)

        # Add rate limit headers to successful responses
        remaining = rate_limiter.get_remaining(rate_limit_key)
        response.headers["X-RateLimit-Limit"] = str(rate_limiter.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Window"] = str(rate_limiter.window_seconds)

        return response

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Security headers middleware
    Adds security headers to all responses
    """

    def __init__(self, app):
        super().__init__(app)
        self.security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: https:; "
                "font-src 'self' data:; "
                "connect-src 'self' https:; "
                "media-src 'self'; "
                "object-src 'none'; "
                "base-uri 'self'"
            ),
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Permissions-Policy": (
                "geolocation=(), "
                "microphone=(), "
                "camera=(), "
                "payment=(), "
                "usb=(), "
                "magnetometer=(), "
                "accelerometer=(), "
                "gyroscope=(), "
                "speaker=()"
            ),
        }

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        # Add security headers
        for header_name, header_value in self.security_headers.items():
            response.headers[header_name] = header_value

        return response

class MetricsMiddleware(BaseHTTPMiddleware):
    """
    Metrics collection middleware
    Collects request/response metrics for monitoring
    """

    def __init__(self, app):
        super().__init__(app)
        # Initialize metrics (would typically use Prometheus in production)
        self.request_count = {}
        self.request_duration = {}
        self.error_count = {}

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        method = request.method
        path = request.url.path

        try:
            response = await call_next(request)
            status_code = response.status_code

            # Record metrics
            duration = time.time() - start_time

            # Update counters (simplified - use proper metrics library in production)
            metric_key = f"{method}:{path}:{status_code}"
            self.request_count[metric_key] = self.request_count.get(metric_key, 0) + 1
            self.request_duration[metric_key] = duration

            if status_code >= 400:
                error_key = f"{method}:{path}:error"
                self.error_count[error_key] = self.error_count.get(error_key, 0) + 1

            return response

        except Exception as exc:
            duration = time.time() - start_time

            # Record error metrics
            error_key = f"{method}:{path}:exception"
            self.error_count[error_key] = self.error_count.get(error_key, 0) + 1

            raise exc

class CORSSecurityMiddleware(BaseHTTPMiddleware):
    """
    Enhanced CORS security middleware
    Validates origins and adds additional security measures
    """

    def __init__(self, app):
        super().__init__(app)
        settings = get_settings()
        self.allowed_origins = settings.CORS_ORIGINS
        self.allowed_methods = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
        self.allowed_headers = [
            "Accept",
            "Accept-Language",
            "Content-Language",
            "Content-Type",
            "Authorization",
            "X-Request-ID",
        ]

    def is_origin_allowed(self, origin: str) -> bool:
        """Check if origin is allowed"""
        if "*" in self.allowed_origins:
            return True

        # Exact match
        if origin in self.allowed_origins:
            return True

        # Pattern matching for subdomains (basic)
        for allowed_origin in self.allowed_origins:
            if allowed_origin.startswith("*."):
                domain = allowed_origin[2:]
                if origin.endswith(f".{domain}") or origin == domain:
                    return True

        return False

    async def dispatch(self, request: Request, call_next):
        origin = request.headers.get("Origin", "")

        # Handle preflight requests
        if request.method == "OPTIONS":
            if origin and not self.is_origin_allowed(origin):
                return JSONResponse(
                    status_code=403,
                    content={"error": "Origin not allowed"},
                )

            return Response(
                status_code=200,
                headers={
                    "Access-Control-Allow-Origin": origin if self.is_origin_allowed(origin) else "",
                    "Access-Control-Allow-Methods": ", ".join(self.allowed_methods),
                    "Access-Control-Allow-Headers": ", ".join(self.allowed_headers),
                    "Access-Control-Max-Age": "86400",  # 24 hours
                },
            )

        # Process regular requests
        response = await call_next(request)

        # Add CORS headers to response
        if origin and self.is_origin_allowed(origin):
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"

        return response