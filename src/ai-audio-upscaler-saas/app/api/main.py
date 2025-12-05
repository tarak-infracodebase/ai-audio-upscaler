"""
FastAPI Main Application - AI Audio Upscaler Pro SaaS
Production-ready API with authentication, monitoring, and async processing
"""

from contextlib import asynccontextmanager
from typing import Dict, Any
import logging
import time

from fastapi import FastAPI, HTTPException, Depends, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import structlog

from app.api.routes import audio, auth, jobs, health
from app.core.config import get_settings
from app.core.security import get_current_user
from app.core.database import engine, create_tables
from app.core.monitoring import setup_monitoring, metrics
from app.api.middleware import RequestLoggingMiddleware, RateLimitMiddleware

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    logger.info("Starting AI Audio Upscaler Pro API")

    # Startup
    settings = get_settings()

    # Initialize database
    await create_tables()
    logger.info("Database tables created/verified")

    # Setup monitoring and telemetry
    setup_monitoring(app)
    logger.info("Monitoring and telemetry configured")

    # Warm up ML models if needed
    try:
        from app.services.audio_processor import AudioProcessorService
        processor = AudioProcessorService()
        await processor.warm_up()
        logger.info("ML models warmed up successfully")
    except Exception as e:
        logger.warning("Model warm-up failed", error=str(e))

    yield

    # Shutdown
    logger.info("Shutting down AI Audio Upscaler Pro API")
    await engine.dispose()

def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    settings = get_settings()

    app = FastAPI(
        title="AI Audio Upscaler Pro",
        description="Production-ready AI-powered audio upscaling SaaS",
        version="2.0.0",
        lifespan=lifespan,
        docs_url="/docs" if settings.ENVIRONMENT != "production" else None,
        redoc_url="/redoc" if settings.ENVIRONMENT != "production" else None,
        openapi_url="/openapi.json" if settings.ENVIRONMENT != "production" else None,
    )

    # Security middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.ALLOWED_HOSTS
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )

    # Custom middleware
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(RateLimitMiddleware)

    # Include routers
    app.include_router(health.router, prefix="/health", tags=["health"])
    app.include_router(auth.router, prefix="/auth", tags=["authentication"])
    app.include_router(
        audio.router,
        prefix="/api/v1/audio",
        tags=["audio"],
        dependencies=[Depends(get_current_user)]
    )
    app.include_router(
        jobs.router,
        prefix="/api/v1/jobs",
        tags=["jobs"],
        dependencies=[Depends(get_current_user)]
    )

    # Global exception handlers
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        logger.error(
            "HTTP exception occurred",
            status_code=exc.status_code,
            detail=exc.detail,
            path=request.url.path,
            method=request.method,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail,
                "status_code": exc.status_code,
                "timestamp": time.time(),
            },
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(
            "Unhandled exception occurred",
            error=str(exc),
            path=request.url.path,
            method=request.method,
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "status_code": 500,
                "timestamp": time.time(),
            },
        )

    # Root endpoint
    @app.get("/")
    async def root():
        return {
            "service": "AI Audio Upscaler Pro",
            "version": "2.0.0",
            "status": "operational",
            "timestamp": time.time(),
        }

    return app

# Create the app instance
app = create_app()

if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "app.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.ENVIRONMENT == "development",
        log_config=None,  # Use our structured logging
    )