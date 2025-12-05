"""
Health Check Routes
Provides health status for load balancers and monitoring systems
"""

from datetime import datetime, timezone
from typing import Dict, Any

from fastapi import APIRouter, status, HTTPException
import structlog
import torch

from app.core.database import engine
from app.core.config import get_settings
from app.services.storage_service import StorageService
from app.worker.celery import celery_app

logger = structlog.get_logger(__name__)
router = APIRouter()

@router.get("/")
async def health_check():
    """
    Basic health check for load balancer
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": "ai-audio-upscaler-pro"
    }

@router.get("/ready")
async def readiness_check():
    """
    Comprehensive readiness check
    Tests all critical dependencies
    """
    checks = {
        "database": False,
        "storage": False,
        "celery": False,
        "gpu": False,
    }

    errors = []
    settings = get_settings()

    try:
        # Check database connection
        async with engine.begin() as conn:
            await conn.execute("SELECT 1")
        checks["database"] = True
        logger.debug("Database connection successful")
    except Exception as e:
        error_msg = f"Database connection failed: {str(e)}"
        errors.append(error_msg)
        logger.error(error_msg)

    try:
        # Check Azure Blob Storage
        storage_service = StorageService()
        await storage_service.health_check()
        checks["storage"] = True
        logger.debug("Storage connection successful")
    except Exception as e:
        error_msg = f"Storage connection failed: {str(e)}"
        errors.append(error_msg)
        logger.error(error_msg)

    try:
        # Check Celery/Redis connection
        celery_inspect = celery_app.control.inspect()
        stats = celery_inspect.stats()
        if stats:
            checks["celery"] = True
            logger.debug("Celery connection successful")
        else:
            errors.append("No Celery workers available")
    except Exception as e:
        error_msg = f"Celery connection failed: {str(e)}"
        errors.append(error_msg)
        logger.error(error_msg)

    try:
        # Check GPU availability (if configured)
        gpu_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if gpu_available else 0
        checks["gpu"] = gpu_available and gpu_count > 0

        if gpu_available:
            logger.debug(f"GPU check successful: {gpu_count} GPUs available")
        else:
            logger.debug("GPU not available, running in CPU mode")
    except Exception as e:
        error_msg = f"GPU check failed: {str(e)}"
        errors.append(error_msg)
        logger.error(error_msg)

    # Determine overall health
    critical_checks = ["database", "storage"]
    critical_healthy = all(checks[check] for check in critical_checks)

    response_data = {
        "status": "ready" if critical_healthy else "not_ready",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "errors": errors if errors else None,
        "version": "2.0.0",
        "environment": settings.ENVIRONMENT,
    }

    if not critical_healthy:
        logger.error("Readiness check failed", checks=checks, errors=errors)
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=response_data
        )

    return response_data

@router.get("/live")
async def liveness_check():
    """
    Basic liveness check - should always return success if app is running
    """
    return {
        "status": "alive",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@router.get("/metrics")
async def metrics_endpoint():
    """
    Application metrics for monitoring
    """
    try:
        # GPU metrics
        gpu_info = {}
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_info = {
                "gpu_count": gpu_count,
                "gpus": []
            }

            for i in range(gpu_count):
                gpu_props = torch.cuda.get_device_properties(i)
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
                memory_cached = torch.cuda.memory_reserved(i) / 1024**3  # GB
                total_memory = gpu_props.total_memory / 1024**3  # GB

                gpu_info["gpus"].append({
                    "device_id": i,
                    "name": gpu_props.name,
                    "memory_allocated_gb": round(memory_allocated, 2),
                    "memory_cached_gb": round(memory_cached, 2),
                    "memory_total_gb": round(total_memory, 2),
                    "memory_utilization": round((memory_allocated / total_memory) * 100, 2),
                    "compute_capability": f"{gpu_props.major}.{gpu_props.minor}",
                })

        # Celery worker stats
        celery_stats = {}
        try:
            celery_inspect = celery_app.control.inspect()
            active = celery_inspect.active()
            scheduled = celery_inspect.scheduled()

            if active:
                celery_stats["active_tasks"] = sum(len(tasks) for tasks in active.values())
                celery_stats["workers"] = list(active.keys())
            else:
                celery_stats["active_tasks"] = 0
                celery_stats["workers"] = []

            if scheduled:
                celery_stats["scheduled_tasks"] = sum(len(tasks) for tasks in scheduled.values())
            else:
                celery_stats["scheduled_tasks"] = 0

        except Exception as e:
            celery_stats["error"] = str(e)

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "gpu": gpu_info,
            "celery": celery_stats,
            "version": "2.0.0",
        }

    except Exception as e:
        logger.error("Failed to collect metrics", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to collect metrics"
        )