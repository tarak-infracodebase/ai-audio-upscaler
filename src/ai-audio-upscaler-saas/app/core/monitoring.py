"""
Monitoring and Observability
Metrics collection, health checks, and telemetry
"""

import time
import psutil
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from fastapi import FastAPI
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
from prometheus_client.exposition import CONTENT_TYPE_LATEST
from starlette.responses import Response
import structlog

logger = structlog.get_logger(__name__)

# Prometheus metrics registry
registry = CollectorRegistry()

# Application metrics
request_count = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status_code'],
    registry=registry
)

request_duration = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint'],
    registry=registry
)

audio_processing_duration = Histogram(
    'audio_processing_duration_seconds',
    'Audio processing duration in seconds',
    ['status', 'mode'],
    registry=registry
)

audio_files_processed = Counter(
    'audio_files_processed_total',
    'Total number of audio files processed',
    ['status', 'mode'],
    registry=registry
)

processing_errors = Counter(
    'processing_errors_total',
    'Total processing errors',
    ['error_type'],
    registry=registry
)

active_jobs = Gauge(
    'active_jobs_total',
    'Number of active processing jobs',
    registry=registry
)

queue_depth = Gauge(
    'queue_depth_total',
    'Number of jobs in processing queue',
    registry=registry
)

# System metrics
system_cpu_usage = Gauge(
    'system_cpu_usage_percent',
    'System CPU usage percentage',
    registry=registry
)

system_memory_usage = Gauge(
    'system_memory_usage_percent',
    'System memory usage percentage',
    registry=registry
)

system_disk_usage = Gauge(
    'system_disk_usage_percent',
    'System disk usage percentage',
    registry=registry
)

gpu_memory_usage = Gauge(
    'gpu_memory_usage_percent',
    'GPU memory usage percentage',
    ['device_id'],
    registry=registry
)

gpu_utilization = Gauge(
    'gpu_utilization_percent',
    'GPU utilization percentage',
    ['device_id'],
    registry=registry
)

# Database metrics
database_connections = Gauge(
    'database_connections_active',
    'Number of active database connections',
    registry=registry
)

database_query_duration = Histogram(
    'database_query_duration_seconds',
    'Database query duration in seconds',
    ['query_type'],
    registry=registry
)

# Cache metrics
cache_hits = Counter(
    'cache_hits_total',
    'Total cache hits',
    ['cache_type'],
    registry=registry
)

cache_misses = Counter(
    'cache_misses_total',
    'Total cache misses',
    ['cache_type'],
    registry=registry
)

class MetricsCollector:
    """
    Centralized metrics collection and monitoring
    """

    def __init__(self):
        self._collection_interval = 30  # seconds
        self._collection_task: Optional[asyncio.Task] = None
        self._running = False

    async def start_collection(self):
        """Start metrics collection background task"""
        if self._running:
            return

        self._running = True
        self._collection_task = asyncio.create_task(self._collect_metrics_loop())
        logger.info("Metrics collection started")

    async def stop_collection(self):
        """Stop metrics collection background task"""
        if not self._running:
            return

        self._running = False
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass

        logger.info("Metrics collection stopped")

    async def _collect_metrics_loop(self):
        """Background task for collecting system metrics"""
        while self._running:
            try:
                await self._collect_system_metrics()
                await self._collect_gpu_metrics()
                await asyncio.sleep(self._collection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Metrics collection failed", error=str(e))
                await asyncio.sleep(self._collection_interval)

    async def _collect_system_metrics(self):
        """Collect system resource metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            system_cpu_usage.set(cpu_percent)

            # Memory usage
            memory = psutil.virtual_memory()
            system_memory_usage.set(memory.percent)

            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            system_disk_usage.set(disk_percent)

        except Exception as e:
            logger.error("Failed to collect system metrics", error=str(e))

    async def _collect_gpu_metrics(self):
        """Collect GPU metrics if available"""
        try:
            import torch

            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    # Memory usage
                    memory_allocated = torch.cuda.memory_allocated(i)
                    memory_total = torch.cuda.get_device_properties(i).total_memory
                    memory_percent = (memory_allocated / memory_total) * 100

                    gpu_memory_usage.labels(device_id=str(i)).set(memory_percent)

                    # Note: GPU utilization requires nvidia-ml-py or similar
                    # For now, we'll set a placeholder
                    gpu_utilization.labels(device_id=str(i)).set(0)

        except ImportError:
            # PyTorch not available, skip GPU metrics
            pass
        except Exception as e:
            logger.error("Failed to collect GPU metrics", error=str(e))

    def record_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record HTTP request metrics"""
        request_count.labels(method=method, endpoint=endpoint, status_code=str(status_code)).inc()
        request_duration.labels(method=method, endpoint=endpoint).observe(duration)

    def record_audio_processing(self, duration: float, status: str, mode: str):
        """Record audio processing metrics"""
        audio_processing_duration.labels(status=status, mode=mode).observe(duration)
        audio_files_processed.labels(status=status, mode=mode).inc()

    def record_processing_error(self, error_type: str):
        """Record processing error"""
        processing_errors.labels(error_type=error_type).inc()

    def set_active_jobs(self, count: int):
        """Set current active jobs count"""
        active_jobs.set(count)

    def set_queue_depth(self, depth: int):
        """Set current queue depth"""
        queue_depth.set(depth)

    def record_database_query(self, query_type: str, duration: float):
        """Record database query metrics"""
        database_query_duration.labels(query_type=query_type).observe(duration)

    def record_cache_hit(self, cache_type: str):
        """Record cache hit"""
        cache_hits.labels(cache_type=cache_type).inc()

    def record_cache_miss(self, cache_type: str):
        """Record cache miss"""
        cache_misses.labels(cache_type=cache_type).inc()

# Global metrics collector instance
metrics = MetricsCollector()

class HealthChecker:
    """
    Application health checking
    """

    def __init__(self):
        self._checks = {}

    def add_check(self, name: str, check_func):
        """Add a health check function"""
        self._checks[name] = check_func

    async def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
        status = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "healthy",
            "checks": {},
            "system": await self._get_system_health(),
        }

        # Run all health checks
        for name, check_func in self._checks.items():
            try:
                check_result = await check_func() if asyncio.iscoroutinefunction(check_func) else check_func()
                status["checks"][name] = {
                    "status": "healthy" if check_result else "unhealthy",
                    "details": check_result
                }
            except Exception as e:
                status["checks"][name] = {
                    "status": "error",
                    "error": str(e)
                }
                status["status"] = "degraded"

        # Determine overall status
        failed_checks = [name for name, check in status["checks"].items() if check["status"] != "healthy"]
        if failed_checks:
            status["status"] = "unhealthy" if len(failed_checks) > len(self._checks) / 2 else "degraded"
            status["failed_checks"] = failed_checks

        return status

    async def _get_system_health(self) -> Dict[str, Any]:
        """Get system health information"""
        try:
            return {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": (psutil.disk_usage('/').used / psutil.disk_usage('/').total) * 100,
                "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None,
                "boot_time": psutil.boot_time(),
            }
        except Exception as e:
            logger.error("Failed to get system health", error=str(e))
            return {"error": str(e)}

# Global health checker instance
health_checker = HealthChecker()

def setup_monitoring(app: FastAPI):
    """
    Setup monitoring and observability for the application
    """

    @app.on_event("startup")
    async def start_monitoring():
        """Start monitoring services"""
        await metrics.start_collection()

        # Add default health checks
        from app.core.database import check_database_health
        health_checker.add_check("database", check_database_health)

        # Add Redis health check
        async def check_redis_health():
            try:
                from app.worker.celery import celery_app
                inspect = celery_app.control.inspect()
                stats = inspect.stats()
                return bool(stats)
            except Exception:
                return False

        health_checker.add_check("redis", check_redis_health)

        # Add storage health check
        async def check_storage_health():
            try:
                from app.services.storage_service import StorageService
                storage = StorageService()
                return await storage.health_check()
            except Exception:
                return False

        health_checker.add_check("storage", check_storage_health)

    @app.on_event("shutdown")
    async def stop_monitoring():
        """Stop monitoring services"""
        await metrics.stop_collection()

    # Metrics endpoint
    @app.get("/metrics", response_class=Response)
    async def get_metrics():
        """Prometheus metrics endpoint"""
        return Response(
            content=generate_latest(registry),
            media_type=CONTENT_TYPE_LATEST
        )

    # Health check endpoints
    @app.get("/health/status")
    async def health_status():
        """Detailed health status"""
        return await health_checker.get_health_status()

    logger.info("Monitoring and observability setup completed")