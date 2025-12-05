"""
Celery Configuration for AI Audio Upscaler Pro
Production-ready async task queue with Redis backend
"""

from celery import Celery
from kombu import Queue
import structlog

from app.core.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()

# Create Celery app
celery_app = Celery("ai_audio_upscaler")

# Celery Configuration
celery_app.conf.update(
    # Broker settings
    broker_url=settings.REDIS_URL,
    result_backend=settings.REDIS_URL,

    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,

    # Task routing and queues
    task_routes={
        "app.worker.tasks.process_audio_task": {"queue": "audio_processing"},
        "app.worker.tasks.cleanup_old_files_task": {"queue": "maintenance"},
        "app.worker.tasks.health_check_task": {"queue": "high_priority"},
    },

    # Define queues with different priorities
    task_queues=(
        Queue("high_priority", routing_key="high_priority"),
        Queue("audio_processing", routing_key="audio_processing"),
        Queue("low_priority", routing_key="low_priority"),
        Queue("maintenance", routing_key="maintenance"),
    ),
    task_default_queue="audio_processing",
    task_default_exchange="ai_audio_upscaler",
    task_default_routing_key="audio_processing",

    # Worker settings
    worker_prefetch_multiplier=1,  # Important for GPU tasks
    task_acks_late=True,
    worker_disable_rate_limits=False,

    # Task execution settings
    task_time_limit=3600,  # 1 hour hard limit
    task_soft_time_limit=3300,  # 55 minutes soft limit
    task_max_retries=3,
    task_default_retry_delay=60,  # 1 minute

    # Result backend settings
    result_expires=86400,  # 24 hours
    result_persistent=True,

    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,

    # Security
    broker_transport_options={
        "master_name": "mymaster",
        "visibility_timeout": 3600,
        "retry_policy": {
            "timeout": 5.0
        }
    } if settings.REDIS_SENTINEL_ENABLED else {},

    # Beat scheduler settings (for periodic tasks)
    beat_schedule={
        "cleanup-old-files": {
            "task": "app.worker.tasks.cleanup_old_files_task",
            "schedule": 3600.0,  # Run every hour
            "options": {"queue": "maintenance"}
        },
        "health-check": {
            "task": "app.worker.tasks.health_check_task",
            "schedule": 300.0,  # Run every 5 minutes
            "options": {"queue": "high_priority"}
        },
    },
)

# Import tasks to register them
from app.worker import tasks  # noqa

@celery_app.task(bind=True)
def debug_task(self):
    """Debug task for testing Celery setup"""
    logger.info(f"Request: {self.request!r}")
    return "Debug task completed"

# Task failure handler
@celery_app.task(bind=True, ignore_result=True)
def task_failure_handler(self, task_id, error, traceback):
    """Handle task failures"""
    logger.error(
        "Task failed",
        task_id=task_id,
        error=str(error),
        traceback=traceback
    )

# Configure task failure callback
celery_app.conf.task_annotations = {
    "*": {"on_failure": task_failure_handler}
}

if __name__ == "__main__":
    celery_app.start()