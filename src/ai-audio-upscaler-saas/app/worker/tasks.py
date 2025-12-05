"""
Celery Tasks for AI Audio Upscaler Pro
Async audio processing and maintenance tasks
"""

import os
import tempfile
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import traceback

from celery import current_task
import structlog
import torch

from app.worker.celery import celery_app
from app.services.audio_processor import AudioProcessorService
from app.services.storage_service import StorageService
from app.services.job_service import JobService
from app.models.job import JobStatus
from app.core.monitoring import metrics

logger = structlog.get_logger(__name__)

@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def process_audio_task(
    self,
    job_id: str,
    input_blob_path: str,
    processing_params: Dict[str, Any],
    user_id: str,
) -> Dict[str, Any]:
    """
    Main audio processing task
    Handles the complete audio upscaling workflow
    """
    logger.info(
        "Starting audio processing task",
        job_id=job_id,
        user_id=user_id,
        task_id=self.request.id,
        processing_params=processing_params
    )

    job_service = None
    storage_service = None
    processor = None
    temp_input_path = None
    temp_output_path = None

    try:
        # Initialize services
        job_service = JobService()
        storage_service = StorageService()
        processor = AudioProcessorService()

        # Update job status to processing
        await job_service.update_job_status(
            job_id,
            JobStatus.PROCESSING,
            task_id=self.request.id
        )

        # Progress callback function
        def progress_callback(progress: float, message: str):
            """Report progress back to the job tracking system"""
            logger.debug(
                "Processing progress update",
                job_id=job_id,
                progress=progress,
                message=message
            )

            # Update task progress
            current_task.update_state(
                state='PROGRESS',
                meta={
                    'progress': progress,
                    'message': message,
                    'job_id': job_id
                }
            )

        # Download input file to temporary location
        progress_callback(0.1, "Downloading input file...")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_input:
            temp_input_path = temp_input.name

        await storage_service.download_file_to_path(input_blob_path, temp_input_path)

        logger.info(
            "Input file downloaded",
            job_id=job_id,
            temp_path=temp_input_path,
            file_size=os.path.getsize(temp_input_path)
        )

        # Process audio
        progress_callback(0.2, "Initializing audio processor...")

        # Ensure GPU resources are available
        gpu_available = torch.cuda.is_available()
        device = "cuda" if gpu_available and processing_params.get("use_ai", False) else "cpu"

        logger.info(
            "Processing configuration",
            job_id=job_id,
            device=device,
            gpu_available=gpu_available,
            use_ai=processing_params.get("use_ai", False)
        )

        # Create temporary output file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_output:
            temp_output_path = temp_output.name

        # Start processing
        progress_callback(0.3, "Starting audio processing...")

        result = await processor.process_audio(
            input_path=temp_input_path,
            output_path=temp_output_path,
            parameters=processing_params,
            progress_callback=progress_callback,
            device=device,
        )

        # Upload result to storage
        progress_callback(0.9, "Uploading processed file...")

        output_filename = f"upscaled_{job_id}.wav"
        output_blob_path = f"outputs/{user_id}/{job_id}/{output_filename}"

        await storage_service.upload_file_from_path(temp_output_path, output_blob_path)

        # Update job with results
        await job_service.update_job_completion(
            job_id=job_id,
            output_blob_path=output_blob_path,
            output_filename=output_filename,
            processing_stats=result.get("stats", {}),
            analysis_data=result.get("analysis", None)
        )

        progress_callback(1.0, "Processing completed successfully")

        # Update metrics
        metrics.jobs_completed.inc()
        processing_duration = result.get("stats", {}).get("duration_seconds", 0)
        if processing_duration > 0:
            metrics.processing_duration.observe(processing_duration)

        logger.info(
            "Audio processing completed successfully",
            job_id=job_id,
            output_blob_path=output_blob_path,
            processing_duration=processing_duration
        )

        return {
            "status": "success",
            "job_id": job_id,
            "output_blob_path": output_blob_path,
            "processing_stats": result.get("stats", {}),
        }

    except Exception as exc:
        error_msg = str(exc)
        logger.error(
            "Audio processing task failed",
            job_id=job_id,
            error=error_msg,
            traceback=traceback.format_exc(),
            exc_info=True
        )

        # Update job status to failed
        if job_service:
            try:
                await job_service.update_job_failure(
                    job_id=job_id,
                    error_message=error_msg
                )
            except Exception as update_error:
                logger.error(
                    "Failed to update job failure status",
                    job_id=job_id,
                    error=str(update_error)
                )

        # Update metrics
        metrics.jobs_failed.inc()

        # Retry logic
        if self.request.retries < self.max_retries:
            retry_delay = min(300, (2 ** self.request.retries) * 60)  # Exponential backoff, max 5 min
            logger.info(
                "Retrying audio processing task",
                job_id=job_id,
                retry_count=self.request.retries + 1,
                retry_delay=retry_delay
            )

            raise self.retry(countdown=retry_delay, exc=exc)

        # Final failure after all retries
        raise exc

    finally:
        # Cleanup temporary files
        if temp_input_path and os.path.exists(temp_input_path):
            try:
                os.unlink(temp_input_path)
                logger.debug("Cleaned up temporary input file", path=temp_input_path)
            except Exception as e:
                logger.warning("Failed to cleanup temporary input file", path=temp_input_path, error=str(e))

        if temp_output_path and os.path.exists(temp_output_path):
            try:
                os.unlink(temp_output_path)
                logger.debug("Cleaned up temporary output file", path=temp_output_path)
            except Exception as e:
                logger.warning("Failed to cleanup temporary output file", path=temp_output_path, error=str(e))

        # Clear GPU memory if used
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("Cleared GPU memory cache")

@celery_app.task(bind=True)
def cleanup_old_files_task(self) -> Dict[str, Any]:
    """
    Periodic task to cleanup old temporary files and expired jobs
    """
    logger.info("Starting cleanup task")

    try:
        storage_service = StorageService()
        job_service = JobService()

        # Cleanup expired jobs (older than 7 days)
        cutoff_date = datetime.utcnow() - timedelta(days=7)
        expired_jobs = await job_service.get_expired_jobs(cutoff_date)

        cleaned_files = 0
        cleaned_jobs = 0

        for job in expired_jobs:
            try:
                # Delete associated files
                if job.input_blob_path:
                    await storage_service.delete_file(job.input_blob_path)
                    cleaned_files += 1

                if job.output_blob_path:
                    await storage_service.delete_file(job.output_blob_path)
                    cleaned_files += 1

                # Delete job record
                await job_service.delete_job(job.id)
                cleaned_jobs += 1

            except Exception as e:
                logger.warning(
                    "Failed to cleanup job",
                    job_id=job.id,
                    error=str(e)
                )

        logger.info(
            "Cleanup task completed",
            cleaned_jobs=cleaned_jobs,
            cleaned_files=cleaned_files
        )

        return {
            "status": "success",
            "cleaned_jobs": cleaned_jobs,
            "cleaned_files": cleaned_files,
        }

    except Exception as exc:
        logger.error(
            "Cleanup task failed",
            error=str(exc),
            exc_info=True
        )
        raise exc

@celery_app.task(bind=True)
def health_check_task(self) -> Dict[str, Any]:
    """
    Periodic health check task for monitoring worker status
    """
    try:
        # Check GPU status
        gpu_status = {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }

        if gpu_status["available"]:
            gpu_status["memory_info"] = []
            for i in range(gpu_status["device_count"]):
                allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
                cached = torch.cuda.memory_reserved(i) / 1024**3     # GB
                gpu_status["memory_info"].append({
                    "device_id": i,
                    "allocated_gb": round(allocated, 2),
                    "cached_gb": round(cached, 2),
                })

        # Check worker status
        worker_status = {
            "task_id": self.request.id,
            "timestamp": datetime.utcnow().isoformat(),
            "gpu": gpu_status,
        }

        logger.debug("Health check completed", status=worker_status)
        return worker_status

    except Exception as exc:
        logger.error(
            "Health check task failed",
            error=str(exc),
            exc_info=True
        )
        raise exc

@celery_app.task(bind=True, max_retries=2)
def batch_process_audio_task(
    self,
    job_ids: list,
    batch_id: str,
    user_id: str,
) -> Dict[str, Any]:
    """
    Batch processing task for multiple audio files
    Optimizes GPU usage by processing multiple files together
    """
    logger.info(
        "Starting batch processing task",
        batch_id=batch_id,
        job_count=len(job_ids),
        user_id=user_id
    )

    results = []
    failed_jobs = []

    try:
        for i, job_id in enumerate(job_ids):
            try:
                progress = (i + 1) / len(job_ids)
                current_task.update_state(
                    state='PROGRESS',
                    meta={
                        'progress': progress,
                        'message': f'Processing job {i+1}/{len(job_ids)}',
                        'batch_id': batch_id,
                        'current_job_id': job_id
                    }
                )

                # Process individual job (this would call the main processing logic)
                # For now, we'll delegate to the individual task
                result = await process_audio_task.apply_async(args=[job_id])
                results.append(result)

            except Exception as e:
                logger.error(
                    "Failed to process job in batch",
                    batch_id=batch_id,
                    job_id=job_id,
                    error=str(e)
                )
                failed_jobs.append(job_id)

        logger.info(
            "Batch processing completed",
            batch_id=batch_id,
            successful=len(results),
            failed=len(failed_jobs)
        )

        return {
            "status": "success" if not failed_jobs else "partial_success",
            "batch_id": batch_id,
            "processed_count": len(results),
            "failed_count": len(failed_jobs),
            "failed_jobs": failed_jobs,
        }

    except Exception as exc:
        logger.error(
            "Batch processing task failed",
            batch_id=batch_id,
            error=str(exc),
            exc_info=True
        )
        raise exc