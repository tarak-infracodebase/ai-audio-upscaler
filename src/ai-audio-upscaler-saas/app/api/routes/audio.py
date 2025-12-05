"""
Audio Processing API Routes
Handles file upload, processing requests, and result retrieval
"""

from typing import Optional, List
import uuid
from datetime import datetime, timezone

from fastapi import (
    APIRouter, Depends, HTTPException, UploadFile, File, Form,
    BackgroundTasks, Query, status
)
from fastapi.responses import StreamingResponse
import structlog

from app.core.security import get_current_user
from app.models.user import User
from app.models.job import JobCreate, JobResponse, JobStatus, ProcessingParameters
from app.services.audio_processor import AudioProcessorService
from app.services.storage_service import StorageService
from app.services.job_service import JobService
from app.worker.tasks import process_audio_task
from app.core.monitoring import metrics

logger = structlog.get_logger(__name__)
router = APIRouter()

@router.post("/upload", response_model=JobResponse)
async def upload_audio(
    file: UploadFile = File(...),
    target_sample_rate: int = Form(48000),
    mode: str = Form("baseline"),
    baseline_method: str = Form("sinc"),
    use_ai: bool = Form(False),
    normalization_mode: str = Form("Peak -1dB"),
    generate_analysis: bool = Form(False),
    # Advanced parameters
    tta: bool = Form(False),
    stereo_mode: str = Form("lr"),
    transient_strength: float = Form(0.0),
    spectral_matching: bool = Form(False),
    qc: bool = Form(False),
    candidate_count: int = Form(8),
    judge_threshold: float = Form(0.5),
    denoising_strength: float = Form(0.6),
    current_user: User = Depends(get_current_user),
    storage_service: StorageService = Depends(),
    job_service: JobService = Depends(),
):
    """
    Upload audio file and start processing job
    """
    logger.info(
        "Audio upload request received",
        user_id=current_user.id,
        filename=file.filename,
        content_type=file.content_type,
        file_size=file.size if hasattr(file, 'size') else 'unknown'
    )

    # Validate file
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file provided"
        )

    # Validate file type
    allowed_types = [
        'audio/wav', 'audio/mpeg', 'audio/flac', 'audio/mp4',
        'audio/x-wav', 'audio/x-flac', 'application/octet-stream'
    ]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {file.content_type}"
        )

    # Validate file size (max 500MB)
    max_size = 500 * 1024 * 1024  # 500MB
    if hasattr(file, 'size') and file.size > max_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File too large. Maximum size: {max_size / (1024*1024):.0f}MB"
        )

    try:
        # Generate job ID
        job_id = str(uuid.uuid4())

        # Upload file to storage
        input_blob_path = await storage_service.upload_file(
            file, f"inputs/{current_user.id}/{job_id}/{file.filename}"
        )

        # Create processing parameters
        processing_params = ProcessingParameters(
            target_sample_rate=target_sample_rate,
            mode=mode,
            baseline_method=baseline_method,
            use_ai=use_ai,
            normalization_mode=normalization_mode,
            generate_analysis=generate_analysis,
            tta=tta,
            stereo_mode=stereo_mode,
            transient_strength=transient_strength,
            spectral_matching=spectral_matching,
            qc=qc,
            candidate_count=candidate_count,
            judge_threshold=judge_threshold,
            denoising_strength=denoising_strength,
        )

        # Create job record
        job_create = JobCreate(
            id=job_id,
            user_id=current_user.id,
            input_filename=file.filename,
            input_blob_path=input_blob_path,
            processing_parameters=processing_params,
        )

        job = await job_service.create_job(job_create)

        # Queue processing task
        task = process_audio_task.delay(
            job_id=job_id,
            input_blob_path=input_blob_path,
            processing_params=processing_params.dict(),
            user_id=current_user.id,
        )

        # Update job with task ID
        await job_service.update_job_task_id(job_id, task.id)

        # Update metrics
        metrics.jobs_created.inc()
        metrics.files_uploaded.inc()

        logger.info(
            "Audio processing job created",
            job_id=job_id,
            task_id=task.id,
            user_id=current_user.id,
            filename=file.filename,
        )

        return JobResponse.from_orm(job)

    except Exception as e:
        logger.error(
            "Failed to create audio processing job",
            error=str(e),
            user_id=current_user.id,
            filename=file.filename,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start audio processing"
        )

@router.get("/jobs", response_model=List[JobResponse])
async def list_jobs(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    status_filter: Optional[JobStatus] = Query(None),
    current_user: User = Depends(get_current_user),
    job_service: JobService = Depends(),
):
    """
    List user's audio processing jobs
    """
    logger.info(
        "Listing jobs request",
        user_id=current_user.id,
        skip=skip,
        limit=limit,
        status_filter=status_filter,
    )

    try:
        jobs = await job_service.get_user_jobs(
            user_id=current_user.id,
            skip=skip,
            limit=limit,
            status_filter=status_filter,
        )
        return [JobResponse.from_orm(job) for job in jobs]

    except Exception as e:
        logger.error(
            "Failed to list jobs",
            error=str(e),
            user_id=current_user.id,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve jobs"
        )

@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(
    job_id: str,
    current_user: User = Depends(get_current_user),
    job_service: JobService = Depends(),
):
    """
    Get specific job details
    """
    logger.info("Get job request", job_id=job_id, user_id=current_user.id)

    try:
        job = await job_service.get_job(job_id)
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found"
            )

        # Verify ownership
        if job.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )

        return JobResponse.from_orm(job)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to get job",
            error=str(e),
            job_id=job_id,
            user_id=current_user.id,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve job"
        )

@router.get("/jobs/{job_id}/download")
async def download_result(
    job_id: str,
    current_user: User = Depends(get_current_user),
    job_service: JobService = Depends(),
    storage_service: StorageService = Depends(),
):
    """
    Download processed audio file
    """
    logger.info("Download request", job_id=job_id, user_id=current_user.id)

    try:
        job = await job_service.get_job(job_id)
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found"
            )

        # Verify ownership
        if job.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )

        # Check if job is completed
        if job.status != JobStatus.COMPLETED:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Job is not completed. Current status: {job.status.value}"
            )

        if not job.output_blob_path:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Output file not found"
            )

        # Get download stream
        file_stream = await storage_service.download_file(job.output_blob_path)
        filename = job.output_filename or f"upscaled_{job.input_filename}"

        # Update metrics
        metrics.files_downloaded.inc()

        logger.info(
            "File download started",
            job_id=job_id,
            user_id=current_user.id,
            filename=filename,
        )

        return StreamingResponse(
            file_stream,
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to download file",
            error=str(e),
            job_id=job_id,
            user_id=current_user.id,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to download file"
        )

@router.delete("/jobs/{job_id}")
async def delete_job(
    job_id: str,
    current_user: User = Depends(get_current_user),
    job_service: JobService = Depends(),
    storage_service: StorageService = Depends(),
):
    """
    Delete job and associated files
    """
    logger.info("Delete job request", job_id=job_id, user_id=current_user.id)

    try:
        job = await job_service.get_job(job_id)
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found"
            )

        # Verify ownership
        if job.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )

        # Delete files from storage
        if job.input_blob_path:
            await storage_service.delete_file(job.input_blob_path)
        if job.output_blob_path:
            await storage_service.delete_file(job.output_blob_path)

        # Delete job record
        await job_service.delete_job(job_id)

        # Update metrics
        metrics.jobs_deleted.inc()

        logger.info(
            "Job deleted successfully",
            job_id=job_id,
            user_id=current_user.id,
        )

        return {"message": "Job deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to delete job",
            error=str(e),
            job_id=job_id,
            user_id=current_user.id,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete job"
        )