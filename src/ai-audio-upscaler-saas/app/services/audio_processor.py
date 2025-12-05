"""
Audio Processing Service
Wraps the AI Audio Upscaler pipeline for production use
"""

import os
import asyncio
from typing import Dict, Any, Optional, Callable
from pathlib import Path
import tempfile

import torch
import structlog

from ai_audio_upscaler.config import UpscalerConfig
from ai_audio_upscaler.pipeline import AudioUpscalerPipeline
from ai_audio_upscaler.security import ValidationError, SecurityError
from app.core.monitoring import metrics
from app.models.job import ProcessingParameters

logger = structlog.get_logger(__name__)

class AudioProcessorService:
    """
    Production service wrapper for AI Audio Upscaler pipeline
    Handles async processing, device management, and monitoring
    """

    def __init__(self):
        self._pipelines: Dict[str, AudioUpscalerPipeline] = {}
        self._device = self._detect_device()
        self._models_loaded = False

    def _detect_device(self) -> str:
        """Detect the best available device for processing"""
        if torch.cuda.is_available():
            device = "cuda"
            gpu_count = torch.cuda.device_count()
            logger.info(f"CUDA detected: {gpu_count} GPUs available")

            # Log GPU information
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                logger.info(
                    f"GPU {i}: {props.name}, "
                    f"Memory: {props.total_memory / 1024**3:.1f}GB, "
                    f"Compute: {props.major}.{props.minor}"
                )
        else:
            device = "cpu"
            logger.info("CUDA not available, using CPU")

        return device

    def _get_pipeline(self, parameters: ProcessingParameters) -> AudioUpscalerPipeline:
        """
        Get or create a pipeline for the given parameters
        Pipelines are cached to avoid repeated initialization
        """
        # Create a cache key based on processing parameters
        cache_key = (
            parameters.mode,
            parameters.baseline_method,
            parameters.use_ai,
            self._device
        )

        if cache_key not in self._pipelines:
            config = UpscalerConfig(
                target_sample_rate=parameters.target_sample_rate,
                mode="ai" if parameters.use_ai else "baseline",
                baseline_method=parameters.baseline_method,
                device=self._device,
                export_format="wav",  # Always export as WAV
                use_advanced_dsp=True,  # Use advanced DSP for better quality
                dsp_quality_preset="quality",
                apply_dither=True,
                noise_shaper="auto",
                # AI-specific settings
                use_diffusion=False,  # Use GAN by default
                diffusion_steps=50,
                model_type="gan",
            )

            pipeline = AudioUpscalerPipeline(config)
            self._pipelines[cache_key] = pipeline

            logger.info(
                "Created new pipeline",
                cache_key=str(cache_key),
                device=self._device,
                mode=config.mode
            )

        return self._pipelines[cache_key]

    async def warm_up(self) -> None:
        """
        Warm up the service by loading models and testing processing
        """
        if self._models_loaded:
            return

        try:
            logger.info("Starting audio processor warm-up")

            # Create a minimal test pipeline
            test_params = ProcessingParameters(
                target_sample_rate=48000,
                mode="baseline",
                baseline_method="sinc",
                use_ai=False,
            )

            pipeline = self._get_pipeline(test_params)

            # Test with a small synthetic audio sample
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                # Create a 1-second 440Hz sine wave test file
                import torchaudio
                import numpy as np

                sample_rate = 44100
                duration = 0.1  # 100ms test
                t = np.linspace(0, duration, int(sample_rate * duration))
                audio = np.sin(2 * np.pi * 440 * t)
                audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)

                torchaudio.save(temp_file.name, audio_tensor, sample_rate)

                # Test processing
                output_file = temp_file.name.replace(".wav", "_out.wav")

                try:
                    await asyncio.to_thread(
                        pipeline.run,
                        temp_file.name,
                        output_file,
                        progress_callback=None,
                        generate_analysis=False
                    )

                    logger.info("Warm-up processing test successful")
                    self._models_loaded = True

                finally:
                    # Cleanup test files
                    for file_path in [temp_file.name, output_file]:
                        if os.path.exists(file_path):
                            os.unlink(file_path)

        except Exception as e:
            logger.warning("Warm-up failed, but continuing", error=str(e))
            # Don't fail startup if warm-up fails

    async def process_audio(
        self,
        input_path: str,
        output_path: str,
        parameters: ProcessingParameters,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        device: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process audio file asynchronously

        Args:
            input_path: Path to input audio file
            output_path: Path for output audio file
            parameters: Processing parameters
            progress_callback: Optional progress reporting function
            device: Optional device override

        Returns:
            Dictionary with processing results and statistics
        """
        start_time = asyncio.get_event_loop().time()
        processing_device = device or self._device

        logger.info(
            "Starting audio processing",
            input_path=input_path,
            output_path=output_path,
            device=processing_device,
            parameters=parameters.dict()
        )

        try:
            # Validate input file exists
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Input file not found: {input_path}")

            # Update processing device in parameters
            pipeline = self._get_pipeline(parameters)
            if hasattr(pipeline.config, 'device'):
                pipeline.config.device = processing_device

            # Async progress wrapper
            async_progress_callback = None
            if progress_callback:
                def sync_progress(progress: float, message: str):
                    """Sync progress callback wrapper"""
                    try:
                        progress_callback(progress, message)
                    except Exception as e:
                        logger.warning("Progress callback failed", error=str(e))

                async_progress_callback = sync_progress

            # Run processing in thread pool to avoid blocking
            result = await asyncio.to_thread(
                self._process_sync,
                pipeline,
                input_path,
                output_path,
                parameters,
                async_progress_callback
            )

            # Calculate processing time
            processing_time = asyncio.get_event_loop().time() - start_time

            # Get file sizes for statistics
            input_size = os.path.getsize(input_path)
            output_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0

            # Update metrics
            metrics.processing_duration.observe(processing_time)
            metrics.audio_files_processed.inc()

            processing_stats = {
                "duration_seconds": processing_time,
                "input_size_bytes": input_size,
                "output_size_bytes": output_size,
                "device": processing_device,
                "parameters": parameters.dict(),
                "success": True,
            }

            # Add analysis data if generated
            analysis_data = result.get("analysis", None)

            logger.info(
                "Audio processing completed successfully",
                input_path=input_path,
                output_path=output_path,
                processing_time=processing_time,
                device=processing_device
            )

            return {
                "status": "success",
                "stats": processing_stats,
                "analysis": analysis_data,
            }

        except (ValidationError, SecurityError) as e:
            logger.error(
                "Audio processing validation failed",
                input_path=input_path,
                error=str(e),
                error_type=type(e).__name__
            )
            metrics.processing_errors.labels(error_type="validation").inc()
            raise

        except FileNotFoundError as e:
            logger.error("Input file not found", input_path=input_path, error=str(e))
            metrics.processing_errors.labels(error_type="file_not_found").inc()
            raise

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(
                    "GPU out of memory during processing",
                    input_path=input_path,
                    device=processing_device,
                    error=str(e)
                )
                metrics.processing_errors.labels(error_type="oom").inc()

                # Try CPU fallback if we were using GPU
                if processing_device == "cuda":
                    logger.info("Retrying processing on CPU")
                    return await self.process_audio(
                        input_path=input_path,
                        output_path=output_path,
                        parameters=parameters,
                        progress_callback=progress_callback,
                        device="cpu"
                    )

            logger.error(
                "Runtime error during processing",
                input_path=input_path,
                device=processing_device,
                error=str(e),
                exc_info=True
            )
            metrics.processing_errors.labels(error_type="runtime").inc()
            raise

        except Exception as e:
            logger.error(
                "Unexpected error during processing",
                input_path=input_path,
                device=processing_device,
                error=str(e),
                exc_info=True
            )
            metrics.processing_errors.labels(error_type="unknown").inc()
            raise

    def _process_sync(
        self,
        pipeline: AudioUpscalerPipeline,
        input_path: str,
        output_path: str,
        parameters: ProcessingParameters,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Synchronous processing wrapper for running in thread pool
        """
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Run the pipeline
            result = pipeline.run(
                input_path=input_path,
                output_path=output_path,
                normalization_mode=parameters.normalization_mode,
                generate_analysis=parameters.generate_analysis,
                progress_callback=progress_callback,
                # Advanced parameters
                tta=parameters.tta,
                stereo_mode=parameters.stereo_mode,
                transient_strength=parameters.transient_strength,
                spectral_matching=parameters.spectral_matching,
                remove_dc=True,
                normalize_input=True,
                limit=True,
                saturation=0.0,
                stereo_width=1.0,
                transient_shaping=0.0,
                # Quality control
                qc=parameters.qc,
                candidate_count=parameters.candidate_count,
                judge_threshold=parameters.judge_threshold,
                denoising_strength=parameters.denoising_strength,
                use_stems=False,  # Not yet supported in production
                use_restoration=False,  # Not yet supported in production
                restoration_strength=0.5,
            )

            return result

        except Exception as e:
            logger.error("Synchronous processing failed", error=str(e), exc_info=True)
            raise

    def get_device_info(self) -> Dict[str, Any]:
        """
        Get information about available processing devices
        """
        info = {
            "current_device": self._device,
            "cuda_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "gpus": [],
        }

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
                memory_cached = torch.cuda.memory_reserved(i) / 1024**3      # GB
                total_memory = props.total_memory / 1024**3                   # GB

                info["gpus"].append({
                    "device_id": i,
                    "name": props.name,
                    "compute_capability": f"{props.major}.{props.minor}",
                    "total_memory_gb": round(total_memory, 2),
                    "allocated_memory_gb": round(memory_allocated, 2),
                    "cached_memory_gb": round(memory_cached, 2),
                    "free_memory_gb": round(total_memory - memory_allocated, 2),
                    "utilization_percent": round((memory_allocated / total_memory) * 100, 2),
                })

        return info

    def cleanup(self) -> None:
        """
        Cleanup resources and clear caches
        """
        logger.info("Cleaning up audio processor")

        # Clear pipeline cache
        self._pipelines.clear()

        # Clear GPU memory if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        logger.info("Audio processor cleanup completed")