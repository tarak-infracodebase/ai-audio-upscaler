import os
import torch
import torchaudio
import subprocess
import uuid
import logging
from typing import Tuple
from pathlib import Path
from .security import validate_audio_file, SecurityError, ValidationError

logger = logging.getLogger(__name__)

def load_audio_robust(file_path: str) -> Tuple[torch.Tensor, int]:
    """
    Robust audio loader with security validation that falls back to FFmpeg for unsupported formats.

    Args:
        file_path (str): Absolute path to the audio file.

    Returns:
        Tuple[torch.Tensor, int]: A tuple containing:
            - waveform (torch.Tensor): Audio data (Channels, Time).
            - sample_rate (int): Sample rate in Hz.

    Raises:
        SecurityError: If file fails security validation.
        ValidationError: If file is invalid or corrupted.
        RuntimeError: If both direct load and FFmpeg fallback fail.
    """
    # Comprehensive security and validation checks
    try:
        validated_path, file_info = validate_audio_file(file_path)
        logger.info(f"Loading validated audio file: {file_info['size_mb']:.1f}MB, "
                   f"{file_info['duration_seconds']:.1f}s")
    except (SecurityError, ValidationError) as e:
        logger.error(f"Audio file validation failed: {e}")
        raise

    # Try direct loading first
    try:
        waveform, sr = torchaudio.load(str(validated_path))

        # Additional runtime validation
        if waveform.numel() == 0:
            raise ValidationError("Loaded audio is empty")

        # Verify the loaded audio matches metadata expectations
        actual_channels = waveform.shape[0]
        if actual_channels != file_info['channels']:
            logger.warning(f"Channel count mismatch: expected {file_info['channels']}, got {actual_channels}")

        if abs(sr - file_info['sample_rate']) > 1:  # Allow small rounding differences
            logger.warning(f"Sample rate mismatch: expected {file_info['sample_rate']}, got {sr}")

        logger.debug(f"Audio loaded successfully: {waveform.shape} at {sr}Hz")
        return waveform, sr

    except Exception as e:
        logger.warning(f"Direct audio loading failed: {e}. Trying FFmpeg fallback...")

        # Fallback to FFmpeg with secure temporary file handling
        temp_dir = validated_path.parent
        temp_wav = temp_dir / f"temp_convert_{uuid.uuid4().hex[:8]}.wav"
        
        try:
            # Convert to WAV using FFmpeg with security considerations
            # -y: overwrite
            # -i: input
            # -vn: disable video (just in case)
            # -acodec pcm_f32le: preserve quality (32-bit float)
            # -t: limit duration to prevent resource exhaustion
            # -loglevel error: reduce output verbosity

            cmd = [
                "ffmpeg", "-y",
                "-i", str(validated_path),
                "-vn",  # no video
                "-acodec", "pcm_f32le",
                "-t", str(file_info['duration_seconds'] + 1),  # add 1s buffer for safety
                "-loglevel", "error",
                str(temp_wav)
            ]

            logger.debug(f"Running FFmpeg command: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=60  # Timeout after 60 seconds
            )

            if not temp_wav.exists():
                raise RuntimeError("FFmpeg conversion failed - no output file created")

            waveform, sr = torchaudio.load(str(temp_wav))

            # Validate converted audio
            if waveform.numel() == 0:
                raise ValidationError("FFmpeg converted audio is empty")

            logger.info(f"Audio successfully converted via FFmpeg: {waveform.shape} at {sr}Hz")
            return waveform, sr
            
        except subprocess.TimeoutExpired:
            logger.error("FFmpeg conversion timed out")
            raise RuntimeError(f"Failed to load audio: {e}. FFmpeg conversion timed out after 60 seconds")
        except subprocess.CalledProcessError as proc_error:
            stderr_output = proc_error.stderr.decode('utf-8', errors='ignore') if proc_error.stderr else "No error details"
            logger.error(f"FFmpeg process failed: {stderr_output}")
            raise RuntimeError(f"Failed to load audio: {e}. FFmpeg failed: {stderr_output}")
        except Exception as ffmpeg_error:
            logger.error(f"FFmpeg fallback failed: {ffmpeg_error}")
            raise RuntimeError(f"Failed to load audio: {e}. FFmpeg fallback failed: {ffmpeg_error}")

        finally:
            # Secure cleanup of temporary file
            if temp_wav.exists():
                try:
                    temp_wav.unlink()
                    logger.debug(f"Cleaned up temporary file: {temp_wav}")
                except OSError as cleanup_error:
                    logger.warning(f"Failed to cleanup temporary file {temp_wav}: {cleanup_error}")
