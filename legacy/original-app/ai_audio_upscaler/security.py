"""
Security and input validation module for AI Audio Upscaler Pro.

This module provides comprehensive security checks, input validation, and
resource management to ensure safe operation in production environments.
"""

import os
import hashlib
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Union
import torch
import torchaudio

logger = logging.getLogger(__name__)

# Security configuration
ALLOWED_AUDIO_EXTENSIONS = {'.wav', '.flac', '.mp3', '.m4a', '.aac', '.ogg', '.opus', '.wma'}
MAX_FILE_SIZE_MB = 500  # Maximum audio file size in MB
MAX_AUDIO_DURATION_SECONDS = 600  # 10 minutes maximum
MIN_SAMPLE_RATE = 8000
MAX_SAMPLE_RATE = 192000
MAX_CHANNELS = 8  # Support up to 7.1 surround sound

class SecurityError(Exception):
    """Raised when security validation fails."""
    pass

class ValidationError(Exception):
    """Raised when input validation fails."""
    pass

def validate_file_path(file_path: str, must_exist: bool = True) -> Path:
    """
    Validate and normalize file paths to prevent path traversal attacks.

    Args:
        file_path: Input file path
        must_exist: Whether the file must exist

    Returns:
        Validated and normalized Path object

    Raises:
        SecurityError: If path validation fails
        ValidationError: If file doesn't exist when required
    """
    if not file_path or not isinstance(file_path, (str, Path)):
        raise ValidationError("File path must be a non-empty string or Path object")

    try:
        path = Path(file_path).resolve()
    except (OSError, ValueError) as e:
        raise SecurityError(f"Invalid file path: {e}")

    # Check for path traversal attempts
    try:
        # Ensure the resolved path is within reasonable bounds
        str_path = str(path)
        if '..' in str_path or '~' in str_path:
            # Additional check after resolution
            raise SecurityError("Path traversal detected")
    except Exception:
        raise SecurityError("Invalid path format")

    # Check if file exists when required
    if must_exist and not path.exists():
        raise ValidationError(f"File does not exist: {path}")

    # Check if it's actually a file (not a directory or special file)
    if must_exist and not path.is_file():
        raise ValidationError(f"Path is not a regular file: {path}")

    return path

def validate_audio_file(file_path: Union[str, Path]) -> Tuple[Path, dict]:
    """
    Comprehensive audio file validation including format, size, and content checks.

    Args:
        file_path: Path to audio file

    Returns:
        Tuple of (validated_path, file_info)

    Raises:
        SecurityError: If security validation fails
        ValidationError: If file validation fails
    """
    # Validate path
    path = validate_file_path(file_path, must_exist=True)

    # Check file extension
    if path.suffix.lower() not in ALLOWED_AUDIO_EXTENSIONS:
        raise ValidationError(
            f"Unsupported file extension: {path.suffix}. "
            f"Allowed extensions: {', '.join(sorted(ALLOWED_AUDIO_EXTENSIONS))}"
        )

    # Check file size
    try:
        file_size = path.stat().st_size
        file_size_mb = file_size / (1024 * 1024)

        if file_size_mb > MAX_FILE_SIZE_MB:
            raise ValidationError(
                f"File too large: {file_size_mb:.1f}MB. Maximum allowed: {MAX_FILE_SIZE_MB}MB"
            )

        if file_size == 0:
            raise ValidationError("File is empty")

    except OSError as e:
        raise ValidationError(f"Cannot access file: {e}")

    # Try to get basic audio info without loading the entire file
    file_info = {
        'path': path,
        'size_bytes': file_size,
        'size_mb': file_size_mb,
        'extension': path.suffix.lower()
    }

    try:
        # Get audio metadata without loading full audio
        info = torchaudio.info(str(path))

        # Validate audio properties
        if info.sample_rate < MIN_SAMPLE_RATE or info.sample_rate > MAX_SAMPLE_RATE:
            raise ValidationError(
                f"Invalid sample rate: {info.sample_rate}Hz. "
                f"Must be between {MIN_SAMPLE_RATE}-{MAX_SAMPLE_RATE}Hz"
            )

        if info.num_channels < 1 or info.num_channels > MAX_CHANNELS:
            raise ValidationError(
                f"Invalid channel count: {info.num_channels}. "
                f"Must be between 1-{MAX_CHANNELS} channels"
            )

        # Calculate duration and validate
        duration_seconds = info.num_frames / info.sample_rate
        if duration_seconds > MAX_AUDIO_DURATION_SECONDS:
            raise ValidationError(
                f"Audio too long: {duration_seconds:.1f}s. "
                f"Maximum allowed: {MAX_AUDIO_DURATION_SECONDS}s"
            )

        if duration_seconds < 0.1:  # Minimum 100ms
            raise ValidationError("Audio too short (minimum 100ms required)")

        # Add metadata to file_info
        file_info.update({
            'sample_rate': info.sample_rate,
            'channels': info.num_channels,
            'frames': info.num_frames,
            'duration_seconds': duration_seconds,
            'bits_per_sample': info.bits_per_sample if hasattr(info, 'bits_per_sample') else None
        })

    except Exception as e:
        # If we can't read the metadata, it might be corrupted or not a valid audio file
        raise ValidationError(f"Cannot read audio file metadata: {e}")

    logger.info(f"Audio file validated: {path.name} ({file_size_mb:.1f}MB, "
               f"{info.sample_rate}Hz, {info.num_channels}ch, {duration_seconds:.1f}s)")

    return path, file_info

def validate_output_path(output_path: Union[str, Path], create_dirs: bool = True) -> Path:
    """
    Validate output path and optionally create parent directories.

    Args:
        output_path: Desired output path
        create_dirs: Whether to create parent directories

    Returns:
        Validated Path object

    Raises:
        SecurityError: If path validation fails
        ValidationError: If path is invalid for writing
    """
    if not output_path or not isinstance(output_path, (str, Path)):
        raise ValidationError("Output path must be a non-empty string or Path object")

    try:
        path = Path(output_path).resolve()
    except (OSError, ValueError) as e:
        raise SecurityError(f"Invalid output path: {e}")

    # Check for path traversal
    str_path = str(path)
    if '..' in str_path or '~' in str_path:
        raise SecurityError("Path traversal detected in output path")

    # Validate parent directory
    parent = path.parent

    if create_dirs:
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise ValidationError(f"Cannot create output directory: {e}")
    elif not parent.exists():
        raise ValidationError(f"Output directory does not exist: {parent}")

    # Check if we can write to the directory
    if not os.access(parent, os.W_OK):
        raise ValidationError(f"No write permission for directory: {parent}")

    # Check if output file already exists and warn
    if path.exists():
        logger.warning(f"Output file already exists and will be overwritten: {path}")

    return path

def validate_sample_rate(sample_rate: int) -> int:
    """
    Validate sample rate parameter.

    Args:
        sample_rate: Target sample rate in Hz

    Returns:
        Validated sample rate

    Raises:
        ValidationError: If sample rate is invalid
    """
    if not isinstance(sample_rate, int):
        raise ValidationError(f"Sample rate must be an integer, got {type(sample_rate)}")

    if sample_rate < MIN_SAMPLE_RATE or sample_rate > MAX_SAMPLE_RATE:
        raise ValidationError(
            f"Invalid sample rate: {sample_rate}Hz. "
            f"Must be between {MIN_SAMPLE_RATE}-{MAX_SAMPLE_RATE}Hz"
        )

    # Check for common sample rates
    common_rates = [8000, 16000, 22050, 44100, 48000, 88200, 96000, 176400, 192000]
    if sample_rate not in common_rates:
        logger.warning(f"Uncommon sample rate: {sample_rate}Hz. Common rates are: {common_rates}")

    return sample_rate

def validate_numeric_parameter(value: Union[int, float], param_name: str,
                             min_val: Optional[float] = None,
                             max_val: Optional[float] = None,
                             allow_none: bool = False) -> Union[int, float, None]:
    """
    Validate numeric parameters with optional range checking.

    Args:
        value: Parameter value to validate
        param_name: Name of the parameter (for error messages)
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)
        allow_none: Whether None is an acceptable value

    Returns:
        Validated parameter value

    Raises:
        ValidationError: If validation fails
    """
    if value is None:
        if allow_none:
            return None
        else:
            raise ValidationError(f"{param_name} cannot be None")

    if not isinstance(value, (int, float)):
        raise ValidationError(f"{param_name} must be a number, got {type(value)}")

    # Check for invalid float values
    if isinstance(value, float):
        if not (float('-inf') < value < float('inf')):
            raise ValidationError(f"{param_name} must be a finite number")

    # Range validation
    if min_val is not None and value < min_val:
        raise ValidationError(f"{param_name} must be >= {min_val}, got {value}")

    if max_val is not None and value > max_val:
        raise ValidationError(f"{param_name} must be <= {max_val}, got {value}")

    return value

def compute_file_hash(file_path: Union[str, Path], algorithm: str = 'sha256') -> str:
    """
    Compute cryptographic hash of a file for integrity checking.

    Args:
        file_path: Path to file
        algorithm: Hash algorithm ('sha256', 'md5', 'sha1')

    Returns:
        Hexadecimal hash string

    Raises:
        ValidationError: If file cannot be read or algorithm is invalid
    """
    path = validate_file_path(file_path, must_exist=True)

    try:
        hasher = hashlib.new(algorithm)
    except ValueError:
        raise ValidationError(f"Unsupported hash algorithm: {algorithm}")

    try:
        with open(path, 'rb') as f:
            # Read in chunks to handle large files
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    except IOError as e:
        raise ValidationError(f"Cannot read file for hashing: {e}")

def check_available_space(path: Union[str, Path], required_mb: float) -> bool:
    """
    Check if there's enough disk space for processing.

    Args:
        path: Directory path to check
        required_mb: Required space in MB

    Returns:
        True if enough space is available

    Raises:
        ValidationError: If cannot check disk space
    """
    try:
        path = Path(path)
        if path.is_file():
            path = path.parent

        stat = os.statvfs(path)
        available_bytes = stat.f_bavail * stat.f_frsize
        available_mb = available_bytes / (1024 * 1024)

        logger.debug(f"Disk space check: {available_mb:.1f}MB available, {required_mb:.1f}MB required")

        return available_mb >= required_mb

    except (OSError, AttributeError):
        # statvfs not available on Windows
        logger.warning("Cannot check disk space on this platform")
        return True  # Assume sufficient space

def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent issues with filesystem restrictions.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename safe for filesystem use
    """
    if not filename:
        return "output"

    # Replace problematic characters
    forbidden_chars = '<>:"/\\|?*'
    sanitized = filename
    for char in forbidden_chars:
        sanitized = sanitized.replace(char, '_')

    # Remove control characters
    sanitized = ''.join(char for char in sanitized if ord(char) >= 32)

    # Trim whitespace and dots (problematic on Windows)
    sanitized = sanitized.strip(' .')

    # Ensure not empty
    if not sanitized:
        sanitized = "output"

    # Limit length
    if len(sanitized) > 200:
        name, ext = os.path.splitext(sanitized)
        sanitized = name[:200-len(ext)] + ext

    return sanitized

class ResourceMonitor:
    """Monitor system resources during processing to prevent resource exhaustion."""

    def __init__(self):
        self.initial_memory = self._get_memory_usage()
        self.peak_memory = self.initial_memory

    def _get_memory_usage(self) -> dict:
        """Get current memory usage statistics."""
        memory_info = {}

        try:
            import psutil
            process = psutil.Process()
            memory_info['process_mb'] = process.memory_info().rss / (1024 * 1024)
            memory_info['system_percent'] = psutil.virtual_memory().percent
            memory_info['available_mb'] = psutil.virtual_memory().available / (1024 * 1024)
        except ImportError:
            logger.debug("psutil not available for memory monitoring")

        # PyTorch GPU memory if available
        if torch.cuda.is_available():
            memory_info['gpu_allocated_mb'] = torch.cuda.memory_allocated() / (1024 * 1024)
            memory_info['gpu_reserved_mb'] = torch.cuda.memory_reserved() / (1024 * 1024)

        return memory_info

    def check_resources(self) -> dict:
        """Check current resource usage and update peak values."""
        current = self._get_memory_usage()

        if 'process_mb' in current and 'process_mb' in self.peak_memory:
            if current['process_mb'] > self.peak_memory['process_mb']:
                self.peak_memory.update(current)

        return current

    def get_resource_summary(self) -> dict:
        """Get summary of resource usage during processing."""
        current = self.check_resources()

        summary = {
            'initial': self.initial_memory,
            'current': current,
            'peak': self.peak_memory
        }

        if 'process_mb' in current and 'process_mb' in self.initial_memory:
            summary['memory_increase_mb'] = current['process_mb'] - self.initial_memory['process_mb']

        return summary