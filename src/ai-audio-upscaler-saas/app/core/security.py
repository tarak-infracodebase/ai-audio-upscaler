"""
Security Utilities
Input validation, rate limiting, and security middleware
"""

import hashlib
import hmac
import secrets
import re
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path
import mimetypes
from urllib.parse import urlparse

from fastapi import HTTPException, Request, status
from fastapi.security.utils import get_authorization_scheme_param
import structlog

logger = structlog.get_logger(__name__)

# Security Constants
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
ALLOWED_AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma'}
ALLOWED_MIME_TYPES = {
    'audio/wav', 'audio/x-wav', 'audio/wave',
    'audio/mpeg', 'audio/mp3',
    'audio/flac', 'audio/x-flac',
    'audio/ogg', 'audio/x-ogg',
    'audio/mp4', 'audio/m4a', 'audio/x-m4a',
    'audio/aac', 'audio/x-aac',
    'audio/x-ms-wma'
}

# Rate limiting storage (in production, use Redis)
_rate_limit_storage = {}

class SecurityError(Exception):
    """Base security error"""
    pass

class ValidationError(SecurityError):
    """Input validation error"""
    pass

class RateLimitError(SecurityError):
    """Rate limit exceeded error"""
    pass

def validate_filename(filename: str) -> str:
    """
    Validate and sanitize filename

    Args:
        filename: Original filename

    Returns:
        Sanitized filename

    Raises:
        ValidationError: If filename is invalid
    """
    if not filename:
        raise ValidationError("Filename cannot be empty")

    if len(filename) > 255:
        raise ValidationError("Filename too long (max 255 characters)")

    # Remove path traversal attempts
    filename = Path(filename).name

    # Remove or replace dangerous characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    filename = re.sub(r'[^\w\-_\.]', '_', filename)

    # Remove leading/trailing dots and spaces
    filename = filename.strip('. ')

    if not filename:
        raise ValidationError("Filename becomes empty after sanitization")

    # Check extension
    if not any(filename.lower().endswith(ext) for ext in ALLOWED_AUDIO_EXTENSIONS):
        raise ValidationError(f"File extension not allowed. Allowed: {', '.join(ALLOWED_AUDIO_EXTENSIONS)}")

    logger.debug("Filename validated", original=filename, sanitized=filename)
    return filename

def validate_file_content(content: bytes, filename: str) -> bool:
    """
    Validate file content for security

    Args:
        content: File content bytes
        filename: Filename for MIME type detection

    Returns:
        True if valid

    Raises:
        ValidationError: If content is invalid
    """
    if not content:
        raise ValidationError("File content is empty")

    if len(content) > MAX_FILE_SIZE:
        raise ValidationError(f"File too large (max {MAX_FILE_SIZE // (1024*1024)}MB)")

    # Check MIME type based on content
    detected_type, _ = mimetypes.guess_type(filename)
    if detected_type not in ALLOWED_MIME_TYPES:
        raise ValidationError(f"File type not allowed: {detected_type}")

    # Basic file signature validation
    if not _validate_file_signature(content, detected_type):
        raise ValidationError("File signature validation failed")

    logger.debug("File content validated", size=len(content), mime_type=detected_type)
    return True

def _validate_file_signature(content: bytes, mime_type: str) -> bool:
    """Validate file signature (magic bytes)"""
    if not content:
        return False

    # Common audio file signatures
    signatures = {
        'audio/wav': [b'RIFF', b'WAVE'],
        'audio/x-wav': [b'RIFF', b'WAVE'],
        'audio/wave': [b'RIFF', b'WAVE'],
        'audio/mpeg': [b'ID3', b'\xff\xfb', b'\xff\xf3', b'\xff\xf2'],
        'audio/mp3': [b'ID3', b'\xff\xfb', b'\xff\xf3', b'\xff\xf2'],
        'audio/flac': [b'fLaC'],
        'audio/x-flac': [b'fLaC'],
        'audio/ogg': [b'OggS'],
        'audio/x-ogg': [b'OggS'],
        'audio/mp4': [b'ftypM4A ', b'ftypisom'],
        'audio/m4a': [b'ftypM4A '],
        'audio/x-m4a': [b'ftypM4A '],
    }

    mime_signatures = signatures.get(mime_type, [])
    if not mime_signatures:
        # If no specific signature, allow (generic validation)
        return True

    # Check if content starts with any valid signature
    for sig in mime_signatures:
        if content.startswith(sig) or sig in content[:64]:  # Check first 64 bytes
            return True

    return False

def validate_processing_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and sanitize processing parameters

    Args:
        params: Processing parameters dict

    Returns:
        Validated parameters

    Raises:
        ValidationError: If parameters are invalid
    """
    validated = {}

    # Target sample rate
    target_sr = params.get('target_sample_rate', 48000)
    if not isinstance(target_sr, int) or target_sr < 8000 or target_sr > 192000:
        raise ValidationError("Invalid target_sample_rate (must be 8000-192000)")
    validated['target_sample_rate'] = target_sr

    # Mode
    mode = params.get('mode', 'baseline')
    if mode not in ['baseline', 'ai']:
        raise ValidationError("Invalid mode (must be 'baseline' or 'ai')")
    validated['mode'] = mode

    # Baseline method
    baseline_method = params.get('baseline_method', 'sinc')
    if baseline_method not in ['sinc', 'linear', 'cubic']:
        raise ValidationError("Invalid baseline_method")
    validated['baseline_method'] = baseline_method

    # Boolean parameters
    bool_params = [
        'use_ai', 'generate_analysis', 'tta', 'qc',
        'spectral_matching', 'remove_dc', 'normalize_input'
    ]

    for param in bool_params:
        value = params.get(param, False)
        if not isinstance(value, bool):
            raise ValidationError(f"Parameter {param} must be boolean")
        validated[param] = value

    # Numeric parameters with ranges
    numeric_params = {
        'transient_strength': (0.0, 1.0, 0.5),
        'denoising_strength': (0.0, 1.0, 0.0),
        'candidate_count': (1, 10, 3),
        'judge_threshold': (0.0, 1.0, 0.7),
    }

    for param, (min_val, max_val, default) in numeric_params.items():
        value = params.get(param, default)
        if not isinstance(value, (int, float)):
            raise ValidationError(f"Parameter {param} must be numeric")
        if not min_val <= value <= max_val:
            raise ValidationError(f"Parameter {param} must be between {min_val} and {max_val}")
        validated[param] = float(value)

    # String parameters
    string_params = {
        'normalization_mode': ['peak', 'rms', 'lufs', 'none'],
        'stereo_mode': ['stereo', 'mono', 'mid_side'],
    }

    for param, allowed_values in string_params.items():
        value = params.get(param, allowed_values[0])
        if value not in allowed_values:
            raise ValidationError(f"Parameter {param} must be one of: {', '.join(allowed_values)}")
        validated[param] = value

    logger.debug("Processing parameters validated", params=validated)
    return validated

def validate_url(url: str, allowed_schemes: List[str] = None) -> bool:
    """
    Validate URL for security

    Args:
        url: URL to validate
        allowed_schemes: List of allowed schemes (default: ['http', 'https'])

    Returns:
        True if valid

    Raises:
        ValidationError: If URL is invalid
    """
    if not url:
        raise ValidationError("URL cannot be empty")

    if len(url) > 2048:
        raise ValidationError("URL too long")

    allowed_schemes = allowed_schemes or ['http', 'https']

    try:
        parsed = urlparse(url)

        if parsed.scheme not in allowed_schemes:
            raise ValidationError(f"URL scheme not allowed: {parsed.scheme}")

        if not parsed.netloc:
            raise ValidationError("URL missing domain")

        # Block localhost and private IPs (basic check)
        if parsed.hostname in ['localhost', '127.0.0.1', '0.0.0.0']:
            raise ValidationError("Local URLs not allowed")

        return True

    except Exception as e:
        raise ValidationError(f"Invalid URL: {str(e)}")

class RateLimiter:
    """Simple rate limiter (use Redis in production)"""

    def __init__(self, max_requests: int = 100, window_seconds: int = 3600):
        self.max_requests = max_requests
        self.window_seconds = window_seconds

    def is_allowed(self, key: str) -> bool:
        """Check if request is allowed"""
        now = datetime.now()
        window_start = now - timedelta(seconds=self.window_seconds)

        # Clean old entries
        if key in _rate_limit_storage:
            _rate_limit_storage[key] = [
                timestamp for timestamp in _rate_limit_storage[key]
                if timestamp > window_start
            ]
        else:
            _rate_limit_storage[key] = []

        # Check limit
        if len(_rate_limit_storage[key]) >= self.max_requests:
            return False

        # Add current request
        _rate_limit_storage[key].append(now)
        return True

    def get_remaining(self, key: str) -> int:
        """Get remaining requests in window"""
        if key not in _rate_limit_storage:
            return self.max_requests

        now = datetime.now()
        window_start = now - timedelta(seconds=self.window_seconds)

        # Count requests in current window
        current_requests = sum(
            1 for timestamp in _rate_limit_storage[key]
            if timestamp > window_start
        )

        return max(0, self.max_requests - current_requests)

def rate_limit_middleware(rate_limiter: RateLimiter):
    """Rate limiting middleware factory"""

    async def middleware(request: Request, call_next):
        # Get client identifier
        client_ip = request.client.host
        user_id = getattr(request.state, 'user_id', None)
        key = f"user:{user_id}" if user_id else f"ip:{client_ip}"

        # Check rate limit
        if not rate_limiter.is_allowed(key):
            remaining = rate_limiter.get_remaining(key)
            logger.warning("Rate limit exceeded", key=key, remaining=remaining)

            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
                headers={"Retry-After": str(rate_limiter.window_seconds)}
            )

        # Add rate limit headers
        response = await call_next(request)
        remaining = rate_limiter.get_remaining(key)

        response.headers["X-RateLimit-Limit"] = str(rate_limiter.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Window"] = str(rate_limiter.window_seconds)

        return response

    return middleware

def generate_secure_token(length: int = 32) -> str:
    """Generate cryptographically secure random token"""
    return secrets.token_urlsafe(length)

def constant_time_compare(a: str, b: str) -> bool:
    """Constant time string comparison to prevent timing attacks"""
    if len(a) != len(b):
        return False
    return hmac.compare_digest(a.encode('utf-8'), b.encode('utf-8'))

def hash_password(password: str, salt: Optional[bytes] = None) -> tuple[str, bytes]:
    """
    Hash password with salt (using scrypt)

    Args:
        password: Plain password
        salt: Optional salt (generated if not provided)

    Returns:
        Tuple of (hashed_password, salt)
    """
    if salt is None:
        salt = secrets.token_bytes(32)

    # Use scrypt for password hashing (memory-hard function)
    hashed = hashlib.scrypt(
        password.encode('utf-8'),
        salt=salt,
        n=16384,  # CPU/memory cost
        r=8,      # block size
        p=1       # parallelization
    )

    return hashed.hex(), salt

def verify_password(password: str, hashed_password: str, salt: bytes) -> bool:
    """Verify password against hash"""
    try:
        computed_hash, _ = hash_password(password, salt)
        return constant_time_compare(computed_hash, hashed_password)
    except Exception:
        return False

def sanitize_log_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize data for logging (remove sensitive information)

    Args:
        data: Data dictionary

    Returns:
        Sanitized data dictionary
    """
    sensitive_keys = {
        'password', 'token', 'secret', 'key', 'authorization',
        'x-api-key', 'x-auth-token', 'access_token', 'refresh_token'
    }

    sanitized = {}
    for key, value in data.items():
        if any(sensitive_key in key.lower() for sensitive_key in sensitive_keys):
            sanitized[key] = "[REDACTED]"
        elif isinstance(value, dict):
            sanitized[key] = sanitize_log_data(value)
        elif isinstance(value, list):
            sanitized[key] = [
                sanitize_log_data(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            sanitized[key] = value

    return sanitized

def get_client_ip(request: Request) -> str:
    """Get real client IP address considering proxies"""
    # Check for forwarded IP headers (in order of preference)
    headers_to_check = [
        'x-forwarded-for',
        'x-real-ip',
        'x-client-ip',
        'cf-connecting-ip',  # Cloudflare
        'x-forwarded',
        'forwarded-for',
        'forwarded'
    ]

    for header in headers_to_check:
        ip = request.headers.get(header)
        if ip:
            # Handle comma-separated list (use first IP)
            ip = ip.split(',')[0].strip()
            if ip and ip != 'unknown':
                return ip

    # Fallback to direct connection IP
    return request.client.host if request.client else 'unknown'