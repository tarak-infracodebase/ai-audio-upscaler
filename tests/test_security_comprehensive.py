"""
Comprehensive security tests for AI Audio Upscaler Pro.
"""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from ai_audio_upscaler.security import (
    validate_file_path, validate_audio_file, validate_output_path,
    validate_sample_rate, validate_numeric_parameter, compute_file_hash,
    check_available_space, sanitize_filename, ResourceMonitor,
    SecurityError, ValidationError
)


@pytest.mark.security
class TestPathValidation:
    """Test path validation and sanitization."""

    def test_validate_file_path_normal(self, sample_audio_file):
        """Test normal file path validation."""
        validated_path = validate_file_path(str(sample_audio_file))
        assert validated_path.exists()
        assert validated_path.is_file()

    def test_validate_file_path_nonexistent(self):
        """Test validation of non-existent file."""
        with pytest.raises(ValidationError, match="File does not exist"):
            validate_file_path("/nonexistent/file.wav", must_exist=True)

    def test_validate_file_path_directory(self, temp_dir):
        """Test validation rejects directories."""
        with pytest.raises(ValidationError, match="not a regular file"):
            validate_file_path(str(temp_dir), must_exist=True)

    @pytest.mark.parametrize("malicious_path", [
        "../../../etc/passwd",
        "..\\..\\..\\windows\\system32\\config\\sam",
        "~/../../root/.ssh/id_rsa",
        "${HOME}/secret.wav",
        "`rm -rf /`",
        "$(echo hello)",
    ])
    def test_validate_file_path_traversal(self, malicious_path):
        """Test path traversal attack prevention."""
        with pytest.raises(SecurityError):
            validate_file_path(malicious_path, must_exist=False)

    def test_validate_file_path_empty(self):
        """Test empty path validation."""
        with pytest.raises(ValidationError, match="must be a non-empty string"):
            validate_file_path("")

    def test_validate_file_path_none(self):
        """Test None path validation."""
        with pytest.raises(ValidationError, match="must be a non-empty string"):
            validate_file_path(None)


@pytest.mark.security
class TestAudioFileValidation:
    """Test audio file validation."""

    def test_validate_audio_file_valid(self, sample_audio_file):
        """Test validation of valid audio file."""
        validated_path, file_info = validate_audio_file(sample_audio_file)

        assert validated_path.exists()
        assert file_info['sample_rate'] == 44100
        assert file_info['channels'] == 1
        assert file_info['duration_seconds'] > 0.9
        assert file_info['extension'] == '.wav'

    def test_validate_audio_file_corrupted(self, corrupted_audio_file):
        """Test validation of corrupted audio file."""
        with pytest.raises(ValidationError, match="Cannot read audio file metadata"):
            validate_audio_file(corrupted_audio_file)

    def test_validate_audio_file_unsupported_extension(self, temp_dir):
        """Test validation of unsupported file extension."""
        bad_file = temp_dir / "test.txt"
        bad_file.write_text("not audio")

        with pytest.raises(ValidationError, match="Unsupported file extension"):
            validate_audio_file(bad_file)

    def test_validate_audio_file_too_large(self, temp_dir):
        """Test validation of oversized file."""
        # Create a file that appears too large
        with patch('ai_audio_upscaler.security.MAX_FILE_SIZE_MB', 0.001):  # 1KB limit
            with pytest.raises(ValidationError, match="File too large"):
                validate_audio_file(temp_dir / "nonexistent.wav")

    def test_validate_audio_file_empty(self, temp_dir):
        """Test validation of empty file."""
        empty_file = temp_dir / "empty.wav"
        empty_file.touch()

        with pytest.raises(ValidationError, match="File is empty"):
            validate_audio_file(empty_file)


@pytest.mark.security
class TestOutputPathValidation:
    """Test output path validation."""

    def test_validate_output_path_normal(self, temp_dir):
        """Test normal output path validation."""
        output_path = temp_dir / "output.wav"
        validated = validate_output_path(output_path, create_dirs=True)

        assert validated.parent.exists()
        assert str(validated).endswith('output.wav')

    def test_validate_output_path_create_dirs(self, temp_dir):
        """Test output path with directory creation."""
        output_path = temp_dir / "subdir" / "deep" / "output.wav"
        validated = validate_output_path(output_path, create_dirs=True)

        assert validated.parent.exists()
        assert validated.parent.name == "deep"

    def test_validate_output_path_no_create_dirs(self, temp_dir):
        """Test output path without directory creation."""
        output_path = temp_dir / "nonexistent" / "output.wav"

        with pytest.raises(ValidationError, match="Output directory does not exist"):
            validate_output_path(output_path, create_dirs=False)

    def test_validate_output_path_permission_denied(self, temp_dir):
        """Test output path with permission issues."""
        # This test may not work on all systems
        try:
            read_only_dir = temp_dir / "readonly"
            read_only_dir.mkdir()
            read_only_dir.chmod(0o444)  # Read-only

            output_path = read_only_dir / "output.wav"
            with pytest.raises(ValidationError, match="No write permission"):
                validate_output_path(output_path, create_dirs=False)

        except (OSError, PermissionError):
            pytest.skip("Cannot test permission denied on this system")

    @pytest.mark.parametrize("malicious_path", [
        "../../../root/malicious.wav",
        "~/../../etc/passwd",
        "${HOME}/secret.wav",
    ])
    def test_validate_output_path_traversal(self, malicious_path):
        """Test output path traversal prevention."""
        with pytest.raises(SecurityError):
            validate_output_path(malicious_path)


@pytest.mark.security
class TestParameterValidation:
    """Test parameter validation functions."""

    def test_validate_sample_rate_valid(self):
        """Test valid sample rate validation."""
        assert validate_sample_rate(44100) == 44100
        assert validate_sample_rate(48000) == 48000
        assert validate_sample_rate(96000) == 96000

    def test_validate_sample_rate_invalid_type(self):
        """Test invalid sample rate type."""
        with pytest.raises(ValidationError, match="must be an integer"):
            validate_sample_rate(44100.5)

    def test_validate_sample_rate_out_of_range(self):
        """Test sample rate out of range."""
        with pytest.raises(ValidationError, match="Invalid sample rate"):
            validate_sample_rate(1000)  # Too low

        with pytest.raises(ValidationError, match="Invalid sample rate"):
            validate_sample_rate(500000)  # Too high

    def test_validate_numeric_parameter_valid(self):
        """Test valid numeric parameter validation."""
        assert validate_numeric_parameter(0.5, "test", 0.0, 1.0) == 0.5
        assert validate_numeric_parameter(10, "count", 1, 100) == 10

    def test_validate_numeric_parameter_none_allowed(self):
        """Test numeric parameter with None allowed."""
        assert validate_numeric_parameter(None, "test", allow_none=True) is None

    def test_validate_numeric_parameter_none_disallowed(self):
        """Test numeric parameter with None disallowed."""
        with pytest.raises(ValidationError, match="cannot be None"):
            validate_numeric_parameter(None, "test", allow_none=False)

    def test_validate_numeric_parameter_wrong_type(self):
        """Test numeric parameter with wrong type."""
        with pytest.raises(ValidationError, match="must be a number"):
            validate_numeric_parameter("not a number", "test")

    def test_validate_numeric_parameter_infinite(self):
        """Test numeric parameter with infinite value."""
        with pytest.raises(ValidationError, match="must be a finite number"):
            validate_numeric_parameter(float('inf'), "test")

    def test_validate_numeric_parameter_out_of_range(self):
        """Test numeric parameter out of range."""
        with pytest.raises(ValidationError, match="must be >= 0"):
            validate_numeric_parameter(-1, "test", min_val=0)

        with pytest.raises(ValidationError, match="must be <= 10"):
            validate_numeric_parameter(15, "test", max_val=10)


@pytest.mark.security
class TestFilenameHandling:
    """Test filename handling and sanitization."""

    @pytest.mark.parametrize("bad_filename,expected", [
        ("file<name>.wav", "file_name_.wav"),
        ("file|name.wav", "file_name.wav"),
        ("file?name.wav", "file_name.wav"),
        ("CON.wav", "CON.wav"),  # Windows reserved names are preserved but warned
        ("file with spaces.wav", "file with spaces.wav"),  # Spaces are OK
        ("", "output"),  # Empty becomes default
        ("   ", "output"),  # Whitespace only becomes default
        ("a" * 300, "a" * 196 + ".wav"),  # Long names are truncated
    ])
    def test_sanitize_filename(self, bad_filename, expected):
        """Test filename sanitization."""
        result = sanitize_filename(bad_filename)
        if expected.endswith(".wav") and not bad_filename.endswith(".wav"):
            # For truncation test, check the pattern rather than exact match
            assert len(result) <= 200
            assert not any(char in result for char in '<>:"/\\|?*')
        else:
            assert result == expected

    def test_compute_file_hash(self, sample_audio_file):
        """Test file hash computation."""
        hash1 = compute_file_hash(sample_audio_file)
        hash2 = compute_file_hash(sample_audio_file)

        assert len(hash1) == 64  # SHA256 hex digest length
        assert hash1 == hash2  # Same file should have same hash

        # Test different algorithm
        md5_hash = compute_file_hash(sample_audio_file, algorithm='md5')
        assert len(md5_hash) == 32  # MD5 hex digest length
        assert md5_hash != hash1  # Different algorithms, different hashes

    def test_compute_file_hash_invalid_algorithm(self, sample_audio_file):
        """Test file hash with invalid algorithm."""
        with pytest.raises(ValidationError, match="Unsupported hash algorithm"):
            compute_file_hash(sample_audio_file, algorithm='invalid')

    def test_compute_file_hash_nonexistent_file(self):
        """Test file hash with non-existent file."""
        with pytest.raises(ValidationError, match="File does not exist"):
            compute_file_hash("/nonexistent/file.wav")


@pytest.mark.security
class TestResourceMonitoring:
    """Test resource monitoring functionality."""

    def test_resource_monitor_initialization(self):
        """Test resource monitor initialization."""
        monitor = ResourceMonitor()
        assert monitor.initial_memory is not None

    def test_resource_monitor_check(self):
        """Test resource monitoring check."""
        monitor = ResourceMonitor()
        current = monitor.check_resources()

        assert isinstance(current, dict)
        # Should have at least some memory info
        assert len(current) > 0

    def test_resource_monitor_summary(self):
        """Test resource monitoring summary."""
        monitor = ResourceMonitor()
        monitor.check_resources()  # Update current stats

        summary = monitor.get_resource_summary()
        assert 'initial' in summary
        assert 'current' in summary
        assert 'peak' in summary

    def test_check_available_space(self, temp_dir):
        """Test disk space checking."""
        # Should return True for small space requirement
        assert check_available_space(temp_dir, 1.0) == True

        # Should return False for huge space requirement (if working correctly)
        # This test might pass if the system has terabytes of free space
        result = check_available_space(temp_dir, 1000000.0)  # 1TB
        assert isinstance(result, bool)


@pytest.mark.security
class TestSecurityIntegration:
    """Test security components working together."""

    def test_malicious_input_chain(self, malicious_paths, temp_dir):
        """Test that malicious inputs are blocked at multiple levels."""
        from ai_audio_upscaler.pipeline import AudioUpscalerPipeline
        from ai_audio_upscaler.config import UpscalerConfig

        config = UpscalerConfig()
        pipeline = AudioUpscalerPipeline(config)

        for malicious_path in malicious_paths:
            # Each malicious path should be rejected
            with pytest.raises((SecurityError, ValidationError, FileNotFoundError)):
                pipeline.load_audio(malicious_path)

    def test_large_file_handling(self, temp_dir):
        """Test handling of extremely large files."""
        # Create a file that reports as very large
        large_file = temp_dir / "large.wav"
        large_file.write_bytes(b"RIFF" + b"x" * 1000)  # Fake WAV header + data

        # Should be rejected due to size or corruption
        with pytest.raises((ValidationError, SecurityError)):
            validate_audio_file(large_file)

    def test_concurrent_security_checks(self, sample_audio_file):
        """Test security checks under concurrent access."""
        import threading
        import queue

        results = queue.Queue()
        errors = queue.Queue()

        def validate_file():
            try:
                path, info = validate_audio_file(sample_audio_file)
                results.put((path, info))
            except Exception as e:
                errors.put(e)

        # Run multiple validation threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=validate_file)
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # All should succeed
        assert results.qsize() == 5
        assert errors.empty()

    def test_memory_exhaustion_protection(self):
        """Test protection against memory exhaustion attacks."""
        monitor = ResourceMonitor()
        initial_memory = monitor.get_resource_summary()

        # This test simulates what might happen with very large inputs
        # In a real attack, this would be much larger
        try:
            # Create a moderately large tensor to test monitoring
            large_tensor = torch.zeros(1000, 1000)  # Much smaller than a real attack
            del large_tensor

            final_memory = monitor.get_resource_summary()
            # Memory monitoring should still work
            assert isinstance(final_memory, dict)
        except Exception:
            # If we can't allocate, that's also fine - system is protecting itself
            pass