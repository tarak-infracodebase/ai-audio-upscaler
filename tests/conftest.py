"""
Pytest configuration and fixtures for AI Audio Upscaler Pro tests.
"""

import os
import tempfile
import pytest
import torch
import numpy as np
from pathlib import Path
from typing import Generator, Tuple

# Import the main modules
from ai_audio_upscaler.config import UpscalerConfig
from ai_audio_upscaler.pipeline import AudioUpscalerPipeline
from ai_audio_upscaler.security import ValidationError, SecurityError


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "security: marks security-related tests")


@pytest.fixture(scope="session")
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory(prefix="ai_upscaler_test_") as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_audio_mono() -> torch.Tensor:
    """Generate a sample mono audio tensor for testing."""
    # Generate 1 second of 440Hz sine wave at 44.1kHz
    sample_rate = 44100
    duration = 1.0
    frequency = 440.0

    t = torch.linspace(0, duration, int(sample_rate * duration))
    audio = torch.sin(2 * torch.pi * frequency * t).unsqueeze(0)  # (1, samples)

    return audio


@pytest.fixture
def sample_audio_stereo() -> torch.Tensor:
    """Generate a sample stereo audio tensor for testing."""
    # Generate 1 second of 440Hz sine wave at 44.1kHz (left) and 880Hz (right)
    sample_rate = 44100
    duration = 1.0

    t = torch.linspace(0, duration, int(sample_rate * duration))
    left = torch.sin(2 * torch.pi * 440.0 * t)
    right = torch.sin(2 * torch.pi * 880.0 * t)

    audio = torch.stack([left, right], dim=0)  # (2, samples)

    return audio


@pytest.fixture
def sample_audio_file(temp_dir: Path, sample_audio_mono: torch.Tensor) -> Path:
    """Create a temporary audio file for testing."""
    import torchaudio

    audio_path = temp_dir / "test_audio.wav"
    torchaudio.save(str(audio_path), sample_audio_mono, sample_rate=44100)

    return audio_path


@pytest.fixture
def corrupted_audio_file(temp_dir: Path) -> Path:
    """Create a corrupted audio file for testing error handling."""
    corrupted_path = temp_dir / "corrupted.wav"
    with open(corrupted_path, "wb") as f:
        f.write(b"This is not a valid audio file")

    return corrupted_path


@pytest.fixture
def large_audio_file(temp_dir: Path) -> Path:
    """Create a large audio file for testing memory management."""
    import torchaudio

    # Generate 30 seconds of audio to test chunking
    sample_rate = 44100
    duration = 30.0
    t = torch.linspace(0, duration, int(sample_rate * duration))
    audio = torch.sin(2 * torch.pi * 440.0 * t).unsqueeze(0)

    large_path = temp_dir / "large_test.wav"
    torchaudio.save(str(large_path), audio, sample_rate=sample_rate)

    return large_path


@pytest.fixture
def default_config() -> UpscalerConfig:
    """Create a default configuration for testing."""
    return UpscalerConfig(
        target_sample_rate=48000,
        mode="baseline",
        baseline_method="sinc",
        device="cpu",  # Use CPU for tests by default
        export_format="wav"
    )


@pytest.fixture
def ai_config() -> UpscalerConfig:
    """Create an AI configuration for testing."""
    return UpscalerConfig(
        target_sample_rate=48000,
        mode="ai",
        baseline_method="sinc",
        device="cpu",
        model_checkpoint=None,  # Will use random weights
        export_format="wav"
    )


@pytest.fixture
def pipeline(default_config: UpscalerConfig) -> AudioUpscalerPipeline:
    """Create a pipeline instance for testing."""
    return AudioUpscalerPipeline(default_config)


@pytest.fixture
def ai_pipeline(ai_config: UpscalerConfig) -> AudioUpscalerPipeline:
    """Create an AI pipeline instance for testing."""
    return AudioUpscalerPipeline(ai_config)


@pytest.fixture
def malicious_paths() -> list:
    """Generate malicious file paths for security testing."""
    return [
        "../../../etc/passwd",
        "..\\..\\..\\windows\\system32\\config\\sam",
        "/dev/null",
        "/proc/self/environ",
        "file://etc/passwd",
        "http://evil.com/malware.wav",
        "\\\\server\\share\\file.wav",
        "CON",  # Windows reserved name
        "PRN",  # Windows reserved name
        "A" * 1000,  # Extremely long filename
        "",  # Empty string
        "   ",  # Whitespace only
        "/tmp/../../../root/.ssh/id_rsa",
        "${HOME}/secret.wav",
        "`rm -rf /`",
        "$(echo hello)",
    ]


@pytest.fixture(scope="session")
def gpu_available() -> bool:
    """Check if GPU is available for testing."""
    return torch.cuda.is_available()


@pytest.fixture
def mock_model_checkpoint(temp_dir: Path) -> Path:
    """Create a mock model checkpoint for testing."""
    checkpoint_path = temp_dir / "mock_model.pth"

    # Create a minimal checkpoint structure
    mock_checkpoint = {
        "config": {
            "base_channels": 32,
            "num_layers": 4,
            "use_spectral": False,
            "use_diffusion": False,
            "diffusion_steps": 1000,
        },
        "state_dict": {}  # Empty state dict for testing
    }

    torch.save(mock_checkpoint, checkpoint_path)
    return checkpoint_path


@pytest.fixture
def performance_monitor():
    """Monitor performance metrics during tests."""
    import psutil
    import time

    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.start_memory = None

        def start(self):
            self.start_time = time.time()
            self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        def stop(self):
            if self.start_time is None:
                return {}

            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

            return {
                "duration": end_time - self.start_time,
                "memory_start_mb": self.start_memory,
                "memory_end_mb": end_memory,
                "memory_increase_mb": end_memory - self.start_memory
            }

    return PerformanceMonitor()


@pytest.fixture
def sample_spectrograms(sample_audio_mono: torch.Tensor) -> dict:
    """Generate sample spectrograms for analysis testing."""
    import torchaudio.transforms as T

    spectrograms = {}

    # Short-time Fourier Transform
    stft = T.Spectrogram(
        n_fft=2048,
        hop_length=512,
        win_length=2048,
        power=2.0
    )

    spectrograms['magnitude'] = stft(sample_audio_mono)

    # Mel-scale spectrogram
    mel_transform = T.MelSpectrogram(
        sample_rate=44100,
        n_fft=2048,
        hop_length=512,
        n_mels=128
    )

    spectrograms['mel'] = mel_transform(sample_audio_mono)

    return spectrograms


@pytest.fixture(autouse=True)
def setup_logging():
    """Configure logging for tests."""
    import logging

    # Set up basic logging configuration for tests
    logging.basicConfig(
        level=logging.WARNING,  # Reduce noise during testing
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Reduce verbosity of specific loggers during tests
    logging.getLogger('matplotlib').setLevel(logging.ERROR)
    logging.getLogger('PIL').setLevel(logging.ERROR)


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Mark slow tests
        if "slow" in item.name or "large" in item.name or "performance" in item.name:
            item.add_marker(pytest.mark.slow)

        # Mark GPU tests
        if "gpu" in item.name or "cuda" in item.name:
            item.add_marker(pytest.mark.gpu)

        # Mark integration tests
        if "integration" in item.name or "end_to_end" in item.name:
            item.add_marker(pytest.mark.integration)

        # Mark security tests
        if "security" in item.name or "malicious" in item.name or "validate" in item.name:
            item.add_marker(pytest.mark.security)


@pytest.fixture
def audio_formats_samples(temp_dir: Path, sample_audio_mono: torch.Tensor) -> dict:
    """Create sample audio files in different formats for testing."""
    import torchaudio

    formats = {}
    base_name = "test_audio"

    # WAV format
    wav_path = temp_dir / f"{base_name}.wav"
    torchaudio.save(str(wav_path), sample_audio_mono, sample_rate=44100, format="wav")
    formats['wav'] = wav_path

    # Try to create other formats if possible
    try:
        flac_path = temp_dir / f"{base_name}.flac"
        torchaudio.save(str(flac_path), sample_audio_mono, sample_rate=44100, format="flac")
        formats['flac'] = flac_path
    except Exception:
        pass  # FLAC not available

    return formats