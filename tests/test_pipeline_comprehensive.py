"""
Comprehensive pipeline tests for AI Audio Upscaler Pro.
"""

import pytest
import torch
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from ai_audio_upscaler.pipeline import AudioUpscalerPipeline
from ai_audio_upscaler.config import UpscalerConfig
from ai_audio_upscaler.security import SecurityError, ValidationError


@pytest.mark.integration
class TestPipelineBasics:
    """Test basic pipeline functionality."""

    def test_pipeline_initialization_baseline(self, default_config):
        """Test pipeline initialization in baseline mode."""
        pipeline = AudioUpscalerPipeline(default_config)

        assert pipeline.config == default_config
        assert pipeline.dsp is not None
        assert pipeline.ai_model is None  # No AI model in baseline mode

    def test_pipeline_initialization_ai(self, ai_config):
        """Test pipeline initialization in AI mode."""
        pipeline = AudioUpscalerPipeline(ai_config)

        assert pipeline.config == ai_config
        assert pipeline.dsp is not None
        assert pipeline.ai_model is not None  # AI model should be loaded

    def test_load_audio_valid(self, pipeline, sample_audio_file):
        """Test loading valid audio file."""
        waveform, sample_rate = pipeline.load_audio(str(sample_audio_file))

        assert isinstance(waveform, torch.Tensor)
        assert waveform.dim() == 2  # (channels, samples)
        assert sample_rate == 44100

    def test_load_audio_nonexistent(self, pipeline):
        """Test loading non-existent file."""
        with pytest.raises(FileNotFoundError):
            pipeline.load_audio("/nonexistent/file.wav")

    def test_save_audio(self, pipeline, temp_dir, sample_audio_mono):
        """Test saving audio file."""
        output_path = temp_dir / "output.wav"
        pipeline.save_audio(sample_audio_mono, str(output_path), 44100)

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_save_audio_directory_creation(self, pipeline, temp_dir, sample_audio_mono):
        """Test saving audio with directory creation."""
        output_path = temp_dir / "subdir" / "deep" / "output.wav"
        pipeline.save_audio(sample_audio_mono, str(output_path), 44100)

        assert output_path.exists()
        assert output_path.parent.exists()


@pytest.mark.integration
class TestPipelineProcessing:
    """Test pipeline processing functionality."""

    def test_run_baseline_default(self, pipeline, sample_audio_file, temp_dir):
        """Test running baseline processing with default settings."""
        output_path = temp_dir / "output.wav"

        result = pipeline.run(
            input_path=str(sample_audio_file),
            output_path=str(output_path)
        )

        assert "output_path" in result
        assert Path(result["output_path"]).exists()

    def test_run_baseline_with_analysis(self, pipeline, sample_audio_file, temp_dir):
        """Test running baseline processing with analysis generation."""
        output_path = temp_dir / "output.wav"

        result = pipeline.run(
            input_path=str(sample_audio_file),
            output_path=str(output_path),
            generate_analysis=True
        )

        assert "output_path" in result
        assert "input_spectrogram" in result
        assert "output_spectrogram" in result
        assert "input_psd" in result
        assert "output_psd" in result

    def test_run_different_sample_rates(self, temp_dir, sample_audio_file):
        """Test running with different target sample rates."""
        sample_rates = [48000, 96000, 192000]

        for target_sr in sample_rates:
            config = UpscalerConfig(target_sample_rate=target_sr)
            pipeline = AudioUpscalerPipeline(config)
            output_path = temp_dir / f"output_{target_sr}.wav"

            result = pipeline.run(
                input_path=str(sample_audio_file),
                output_path=str(output_path)
            )

            assert Path(result["output_path"]).exists()

    def test_run_different_normalization_modes(self, pipeline, sample_audio_file, temp_dir):
        """Test running with different normalization modes."""
        modes = ["Peak -1dB", "Peak -3dB", "None"]

        for mode in modes:
            output_path = temp_dir / f"output_{mode.replace(' ', '_').replace('-', 'minus')}.wav"

            result = pipeline.run(
                input_path=str(sample_audio_file),
                output_path=str(output_path),
                normalization_mode=mode
            )

            assert Path(result["output_path"]).exists()

    @pytest.mark.slow
    def test_run_large_file(self, pipeline, large_audio_file, temp_dir):
        """Test processing large audio file."""
        output_path = temp_dir / "large_output.wav"

        result = pipeline.run(
            input_path=str(large_audio_file),
            output_path=str(output_path)
        )

        assert Path(result["output_path"]).exists()

    def test_run_with_progress_callback(self, pipeline, sample_audio_file, temp_dir):
        """Test running with progress callback."""
        output_path = temp_dir / "output.wav"
        progress_calls = []

        def progress_callback(progress, message):
            progress_calls.append((progress, message))

        result = pipeline.run(
            input_path=str(sample_audio_file),
            output_path=str(output_path),
            progress_callback=progress_callback
        )

        assert len(progress_calls) > 0
        assert all(0.0 <= p <= 1.0 for p, _ in progress_calls)
        assert all(isinstance(m, str) for _, m in progress_calls)


@pytest.mark.integration
class TestPipelineParameterValidation:
    """Test pipeline parameter validation."""

    def test_run_invalid_input_path(self, pipeline, temp_dir):
        """Test running with invalid input path."""
        output_path = temp_dir / "output.wav"

        with pytest.raises((ValidationError, SecurityError, FileNotFoundError)):
            pipeline.run(
                input_path="/nonexistent/file.wav",
                output_path=str(output_path)
            )

    def test_run_invalid_output_path(self, pipeline, sample_audio_file):
        """Test running with invalid output path."""
        with pytest.raises((ValidationError, SecurityError)):
            pipeline.run(
                input_path=str(sample_audio_file),
                output_path="../../../etc/passwd"
            )

    def test_run_invalid_normalization_mode(self, pipeline, sample_audio_file, temp_dir):
        """Test running with invalid normalization mode."""
        output_path = temp_dir / "output.wav"

        with pytest.raises(ValidationError, match="Invalid normalization mode"):
            pipeline.run(
                input_path=str(sample_audio_file),
                output_path=str(output_path),
                normalization_mode="Invalid Mode"
            )

    def test_run_invalid_numeric_parameters(self, pipeline, sample_audio_file, temp_dir):
        """Test running with invalid numeric parameters."""
        output_path = temp_dir / "output.wav"

        # Test various invalid numeric parameters
        invalid_params = [
            {"transient_strength": -0.1},  # Below minimum
            {"saturation": 1.5},  # Above maximum
            {"stereo_width": -1.0},  # Below minimum
            {"judge_threshold": 2.0},  # Above maximum
            {"candidate_count": 0},  # Below minimum
        ]

        for params in invalid_params:
            with pytest.raises(ValidationError):
                pipeline.run(
                    input_path=str(sample_audio_file),
                    output_path=str(output_path),
                    **params
                )

    def test_run_invalid_stereo_mode(self, pipeline, sample_audio_file, temp_dir):
        """Test running with invalid stereo mode."""
        output_path = temp_dir / "output.wav"

        with pytest.raises(ValidationError, match="Invalid stereo_mode"):
            pipeline.run(
                input_path=str(sample_audio_file),
                output_path=str(output_path),
                stereo_mode="invalid_mode"
            )


@pytest.mark.integration
class TestPipelineAdvancedFeatures:
    """Test advanced pipeline features."""

    def test_run_advanced_dsp(self, sample_audio_file, temp_dir):
        """Test running with advanced DSP features."""
        config = UpscalerConfig(
            use_advanced_dsp=True,
            dsp_quality_preset="quality",
            baseline_method="poly-sinc-hq"
        )
        pipeline = AudioUpscalerPipeline(config)
        output_path = temp_dir / "output_advanced.wav"

        result = pipeline.run(
            input_path=str(sample_audio_file),
            output_path=str(output_path)
        )

        assert Path(result["output_path"]).exists()

    def test_run_with_mastering_chain(self, pipeline, sample_audio_file, temp_dir):
        """Test running with mastering chain parameters."""
        output_path = temp_dir / "output_mastered.wav"

        result = pipeline.run(
            input_path=str(sample_audio_file),
            output_path=str(output_path),
            remove_dc=True,
            normalize_input=True,
            limit=True,
            saturation=0.1,
            stereo_width=1.2,
            transient_shaping=0.2
        )

        assert Path(result["output_path"]).exists()

    @pytest.mark.gpu
    def test_run_ai_mode_cpu(self, sample_audio_file, temp_dir):
        """Test AI mode processing on CPU."""
        config = UpscalerConfig(mode="ai", device="cpu")
        pipeline = AudioUpscalerPipeline(config)
        output_path = temp_dir / "output_ai_cpu.wav"

        result = pipeline.run(
            input_path=str(sample_audio_file),
            output_path=str(output_path)
        )

        assert Path(result["output_path"]).exists()

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_run_ai_mode_cuda(self, sample_audio_file, temp_dir):
        """Test AI mode processing on CUDA."""
        config = UpscalerConfig(mode="ai", device="cuda")
        pipeline = AudioUpscalerPipeline(config)
        output_path = temp_dir / "output_ai_cuda.wav"

        result = pipeline.run(
            input_path=str(sample_audio_file),
            output_path=str(output_path)
        )

        assert Path(result["output_path"]).exists()


@pytest.mark.integration
class TestPipelineErrorHandling:
    """Test pipeline error handling."""

    def test_run_corrupted_audio(self, pipeline, corrupted_audio_file, temp_dir):
        """Test handling of corrupted audio files."""
        output_path = temp_dir / "output.wav"

        with pytest.raises(ValidationError):
            pipeline.run(
                input_path=str(corrupted_audio_file),
                output_path=str(output_path)
            )

    def test_run_permission_error(self, pipeline, sample_audio_file, temp_dir):
        """Test handling of permission errors."""
        # Create a read-only directory
        try:
            readonly_dir = temp_dir / "readonly"
            readonly_dir.mkdir()
            readonly_dir.chmod(0o444)

            output_path = readonly_dir / "output.wav"

            with pytest.raises(ValidationError):
                pipeline.run(
                    input_path=str(sample_audio_file),
                    output_path=str(output_path)
                )
        except (OSError, PermissionError):
            pytest.skip("Cannot test permission errors on this system")

    def test_run_disk_space_exhaustion(self, pipeline, sample_audio_file, temp_dir):
        """Test handling of disk space exhaustion."""
        output_path = temp_dir / "output.wav"

        # Mock check_available_space to return False
        with patch('ai_audio_upscaler.security.check_available_space', return_value=False):
            # The pipeline should still work, but might warn about space
            result = pipeline.run(
                input_path=str(sample_audio_file),
                output_path=str(output_path)
            )
            # Should still succeed since we're not actually out of space
            assert Path(result["output_path"]).exists()

    def test_run_memory_pressure(self, pipeline, sample_audio_file, temp_dir):
        """Test pipeline behavior under memory pressure."""
        output_path = temp_dir / "output.wav"

        # Mock memory manager to report low memory
        with patch.object(pipeline, 'memory_manager') as mock_manager:
            mock_manager.get_memory_info.return_value = {
                'available': 0.1,  # Very low available memory
                'allocated': 1.0,
                'reserved': 1.0,
                'free': 0.1,
            }
            mock_manager.can_allocate.return_value = False

            # Should still work with fallbacks
            result = pipeline.run(
                input_path=str(sample_audio_file),
                output_path=str(output_path)
            )

            assert Path(result["output_path"]).exists()


@pytest.mark.performance
class TestPipelinePerformance:
    """Test pipeline performance characteristics."""

    def test_processing_speed_baseline(self, pipeline, sample_audio_file, temp_dir, performance_monitor):
        """Test baseline processing speed."""
        output_path = temp_dir / "output.wav"

        performance_monitor.start()

        result = pipeline.run(
            input_path=str(sample_audio_file),
            output_path=str(output_path)
        )

        metrics = performance_monitor.stop()

        assert Path(result["output_path"]).exists()
        assert metrics["duration"] < 10.0  # Should complete within 10 seconds for 1s audio
        assert metrics["memory_increase_mb"] < 500  # Reasonable memory usage

    @pytest.mark.slow
    def test_memory_usage_large_file(self, pipeline, large_audio_file, temp_dir, performance_monitor):
        """Test memory usage with large files."""
        output_path = temp_dir / "large_output.wav"

        performance_monitor.start()

        result = pipeline.run(
            input_path=str(large_audio_file),
            output_path=str(output_path)
        )

        metrics = performance_monitor.stop()

        assert Path(result["output_path"]).exists()
        # Memory should be reasonable even for large files
        assert metrics["memory_increase_mb"] < 2000  # 2GB limit

    def test_concurrent_processing(self, default_config, sample_audio_file, temp_dir):
        """Test concurrent pipeline processing."""
        import threading
        import queue

        results_queue = queue.Queue()
        errors_queue = queue.Queue()

        def process_audio(thread_id):
            try:
                pipeline = AudioUpscalerPipeline(default_config)
                output_path = temp_dir / f"output_{thread_id}.wav"

                result = pipeline.run(
                    input_path=str(sample_audio_file),
                    output_path=str(output_path)
                )
                results_queue.put((thread_id, result))
            except Exception as e:
                errors_queue.put((thread_id, e))

        # Start multiple threads
        threads = []
        for i in range(3):  # Limited number to avoid overwhelming system
            thread = threading.Thread(target=process_audio, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join(timeout=30)

        # Check results
        assert results_queue.qsize() == 3
        assert errors_queue.empty()

        # Verify all output files exist
        for i in range(3):
            output_path = temp_dir / f"output_{i}.wav"
            assert output_path.exists()


@pytest.mark.integration
class TestPipelineFormats:
    """Test pipeline with different audio formats."""

    def test_run_different_input_formats(self, audio_formats_samples, temp_dir):
        """Test processing different input audio formats."""
        config = UpscalerConfig()
        pipeline = AudioUpscalerPipeline(config)

        for format_name, input_path in audio_formats_samples.items():
            output_path = temp_dir / f"output_{format_name}.wav"

            result = pipeline.run(
                input_path=str(input_path),
                output_path=str(output_path)
            )

            assert Path(result["output_path"]).exists()

    def test_run_different_output_formats(self, pipeline, sample_audio_file, temp_dir):
        """Test generating different output formats."""
        formats = ["wav", "flac"]  # Only test formats likely to be available

        for fmt in formats:
            config = UpscalerConfig(export_format=fmt)
            format_pipeline = AudioUpscalerPipeline(config)
            output_path = temp_dir / f"output.{fmt}"

            try:
                result = format_pipeline.run(
                    input_path=str(sample_audio_file),
                    output_path=str(output_path)
                )

                # The pipeline might adjust the extension
                assert any(Path(p).exists() for p in [
                    str(output_path),
                    str(output_path.with_suffix(f".{fmt}"))
                ])
            except Exception as e:
                # Some formats might not be available on all systems
                if "format" in str(e).lower() or "codec" in str(e).lower():
                    pytest.skip(f"Format {fmt} not available on this system")
                else:
                    raise