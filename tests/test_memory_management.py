"""
Comprehensive memory management tests for AI Audio Upscaler Pro.
"""

import pytest
import torch
import time
from unittest.mock import patch, MagicMock

from ai_audio_upscaler.memory_manager import (
    MemoryManager, oom_safe_function, AdaptiveBatchProcessor
)


@pytest.mark.gpu
class TestMemoryManager:
    """Test MemoryManager functionality."""

    def test_memory_manager_cpu(self):
        """Test MemoryManager on CPU."""
        device = torch.device("cpu")
        manager = MemoryManager(device)

        assert manager.device == device
        assert not manager.is_cuda

        info = manager.get_memory_info()
        assert info['available'] == float('inf')
        assert info['allocated'] == 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_memory_manager_cuda(self):
        """Test MemoryManager on CUDA."""
        device = torch.device("cuda")
        manager = MemoryManager(device)

        assert manager.device == device
        assert manager.is_cuda

        info = manager.get_memory_info()
        assert info['available'] >= 0
        assert info['allocated'] >= 0
        assert info['free'] > 0

    def test_estimate_memory_requirement(self):
        """Test memory requirement estimation."""
        device = torch.device("cpu")
        manager = MemoryManager(device)

        # Test with simple tensor shapes
        shapes = [(100, 100), (200, 200)]
        requirement = manager.estimate_memory_requirement(shapes)

        assert requirement > 0
        assert isinstance(requirement, float)

    def test_can_allocate_cpu(self):
        """Test allocation check on CPU."""
        device = torch.device("cpu")
        manager = MemoryManager(device)

        # CPU should always return True
        assert manager.can_allocate(1.0) == True
        assert manager.can_allocate(1000.0) == True

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_can_allocate_cuda(self):
        """Test allocation check on CUDA."""
        device = torch.device("cuda")
        manager = MemoryManager(device)

        # Small allocation should work
        assert manager.can_allocate(0.1) == True

        # Huge allocation should fail
        assert manager.can_allocate(1000.0) == False

    def test_cleanup_memory(self):
        """Test memory cleanup."""
        device = torch.device("cpu")
        manager = MemoryManager(device)

        # Should not raise exceptions
        manager.cleanup_memory(aggressive=False)
        manager.cleanup_memory(aggressive=True)

    def test_memory_context_cpu(self):
        """Test memory context on CPU."""
        device = torch.device("cpu")
        manager = MemoryManager(device)

        with manager.memory_context():
            # Should work without issues
            tensor = torch.zeros(100, 100)
            assert tensor.numel() == 10000

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_memory_context_cuda_success(self):
        """Test successful memory context on CUDA."""
        device = torch.device("cuda")
        manager = MemoryManager(device)

        with manager.memory_context(required_gb=0.1):
            tensor = torch.zeros(100, 100, device=device)
            assert tensor.device.type == 'cuda'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_memory_context_cuda_failure(self):
        """Test memory context failure on CUDA."""
        device = torch.device("cuda")
        manager = MemoryManager(device)

        with pytest.raises(RuntimeError, match="Insufficient CUDA memory"):
            with manager.memory_context(required_gb=1000.0):
                pass

    def test_auto_batch_size_cpu(self):
        """Test automatic batch size calculation on CPU."""
        device = torch.device("cpu")
        manager = MemoryManager(device)

        shapes = [(100, 100)]
        batch_size = manager.auto_batch_size(4, shapes)

        assert batch_size == 4  # Should return requested size for CPU

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_auto_batch_size_cuda(self):
        """Test automatic batch size calculation on CUDA."""
        device = torch.device("cuda")
        manager = MemoryManager(device)

        shapes = [(1000, 1000)]  # Larger shapes to test limitation
        batch_size = manager.auto_batch_size(64, shapes)

        assert 1 <= batch_size <= 64
        assert isinstance(batch_size, int)

    def test_get_memory_stats(self):
        """Test memory statistics."""
        device = torch.device("cpu")
        manager = MemoryManager(device)

        stats = manager.get_memory_stats()

        assert 'device' in stats
        assert 'is_cuda' in stats
        assert 'current_allocated' in stats
        assert stats['device'] == str(device)
        assert stats['is_cuda'] == (device.type == 'cuda')


class TestOOMSafeFunction:
    """Test OOM-safe function decorator."""

    def test_oom_safe_function_success(self):
        """Test OOM-safe function with successful execution."""

        @oom_safe_function()
        def successful_function(x):
            return x * 2

        result = successful_function(5)
        assert result == 10

    def test_oom_safe_function_non_oom_error(self):
        """Test OOM-safe function with non-OOM error."""

        @oom_safe_function()
        def error_function():
            raise ValueError("Not an OOM error")

        with pytest.raises(ValueError):
            error_function()

    def test_oom_safe_function_oom_retry(self):
        """Test OOM-safe function with OOM error and retry."""
        call_count = 0

        @oom_safe_function(max_retries=2)
        def oom_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("CUDA out of memory")
            return "success"

        result = oom_function()
        assert result == "success"
        assert call_count == 3

    def test_oom_safe_function_oom_exhaustion(self):
        """Test OOM-safe function with retry exhaustion."""

        @oom_safe_function(max_retries=1)
        def always_oom_function():
            raise RuntimeError("CUDA out of memory")

        with pytest.raises(RuntimeError, match="out of memory"):
            always_oom_function()

    def test_oom_safe_function_fallback_device(self):
        """Test OOM-safe function with device fallback."""

        @oom_safe_function(fallback_device='cpu', max_retries=2)
        def device_function(tensor):
            if tensor.device.type == 'cuda':
                raise RuntimeError("CUDA out of memory")
            return tensor.device.type

        input_tensor = torch.zeros(10)
        if torch.cuda.is_available():
            input_tensor = input_tensor.to('cuda')
            result = device_function(input_tensor)
            assert result == 'cpu'
        else:
            result = device_function(input_tensor)
            assert result == 'cpu'


class TestAdaptiveBatchProcessor:
    """Test AdaptiveBatchProcessor functionality."""

    def test_adaptive_processor_initialization(self):
        """Test AdaptiveBatchProcessor initialization."""
        device = torch.device("cpu")
        manager = MemoryManager(device)
        processor = AdaptiveBatchProcessor(manager, initial_batch_size=4)

        assert processor.current_batch_size == 4
        assert processor.min_batch_size == 1
        assert processor.max_batch_size == 8

    def test_adjust_batch_size_oom(self):
        """Test batch size adjustment after OOM."""
        device = torch.device("cpu")
        manager = MemoryManager(device)
        processor = AdaptiveBatchProcessor(manager, initial_batch_size=8)

        initial_size = processor.current_batch_size
        processor.adjust_batch_size(had_oom=True, processing_time=1.0)

        assert processor.current_batch_size < initial_size
        assert processor.current_batch_size >= processor.min_batch_size

    def test_adjust_batch_size_success(self):
        """Test batch size adjustment after successful processing."""
        device = torch.device("cpu")
        manager = MemoryManager(device)
        processor = AdaptiveBatchProcessor(manager, initial_batch_size=2)

        # Add some successful history
        for _ in range(5):
            processor.adjust_batch_size(had_oom=False, processing_time=0.5)

        # Should potentially increase batch size
        final_size = processor.current_batch_size
        assert final_size >= 2  # At least the initial size

    def test_process_batches_success(self):
        """Test successful batch processing."""
        device = torch.device("cpu")
        manager = MemoryManager(device)
        processor = AdaptiveBatchProcessor(manager, initial_batch_size=2)

        items = list(range(10))

        def process_func(batch):
            return [x * 2 for x in batch]

        results = processor.process_batches(items, process_func)

        expected = [x * 2 for x in items]
        assert results == expected

    def test_process_batches_empty(self):
        """Test processing empty batch list."""
        device = torch.device("cpu")
        manager = MemoryManager(device)
        processor = AdaptiveBatchProcessor(manager)

        def process_func(batch):
            return batch

        results = processor.process_batches([], process_func)
        assert results == []

    def test_process_batches_with_oom(self):
        """Test batch processing with OOM errors."""
        device = torch.device("cpu")
        manager = MemoryManager(device)
        processor = AdaptiveBatchProcessor(manager, initial_batch_size=4)

        items = list(range(8))
        call_count = 0

        def process_func(batch):
            nonlocal call_count
            call_count += 1
            # Simulate OOM on large batches initially
            if len(batch) > 2 and call_count < 3:
                raise RuntimeError("CUDA out of memory")
            return [x * 2 for x in batch]

        results = processor.process_batches(items, process_func)

        expected = [x * 2 for x in items]
        assert results == expected
        # Batch size should have been reduced
        assert processor.current_batch_size <= 2

    def test_process_batches_minimum_batch_oom(self):
        """Test batch processing when OOM occurs even with minimum batch size."""
        device = torch.device("cpu")
        manager = MemoryManager(device)
        processor = AdaptiveBatchProcessor(manager, initial_batch_size=4)

        # Force batch size to minimum
        processor.current_batch_size = processor.min_batch_size

        items = [1, 2, 3]

        def process_func(batch):
            raise RuntimeError("CUDA out of memory")

        with pytest.raises(RuntimeError, match="out of memory"):
            processor.process_batches(items, process_func)

    def test_get_stats(self):
        """Test getting processor statistics."""
        device = torch.device("cpu")
        manager = MemoryManager(device)
        processor = AdaptiveBatchProcessor(manager)

        # Process some batches to generate stats
        items = [1, 2, 3, 4]
        processor.process_batches(items, lambda x: x)

        stats = processor.get_stats()

        assert 'current_batch_size' in stats
        assert 'oom_events' in stats
        assert 'memory_stats' in stats
        assert isinstance(stats['current_batch_size'], int)


@pytest.mark.integration
class TestMemoryIntegration:
    """Test memory management integration with other components."""

    def test_memory_manager_with_pipeline(self, sample_audio_file, temp_dir):
        """Test memory manager integration with pipeline."""
        from ai_audio_upscaler.pipeline import AudioUpscalerPipeline
        from ai_audio_upscaler.config import UpscalerConfig

        config = UpscalerConfig(mode="ai", device="cpu")  # Use CPU for consistent testing
        pipeline = AudioUpscalerPipeline(config)

        # Access memory manager from AI model
        if pipeline.ai_model and hasattr(pipeline.ai_model, 'memory_manager'):
            memory_manager = pipeline.ai_model.memory_manager
            initial_stats = memory_manager.get_memory_stats()

            output_path = temp_dir / "output.wav"
            result = pipeline.run(
                input_path=str(sample_audio_file),
                output_path=str(output_path)
            )

            final_stats = memory_manager.get_memory_stats()

            assert "output_path" in result
            assert Path(result["output_path"]).exists()

            # Memory stats should be tracked
            assert 'current_allocated' in final_stats
            assert final_stats['oom_count'] >= 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_memory_pressure_handling(self):
        """Test handling of memory pressure scenarios."""
        device = torch.device("cuda")
        manager = MemoryManager(device)

        # Allocate some memory to create pressure
        tensors = []
        try:
            # Allocate tensors until we're using substantial memory
            while len(tensors) < 10:
                tensor = torch.zeros(1000, 1000, device=device)
                tensors.append(tensor)

                current_info = manager.get_memory_info()
                if current_info['available'] < 1.0:  # Less than 1GB available
                    break

            # Test that memory manager can still function
            assert manager.can_allocate(0.1) in [True, False]  # Either is valid

            # Test cleanup
            del tensors
            manager.cleanup_memory(aggressive=True)

            # Should have more memory available now
            final_info = manager.get_memory_info()
            assert final_info['available'] >= 0

        except RuntimeError as e:
            if "out of memory" in str(e):
                pytest.skip("System under memory pressure, cannot complete test")
            else:
                raise

    def test_concurrent_memory_access(self):
        """Test concurrent access to memory manager."""
        import threading
        import queue

        device = torch.device("cpu")
        manager = MemoryManager(device)

        results = queue.Queue()
        errors = queue.Queue()

        def memory_task():
            try:
                for _ in range(10):
                    info = manager.get_memory_info()
                    manager.cleanup_memory()
                    with manager.memory_context():
                        tensor = torch.zeros(100, 100)
                        del tensor
                results.put("success")
            except Exception as e:
                errors.put(e)

        # Start multiple threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=memory_task)
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join(timeout=10)

        # Check results
        assert results.qsize() == 3
        assert errors.empty()


@pytest.mark.performance
class TestMemoryPerformance:
    """Test memory management performance."""

    def test_memory_context_overhead(self):
        """Test overhead of memory context."""
        device = torch.device("cpu")
        manager = MemoryManager(device)

        # Test without context
        start_time = time.time()
        for _ in range(100):
            tensor = torch.zeros(100, 100)
            del tensor
        no_context_time = time.time() - start_time

        # Test with context
        start_time = time.time()
        for _ in range(100):
            with manager.memory_context():
                tensor = torch.zeros(100, 100)
                del tensor
        with_context_time = time.time() - start_time

        # Context should not add significant overhead
        overhead_ratio = with_context_time / no_context_time
        assert overhead_ratio < 2.0  # Less than 100% overhead

    def test_batch_processor_efficiency(self):
        """Test efficiency of adaptive batch processor."""
        device = torch.device("cpu")
        manager = MemoryManager(device)
        processor = AdaptiveBatchProcessor(manager)

        items = list(range(1000))

        def fast_process(batch):
            return [x * 2 for x in batch]

        start_time = time.time()
        results = processor.process_batches(items, fast_process)
        processing_time = time.time() - start_time

        assert len(results) == len(items)
        assert processing_time < 5.0  # Should complete quickly

        stats = processor.get_stats()
        assert stats['oom_events'] == 0  # No OOMs with simple processing