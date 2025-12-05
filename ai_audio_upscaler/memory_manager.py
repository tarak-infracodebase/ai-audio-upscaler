"""
Advanced CUDA memory management and OOM fallback system.

This module provides comprehensive memory management for AI Audio Upscaler Pro,
including OOM prevention, graceful fallbacks, and memory optimization strategies.
"""

import torch
import logging
import gc
import time
from typing import Optional, Dict, Any, Callable, TypeVar, Tuple
from contextlib import contextmanager
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')

class MemoryManager:
    """
    Advanced CUDA memory manager with automatic fallbacks and optimization.
    """

    def __init__(self, device: torch.device, safety_margin_gb: float = 1.5):
        self.device = device
        self.safety_margin_bytes = int(safety_margin_gb * 1024**3)
        self.is_cuda = device.type == 'cuda'
        self.memory_stats = {'peak_allocated': 0, 'peak_reserved': 0, 'oom_count': 0, 'fallback_count': 0}

        if self.is_cuda:
            self._log_initial_memory_state()

    def _log_initial_memory_state(self):
        """Log initial GPU memory state."""
        if not self.is_cuda:
            return

        try:
            total_memory = torch.cuda.get_device_properties(self.device).total_memory
            free_memory, used_memory = torch.cuda.mem_get_info(self.device)

            logger.info(f"CUDA Memory State - Total: {total_memory/1024**3:.2f}GB, "
                       f"Free: {free_memory/1024**3:.2f}GB, "
                       f"Used: {(total_memory - free_memory)/1024**3:.2f}GB")

        except Exception as e:
            logger.warning(f"Could not get initial CUDA memory state: {e}")

    def get_memory_info(self) -> Dict[str, float]:
        """
        Get current memory usage information.

        Returns:
            Dictionary with memory info in GB
        """
        if not self.is_cuda:
            return {'available': float('inf'), 'allocated': 0, 'reserved': 0, 'free': float('inf')}

        try:
            # PyTorch memory info
            allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            reserved = torch.cuda.memory_reserved(self.device) / 1024**3

            # System memory info
            free_mem, total_mem = torch.cuda.mem_get_info(self.device)
            free = free_mem / 1024**3
            total = total_mem / 1024**3

            # Update peak tracking
            self.memory_stats['peak_allocated'] = max(self.memory_stats['peak_allocated'], allocated)
            self.memory_stats['peak_reserved'] = max(self.memory_stats['peak_reserved'], reserved)

            return {
                'allocated': allocated,
                'reserved': reserved,
                'free': free,
                'total': total,
                'available': max(0, free - self.safety_margin_bytes / 1024**3)
            }
        except Exception as e:
            logger.warning(f"Could not get memory info: {e}")
            return {'available': 0, 'allocated': 0, 'reserved': 0, 'free': 0}

    def estimate_memory_requirement(self, tensor_shapes: list, dtype: torch.dtype = torch.float32,
                                  overhead_factor: float = 2.0) -> float:
        """
        Estimate memory requirement for given tensor shapes.

        Args:
            tensor_shapes: List of tensor shapes (tuples)
            dtype: Data type of tensors
            overhead_factor: Multiplicative factor for computation overhead

        Returns:
            Estimated memory requirement in GB
        """
        bytes_per_element = torch.tensor([], dtype=dtype).element_size()
        total_elements = sum(torch.tensor(shape).prod().item() for shape in tensor_shapes)
        base_memory_bytes = total_elements * bytes_per_element
        estimated_bytes = base_memory_bytes * overhead_factor

        return estimated_bytes / 1024**3

    def can_allocate(self, required_gb: float) -> bool:
        """
        Check if we can safely allocate the required memory.

        Args:
            required_gb: Required memory in GB

        Returns:
            True if allocation is safe
        """
        if not self.is_cuda:
            return True  # Assume CPU has sufficient memory

        memory_info = self.get_memory_info()
        available = memory_info['available']

        can_alloc = available >= required_gb
        logger.debug(f"Memory check: Required {required_gb:.2f}GB, Available {available:.2f}GB, "
                    f"Can allocate: {can_alloc}")

        return can_alloc

    def cleanup_memory(self, aggressive: bool = False):
        """
        Clean up CUDA memory.

        Args:
            aggressive: Whether to perform aggressive cleanup
        """
        if not self.is_cuda:
            return

        logger.debug(f"Memory cleanup ({'aggressive' if aggressive else 'standard'})")

        # Python garbage collection
        gc.collect()

        # CUDA cache cleanup
        torch.cuda.empty_cache()

        if aggressive:
            # Force synchronization and additional cleanup
            torch.cuda.synchronize(self.device)

            # Try to release all unused cached memory
            try:
                torch.cuda.memory.empty_cache()
            except AttributeError:
                pass  # Not available in all PyTorch versions

            # Second round after sync
            gc.collect()
            torch.cuda.empty_cache()

    @contextmanager
    def memory_context(self, required_gb: Optional[float] = None, cleanup_after: bool = True):
        """
        Context manager for memory-aware operations.

        Args:
            required_gb: Required memory in GB (optional)
            cleanup_after: Whether to cleanup after context exits

        Raises:
            RuntimeError: If insufficient memory is available
        """
        if required_gb and not self.can_allocate(required_gb):
            # Try cleanup and check again
            self.cleanup_memory(aggressive=True)
            if not self.can_allocate(required_gb):
                available = self.get_memory_info()['available']
                raise RuntimeError(f"Insufficient CUDA memory: need {required_gb:.2f}GB, "
                                 f"available {available:.2f}GB")

        memory_before = self.get_memory_info()

        try:
            yield
        finally:
            if cleanup_after:
                self.cleanup_memory()

            if logger.isEnabledFor(logging.DEBUG):
                memory_after = self.get_memory_info()
                allocated_diff = memory_after['allocated'] - memory_before['allocated']
                logger.debug(f"Memory context completed. Allocated change: {allocated_diff:+.2f}GB")

    def auto_batch_size(self, base_batch_size: int, tensor_shapes: list,
                       max_memory_gb: Optional[float] = None, min_batch_size: int = 1) -> int:
        """
        Automatically determine optimal batch size based on available memory.

        Args:
            base_batch_size: Desired batch size
            tensor_shapes: List of tensor shapes per batch item
            max_memory_gb: Maximum memory to use (None for auto)
            min_batch_size: Minimum acceptable batch size

        Returns:
            Optimized batch size
        """
        if not self.is_cuda:
            return base_batch_size

        memory_info = self.get_memory_info()

        if max_memory_gb is None:
            max_memory_gb = memory_info['available']

        # Estimate memory per batch item
        memory_per_item = self.estimate_memory_requirement(tensor_shapes, overhead_factor=2.5)

        if memory_per_item <= 0:
            return base_batch_size

        # Calculate maximum possible batch size
        max_batch_size = int(max_memory_gb / memory_per_item)
        optimal_batch_size = max(min_batch_size, min(base_batch_size, max_batch_size))

        logger.debug(f"Auto batch sizing: {memory_per_item:.3f}GB per item, "
                    f"max memory {max_memory_gb:.2f}GB, "
                    f"optimal batch: {optimal_batch_size} (requested: {base_batch_size})")

        return optimal_batch_size

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        stats = self.memory_stats.copy()
        current_info = self.get_memory_info()

        stats.update({
            'current_allocated': current_info['allocated'],
            'current_reserved': current_info['reserved'],
            'current_free': current_info['free'],
            'current_available': current_info['available'],
            'device': str(self.device),
            'is_cuda': self.is_cuda
        })

        return stats


def oom_safe_function(fallback_device: str = 'cpu', max_retries: int = 2,
                     cleanup_between_retries: bool = True):
    """
    Decorator for OOM-safe function execution with automatic fallback.

    Args:
        fallback_device: Device to fallback to on OOM
        max_retries: Maximum number of retry attempts
        cleanup_between_retries: Whether to cleanup memory between retries

    Returns:
        Decorated function with OOM safety
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            memory_manager = None

            # Try to find memory manager in args/kwargs
            for arg in args:
                if isinstance(arg, MemoryManager):
                    memory_manager = arg
                    break

            if memory_manager is None:
                for value in kwargs.values():
                    if isinstance(value, MemoryManager):
                        memory_manager = value
                        break

            retry_count = 0
            last_exception = None

            while retry_count <= max_retries:
                try:
                    if retry_count > 0:
                        logger.info(f"Retrying {func.__name__} (attempt {retry_count + 1}/{max_retries + 1})")

                        if cleanup_between_retries and memory_manager:
                            memory_manager.cleanup_memory(aggressive=True)

                    return func(*args, **kwargs)

                except RuntimeError as e:
                    if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                        if memory_manager:
                            memory_manager.memory_stats['oom_count'] += 1

                        logger.warning(f"OOM error in {func.__name__}: {e}")
                        last_exception = e
                        retry_count += 1

                        if retry_count <= max_retries:
                            # Try fallback strategies
                            if retry_count == 1 and memory_manager:
                                # First retry: aggressive cleanup
                                logger.info("Attempting aggressive memory cleanup...")
                                memory_manager.cleanup_memory(aggressive=True)
                                continue
                            elif retry_count == 2 and fallback_device != 'cuda':
                                # Second retry: device fallback
                                logger.warning(f"Falling back to {fallback_device} device")
                                if memory_manager:
                                    memory_manager.memory_stats['fallback_count'] += 1

                                # Modify device in args/kwargs
                                new_args = []
                                for arg in args:
                                    if hasattr(arg, 'device') and hasattr(arg, 'to'):
                                        new_args.append(arg.to(fallback_device))
                                    else:
                                        new_args.append(arg)

                                new_kwargs = {}
                                for key, value in kwargs.items():
                                    if key == 'device':
                                        new_kwargs[key] = torch.device(fallback_device)
                                    elif hasattr(value, 'device') and hasattr(value, 'to'):
                                        new_kwargs[key] = value.to(fallback_device)
                                    else:
                                        new_kwargs[key] = value

                                return func(*new_args, **new_kwargs)
                    else:
                        # Non-memory error, re-raise immediately
                        raise

            # All retries exhausted
            if last_exception:
                logger.error(f"All retry attempts exhausted for {func.__name__}")
                raise last_exception
            else:
                raise RuntimeError(f"Unexpected error in {func.__name__} after {max_retries} retries")

        return wrapper
    return decorator


class AdaptiveBatchProcessor:
    """
    Processor that automatically adjusts batch size based on available memory and OOM events.
    """

    def __init__(self, memory_manager: MemoryManager, initial_batch_size: int = 4):
        self.memory_manager = memory_manager
        self.current_batch_size = initial_batch_size
        self.min_batch_size = 1
        self.max_batch_size = initial_batch_size * 2
        self.oom_history = []
        self.success_history = []

    def adjust_batch_size(self, had_oom: bool, processing_time: float):
        """
        Adjust batch size based on OOM events and processing efficiency.

        Args:
            had_oom: Whether an OOM occurred
            processing_time: Time taken for processing
        """
        if had_oom:
            self.oom_history.append(self.current_batch_size)
            # Reduce batch size aggressively
            self.current_batch_size = max(self.min_batch_size, self.current_batch_size // 2)
            logger.info(f"OOM detected: reducing batch size to {self.current_batch_size}")
        else:
            self.success_history.append((self.current_batch_size, processing_time))

            # Consider increasing batch size if:
            # 1. No recent OOMs
            # 2. Current batch size is below max
            # 3. Processing time is reasonable (not CPU-bound)
            recent_ooms = [bs for bs in self.oom_history[-5:] if bs >= self.current_batch_size]

            if (not recent_ooms and
                self.current_batch_size < self.max_batch_size and
                len(self.success_history) >= 3):

                # Check if increasing batch size would be beneficial
                avg_time_per_item = processing_time / self.current_batch_size
                memory_info = self.memory_manager.get_memory_info()

                if (avg_time_per_item < 2.0 and  # Not too slow per item
                    memory_info['available'] > 1.0):  # Have some memory headroom

                    new_batch_size = min(self.max_batch_size, self.current_batch_size + 1)
                    logger.info(f"Increasing batch size from {self.current_batch_size} to {new_batch_size}")
                    self.current_batch_size = new_batch_size

        # Trim history to prevent unbounded growth
        self.oom_history = self.oom_history[-10:]
        self.success_history = self.success_history[-10:]

    def process_batches(self, items: list, process_func: Callable,
                       tensor_shapes: Optional[list] = None) -> list:
        """
        Process items in adaptive batches.

        Args:
            items: List of items to process
            process_func: Function to process each batch
            tensor_shapes: Expected tensor shapes for memory estimation

        Returns:
            List of processing results
        """
        if not items:
            return []

        results = []
        total_items = len(items)
        processed_items = 0

        while processed_items < total_items:
            # Adjust batch size based on available memory
            if tensor_shapes:
                available_memory = self.memory_manager.get_memory_info()['available']
                memory_batch_size = self.memory_manager.auto_batch_size(
                    self.current_batch_size, tensor_shapes, available_memory * 0.8
                )
                effective_batch_size = min(self.current_batch_size, memory_batch_size)
            else:
                effective_batch_size = self.current_batch_size

            # Get current batch
            end_idx = min(processed_items + effective_batch_size, total_items)
            batch = items[processed_items:end_idx]

            logger.debug(f"Processing batch {processed_items//effective_batch_size + 1}: "
                        f"items {processed_items+1}-{end_idx}/{total_items} "
                        f"(batch_size={len(batch)})")

            # Process with OOM handling
            start_time = time.time()
            had_oom = False

            try:
                with self.memory_manager.memory_context(cleanup_after=True):
                    batch_results = process_func(batch)
                    results.extend(batch_results if isinstance(batch_results, list) else [batch_results])

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    had_oom = True
                    logger.warning(f"OOM in batch processing: {e}")

                    # If batch size is already at minimum, we can't reduce further
                    if effective_batch_size <= self.min_batch_size:
                        logger.error("OOM with minimum batch size - cannot continue")
                        raise

                    # Don't update processed_items, retry with smaller batch
                    self.adjust_batch_size(had_oom=True, processing_time=0)
                    continue
                else:
                    raise

            # Update stats and batch size
            processing_time = time.time() - start_time
            self.adjust_batch_size(had_oom, processing_time)

            processed_items = end_idx

        logger.info(f"Adaptive batch processing completed: {total_items} items, "
                   f"final batch size: {self.current_batch_size}")

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get adaptive processing statistics."""
        return {
            'current_batch_size': self.current_batch_size,
            'oom_events': len(self.oom_history),
            'recent_oom_batch_sizes': self.oom_history[-5:],
            'avg_processing_time': (
                sum(time for _, time in self.success_history) / len(self.success_history)
                if self.success_history else 0
            ),
            'memory_stats': self.memory_manager.get_memory_stats()
        }