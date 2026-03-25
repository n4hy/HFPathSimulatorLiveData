"""GPU profiling utilities for CUDA kernel timing.

Provides CUDA event-based timing for accurate GPU kernel measurement,
memory profiling, and kernel statistics collection.
"""

import time
import functools
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Tuple
from contextlib import contextmanager
import threading
import numpy as np

# GPU availability flags
_cupy_available = False
_cuda_available = False

try:
    import cupy as cp
    _cupy_available = True
    _cuda_available = True
except ImportError:
    pass


# Global kernel statistics
_kernel_stats: Dict[str, List[float]] = {}
_kernel_lock = threading.Lock()


@dataclass
class KernelStats:
    """Statistics for a GPU kernel."""

    name: str
    count: int
    total_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float
    std_ms: float
    last_ms: float
    throughput_gsps: Optional[float] = None  # Giga-samples per second

    def __str__(self) -> str:
        tp_str = f", throughput={self.throughput_gsps:.2f} Gsps" if self.throughput_gsps else ""
        return (
            f"{self.name}: count={self.count}, "
            f"mean={self.mean_ms:.3f}ms, "
            f"min={self.min_ms:.3f}ms, max={self.max_ms:.3f}ms{tp_str}"
        )


@dataclass
class GPUMemoryInfo:
    """GPU memory usage information."""

    total_bytes: int
    used_bytes: int
    free_bytes: int
    pool_used_bytes: int = 0
    pool_total_bytes: int = 0

    @property
    def total_gb(self) -> float:
        return self.total_bytes / (1024 ** 3)

    @property
    def used_gb(self) -> float:
        return self.used_bytes / (1024 ** 3)

    @property
    def free_gb(self) -> float:
        return self.free_bytes / (1024 ** 3)

    @property
    def utilization_pct(self) -> float:
        if self.total_bytes == 0:
            return 0.0
        return 100.0 * self.used_bytes / self.total_bytes

    def __str__(self) -> str:
        return (
            f"GPU Memory: {self.used_gb:.2f}/{self.total_gb:.2f} GB "
            f"({self.utilization_pct:.1f}% used), {self.free_gb:.2f} GB free"
        )


class CUDATimer:
    """CUDA event-based timer for accurate GPU kernel timing.

    Uses CUDA events for precise GPU timing that accounts for
    asynchronous execution and kernel overlap.

    Example:
        timer = CUDATimer()
        timer.start()
        # GPU kernel launches
        kernel_func(...)
        timer.stop()
        print(f"Kernel took {timer.elapsed_ms:.3f}ms")
    """

    def __init__(self, name: Optional[str] = None, record: bool = True):
        """Initialize CUDA timer.

        Args:
            name: Optional name for statistics tracking
            record: Whether to record to global statistics
        """
        self.name = name
        self.record = record
        self._elapsed_ms: float = 0.0

        if _cupy_available:
            self._start_event = cp.cuda.Event()
            self._end_event = cp.cuda.Event()
        else:
            self._start_event = None
            self._end_event = None
            self._start_time: Optional[float] = None
            self._end_time: Optional[float] = None

    def start(self) -> "CUDATimer":
        """Start the timer by recording a CUDA event."""
        if _cupy_available:
            self._start_event.record()
        else:
            self._start_time = time.perf_counter()
        return self

    def stop(self) -> float:
        """Stop the timer and return elapsed time in milliseconds."""
        if _cupy_available:
            self._end_event.record()
            self._end_event.synchronize()
            self._elapsed_ms = cp.cuda.get_elapsed_time(
                self._start_event, self._end_event
            )
        else:
            self._end_time = time.perf_counter()
            self._elapsed_ms = (self._end_time - self._start_time) * 1000

        if self.record and self.name:
            _record_kernel_timing(self.name, self._elapsed_ms)

        return self._elapsed_ms

    @property
    def elapsed_ms(self) -> float:
        """Elapsed time in milliseconds."""
        return self._elapsed_ms

    def __enter__(self) -> "CUDATimer":
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()


@contextmanager
def gpu_timer(name: Optional[str] = None, record: bool = True):
    """Context manager for GPU kernel timing.

    Args:
        name: Optional name for statistics tracking
        record: Whether to record to global statistics

    Yields:
        CUDATimer instance

    Example:
        with gpu_timer("fft_kernel") as t:
            cp.fft.fft(data)
        print(f"FFT took {t.elapsed_ms:.3f}ms")
    """
    t = CUDATimer(name=name, record=record)
    try:
        yield t.start()
    finally:
        t.stop()


def profile_kernel(
    name: Optional[str] = None,
    record: bool = True,
    print_result: bool = False,
    n_samples: Optional[int] = None,
) -> Callable:
    """Decorator to profile GPU kernel execution time.

    Args:
        name: Custom name (defaults to function name)
        record: Whether to record to global statistics
        print_result: Whether to print timing after each call
        n_samples: Number of samples processed (for throughput calc)

    Returns:
        Decorated function

    Example:
        @profile_kernel(print_result=True, n_samples=4096)
        def my_kernel(data):
            return cp.fft.fft(data)
    """
    def decorator(func: Callable) -> Callable:
        func_name = name or func.__qualname__

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            with CUDATimer(name=func_name, record=record) as t:
                result = func(*args, **kwargs)

            if print_result:
                msg = f"{func_name}: {t.elapsed_ms:.3f}ms"
                if n_samples:
                    throughput = n_samples / t.elapsed_ms / 1000  # Msps
                    msg += f" ({throughput:.2f} Msps)"
                print(msg)

            return result

        wrapper._profiled = True
        wrapper._profile_name = func_name

        return wrapper
    return decorator


class GPUProfiler:
    """Comprehensive GPU profiler for tracking kernel execution.

    Provides detailed profiling with memory tracking, kernel timing,
    and performance analysis.

    Example:
        profiler = GPUProfiler()
        profiler.start_session("channel_processing")

        with profiler.profile("fft"):
            result = cp.fft.fft(data)

        with profiler.profile("multiply"):
            result = result * H

        report = profiler.end_session()
        print(report)
    """

    def __init__(self):
        self._sessions: Dict[str, Dict] = {}
        self._current_session: Optional[str] = None
        self._start_memory: Optional[GPUMemoryInfo] = None

    def start_session(self, name: str) -> None:
        """Start a profiling session.

        Args:
            name: Session name for identification
        """
        self._current_session = name
        self._start_memory = get_gpu_memory_info()
        self._sessions[name] = {
            "start_time": time.perf_counter(),
            "kernels": [],
            "memory_start": self._start_memory,
            "memory_end": None,
            "total_kernel_time_ms": 0.0,
        }

    def end_session(self) -> Dict[str, Any]:
        """End the current profiling session.

        Returns:
            Session summary with timing and memory info
        """
        if self._current_session is None:
            raise RuntimeError("No active session")

        session = self._sessions[self._current_session]
        session["end_time"] = time.perf_counter()
        session["memory_end"] = get_gpu_memory_info()
        session["wall_time_ms"] = (
            session["end_time"] - session["start_time"]
        ) * 1000

        # Calculate total kernel time
        session["total_kernel_time_ms"] = sum(
            k["elapsed_ms"] for k in session["kernels"]
        )

        # Memory delta
        if session["memory_start"] and session["memory_end"]:
            session["memory_delta_bytes"] = (
                session["memory_end"].used_bytes -
                session["memory_start"].used_bytes
            )

        result = session.copy()
        self._current_session = None
        return result

    @contextmanager
    def profile(self, kernel_name: str, n_samples: Optional[int] = None):
        """Context manager to profile a kernel within a session.

        Args:
            kernel_name: Name of the kernel/operation
            n_samples: Number of samples for throughput calculation

        Yields:
            CUDATimer instance
        """
        if self._current_session is None:
            raise RuntimeError("No active session - call start_session first")

        timer = CUDATimer(name=kernel_name, record=False)
        try:
            yield timer.start()
        finally:
            elapsed = timer.stop()

            kernel_info = {
                "name": kernel_name,
                "elapsed_ms": elapsed,
                "n_samples": n_samples,
            }

            if n_samples:
                kernel_info["throughput_msps"] = n_samples / elapsed / 1000

            self._sessions[self._current_session]["kernels"].append(kernel_info)

    def get_session_summary(self, name: str) -> Optional[Dict]:
        """Get summary for a completed session."""
        return self._sessions.get(name)

    def print_session_report(self, name: str) -> None:
        """Print a detailed report for a session."""
        session = self._sessions.get(name)
        if not session:
            print(f"No session found: {name}")
            return

        print("\n" + "=" * 70)
        print(f"GPU PROFILING SESSION: {name}")
        print("=" * 70)

        print(f"\nWall time: {session['wall_time_ms']:.3f}ms")
        print(f"Total kernel time: {session['total_kernel_time_ms']:.3f}ms")

        if session.get("memory_delta_bytes"):
            delta_mb = session["memory_delta_bytes"] / (1024 ** 2)
            print(f"Memory delta: {delta_mb:+.2f} MB")

        print("\nKernel breakdown:")
        print("-" * 70)
        print(f"{'Kernel':<40} {'Time':>12} {'Throughput':>15}")
        print("-" * 70)

        for k in session["kernels"]:
            tp_str = ""
            if k.get("throughput_msps"):
                tp_str = f"{k['throughput_msps']:.2f} Msps"
            print(f"{k['name']:<40} {k['elapsed_ms']:>10.3f}ms {tp_str:>15}")

        print("=" * 70)


def _record_kernel_timing(name: str, elapsed_ms: float) -> None:
    """Record a kernel timing measurement to global statistics."""
    with _kernel_lock:
        if name not in _kernel_stats:
            _kernel_stats[name] = []
        _kernel_stats[name].append(elapsed_ms)


def get_kernel_stats(name: Optional[str] = None) -> Dict[str, KernelStats]:
    """Get kernel timing statistics.

    Args:
        name: Specific kernel name, or None for all

    Returns:
        Dictionary mapping kernel names to KernelStats
    """
    with _kernel_lock:
        if name is not None:
            if name not in _kernel_stats:
                return {}
            names = [name]
        else:
            names = list(_kernel_stats.keys())

        result = {}
        for n in names:
            times = _kernel_stats[n]
            if not times:
                continue

            result[n] = KernelStats(
                name=n,
                count=len(times),
                total_ms=sum(times),
                mean_ms=statistics.mean(times),
                min_ms=min(times),
                max_ms=max(times),
                std_ms=statistics.stdev(times) if len(times) > 1 else 0.0,
                last_ms=times[-1],
            )

        return result


def reset_kernel_stats(name: Optional[str] = None) -> None:
    """Reset kernel timing statistics."""
    with _kernel_lock:
        if name is not None:
            _kernel_stats.pop(name, None)
        else:
            _kernel_stats.clear()


def get_gpu_memory_info() -> GPUMemoryInfo:
    """Get current GPU memory usage.

    Returns:
        GPUMemoryInfo with memory statistics
    """
    if not _cupy_available:
        return GPUMemoryInfo(
            total_bytes=0,
            used_bytes=0,
            free_bytes=0,
        )

    mempool = cp.get_default_memory_pool()
    device = cp.cuda.Device()

    total = device.mem_info[1]
    free = device.mem_info[0]
    used = total - free

    return GPUMemoryInfo(
        total_bytes=total,
        used_bytes=used,
        free_bytes=free,
        pool_used_bytes=mempool.used_bytes(),
        pool_total_bytes=mempool.total_bytes(),
    )


def synchronize_gpu() -> None:
    """Synchronize GPU to ensure all kernels complete."""
    if _cupy_available:
        cp.cuda.Device().synchronize()


def clear_gpu_memory() -> None:
    """Clear GPU memory pool and cache."""
    if _cupy_available:
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()


class KernelProfiler:
    """Profile individual CUDA kernels with detailed metrics.

    Tracks kernel launch configurations, occupancy, and throughput.
    """

    def __init__(self):
        self._profiles: Dict[str, List[Dict]] = {}

    def profile_launch(
        self,
        name: str,
        grid: Tuple[int, ...],
        block: Tuple[int, ...],
        n_elements: int,
        elapsed_ms: float,
    ) -> None:
        """Record a kernel launch profile.

        Args:
            name: Kernel name
            grid: Grid dimensions (blocks)
            block: Block dimensions (threads)
            n_elements: Number of elements processed
            elapsed_ms: Execution time in milliseconds
        """
        total_threads = 1
        for g in grid:
            total_threads *= g
        for b in block:
            total_threads *= b

        profile = {
            "grid": grid,
            "block": block,
            "total_threads": total_threads,
            "n_elements": n_elements,
            "elapsed_ms": elapsed_ms,
            "throughput_elements_per_ms": n_elements / elapsed_ms if elapsed_ms > 0 else 0,
            "elements_per_thread": n_elements / total_threads if total_threads > 0 else 0,
        }

        if name not in self._profiles:
            self._profiles[name] = []
        self._profiles[name].append(profile)

    def get_summary(self, name: str) -> Optional[Dict]:
        """Get profiling summary for a kernel."""
        if name not in self._profiles:
            return None

        profiles = self._profiles[name]
        elapsed_times = [p["elapsed_ms"] for p in profiles]
        throughputs = [p["throughput_elements_per_ms"] for p in profiles]

        return {
            "name": name,
            "count": len(profiles),
            "mean_elapsed_ms": statistics.mean(elapsed_times),
            "mean_throughput_elements_per_ms": statistics.mean(throughputs),
            "mean_throughput_msps": statistics.mean(throughputs) / 1000,
            "typical_grid": profiles[-1]["grid"],
            "typical_block": profiles[-1]["block"],
        }
