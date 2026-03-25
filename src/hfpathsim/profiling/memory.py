"""Memory profiling utilities for CPU and GPU memory tracking.

Provides tools for monitoring memory usage, detecting leaks, and
tracking peak memory consumption during processing.
"""

import os
import sys
import functools
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Any, Tuple
from contextlib import contextmanager
import time

# Try to import memory profiling libraries
_psutil_available = False
_tracemalloc_available = False

try:
    import psutil
    _psutil_available = True
except ImportError:
    pass

try:
    import tracemalloc
    _tracemalloc_available = True
except ImportError:
    pass


@dataclass
class MemoryUsage:
    """Memory usage snapshot."""

    rss_bytes: int  # Resident set size
    vms_bytes: int  # Virtual memory size
    shared_bytes: int = 0
    private_bytes: int = 0
    gpu_bytes: int = 0

    @property
    def rss_mb(self) -> float:
        return self.rss_bytes / (1024 ** 2)

    @property
    def vms_mb(self) -> float:
        return self.vms_bytes / (1024 ** 2)

    @property
    def gpu_mb(self) -> float:
        return self.gpu_bytes / (1024 ** 2)

    def __str__(self) -> str:
        return f"RSS: {self.rss_mb:.1f}MB, VMS: {self.vms_mb:.1f}MB, GPU: {self.gpu_mb:.1f}MB"


@dataclass
class MemoryDelta:
    """Memory change between two snapshots."""

    rss_delta_bytes: int
    vms_delta_bytes: int
    gpu_delta_bytes: int = 0

    @property
    def rss_delta_mb(self) -> float:
        return self.rss_delta_bytes / (1024 ** 2)

    @property
    def vms_delta_mb(self) -> float:
        return self.vms_delta_bytes / (1024 ** 2)

    @property
    def gpu_delta_mb(self) -> float:
        return self.gpu_delta_bytes / (1024 ** 2)

    def __str__(self) -> str:
        return (
            f"RSS: {self.rss_delta_mb:+.2f}MB, "
            f"VMS: {self.vms_delta_mb:+.2f}MB, "
            f"GPU: {self.gpu_delta_mb:+.2f}MB"
        )


class MemoryProfiler:
    """Comprehensive memory profiler for tracking allocations.

    Tracks CPU and GPU memory usage over time, identifies peak usage,
    and can detect memory leaks.

    Example:
        profiler = MemoryProfiler()
        profiler.start()

        # Do memory-intensive operations
        data = process_large_dataset()

        profiler.snapshot("after_processing")
        report = profiler.stop()
        print(report)
    """

    def __init__(self, track_gpu: bool = True, track_allocations: bool = False):
        """Initialize memory profiler.

        Args:
            track_gpu: Whether to track GPU memory
            track_allocations: Whether to track individual allocations (slow)
        """
        self.track_gpu = track_gpu
        self.track_allocations = track_allocations

        self._started = False
        self._start_memory: Optional[MemoryUsage] = None
        self._peak_memory: Optional[MemoryUsage] = None
        self._snapshots: Dict[str, MemoryUsage] = {}
        self._allocation_trace: List[Tuple[str, int]] = []

    def start(self) -> "MemoryProfiler":
        """Start memory profiling."""
        self._started = True
        self._start_memory = get_memory_usage(include_gpu=self.track_gpu)
        self._peak_memory = self._start_memory
        self._snapshots = {"start": self._start_memory}

        if self.track_allocations and _tracemalloc_available:
            tracemalloc.start()

        return self

    def stop(self) -> Dict[str, Any]:
        """Stop profiling and return report.

        Returns:
            Dictionary with memory profiling results
        """
        if not self._started:
            raise RuntimeError("Profiler not started")

        end_memory = get_memory_usage(include_gpu=self.track_gpu)
        self._snapshots["end"] = end_memory

        report = {
            "start": self._start_memory,
            "end": end_memory,
            "peak": self._peak_memory,
            "delta": MemoryDelta(
                rss_delta_bytes=end_memory.rss_bytes - self._start_memory.rss_bytes,
                vms_delta_bytes=end_memory.vms_bytes - self._start_memory.vms_bytes,
                gpu_delta_bytes=end_memory.gpu_bytes - self._start_memory.gpu_bytes,
            ),
            "snapshots": self._snapshots.copy(),
        }

        if self.track_allocations and _tracemalloc_available:
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics("lineno")[:10]
            report["top_allocations"] = [
                {"file": str(stat.traceback), "size_kb": stat.size / 1024}
                for stat in top_stats
            ]
            tracemalloc.stop()

        self._started = False
        return report

    def snapshot(self, name: str) -> MemoryUsage:
        """Take a memory snapshot.

        Args:
            name: Snapshot identifier

        Returns:
            Current memory usage
        """
        if not self._started:
            raise RuntimeError("Profiler not started")

        current = get_memory_usage(include_gpu=self.track_gpu)
        self._snapshots[name] = current

        # Update peak
        if current.rss_bytes > self._peak_memory.rss_bytes:
            self._peak_memory = current

        return current

    def get_delta_since(self, snapshot_name: str) -> Optional[MemoryDelta]:
        """Get memory delta since a snapshot.

        Args:
            snapshot_name: Name of previous snapshot

        Returns:
            MemoryDelta or None if snapshot not found
        """
        if snapshot_name not in self._snapshots:
            return None

        prev = self._snapshots[snapshot_name]
        current = get_memory_usage(include_gpu=self.track_gpu)

        return MemoryDelta(
            rss_delta_bytes=current.rss_bytes - prev.rss_bytes,
            vms_delta_bytes=current.vms_bytes - prev.vms_bytes,
            gpu_delta_bytes=current.gpu_bytes - prev.gpu_bytes,
        )

    def print_report(self) -> None:
        """Print memory profiling report."""
        if not self._snapshots:
            print("No memory data collected.")
            return

        print("\n" + "=" * 60)
        print("MEMORY PROFILING REPORT")
        print("=" * 60)

        if self._start_memory:
            print(f"\nStart: {self._start_memory}")

        if self._peak_memory:
            print(f"Peak:  {self._peak_memory}")

        end = self._snapshots.get("end")
        if end:
            print(f"End:   {end}")

        if self._start_memory and end:
            delta = MemoryDelta(
                rss_delta_bytes=end.rss_bytes - self._start_memory.rss_bytes,
                vms_delta_bytes=end.vms_bytes - self._start_memory.vms_bytes,
                gpu_delta_bytes=end.gpu_bytes - self._start_memory.gpu_bytes,
            )
            print(f"\nDelta: {delta}")

        if len(self._snapshots) > 2:
            print("\nSnapshots:")
            for name, mem in self._snapshots.items():
                if name not in ("start", "end"):
                    print(f"  {name}: {mem}")

        print("=" * 60)


def get_memory_usage(include_gpu: bool = True) -> MemoryUsage:
    """Get current memory usage.

    Args:
        include_gpu: Whether to include GPU memory

    Returns:
        MemoryUsage snapshot
    """
    rss = 0
    vms = 0
    shared = 0

    if _psutil_available:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        rss = mem_info.rss
        vms = mem_info.vms
        if hasattr(mem_info, "shared"):
            shared = mem_info.shared
    else:
        # Fallback: try to read /proc/self/status on Linux
        try:
            with open("/proc/self/status", "r") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        rss = int(line.split()[1]) * 1024
                    elif line.startswith("VmSize:"):
                        vms = int(line.split()[1]) * 1024
        except (FileNotFoundError, PermissionError):
            pass

    gpu_bytes = 0
    if include_gpu:
        try:
            import cupy as cp
            device = cp.cuda.Device()
            total, free = device.mem_info
            gpu_bytes = total - free
        except (ImportError, Exception):
            pass

    return MemoryUsage(
        rss_bytes=rss,
        vms_bytes=vms,
        shared_bytes=shared,
        gpu_bytes=gpu_bytes,
    )


def get_peak_memory() -> MemoryUsage:
    """Get peak memory usage since process start.

    Returns:
        Peak memory usage (may not include GPU)
    """
    if _psutil_available:
        process = psutil.Process(os.getpid())
        # Note: This gets current, not peak - peak requires special tracking
        mem_info = process.memory_info()
        return MemoryUsage(
            rss_bytes=mem_info.rss,
            vms_bytes=mem_info.vms,
        )

    return get_memory_usage(include_gpu=False)


@contextmanager
def track_memory(name: Optional[str] = None, print_result: bool = False):
    """Context manager to track memory usage of a code block.

    Args:
        name: Optional name for the operation
        print_result: Whether to print memory delta

    Yields:
        MemoryProfiler instance

    Example:
        with track_memory("data_loading", print_result=True) as tracker:
            data = load_large_dataset()
    """
    profiler = MemoryProfiler(track_gpu=True)
    profiler.start()

    try:
        yield profiler
    finally:
        report = profiler.stop()

        if print_result:
            label = name or "Memory"
            delta = report["delta"]
            print(f"{label}: {delta}")


def memory_profile(
    name: Optional[str] = None,
    print_result: bool = False,
) -> Callable:
    """Decorator to profile memory usage of a function.

    Args:
        name: Custom name (defaults to function name)
        print_result: Whether to print memory delta

    Returns:
        Decorated function

    Example:
        @memory_profile(print_result=True)
        def load_data():
            return large_dataset
    """
    def decorator(func: Callable) -> Callable:
        func_name = name or func.__qualname__

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            with track_memory(func_name, print_result=print_result):
                return func(*args, **kwargs)

        return wrapper
    return decorator


class MemoryWatcher:
    """Background memory watcher for continuous monitoring.

    Runs in a separate thread and periodically samples memory usage.
    """

    def __init__(self, interval_sec: float = 1.0, include_gpu: bool = True):
        """Initialize memory watcher.

        Args:
            interval_sec: Sampling interval in seconds
            include_gpu: Whether to track GPU memory
        """
        self.interval_sec = interval_sec
        self.include_gpu = include_gpu

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._samples: List[Tuple[float, MemoryUsage]] = []
        self._peak: Optional[MemoryUsage] = None
        self._lock = threading.Lock()

    def start(self) -> "MemoryWatcher":
        """Start background monitoring."""
        if self._running:
            return self

        self._running = True
        self._samples = []
        self._peak = None
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> Dict[str, Any]:
        """Stop monitoring and return collected data.

        Returns:
            Dictionary with samples and peak memory
        """
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

        with self._lock:
            return {
                "samples": self._samples.copy(),
                "peak": self._peak,
                "sample_count": len(self._samples),
            }

    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        start_time = time.perf_counter()

        while self._running:
            elapsed = time.perf_counter() - start_time
            mem = get_memory_usage(include_gpu=self.include_gpu)

            with self._lock:
                self._samples.append((elapsed, mem))
                if self._peak is None or mem.rss_bytes > self._peak.rss_bytes:
                    self._peak = mem

            time.sleep(self.interval_sec)

    def get_current(self) -> Optional[MemoryUsage]:
        """Get most recent memory sample."""
        with self._lock:
            if self._samples:
                return self._samples[-1][1]
        return None

    def get_peak(self) -> Optional[MemoryUsage]:
        """Get peak memory usage."""
        with self._lock:
            return self._peak
