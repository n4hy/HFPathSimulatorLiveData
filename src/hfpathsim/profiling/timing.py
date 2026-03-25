"""CPU timing utilities for profiling.

Provides decorators, context managers, and statistics collection for
measuring execution time of functions and code blocks.
"""

import time
import functools
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from contextlib import contextmanager
import threading

# Thread-local storage for nested timing
_local = threading.local()

# Global timing statistics
_timing_stats: Dict[str, List[float]] = {}
_timing_lock = threading.Lock()


@dataclass
class TimingStats:
    """Statistics for a timed operation."""

    name: str
    count: int
    total_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float
    std_ms: float
    last_ms: float

    def __str__(self) -> str:
        return (
            f"{self.name}: count={self.count}, "
            f"mean={self.mean_ms:.3f}ms, "
            f"min={self.min_ms:.3f}ms, max={self.max_ms:.3f}ms, "
            f"std={self.std_ms:.3f}ms"
        )


class Timer:
    """High-resolution timer for performance measurement.

    Can be used as a context manager or manually started/stopped.

    Examples:
        # As context manager
        with Timer() as t:
            do_something()
        print(f"Took {t.elapsed_ms:.2f}ms")

        # Manual control
        timer = Timer()
        timer.start()
        do_something()
        timer.stop()
        print(f"Took {timer.elapsed_ms:.2f}ms")
    """

    def __init__(self, name: Optional[str] = None, record: bool = True):
        """Initialize timer.

        Args:
            name: Optional name for statistics tracking
            record: Whether to record to global statistics
        """
        self.name = name
        self.record = record
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._elapsed: float = 0.0

    def start(self) -> "Timer":
        """Start the timer."""
        self._start_time = time.perf_counter()
        self._end_time = None
        return self

    def stop(self) -> float:
        """Stop the timer and return elapsed time in seconds."""
        if self._start_time is None:
            raise RuntimeError("Timer not started")
        self._end_time = time.perf_counter()
        self._elapsed = self._end_time - self._start_time

        if self.record and self.name:
            _record_timing(self.name, self._elapsed * 1000)

        return self._elapsed

    def reset(self) -> "Timer":
        """Reset the timer."""
        self._start_time = None
        self._end_time = None
        self._elapsed = 0.0
        return self

    @property
    def elapsed(self) -> float:
        """Elapsed time in seconds."""
        if self._start_time is None:
            return 0.0
        if self._end_time is None:
            return time.perf_counter() - self._start_time
        return self._elapsed

    @property
    def elapsed_ms(self) -> float:
        """Elapsed time in milliseconds."""
        return self.elapsed * 1000

    @property
    def elapsed_us(self) -> float:
        """Elapsed time in microseconds."""
        return self.elapsed * 1_000_000

    def __enter__(self) -> "Timer":
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()


@contextmanager
def timer(name: Optional[str] = None, record: bool = True):
    """Context manager for timing code blocks.

    Args:
        name: Optional name for statistics tracking
        record: Whether to record to global statistics

    Yields:
        Timer instance

    Example:
        with timer("data_processing") as t:
            process_data()
        print(f"Processing took {t.elapsed_ms:.2f}ms")
    """
    t = Timer(name=name, record=record)
    try:
        yield t.start()
    finally:
        t.stop()


def profile_function(
    name: Optional[str] = None,
    record: bool = True,
    print_result: bool = False,
) -> Callable:
    """Decorator to profile function execution time.

    Args:
        name: Custom name (defaults to function name)
        record: Whether to record to global statistics
        print_result: Whether to print timing after each call

    Returns:
        Decorated function

    Example:
        @profile_function(print_result=True)
        def my_function():
            ...
    """
    def decorator(func: Callable) -> Callable:
        func_name = name or func.__qualname__

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            with Timer(name=func_name, record=record) as t:
                result = func(*args, **kwargs)

            if print_result:
                print(f"{func_name}: {t.elapsed_ms:.3f}ms")

            return result

        # Attach timing info to function
        wrapper._profiled = True
        wrapper._profile_name = func_name

        return wrapper
    return decorator


def _record_timing(name: str, elapsed_ms: float) -> None:
    """Record a timing measurement to global statistics."""
    with _timing_lock:
        if name not in _timing_stats:
            _timing_stats[name] = []
        _timing_stats[name].append(elapsed_ms)


def get_timing_stats(name: Optional[str] = None) -> Dict[str, TimingStats]:
    """Get timing statistics.

    Args:
        name: Specific operation name, or None for all

    Returns:
        Dictionary mapping operation names to TimingStats
    """
    with _timing_lock:
        if name is not None:
            if name not in _timing_stats:
                return {}
            names = [name]
        else:
            names = list(_timing_stats.keys())

        result = {}
        for n in names:
            times = _timing_stats[n]
            if not times:
                continue

            result[n] = TimingStats(
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


def reset_timing_stats(name: Optional[str] = None) -> None:
    """Reset timing statistics.

    Args:
        name: Specific operation to reset, or None for all
    """
    with _timing_lock:
        if name is not None:
            _timing_stats.pop(name, None)
        else:
            _timing_stats.clear()


def print_timing_report(name: Optional[str] = None) -> None:
    """Print a timing report to stdout.

    Args:
        name: Specific operation, or None for all
    """
    stats = get_timing_stats(name)

    if not stats:
        print("No timing data recorded.")
        return

    print("\n" + "=" * 70)
    print("TIMING REPORT")
    print("=" * 70)

    # Sort by total time
    sorted_stats = sorted(
        stats.values(),
        key=lambda s: s.total_ms,
        reverse=True
    )

    print(f"{'Operation':<40} {'Count':>8} {'Mean':>10} {'Total':>12}")
    print("-" * 70)

    for s in sorted_stats:
        print(
            f"{s.name:<40} {s.count:>8} "
            f"{s.mean_ms:>9.3f}ms {s.total_ms:>11.1f}ms"
        )

    print("=" * 70)


class ScopedTimer:
    """Timer that automatically records when it goes out of scope.

    Useful for timing in functions with multiple return paths.
    """

    def __init__(self, name: str):
        self.name = name
        self._timer = Timer(name=name, record=True)
        self._timer.start()

    def __del__(self):
        if self._timer._start_time is not None and self._timer._end_time is None:
            self._timer.stop()


class AccumulatingTimer:
    """Timer that accumulates time across multiple start/stop cycles.

    Useful for measuring total time spent in repeated operations.
    """

    def __init__(self, name: Optional[str] = None):
        self.name = name
        self._total_elapsed: float = 0.0
        self._current_start: Optional[float] = None
        self._count: int = 0

    def start(self) -> "AccumulatingTimer":
        """Start or resume timing."""
        self._current_start = time.perf_counter()
        return self

    def stop(self) -> float:
        """Stop timing and accumulate elapsed time."""
        if self._current_start is None:
            raise RuntimeError("Timer not started")
        elapsed = time.perf_counter() - self._current_start
        self._total_elapsed += elapsed
        self._current_start = None
        self._count += 1
        return elapsed

    def reset(self) -> "AccumulatingTimer":
        """Reset accumulated time."""
        self._total_elapsed = 0.0
        self._current_start = None
        self._count = 0
        return self

    @property
    def total_elapsed(self) -> float:
        """Total accumulated elapsed time in seconds."""
        return self._total_elapsed

    @property
    def total_elapsed_ms(self) -> float:
        """Total accumulated elapsed time in milliseconds."""
        return self._total_elapsed * 1000

    @property
    def count(self) -> int:
        """Number of timing cycles."""
        return self._count

    @property
    def mean_elapsed_ms(self) -> float:
        """Mean elapsed time per cycle in milliseconds."""
        if self._count == 0:
            return 0.0
        return self.total_elapsed_ms / self._count

    def __enter__(self) -> "AccumulatingTimer":
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()

    def record(self) -> None:
        """Record accumulated statistics to global stats."""
        if self.name and self._count > 0:
            _record_timing(self.name, self.mean_elapsed_ms)
