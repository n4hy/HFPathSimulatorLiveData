"""Benchmarking framework for HF Path Simulator.

Provides tools for measuring throughput, latency, and scalability
of channel processing operations.
"""

import time
import functools
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Tuple
import numpy as np

from .timing import Timer
from .gpu_profiler import CUDATimer, get_gpu_memory_info, synchronize_gpu


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    name: str
    iterations: int
    total_time_sec: float
    mean_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    samples_processed: int = 0
    throughput_msps: float = 0.0  # Mega-samples per second
    throughput_mbps: float = 0.0  # Megabytes per second
    latency_us: float = 0.0  # Mean latency in microseconds

    def __str__(self) -> str:
        lines = [
            f"Benchmark: {self.name}",
            f"  Iterations: {self.iterations}",
            f"  Mean time: {self.mean_time_ms:.3f}ms (std: {self.std_time_ms:.3f}ms)",
            f"  Min/Max: {self.min_time_ms:.3f}ms / {self.max_time_ms:.3f}ms",
        ]
        if self.throughput_msps > 0:
            lines.append(f"  Throughput: {self.throughput_msps:.2f} Msps")
        if self.throughput_mbps > 0:
            lines.append(f"  Bandwidth: {self.throughput_mbps:.2f} MB/s")
        if self.latency_us > 0:
            lines.append(f"  Latency: {self.latency_us:.1f} us")
        return "\n".join(lines)


class Benchmark:
    """Single benchmark test case.

    Example:
        def process_block(data):
            return np.fft.fft(data)

        bench = Benchmark(
            name="fft_4096",
            func=process_block,
            setup=lambda: np.random.randn(4096) + 1j*np.random.randn(4096),
            n_samples=4096,
        )

        result = bench.run(iterations=100)
        print(result)
    """

    def __init__(
        self,
        name: str,
        func: Callable,
        setup: Optional[Callable] = None,
        teardown: Optional[Callable] = None,
        n_samples: int = 0,
        bytes_per_sample: int = 8,  # complex64 = 8 bytes
        use_gpu: bool = False,
        warmup_iterations: int = 5,
    ):
        """Initialize benchmark.

        Args:
            name: Benchmark name
            func: Function to benchmark (receives setup() output as args)
            setup: Optional setup function called before each iteration
            teardown: Optional cleanup function called after each iteration
            n_samples: Number of samples processed per iteration
            bytes_per_sample: Bytes per sample for bandwidth calculation
            use_gpu: Whether to use GPU timing
            warmup_iterations: Warmup iterations before measurement
        """
        self.name = name
        self.func = func
        self.setup = setup
        self.teardown = teardown
        self.n_samples = n_samples
        self.bytes_per_sample = bytes_per_sample
        self.use_gpu = use_gpu
        self.warmup_iterations = warmup_iterations

    def run(self, iterations: int = 100) -> BenchmarkResult:
        """Run the benchmark.

        Args:
            iterations: Number of measured iterations

        Returns:
            BenchmarkResult with timing statistics
        """
        times_ms: List[float] = []

        # Warmup
        for _ in range(self.warmup_iterations):
            args = self.setup() if self.setup else ()
            if not isinstance(args, tuple):
                args = (args,)
            _ = self.func(*args)
            if self.teardown:
                self.teardown()

        # Synchronize GPU before measurement
        if self.use_gpu:
            synchronize_gpu()

        # Measured iterations
        total_start = time.perf_counter()

        for _ in range(iterations):
            args = self.setup() if self.setup else ()
            if not isinstance(args, tuple):
                args = (args,)

            if self.use_gpu:
                timer = CUDATimer()
                timer.start()
                _ = self.func(*args)
                elapsed_ms = timer.stop()
            else:
                start = time.perf_counter()
                _ = self.func(*args)
                elapsed_ms = (time.perf_counter() - start) * 1000

            times_ms.append(elapsed_ms)

            if self.teardown:
                self.teardown()

        total_time = time.perf_counter() - total_start

        # Calculate statistics
        mean_ms = statistics.mean(times_ms)
        std_ms = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0

        # Throughput calculations
        throughput_msps = 0.0
        throughput_mbps = 0.0
        if self.n_samples > 0 and mean_ms > 0:
            throughput_msps = self.n_samples / mean_ms / 1000  # Million samples/sec
            throughput_mbps = (
                self.n_samples * self.bytes_per_sample / mean_ms / 1000
            )  # MB/s

        return BenchmarkResult(
            name=self.name,
            iterations=iterations,
            total_time_sec=total_time,
            mean_time_ms=mean_ms,
            std_time_ms=std_ms,
            min_time_ms=min(times_ms),
            max_time_ms=max(times_ms),
            samples_processed=self.n_samples * iterations,
            throughput_msps=throughput_msps,
            throughput_mbps=throughput_mbps,
            latency_us=mean_ms * 1000,  # Convert ms to us
        )


class BenchmarkSuite:
    """Collection of benchmarks for comprehensive testing.

    Example:
        suite = BenchmarkSuite("channel_processing")
        suite.add_benchmark(fft_bench)
        suite.add_benchmark(convolution_bench)
        results = suite.run_all()
        suite.print_report()
    """

    def __init__(self, name: str):
        """Initialize benchmark suite.

        Args:
            name: Suite name
        """
        self.name = name
        self.benchmarks: List[Benchmark] = []
        self.results: Dict[str, BenchmarkResult] = {}

    def add_benchmark(self, benchmark: Benchmark) -> "BenchmarkSuite":
        """Add a benchmark to the suite.

        Args:
            benchmark: Benchmark instance

        Returns:
            Self for chaining
        """
        self.benchmarks.append(benchmark)
        return self

    def add(
        self,
        name: str,
        func: Callable,
        setup: Optional[Callable] = None,
        n_samples: int = 0,
        use_gpu: bool = False,
    ) -> "BenchmarkSuite":
        """Add a benchmark with simplified interface.

        Args:
            name: Benchmark name
            func: Function to benchmark
            setup: Optional setup function
            n_samples: Samples per iteration
            use_gpu: Use GPU timing

        Returns:
            Self for chaining
        """
        self.benchmarks.append(Benchmark(
            name=name,
            func=func,
            setup=setup,
            n_samples=n_samples,
            use_gpu=use_gpu,
        ))
        return self

    def run_all(self, iterations: int = 100) -> Dict[str, BenchmarkResult]:
        """Run all benchmarks in the suite.

        Args:
            iterations: Iterations per benchmark

        Returns:
            Dictionary mapping benchmark names to results
        """
        self.results = {}

        for bench in self.benchmarks:
            print(f"Running benchmark: {bench.name}...")
            result = bench.run(iterations)
            self.results[bench.name] = result

        return self.results

    def print_report(self) -> None:
        """Print benchmark results report."""
        print("\n" + "=" * 70)
        print(f"BENCHMARK SUITE: {self.name}")
        print("=" * 70)

        if not self.results:
            print("No results. Run run_all() first.")
            return

        print(f"\n{'Benchmark':<30} {'Mean (ms)':>12} {'Std':>10} {'Throughput':>15}")
        print("-" * 70)

        for name, result in self.results.items():
            tp_str = ""
            if result.throughput_msps > 0:
                tp_str = f"{result.throughput_msps:.2f} Msps"

            print(
                f"{name:<30} {result.mean_time_ms:>10.3f}ms "
                f"{result.std_time_ms:>9.3f} {tp_str:>15}"
            )

        print("=" * 70)

    def compare(self, baseline_name: str) -> Dict[str, float]:
        """Compare benchmarks against a baseline.

        Args:
            baseline_name: Name of baseline benchmark

        Returns:
            Dictionary of speedup ratios (>1 means faster than baseline)
        """
        if baseline_name not in self.results:
            raise ValueError(f"Baseline '{baseline_name}' not found")

        baseline = self.results[baseline_name]
        comparisons = {}

        for name, result in self.results.items():
            if name != baseline_name:
                speedup = baseline.mean_time_ms / result.mean_time_ms
                comparisons[name] = speedup

        return comparisons


def run_throughput_benchmark(
    func: Callable,
    sample_sizes: List[int] = None,
    iterations: int = 50,
    use_gpu: bool = False,
) -> Dict[int, BenchmarkResult]:
    """Run throughput scaling benchmark across sample sizes.

    Args:
        func: Function to benchmark (receives array as first arg)
        sample_sizes: List of sample sizes to test
        iterations: Iterations per size
        use_gpu: Use GPU timing

    Returns:
        Dictionary mapping sample sizes to results
    """
    if sample_sizes is None:
        sample_sizes = [1024, 4096, 16384, 65536, 262144]

    results = {}

    for n in sample_sizes:
        def setup():
            return (np.random.randn(n) + 1j * np.random.randn(n)).astype(np.complex64)

        bench = Benchmark(
            name=f"n={n}",
            func=func,
            setup=setup,
            n_samples=n,
            use_gpu=use_gpu,
        )

        results[n] = bench.run(iterations)
        print(f"  n={n}: {results[n].throughput_msps:.2f} Msps")

    return results


def run_latency_benchmark(
    func: Callable,
    n_samples: int = 4096,
    iterations: int = 1000,
    percentiles: List[int] = None,
) -> Dict[str, float]:
    """Run latency benchmark with percentile analysis.

    Args:
        func: Function to benchmark
        n_samples: Fixed sample size
        iterations: Number of iterations
        percentiles: Percentiles to compute (default: [50, 90, 95, 99])

    Returns:
        Dictionary with mean, min, max, and percentiles in microseconds
    """
    if percentiles is None:
        percentiles = [50, 90, 95, 99]

    latencies_us: List[float] = []

    # Warmup
    for _ in range(10):
        data = (np.random.randn(n_samples) + 1j * np.random.randn(n_samples)).astype(np.complex64)
        _ = func(data)

    # Measure
    for _ in range(iterations):
        data = (np.random.randn(n_samples) + 1j * np.random.randn(n_samples)).astype(np.complex64)
        start = time.perf_counter()
        _ = func(data)
        elapsed_us = (time.perf_counter() - start) * 1_000_000
        latencies_us.append(elapsed_us)

    result = {
        "mean_us": statistics.mean(latencies_us),
        "min_us": min(latencies_us),
        "max_us": max(latencies_us),
        "std_us": statistics.stdev(latencies_us),
    }

    # Compute percentiles
    sorted_latencies = sorted(latencies_us)
    for p in percentiles:
        idx = int(len(sorted_latencies) * p / 100)
        result[f"p{p}_us"] = sorted_latencies[min(idx, len(sorted_latencies) - 1)]

    return result


def benchmark(
    name: Optional[str] = None,
    n_samples: int = 0,
    iterations: int = 100,
    use_gpu: bool = False,
    print_result: bool = True,
) -> Callable:
    """Decorator to benchmark a function.

    Args:
        name: Custom name (defaults to function name)
        n_samples: Samples processed per call
        iterations: Benchmark iterations
        use_gpu: Use GPU timing
        print_result: Print result after first run

    Returns:
        Decorated function with .benchmark() method

    Example:
        @benchmark(n_samples=4096, print_result=True)
        def process_signal(data):
            return np.fft.fft(data)

        # Normal use
        result = process_signal(my_data)

        # Run benchmark
        bench_result = process_signal.benchmark(iterations=100)
    """
    def decorator(func: Callable) -> Callable:
        func_name = name or func.__qualname__

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            return func(*args, **kwargs)

        def run_benchmark(iters: int = iterations) -> BenchmarkResult:
            """Run benchmark on this function."""
            bench = Benchmark(
                name=func_name,
                func=func,
                n_samples=n_samples,
                use_gpu=use_gpu,
            )
            result = bench.run(iters)
            if print_result:
                print(result)
            return result

        wrapper.benchmark = run_benchmark
        wrapper._benchmarked = True

        return wrapper
    return decorator


# Pre-built benchmarks for common operations
def create_standard_benchmarks() -> BenchmarkSuite:
    """Create standard benchmark suite for HF channel processing.

    Returns:
        BenchmarkSuite with common operation benchmarks
    """
    suite = BenchmarkSuite("hf_channel_standard")

    # FFT benchmarks
    for n in [1024, 4096, 16384]:
        def make_setup(size):
            return lambda: (np.random.randn(size) + 1j * np.random.randn(size)).astype(np.complex64)

        suite.add_benchmark(Benchmark(
            name=f"numpy_fft_{n}",
            func=np.fft.fft,
            setup=make_setup(n),
            n_samples=n,
        ))

    # Overlap-save convolution
    def overlap_save_bench(data, H):
        """Simple overlap-save implementation for benchmarking."""
        block_size = len(H)
        overlap = block_size // 4
        output_size = block_size - overlap
        n_blocks = (len(data) + output_size - 1) // output_size

        output = np.zeros(n_blocks * output_size, dtype=np.complex64)
        padded = np.zeros(n_blocks * output_size + overlap, dtype=np.complex64)
        padded[overlap:overlap + len(data)] = data

        for i in range(n_blocks):
            start = i * output_size
            block = padded[start:start + block_size]
            X = np.fft.fft(block)
            Y = X * H
            y = np.fft.ifft(Y)
            output[i * output_size:(i + 1) * output_size] = y[overlap:]

        return output[:len(data)]

    def setup_overlap_save():
        data = (np.random.randn(65536) + 1j * np.random.randn(65536)).astype(np.complex64)
        H = np.ones(4096, dtype=np.complex64)
        return (data, H)

    suite.add_benchmark(Benchmark(
        name="overlap_save_65536",
        func=lambda d, h: overlap_save_bench(d, h),
        setup=setup_overlap_save,
        n_samples=65536,
    ))

    return suite
