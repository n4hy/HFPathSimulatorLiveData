"""Tests for profiling and benchmarking infrastructure."""

import time
import numpy as np
import pytest

from hfpathsim.profiling import (
    # Timing
    Timer,
    timer,
    profile_function,
    get_timing_stats,
    reset_timing_stats,
    # GPU profiling
    GPUProfiler,
    gpu_timer,
    CUDATimer,
    get_gpu_memory_info,
    # Memory
    MemoryProfiler,
    get_memory_usage,
    track_memory,
    # Benchmarks
    Benchmark,
    BenchmarkSuite,
    run_throughput_benchmark,
    run_latency_benchmark,
    # Reports
    generate_report,
    export_report_json,
)


class TestTimer:
    """Tests for CPU timing utilities."""

    def test_timer_basic(self):
        """Test basic timer functionality."""
        t = Timer()
        t.start()
        time.sleep(0.01)  # 10ms
        t.stop()

        assert t.elapsed_ms >= 9  # Allow some tolerance
        assert t.elapsed_ms < 50  # Should not be too long

    def test_timer_context_manager(self):
        """Test timer as context manager."""
        with Timer() as t:
            time.sleep(0.01)

        assert t.elapsed_ms >= 9
        assert t.elapsed_ms < 50

    def test_timer_with_name(self):
        """Test timer records to global stats."""
        reset_timing_stats()

        with Timer(name="test_operation") as t:
            time.sleep(0.005)

        stats = get_timing_stats("test_operation")
        assert "test_operation" in stats
        assert stats["test_operation"].count == 1

    def test_timer_context_function(self):
        """Test timer() context function."""
        with timer("ctx_test") as t:
            time.sleep(0.005)

        assert t.elapsed_ms >= 4


class TestProfileFunction:
    """Tests for function profiling decorator."""

    def test_profile_decorator(self):
        """Test profile_function decorator."""
        reset_timing_stats()

        @profile_function(name="test_func")
        def my_function():
            time.sleep(0.005)
            return 42

        result = my_function()

        assert result == 42
        stats = get_timing_stats("test_func")
        assert "test_func" in stats
        assert stats["test_func"].count == 1

    def test_profile_multiple_calls(self):
        """Test profiling accumulates across calls."""
        reset_timing_stats()

        @profile_function(name="multi_call")
        def func():
            pass

        for _ in range(10):
            func()

        stats = get_timing_stats("multi_call")
        assert stats["multi_call"].count == 10


class TestCUDATimer:
    """Tests for GPU timing utilities."""

    def test_cuda_timer_basic(self):
        """Test CUDA timer (falls back to CPU if no GPU)."""
        t = CUDATimer()
        t.start()
        time.sleep(0.01)
        elapsed = t.stop()

        assert elapsed >= 9  # milliseconds
        assert elapsed < 100

    def test_cuda_timer_context(self):
        """Test CUDA timer as context manager."""
        with CUDATimer() as t:
            time.sleep(0.01)

        assert t.elapsed_ms >= 9

    def test_gpu_timer_context_function(self):
        """Test gpu_timer() context function."""
        with gpu_timer("gpu_test") as t:
            time.sleep(0.005)

        assert t.elapsed_ms >= 4


class TestGPUMemoryInfo:
    """Tests for GPU memory information."""

    def test_get_gpu_memory_info(self):
        """Test GPU memory info retrieval."""
        info = get_gpu_memory_info()

        # Should have valid structure even without GPU
        assert hasattr(info, "total_bytes")
        assert hasattr(info, "used_bytes")
        assert hasattr(info, "free_bytes")
        assert info.total_bytes >= 0


class TestGPUProfiler:
    """Tests for comprehensive GPU profiler."""

    def test_profiler_session(self):
        """Test GPU profiler session management."""
        profiler = GPUProfiler()
        profiler.start_session("test_session")

        with profiler.profile("operation1", n_samples=1000):
            time.sleep(0.005)

        with profiler.profile("operation2", n_samples=2000):
            time.sleep(0.005)

        report = profiler.end_session()

        assert "wall_time_ms" in report
        assert "kernels" in report
        assert len(report["kernels"]) == 2
        assert report["kernels"][0]["name"] == "operation1"
        assert report["kernels"][1]["name"] == "operation2"


class TestMemoryProfiler:
    """Tests for memory profiling."""

    def test_get_memory_usage(self):
        """Test memory usage retrieval."""
        mem = get_memory_usage()

        assert mem.rss_bytes > 0
        assert mem.rss_mb > 0

    def test_memory_profiler_basic(self):
        """Test memory profiler basic usage."""
        profiler = MemoryProfiler()
        profiler.start()

        # Allocate some memory
        data = np.zeros(1000000, dtype=np.float64)  # ~8MB

        profiler.snapshot("after_alloc")
        report = profiler.stop()

        assert "start" in report
        assert "end" in report
        assert "delta" in report
        assert "snapshots" in report
        assert "after_alloc" in report["snapshots"]

    def test_track_memory_context(self):
        """Test track_memory context manager."""
        with track_memory("allocation_test") as profiler:
            data = np.zeros(100000, dtype=np.float64)

        # Should complete without error


class TestBenchmark:
    """Tests for benchmarking framework."""

    def test_benchmark_basic(self):
        """Test basic benchmark functionality."""
        def simple_func(data):
            return np.fft.fft(data)

        bench = Benchmark(
            name="fft_test",
            func=simple_func,
            setup=lambda: (np.random.randn(1024) + 1j * np.random.randn(1024)).astype(np.complex64),
            n_samples=1024,
        )

        result = bench.run(iterations=10)

        assert result.name == "fft_test"
        assert result.iterations == 10
        assert result.mean_time_ms > 0
        assert result.throughput_msps > 0

    def test_benchmark_suite(self):
        """Test benchmark suite."""
        suite = BenchmarkSuite("test_suite")

        suite.add(
            name="func1",
            func=lambda x: x * 2,
            setup=lambda: np.random.randn(1000),
            n_samples=1000,
        )

        suite.add(
            name="func2",
            func=lambda x: np.fft.fft(x),
            setup=lambda: (np.random.randn(1024) + 1j * np.random.randn(1024)).astype(np.complex64),
            n_samples=1024,
        )

        results = suite.run_all(iterations=5)

        assert "func1" in results
        assert "func2" in results


class TestThroughputBenchmark:
    """Tests for throughput benchmarking."""

    def test_run_throughput_benchmark(self):
        """Test throughput scaling benchmark."""
        def process(data):
            return np.fft.fft(data)

        results = run_throughput_benchmark(
            func=process,
            sample_sizes=[256, 512, 1024],
            iterations=5,
        )

        assert 256 in results
        assert 512 in results
        assert 1024 in results

        # Throughput should generally be positive
        for size, result in results.items():
            assert result.throughput_msps > 0


class TestLatencyBenchmark:
    """Tests for latency benchmarking."""

    def test_run_latency_benchmark(self):
        """Test latency benchmark with percentiles."""
        def process(data):
            return data * 2

        results = run_latency_benchmark(
            func=process,
            n_samples=1024,
            iterations=100,
            percentiles=[50, 90, 99],
        )

        assert "mean_us" in results
        assert "min_us" in results
        assert "max_us" in results
        assert "p50_us" in results
        assert "p90_us" in results
        assert "p99_us" in results

        # p50 <= p90 <= p99
        assert results["p50_us"] <= results["p90_us"]
        assert results["p90_us"] <= results["p99_us"]


class TestReports:
    """Tests for report generation."""

    def test_generate_report(self):
        """Test report generation."""
        reset_timing_stats()

        # Generate some timing data
        with timer("report_test"):
            time.sleep(0.005)

        report = generate_report()

        assert report.timestamp is not None
        assert "report_test" in report.timing_stats

    def test_export_report_json(self, tmp_path):
        """Test JSON export."""
        report = generate_report()
        filepath = tmp_path / "report.json"

        export_report_json(report, str(filepath))

        assert filepath.exists()

        # Verify it's valid JSON
        import json
        with open(filepath) as f:
            data = json.load(f)

        assert "timestamp" in data
        assert "system_info" in data


@pytest.mark.benchmark
class TestPerformanceBenchmarks:
    """Performance benchmark tests."""

    def test_fft_throughput(self):
        """Benchmark FFT throughput."""
        def fft_process(data):
            return np.fft.fft(data)

        results = run_throughput_benchmark(
            func=fft_process,
            sample_sizes=[4096, 16384],
            iterations=20,
        )

        # FFT should achieve reasonable throughput
        for size, result in results.items():
            assert result.throughput_msps > 1  # At least 1 Msps

    def test_overlap_save_throughput(self):
        """Benchmark overlap-save convolution."""
        def overlap_save(data):
            block_size = 4096
            H = np.ones(block_size, dtype=np.complex64)

            # Simple overlap-save
            output = np.zeros_like(data)
            n_blocks = len(data) // block_size

            for i in range(n_blocks):
                start = i * block_size
                block = data[start:start + block_size]
                X = np.fft.fft(block)
                Y = X * H
                y = np.fft.ifft(Y)
                output[start:start + block_size] = y

            return output

        bench = Benchmark(
            name="overlap_save",
            func=overlap_save,
            setup=lambda: (np.random.randn(65536) + 1j * np.random.randn(65536)).astype(np.complex64),
            n_samples=65536,
        )

        result = bench.run(iterations=10)

        assert result.throughput_msps > 0
        print(f"\nOverlap-save throughput: {result.throughput_msps:.2f} Msps")
