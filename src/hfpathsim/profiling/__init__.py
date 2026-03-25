"""Profiling and benchmarking infrastructure for HF Path Simulator.

This module provides comprehensive performance profiling capabilities:
- CPU timing with decorators and context managers
- GPU kernel profiling with CUDA events
- Memory profiling (CPU and GPU)
- Throughput benchmarking
- Performance report generation

Phase 9: Performance optimization and profiling infrastructure.
"""

from .timing import (
    Timer,
    timer,
    profile_function,
    get_timing_stats,
    reset_timing_stats,
    print_timing_report,
)

from .gpu_profiler import (
    GPUProfiler,
    gpu_timer,
    CUDATimer,
    get_gpu_memory_info,
    profile_kernel,
    get_kernel_stats,
)

from .memory import (
    MemoryProfiler,
    get_memory_usage,
    track_memory,
    get_peak_memory,
    memory_profile,
)

from .benchmarks import (
    Benchmark,
    BenchmarkSuite,
    run_throughput_benchmark,
    run_latency_benchmark,
    benchmark,
)

from .reports import (
    PerformanceReport,
    generate_report,
    export_report_json,
    export_report_html,
)

__all__ = [
    # Timing
    "Timer",
    "timer",
    "profile_function",
    "get_timing_stats",
    "reset_timing_stats",
    "print_timing_report",
    # GPU profiling
    "GPUProfiler",
    "gpu_timer",
    "CUDATimer",
    "get_gpu_memory_info",
    "profile_kernel",
    "get_kernel_stats",
    # Memory
    "MemoryProfiler",
    "get_memory_usage",
    "track_memory",
    "get_peak_memory",
    "memory_profile",
    # Benchmarks
    "Benchmark",
    "BenchmarkSuite",
    "run_throughput_benchmark",
    "run_latency_benchmark",
    "benchmark",
    # Reports
    "PerformanceReport",
    "generate_report",
    "export_report_json",
    "export_report_html",
]
