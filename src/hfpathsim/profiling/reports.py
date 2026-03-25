"""Performance report generation and export.

Provides formatted reports in text, JSON, and HTML formats
for profiling results.
"""

import json
import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path

from .timing import get_timing_stats, TimingStats
from .gpu_profiler import get_kernel_stats, get_gpu_memory_info, KernelStats, GPUMemoryInfo
from .memory import get_memory_usage, MemoryUsage
from .benchmarks import BenchmarkResult


@dataclass
class PerformanceReport:
    """Comprehensive performance report."""

    timestamp: str
    system_info: Dict[str, Any]
    timing_stats: Dict[str, Dict]
    kernel_stats: Dict[str, Dict]
    memory_info: Dict[str, Any]
    benchmark_results: Dict[str, Dict]
    custom_metrics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "timestamp": self.timestamp,
            "system_info": self.system_info,
            "timing_stats": self.timing_stats,
            "kernel_stats": self.kernel_stats,
            "memory_info": self.memory_info,
            "benchmark_results": self.benchmark_results,
            "custom_metrics": self.custom_metrics,
        }


def generate_report(
    benchmark_results: Optional[Dict[str, BenchmarkResult]] = None,
    custom_metrics: Optional[Dict[str, Any]] = None,
    include_system_info: bool = True,
) -> PerformanceReport:
    """Generate a comprehensive performance report.

    Args:
        benchmark_results: Optional benchmark results to include
        custom_metrics: Optional custom metrics
        include_system_info: Whether to gather system information

    Returns:
        PerformanceReport with all collected data
    """
    timestamp = datetime.datetime.now().isoformat()

    # Gather system info
    system_info = {}
    if include_system_info:
        system_info = _gather_system_info()

    # Timing statistics
    timing_raw = get_timing_stats()
    timing_stats = {
        name: {
            "count": s.count,
            "total_ms": s.total_ms,
            "mean_ms": s.mean_ms,
            "min_ms": s.min_ms,
            "max_ms": s.max_ms,
            "std_ms": s.std_ms,
        }
        for name, s in timing_raw.items()
    }

    # Kernel statistics
    kernel_raw = get_kernel_stats()
    kernel_stats = {
        name: {
            "count": s.count,
            "total_ms": s.total_ms,
            "mean_ms": s.mean_ms,
            "min_ms": s.min_ms,
            "max_ms": s.max_ms,
            "std_ms": s.std_ms,
        }
        for name, s in kernel_raw.items()
    }

    # Memory info
    cpu_mem = get_memory_usage(include_gpu=False)
    gpu_mem = get_gpu_memory_info()

    memory_info = {
        "cpu": {
            "rss_mb": cpu_mem.rss_mb,
            "vms_mb": cpu_mem.vms_mb,
        },
        "gpu": {
            "total_gb": gpu_mem.total_gb,
            "used_gb": gpu_mem.used_gb,
            "free_gb": gpu_mem.free_gb,
            "utilization_pct": gpu_mem.utilization_pct,
        },
    }

    # Benchmark results
    bench_dict = {}
    if benchmark_results:
        for name, result in benchmark_results.items():
            bench_dict[name] = {
                "iterations": result.iterations,
                "mean_time_ms": result.mean_time_ms,
                "std_time_ms": result.std_time_ms,
                "min_time_ms": result.min_time_ms,
                "max_time_ms": result.max_time_ms,
                "throughput_msps": result.throughput_msps,
                "throughput_mbps": result.throughput_mbps,
            }

    return PerformanceReport(
        timestamp=timestamp,
        system_info=system_info,
        timing_stats=timing_stats,
        kernel_stats=kernel_stats,
        memory_info=memory_info,
        benchmark_results=bench_dict,
        custom_metrics=custom_metrics or {},
    )


def _gather_system_info() -> Dict[str, Any]:
    """Gather system information for the report."""
    import platform
    import sys

    info = {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "platform_version": platform.version(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "python_version": sys.version,
    }

    # CPU info
    try:
        import psutil
        info["cpu_count_physical"] = psutil.cpu_count(logical=False)
        info["cpu_count_logical"] = psutil.cpu_count(logical=True)
        info["cpu_freq_mhz"] = psutil.cpu_freq().current if psutil.cpu_freq() else None
        mem = psutil.virtual_memory()
        info["total_ram_gb"] = mem.total / (1024 ** 3)
    except ImportError:
        pass

    # GPU info
    try:
        import cupy as cp
        device = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(device.id)
        info["gpu_name"] = props["name"].decode()
        info["gpu_compute_capability"] = f"{props['major']}.{props['minor']}"
        info["gpu_memory_gb"] = props["totalGlobalMem"] / (1024 ** 3)
        info["gpu_multiprocessors"] = props["multiProcessorCount"]
    except (ImportError, Exception):
        info["gpu_name"] = "Not available"

    # NumPy info
    try:
        import numpy as np
        info["numpy_version"] = np.__version__

        # Check for optimized BLAS
        config = np.__config__
        if hasattr(config, "show"):
            # NumPy 1.x
            pass
        info["numpy_blas"] = "unknown"
    except Exception:
        pass

    return info


def export_report_json(
    report: PerformanceReport,
    filepath: str,
    indent: int = 2,
) -> None:
    """Export report to JSON file.

    Args:
        report: PerformanceReport to export
        filepath: Output file path
        indent: JSON indentation
    """
    with open(filepath, "w") as f:
        json.dump(report.to_dict(), f, indent=indent, default=str)


def export_report_html(
    report: PerformanceReport,
    filepath: str,
    title: str = "HF Path Simulator Performance Report",
) -> None:
    """Export report to HTML file.

    Args:
        report: PerformanceReport to export
        filepath: Output file path
        title: HTML page title
    """
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1 {{ color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .section {{
            background: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{ background: #f8f9fa; font-weight: 600; }}
        tr:hover {{ background: #f5f5f5; }}
        .metric {{ font-size: 24px; font-weight: bold; color: #007bff; }}
        .metric-label {{ color: #666; font-size: 14px; }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }}
        .metric-card {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            text-align: center;
        }}
        .timestamp {{ color: #888; font-size: 14px; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <p class="timestamp">Generated: {report.timestamp}</p>
"""

    # System info section
    if report.system_info:
        html += """
    <div class="section">
        <h2>System Information</h2>
        <div class="metrics-grid">
"""
        for key, value in report.system_info.items():
            if value is not None:
                label = key.replace("_", " ").title()
                html += f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric">{value}</div>
            </div>
"""
        html += """
        </div>
    </div>
"""

    # Memory section
    html += """
    <div class="section">
        <h2>Memory Usage</h2>
        <div class="metrics-grid">
"""
    cpu_mem = report.memory_info.get("cpu", {})
    gpu_mem = report.memory_info.get("gpu", {})

    html += f"""
            <div class="metric-card">
                <div class="metric-label">CPU RSS</div>
                <div class="metric">{cpu_mem.get('rss_mb', 0):.1f} MB</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">GPU Used</div>
                <div class="metric">{gpu_mem.get('used_gb', 0):.2f} GB</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">GPU Total</div>
                <div class="metric">{gpu_mem.get('total_gb', 0):.2f} GB</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">GPU Utilization</div>
                <div class="metric">{gpu_mem.get('utilization_pct', 0):.1f}%</div>
            </div>
        </div>
    </div>
"""

    # Timing stats section
    if report.timing_stats:
        html += """
    <div class="section">
        <h2>CPU Timing Statistics</h2>
        <table>
            <tr>
                <th>Operation</th>
                <th>Count</th>
                <th>Mean (ms)</th>
                <th>Std (ms)</th>
                <th>Min (ms)</th>
                <th>Max (ms)</th>
                <th>Total (ms)</th>
            </tr>
"""
        for name, stats in sorted(report.timing_stats.items(), key=lambda x: -x[1]["total_ms"]):
            html += f"""
            <tr>
                <td>{name}</td>
                <td>{stats['count']}</td>
                <td>{stats['mean_ms']:.3f}</td>
                <td>{stats['std_ms']:.3f}</td>
                <td>{stats['min_ms']:.3f}</td>
                <td>{stats['max_ms']:.3f}</td>
                <td>{stats['total_ms']:.1f}</td>
            </tr>
"""
        html += """
        </table>
    </div>
"""

    # Kernel stats section
    if report.kernel_stats:
        html += """
    <div class="section">
        <h2>GPU Kernel Statistics</h2>
        <table>
            <tr>
                <th>Kernel</th>
                <th>Count</th>
                <th>Mean (ms)</th>
                <th>Std (ms)</th>
                <th>Min (ms)</th>
                <th>Max (ms)</th>
            </tr>
"""
        for name, stats in sorted(report.kernel_stats.items(), key=lambda x: -x[1]["total_ms"]):
            html += f"""
            <tr>
                <td>{name}</td>
                <td>{stats['count']}</td>
                <td>{stats['mean_ms']:.3f}</td>
                <td>{stats['std_ms']:.3f}</td>
                <td>{stats['min_ms']:.3f}</td>
                <td>{stats['max_ms']:.3f}</td>
            </tr>
"""
        html += """
        </table>
    </div>
"""

    # Benchmark results section
    if report.benchmark_results:
        html += """
    <div class="section">
        <h2>Benchmark Results</h2>
        <table>
            <tr>
                <th>Benchmark</th>
                <th>Iterations</th>
                <th>Mean (ms)</th>
                <th>Std (ms)</th>
                <th>Throughput (Msps)</th>
                <th>Bandwidth (MB/s)</th>
            </tr>
"""
        for name, result in report.benchmark_results.items():
            html += f"""
            <tr>
                <td>{name}</td>
                <td>{result['iterations']}</td>
                <td>{result['mean_time_ms']:.3f}</td>
                <td>{result['std_time_ms']:.3f}</td>
                <td>{result['throughput_msps']:.2f}</td>
                <td>{result['throughput_mbps']:.2f}</td>
            </tr>
"""
        html += """
        </table>
    </div>
"""

    # Custom metrics section
    if report.custom_metrics:
        html += """
    <div class="section">
        <h2>Custom Metrics</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
"""
        for name, value in report.custom_metrics.items():
            html += f"""
            <tr>
                <td>{name}</td>
                <td>{value}</td>
            </tr>
"""
        html += """
        </table>
    </div>
"""

    html += """
</body>
</html>
"""

    with open(filepath, "w") as f:
        f.write(html)


def print_summary_report() -> None:
    """Print a quick summary of collected profiling data."""
    report = generate_report(include_system_info=False)

    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)

    # Memory
    cpu = report.memory_info.get("cpu", {})
    gpu = report.memory_info.get("gpu", {})
    print(f"\nMemory: CPU RSS {cpu.get('rss_mb', 0):.1f}MB, GPU {gpu.get('used_gb', 0):.2f}/{gpu.get('total_gb', 0):.2f}GB")

    # Top timing operations
    if report.timing_stats:
        print("\nTop CPU Operations:")
        sorted_stats = sorted(report.timing_stats.items(), key=lambda x: -x[1]["total_ms"])
        for name, stats in sorted_stats[:5]:
            print(f"  {name}: {stats['mean_ms']:.3f}ms (x{stats['count']})")

    # Top kernels
    if report.kernel_stats:
        print("\nTop GPU Kernels:")
        sorted_kernels = sorted(report.kernel_stats.items(), key=lambda x: -x[1]["total_ms"])
        for name, stats in sorted_kernels[:5]:
            print(f"  {name}: {stats['mean_ms']:.3f}ms (x{stats['count']})")

    print("=" * 70)
