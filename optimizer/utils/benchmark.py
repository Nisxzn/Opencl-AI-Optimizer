"""
Benchmark System
================
Measures and compares execution time between CPU (NumPy) and OpenCL
implementations of the same operation / model inference.

Usage:
    bench = Benchmark()
    result = bench.compare(
        fn_cpu=lambda: model_cpu.predict(x),
        fn_gpu=lambda: model_gpu.predict(x),
        label="Dense NN inference",
        runs=20,
    )
    bench.print_result(result)
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Dict, Any
from optimizer.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TimingResult:
    """Raw timing data for a single backend."""
    label:       str
    times_ms:    List[float]          # per-run times in milliseconds
    mean_ms:     float = 0.0
    std_ms:      float = 0.0
    min_ms:      float = 0.0
    max_ms:      float = 0.0
    throughput:  float = 0.0          # runs per second

    def compute_stats(self) -> None:
        arr = np.array(self.times_ms)
        self.mean_ms    = float(arr.mean())
        self.std_ms     = float(arr.std())
        self.min_ms     = float(arr.min())
        self.max_ms     = float(arr.max())
        self.throughput = 1000.0 / self.mean_ms if self.mean_ms > 0 else 0.0


@dataclass
class BenchmarkResult:
    """Full comparison between CPU and GPU backends."""
    label:       str
    cpu:         TimingResult
    gpu:         TimingResult
    speedup:     float = 0.0          # cpu_mean / gpu_mean
    winner:      str  = ""
    runs:        int  = 0
    warmup_runs: int  = 0
    metadata:    Dict[str, Any] = field(default_factory=dict)

    def compute(self) -> None:
        self.cpu.compute_stats()
        self.gpu.compute_stats()
        if self.gpu.mean_ms > 0:
            self.speedup = self.cpu.mean_ms / self.gpu.mean_ms
        self.winner = "OpenCL" if self.speedup >= 1.0 else "CPU"


# ---------------------------------------------------------------------------
# Benchmark class
# ---------------------------------------------------------------------------

class Benchmark:
    """
    Utility class for timing and comparing CPU vs OpenCL inference.

    Example::

        bench = Benchmark()

        result = bench.compare(
            fn_cpu    = lambda: cpu_model.predict(x),
            fn_gpu    = lambda: ocl_model.predict(x),
            label     = "CNN inference (batch=32)",
            runs      = 50,
            warmup    = 5,
        )

        bench.print_result(result)
        bench.export_json(result, "benchmark_cnn.json")
    """

    def __init__(self):
        self._results: List[BenchmarkResult] = []

    # ------------------------------------------------------------------
    # Core timing primitive
    # ------------------------------------------------------------------

    @staticmethod
    def time_function(fn: Callable, runs: int = 10, warmup: int = 3) -> List[float]:
        """
        Execute *fn* repeatedly and return per-run times in milliseconds.

        Args:
            fn     : Zero-argument callable to benchmark.
            runs   : Number of timed iterations.
            warmup : Number of un-timed warm-up iterations.

        Returns:
            List of elapsed times in milliseconds.
        """
        # Warm-up (lets JIT / OpenCL kernel compilation settle)
        for _ in range(warmup):
            try:
                fn()
            except Exception as e:
                logger.warning(f"Warm-up raised: {e}")
                return [float("inf")] * runs

        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            fn()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)   # → ms
        return times

    # ------------------------------------------------------------------
    # CPU vs GPU comparison
    # ------------------------------------------------------------------

    def compare(
        self,
        fn_cpu:   Callable,
        fn_gpu:   Callable,
        label:    str = "benchmark",
        runs:     int = 20,
        warmup:   int = 3,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BenchmarkResult:
        """
        Time both CPU and GPU implementations and return a BenchmarkResult.

        Args:
            fn_cpu   : CPU callable (no arguments).
            fn_gpu   : GPU/OpenCL callable (no arguments).
            label    : Human-readable benchmark label.
            runs     : Number of timed runs per backend.
            warmup   : Warm-up runs (not counted).
            metadata : Optional dict of extra info to embed in result.

        Returns:
            A fully computed BenchmarkResult.
        """
        logger.info(f"Benchmarking '{label}' — {runs} runs, {warmup} warm-up …")

        cpu_times = self.time_function(fn_cpu, runs=runs, warmup=warmup)
        gpu_times = self.time_function(fn_gpu, runs=runs, warmup=warmup)

        result = BenchmarkResult(
            label       = label,
            cpu         = TimingResult(label="CPU (NumPy)", times_ms=cpu_times),
            gpu         = TimingResult(label="OpenCL",      times_ms=gpu_times),
            runs        = runs,
            warmup_runs = warmup,
            metadata    = metadata or {},
        )
        result.compute()
        self._results.append(result)
        return result

    # ------------------------------------------------------------------
    # Single-backend timing helper
    # ------------------------------------------------------------------

    def time_single(
        self,
        fn:     Callable,
        label:  str = "run",
        runs:   int = 10,
        warmup: int = 2,
    ) -> TimingResult:
        """Time a single callable and return a TimingResult."""
        times = self.time_function(fn, runs=runs, warmup=warmup)
        tr = TimingResult(label=label, times_ms=times)
        tr.compute_stats()
        return tr

    # ------------------------------------------------------------------
    # Pretty printing
    # ------------------------------------------------------------------

    @staticmethod
    def print_result(result: BenchmarkResult) -> None:
        """Print a formatted benchmark comparison table to stdout."""
        w = 60
        sep = "─" * w

        print(f"\n┌{sep}┐")
        print(f"│  📊  BENCHMARK: {result.label:<{w - 18}}│")
        print(f"├{sep}┤")
        print(f"│  {'Runs':<20} {result.runs:<10} {'Warm-up':<10} {result.warmup_runs:<8}│")
        print(f"├{sep}┤")

        for tr in [result.cpu, result.gpu]:
            print(f"│  {tr.label:<20}                                    │")
            print(f"│    Mean  : {tr.mean_ms:>8.3f} ms   Std: {tr.std_ms:>8.3f} ms            │")
            print(f"│    Min   : {tr.min_ms:>8.3f} ms   Max: {tr.max_ms:>8.3f} ms            │")
            print(f"│    Speed : {tr.throughput:>8.1f} calls/sec                        │")
            print(f"├{sep}┤")

        winner_icon = "🚀" if result.winner == "OpenCL" else "💻"
        print(f"│  {winner_icon}  Speedup  : {result.speedup:.2f}x   Winner: {result.winner:<30} │")
        print(f"└{sep}┘\n")

    def print_all(self) -> None:
        """Print all stored benchmark results."""
        if not self._results:
            print("No benchmarks recorded yet.")
            return
        for r in self._results:
            self.print_result(r)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_json(self, result: BenchmarkResult, path: str) -> None:
        """Serialize a BenchmarkResult to a JSON file."""
        import json

        data = {
            "label":   result.label,
            "runs":    result.runs,
            "warmup":  result.warmup_runs,
            "speedup": result.speedup,
            "winner":  result.winner,
            "cpu": {
                "mean_ms":  result.cpu.mean_ms,
                "std_ms":   result.cpu.std_ms,
                "min_ms":   result.cpu.min_ms,
                "max_ms":   result.cpu.max_ms,
                "times_ms": result.cpu.times_ms,
            },
            "gpu": {
                "mean_ms":  result.gpu.mean_ms,
                "std_ms":   result.gpu.std_ms,
                "min_ms":   result.gpu.min_ms,
                "max_ms":   result.gpu.max_ms,
                "times_ms": result.gpu.times_ms,
            },
            "metadata": result.metadata,
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Benchmark result exported to: {path}")

    def export_all_json(self, path: str) -> None:
        """Export all stored results to a single JSON file."""
        import json

        all_data = []
        for r in self._results:
            all_data.append({
                "label":   r.label,
                "speedup": r.speedup,
                "winner":  r.winner,
                "cpu_mean_ms": r.cpu.mean_ms,
                "gpu_mean_ms": r.gpu.mean_ms,
            })

        with open(path, "w") as f:
            json.dump(all_data, f, indent=2)
        logger.info(f"All benchmark results exported to: {path}")
