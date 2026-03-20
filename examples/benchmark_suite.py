"""
Comprehensive Benchmark Suite
==============================
Runs a battery of micro-benchmarks comparing CPU (NumPy) vs OpenCL
performance across all supported operations and model sizes.

Results are printed to console AND saved as:
  - reports/benchmark_suite_results.json
  - reports/benchmark_suite_report.md

Run:
    python examples/benchmark_suite.py
"""

import numpy as np
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from optimizer.backend.opencl_device import get_device
from optimizer.backend.fallback import CPUFallback
from optimizer.ops.matmul      import MatMulOp
from optimizer.ops.conv2d      import Conv2DOp
from optimizer.ops.activations import ActivationOp
from optimizer.ops.pooling     import PoolingOp
from optimizer.utils.benchmark import Benchmark
from optimizer.utils.report    import generate_report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def header(title: str) -> None:
    print(f"\n{'━'*55}")
    print(f"  {title}")
    print(f"{'━'*55}")


# ---------------------------------------------------------------------------
# Op-level benchmarks
# ---------------------------------------------------------------------------

def bench_matmul(bench: Benchmark, op: MatMulOp) -> list:
    results = []
    configs = [
        (64,   128,  64),
        (256,  512, 256),
        (512, 1024, 512),
        (1024, 2048, 1024),
    ]
    header("MatMul Benchmarks")
    for M, K, N in configs:
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)
        r = bench.compare(
            fn_cpu = lambda A=A, B=B: CPUFallback.matmul(A, B),
            fn_gpu = lambda A=A, B=B: op.matmul(A, B),
            label  = f"MatMul ({M}×{K}) @ ({K}×{N})",
            runs   = 20,
            warmup = 3,
        )
        bench.print_result(r)
        results.append(r)
    return results


def bench_dense(bench: Benchmark, op: MatMulOp) -> list:
    results = []
    configs = [
        (32,  128,  64),
        (64,  512, 256),
        (128, 1024, 512),
        (256, 2048, 1024),
    ]
    header("Dense Layer Forward Benchmarks")
    for B, I, O in configs:
        x = np.random.randn(B, I).astype(np.float32)
        W = np.random.randn(I, O).astype(np.float32)
        b = np.zeros(O, dtype=np.float32)
        r = bench.compare(
            fn_cpu = lambda x=x,W=W,b=b: CPUFallback.dense_forward(x, W, b),
            fn_gpu = lambda x=x,W=W,b=b: op.dense_forward(x, W, b),
            label  = f"Dense  batch={B}  in={I}  out={O}",
            runs   = 20,
            warmup = 3,
        )
        bench.print_result(r)
        results.append(r)
    return results


def bench_activations(bench: Benchmark, act_op: ActivationOp) -> list:
    results = []
    sizes = [10_000, 100_000, 1_000_000]
    activations = ["relu", "sigmoid", "tanh"]
    header("Activation Benchmarks")
    for act in activations:
        for n in sizes:
            x = np.random.randn(n).astype(np.float32)
            r = bench.compare(
                fn_cpu = lambda x=x, a=act: ActivationOp._cpu_apply(a, x, 0.01),
                fn_gpu = lambda x=x, a=act: act_op.apply(a, x),
                label  = f"{act.upper()}  n={n:,}",
                runs   = 15,
                warmup = 3,
            )
            bench.print_result(r)
            results.append(r)
    return results


def bench_conv2d(bench: Benchmark, conv_op: Conv2DOp) -> list:
    results = []
    configs = [
        # (N, C_in, H, W, C_out, K, stride, pad)
        (1,  1,  28, 28,  8, 3, 1, 1),
        (4,  8,  14, 14, 16, 3, 1, 1),
        (8, 16,   7,  7, 32, 3, 1, 1),
    ]
    header("Conv2D Benchmarks")
    for N, Ci, H, W, Co, K, st, pd in configs:
        x  = np.random.randn(N, Ci, H, W).astype(np.float32)
        wt = np.random.randn(Co, Ci, K, K).astype(np.float32) * 0.05
        b  = np.zeros(Co, dtype=np.float32)
        r  = bench.compare(
            fn_cpu = lambda x=x,wt=wt,b=b,st=st,pd=pd: CPUFallback.conv2d(x,wt,b,st,pd),
            fn_gpu = lambda x=x,wt=wt,b=b,st=st,pd=pd: conv_op.conv2d(x,wt,b,st,pd),
            label  = f"Conv2D  N={N} {Ci}→{Co}ch  {H}×{W}  k={K}",
            runs   = 10,
            warmup = 2,
        )
        bench.print_result(r)
        results.append(r)
    return results


def bench_pooling(bench: Benchmark, pool_op: PoolingOp) -> list:
    results = []
    configs = [
        (1, 8, 28, 28),
        (4, 16, 14, 14),
        (8, 32, 7, 7),
    ]
    header("Max Pooling Benchmarks")
    for N, C, H, W in configs:
        x = np.random.randn(N, C, H, W).astype(np.float32)
        r = bench.compare(
            fn_cpu = lambda x=x: CPUFallback.max_pool2d(x, 2, 2),
            fn_gpu = lambda x=x: pool_op.max_pool(x, 2, 2),
            label  = f"MaxPool2D  N={N} {C}ch {H}×{W}",
            runs   = 20,
            warmup = 3,
        )
        bench.print_result(r)
        results.append(r)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 55)
    print("  OpenCL AI Optimizer — Benchmark Suite")
    print("=" * 55)

    np.random.seed(123)
    device = get_device()
    print(f"\nDevice: {device}\n")

    # Instantiate ops
    matmul_op = MatMulOp(device=device)
    conv_op   = Conv2DOp(device=device)
    act_op    = ActivationOp(device=device)
    pool_op   = PoolingOp(device=device)

    bench = Benchmark()
    all_results = []

    # Run all benchmarks
    all_results += bench_matmul(bench, matmul_op)
    all_results += bench_dense(bench, matmul_op)
    all_results += bench_activations(bench, act_op)
    all_results += bench_conv2d(bench, conv_op)
    all_results += bench_pooling(bench, pool_op)

    # Summary
    header("Overall Summary")
    ocl_wins  = sum(1 for r in all_results if r.winner == "OpenCL")
    cpu_wins  = len(all_results) - ocl_wins
    avg_speedup = np.mean([r.speedup for r in all_results])
    max_speedup = max(r.speedup for r in all_results)
    best_label  = max(all_results, key=lambda r: r.speedup).label

    print(f"  Total benchmarks  : {len(all_results)}")
    print(f"  OpenCL wins       : {ocl_wins}")
    print(f"  CPU wins          : {cpu_wins}")
    print(f"  Average speedup   : {avg_speedup:.2f}x")
    print(f"  Best speedup      : {max_speedup:.2f}x  ({best_label})")

    # Export reports
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "reports"), exist_ok=True)

    bench.export_all_json(
        os.path.join(os.path.dirname(__file__), "..", "reports", "benchmark_suite_results.json")
    )

    generate_report(
        results     = all_results,
        device_info = device.device_info(),
        output_path = os.path.join(
            os.path.dirname(__file__), "..", "reports", "benchmark_suite_report.md"
        ),
        title       = "OpenCL AI Optimizer — Full Benchmark Suite",
    )

    print("\n✅ Benchmark suite complete. Reports saved to reports/")


if __name__ == "__main__":
    main()
