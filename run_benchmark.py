#!/usr/bin/env python
"""
run_benchmark.py — CLI Benchmark Tool
======================================
Runs a comprehensive CPU vs OpenCL benchmark across all supported tensor
operations and model sizes.  Saves results to reports/.

Usage:
    python run_benchmark.py              # full suite (all ops)
    python run_benchmark.py --quick      # fast subset (3 sizes each)
    python run_benchmark.py --op matmul  # single op only
    python run_benchmark.py --no-report  # skip report generation

Output:
    • Live table printed to console
    • reports/benchmark_report.md
    • reports/benchmark_results.json
"""

from __future__ import annotations

import argparse
import os
import sys
import numpy as np
import time

# Allow running from any directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from optimizer.backend.opencl_device import get_device
from optimizer.backend.fallback import CPUFallback
from optimizer.ops.matmul      import MatMulOp
from optimizer.ops.conv2d      import Conv2DOp
from optimizer.ops.activations import ActivationOp
from optimizer.ops.pooling     import PoolingOp
from optimizer.utils.benchmark import Benchmark
from optimizer.utils.report    import generate_report


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _sep(title: str = "", width: int = 62) -> None:
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{'─' * pad} {title} {'─' * pad}")
    else:
        print("─" * width)


def _runs(quick: bool) -> tuple[int, int]:
    """Return (runs, warmup) based on speed preference."""
    return (6, 1) if quick else (20, 3)


# ──────────────────────────────────────────────────────────────────────────────
# Per-op benchmark functions
# ──────────────────────────────────────────────────────────────────────────────

def bench_matmul(bench: Benchmark, op: MatMulOp, quick: bool) -> list:
    runs, warmup = _runs(quick)
    configs = [(256, 512, 256), (512, 1024, 512), (1024, 2048, 1024)] if quick else  [
        (64,    128,   64),
        (256,   512,  256),
        (512,  1024,  512),
        (1024, 2048, 1024),
    ]
    _sep("MatMul  A @ B")
    results = []
    for M, K, N in configs:
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)
        r = bench.compare(
            fn_cpu = lambda A=A, B=B: CPUFallback.matmul(A, B),
            fn_gpu = lambda A=A, B=B: op.matmul(A, B),
            label  = f"MatMul ({M}×{K}) @ ({K}×{N})",
            runs=runs, warmup=warmup,
        )
        bench.print_result(r)
        results.append(r)
    return results


def bench_dense(bench: Benchmark, op: MatMulOp, quick: bool) -> list:
    runs, warmup = _runs(quick)
    configs = [(64, 512, 256), (128, 1024, 512), (256, 2048, 1024)] if quick else [
        (32,  128,   64),
        (64,  512,  256),
        (128, 1024,  512),
        (256, 2048, 1024),
    ]
    _sep("Dense Layer  (batch @ weights + bias)")
    results = []
    for B, I, O in configs:
        x = np.random.randn(B, I).astype(np.float32)
        W = np.random.randn(I, O).astype(np.float32)
        b = np.zeros(O, dtype=np.float32)
        r = bench.compare(
            fn_cpu = lambda x=x, W=W, b=b: CPUFallback.dense_forward(x, W, b),
            fn_gpu = lambda x=x, W=W, b=b: op.dense_forward(x, W, b),
            label  = f"Dense  B={B}  in={I}  out={O}",
            runs=runs, warmup=warmup,
        )
        bench.print_result(r)
        results.append(r)
    return results


def bench_activations(bench: Benchmark, act_op: ActivationOp, quick: bool) -> list:
    runs, warmup = _runs(quick)
    sizes = [100_000, 1_000_000] if quick else [10_000, 100_000, 1_000_000]
    acts  = ["relu", "sigmoid", "softmax"]
    _sep("Activation Functions")
    results = []
    for act in acts:
        for n in sizes:
            x = np.random.randn(n).astype(np.float32)
            r = bench.compare(
                fn_cpu = lambda x=x, a=act: ActivationOp._cpu_apply(a, x, 0.01),
                fn_gpu = lambda x=x, a=act: act_op.apply(a, x),
                label  = f"{act.upper():10s}  n={n:>9,}",
                runs=runs, warmup=warmup,
            )
            bench.print_result(r)
            results.append(r)
    return results


def bench_conv2d(bench: Benchmark, conv_op: Conv2DOp, quick: bool) -> list:
    runs, warmup = _runs(quick)
    configs = [
        (4,  8, 14, 14, 16, 3, 1, 1),
        (8, 16,  7,  7, 32, 3, 1, 1),
    ] if quick else [
        (1,  1, 28, 28,  8, 3, 1, 1),
        (4,  8, 14, 14, 16, 3, 1, 1),
        (8, 16,  7,  7, 32, 3, 1, 1),
    ]
    _sep("Conv2D  (NCHW format)")
    results = []
    for N, Ci, H, W, Co, K, st, pd in configs:
        x  = np.random.randn(N, Ci, H, W).astype(np.float32)
        wt = np.random.randn(Co, Ci, K, K).astype(np.float32) * 0.05
        b  = np.zeros(Co, dtype=np.float32)
        r  = bench.compare(
            fn_cpu = lambda x=x, wt=wt, b=b, st=st, pd=pd: CPUFallback.conv2d(x, wt, b, st, pd),
            fn_gpu = lambda x=x, wt=wt, b=b, st=st, pd=pd: conv_op.conv2d(x, wt, b, st, pd),
            label  = f"Conv2D  N={N}  {Ci}→{Co}ch  {H}×{W}  k={K}×{K}",
            runs=runs, warmup=warmup,
        )
        bench.print_result(r)
        results.append(r)
    return results


def bench_pooling(bench: Benchmark, pool_op: PoolingOp, quick: bool) -> list:
    runs, warmup = _runs(quick)
    configs = [(4, 16, 14, 14), (8, 32, 7, 7)] if quick else [
        (1,  8, 28, 28),
        (4, 16, 14, 14),
        (8, 32,  7,  7),
    ]
    _sep("Max Pooling 2D  (NCHW)")
    results = []
    for N, C, H, W in configs:
        x = np.random.randn(N, C, H, W).astype(np.float32)
        r = bench.compare(
            fn_cpu = lambda x=x: CPUFallback.max_pool2d(x, 2, 2),
            fn_gpu = lambda x=x: pool_op.max_pool(x, 2, 2),
            label  = f"MaxPool2D  N={N}  {C}ch  {H}×{W}",
            runs=runs, warmup=warmup,
        )
        bench.print_result(r)
        results.append(r)
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

OP_MAP = {
    "matmul":      bench_matmul,
    "dense":       bench_dense,
    "activations": bench_activations,
    "conv2d":      bench_conv2d,
    "pooling":     bench_pooling,
}


def main():
    parser = argparse.ArgumentParser(
        description="OpenCL AI Optimizer — CPU vs GPU Benchmark CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--op",
        choices=list(OP_MAP),
        default=None,
        help="Benchmark a single operation (default: all)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fast subset — fewer sizes, fewer runs (good for development)",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip writing Markdown / JSON report files",
    )
    args = parser.parse_args()

    SEP = "=" * 62
    print(f"\n{SEP}")
    print("  OpenCL AI Optimizer — Benchmark Suite")
    print(f"{SEP}")

    np.random.seed(123)

    device = get_device()
    print(f"\n  Device : {device}")
    info = device.device_info()
    if info.get("available"):
        print(f"  Local  : {info.get('local_mem_kb', '?')} KB  |  "
              f"Max WG size: {info.get('max_work_group_size', '?')}\n")
    else:
        print("  ⚠️  OpenCL unavailable — all results will show CPU-only.\n")

    # Instantiate ops
    matmul_op = MatMulOp(device=device)
    conv_op   = Conv2DOp(device=device)
    act_op    = ActivationOp(device=device)
    pool_op   = PoolingOp(device=device)

    bench = Benchmark()
    all_results = []

    op_funcs = {
        "matmul":      (bench_matmul,      (bench, matmul_op, args.quick)),
        "dense":       (bench_dense,       (bench, matmul_op, args.quick)),
        "activations": (bench_activations, (bench, act_op,    args.quick)),
        "conv2d":      (bench_conv2d,      (bench, conv_op,   args.quick)),
        "pooling":     (bench_pooling,     (bench, pool_op,   args.quick)),
    }

    ops_to_run = [args.op] if args.op else list(op_funcs)
    for key in ops_to_run:
        fn, fn_args = op_funcs[key]
        all_results += fn(*fn_args)

    # ── Summary ──────────────────────────────────────────────────────────────
    _sep("Summary")
    ocl_wins    = sum(1 for r in all_results if r.winner == "OpenCL")
    cpu_wins    = len(all_results) - ocl_wins
    avg_speedup = float(np.mean([r.speedup for r in all_results]))
    best        = max(all_results, key=lambda r: r.speedup)

    print(f"\n  Total benchmarks : {len(all_results)}")
    print(f"  OpenCL wins      : {ocl_wins}")
    print(f"  CPU wins         : {cpu_wins}")
    print(f"  Avg speedup      : {avg_speedup:.2f}x")
    print(f"  Best speedup     : {best.speedup:.2f}x  ({best.label})")
    print()

    # ── Reports ───────────────────────────────────────────────────────────────
    if not args.no_report:
        reports_dir = os.path.join(os.path.dirname(__file__), "reports")
        os.makedirs(reports_dir, exist_ok=True)

        bench.export_all_json(os.path.join(reports_dir, "benchmark_results.json"))
        generate_report(
            results     = all_results,
            device_info = device.device_info(),
            output_path = os.path.join(reports_dir, "benchmark_report.md"),
            title       = "OpenCL AI Optimizer — Benchmark Report",
        )
        print(f"  📄  Reports saved → {reports_dir}/")

    print(f"\n{SEP}")
    print("  Benchmark complete ✅")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
