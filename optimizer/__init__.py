"""
OpenCL AI Inference Optimizer
=============================
A lightweight, plug-and-play optimization engine that accelerates ML inference
using OpenCL-based parallel execution on GPU/CPU devices.

Quick Start:
    from optimizer import optimize_model
    optimized = optimize_model(model)
    output = optimized.predict(input_data)
"""

from optimizer.core import optimize_model, OptimizerConfig
from optimizer.utils.benchmark import Benchmark
from optimizer.utils.report import generate_report

__version__ = "1.0.0"
__author__ = "OpenCL AI Optimizer Team"
__license__ = "MIT"

__all__ = [
    "optimize_model",
    "OptimizerConfig",
    "Benchmark",
    "generate_report",
]