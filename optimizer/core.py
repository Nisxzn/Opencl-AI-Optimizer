"""
Core Optimizer Pipeline
========================
The main public entry point for the OpenCL AI Inference Optimizer.

One-liner API:
    from optimizer import optimize_model
    optimized = optimize_model(model)
    output    = optimized.predict(input_data)

What happens internally:
    1. ModelParser  — extracts LayerSpecs from the input model
    2. GraphOptimizer — applies fusion + dead-layer elimination
    3. Layer factory — instantiates optimized DenseLayer / Conv2DLayer /
                       PoolingLayer / ActivationLayer backed by OpenCL kernels
    4. OptimizedModel — wraps the layer list into a predict()-able object
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Any, List, Optional, Dict

from optimizer.backend.opencl_device import OpenCLDevice, get_device
from optimizer.compiler.model_parser  import ModelParser, LayerSpec
from optimizer.compiler.graph_optimizer import GraphOptimizer
from optimizer.layers.dense      import DenseLayer
from optimizer.layers.conv2d     import Conv2DLayer
from optimizer.layers.activation import ActivationLayer
from optimizer.layers.pooling    import PoolingLayer
from optimizer.layers.base       import BaseLayer
from optimizer.utils.logger      import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class OptimizerConfig:
    """
    Configuration object for the optimize_model() call.

    Attributes:
        prefer_gpu      : Try to select a GPU device first (default True).
        platform_index  : Override platform index (default 0 = auto).
        device_index    : Override device index (default 0 = auto).
        enable_fusion   : Run activation-fusion graph pass (default True).
        verbose         : Print detailed layer summary after optimization.
        fallback_to_cpu : If True (default), silently use NumPy when OpenCL
                          fails rather than raising an exception.
    """
    prefer_gpu:     bool = True
    platform_index: int  = 0
    device_index:   int  = 0
    enable_fusion:  bool = True
    verbose:        bool = True
    fallback_to_cpu:bool = True


# ---------------------------------------------------------------------------
# Optimized Model wrapper
# ---------------------------------------------------------------------------

class OptimizedModel:
    """
    A self-contained, inference-ready model composed of OpenCL-backed layers.

    Create via optimize_model(), not directly.

    Methods:
        predict(inputs)      → np.ndarray
        summary()            → prints layer table
        benchmark(inputs, …) → BenchmarkResult vs the original model
    """

    _is_opencl_optimized = True   # marker for ModelParser

    def __init__(
        self,
        layers: List[BaseLayer],
        device: OpenCLDevice,
        original_model: Any = None,
        metadata: Optional[Dict] = None,
    ):
        self._layers        = layers
        self._device        = device
        self._original      = original_model
        self._metadata      = metadata or {}

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        Run a full forward pass through all optimized layers.

        Args:
            inputs : Input numpy array. Shape depends on model type:
                     - Dense NN : (batch, features)  or (features,)
                     - CNN      : (batch, C, H, W)   — NCHW format

        Returns:
            Output numpy array.
        """
        x = np.asarray(inputs, dtype=np.float32)

        for layer in self._layers:
            layer_type = type(layer).__name__

            # Handle flatten implicitly (between Conv and Dense blocks)
            if layer_type == "_FlattenLayer":
                N = x.shape[0]
                x = x.reshape(N, -1)
                continue

            x = layer.forward(x)

        return x

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        return self.predict(inputs)

    # ------------------------------------------------------------------
    # Benchmarking convenience method
    # ------------------------------------------------------------------

    def benchmark(
        self,
        inputs: np.ndarray,
        runs:   int = 20,
        warmup: int = 3,
        label:  str = "OptimizedModel",
    ):
        """
        Benchmark this optimized model vs the original (if available).

        Args:
            inputs : Sample input array used for timing.
            runs   : Number of timed runs.
            warmup : Warm-up runs.
            label  : Label for the benchmark result.

        Returns:
            A BenchmarkResult, or a single TimingResult if no original model.
        """
        from optimizer.utils.benchmark import Benchmark
        bench = Benchmark()

        if self._original is not None and hasattr(self._original, "predict"):
            result = bench.compare(
                fn_cpu = lambda: self._original.predict(inputs),
                fn_gpu = lambda: self.predict(inputs),
                label  = label,
                runs   = runs,
                warmup = warmup,
            )
            bench.print_result(result)
            return result
        else:
            tr = bench.time_single(
                fn     = lambda: self.predict(inputs),
                label  = label,
                runs   = runs,
                warmup = warmup,
            )
            logger.info(
                f"Benchmarked '{label}': mean={tr.mean_ms:.3f} ms, "
                f"std={tr.std_ms:.3f} ms, throughput={tr.throughput:.1f} calls/sec"
            )
            return tr

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    def summary(self) -> None:
        """Print a formatted summary of all optimized layers."""
        w = 70
        backend = "OpenCL" if self._device.available else "CPU Fallback"
        print(f"\n{'─'*w}")
        print(f"  OptimizedModel — {len(self._layers)} layers  |  Backend: {backend}")
        print(f"{'─'*w}")
        for i, layer in enumerate(self._layers):
            print(f"  [{i:02d}] {layer.summary()}")
        print(f"{'─'*w}\n")

    @property
    def layers(self) -> List[BaseLayer]:
        return self._layers

    @property
    def device_info(self) -> Dict:
        return self._device.device_info()

    def __repr__(self) -> str:
        return (
            f"OptimizedModel("
            f"layers={len(self._layers)}, "
            f"device='{self._device}')"
        )


# ---------------------------------------------------------------------------
# Internal Flatten shim
# ---------------------------------------------------------------------------

class _FlattenLayer(BaseLayer):
    """Thin shim that reshapes (N, C, H, W) → (N, C*H*W) on forward()."""

    def __init__(self):
        # Avoid calling get_device() here (unnecessary overhead)
        self.name = "flatten"
        self.device = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        N = inputs.shape[0]
        return inputs.reshape(N, -1).astype(np.float32)

    def summary(self) -> str:
        return "FlattenLayer(name='flatten')"


# ---------------------------------------------------------------------------
# Layer factory
# ---------------------------------------------------------------------------

def _build_layer(spec: LayerSpec, device: OpenCLDevice) -> Optional[BaseLayer]:
    """
    Convert a LayerSpec into a concrete optimized layer.

    Returns None for layer types that are skipped (input, unknown, etc.).
    """
    t   = spec.layer_type
    cfg = spec.config

    # --- Dense ---
    if t == "dense":
        if len(spec.weights) < 2 or spec.weights[0] is None:
            logger.warning(f"  Skipping dense layer '{spec.name}' — no weights found.")
            return None
        W, b = spec.weights[0], spec.weights[1]
        return DenseLayer(
            weights    = W,
            bias       = b,
            activation = cfg.get("activation", None),
            name       = spec.name,
            device     = device,
        )

    # --- Conv2D ---
    if t == "conv2d":
        if len(spec.weights) < 2 or spec.weights[0] is None:
            logger.warning(f"  Skipping conv2d layer '{spec.name}' — no weights found.")
            return None
        W, b = spec.weights[0], spec.weights[1]

        # Normalise Keras padding string → int
        pad_raw = cfg.get("padding", "valid")
        if isinstance(pad_raw, str):
            if pad_raw == "same":
                kH = W.shape[2]
                padding = (kH - 1) // 2
            else:   # 'valid'
                padding = 0
        else:
            padding = int(pad_raw)

        strides = cfg.get("strides", (1, 1))
        stride  = strides[0] if isinstance(strides, (tuple, list)) else int(strides)

        return Conv2DLayer(
            weights    = W,
            bias       = b,
            stride     = stride,
            padding    = padding,
            activation = cfg.get("activation", None),
            name       = spec.name,
            device     = device,
        )

    # --- Activation ---
    if t == "activation":
        act = cfg.get("activation", "linear")
        if act == "linear":
            return None   # no-op
        return ActivationLayer(activation=act, name=spec.name, device=device)

    # --- Pooling ---
    if t == "pooling":
        mode  = cfg.get("mode", "max")
        ps    = cfg.get("pool_size", (2, 2))
        pool_size = ps[0] if isinstance(ps, (tuple, list)) else int(ps)
        st    = cfg.get("strides", (2, 2))
        stride = st[0] if isinstance(st, (tuple, list)) else int(st)
        return PoolingLayer(
            mode      = mode,
            pool_size = pool_size,
            stride    = stride,
            name      = spec.name,
            device    = device,
        )

    # --- Flatten  ---
    if t == "flatten":
        return _FlattenLayer()

    # --- Skip unknowns / inputs ---
    if t not in ("unknown", "input_layer", "input"):
        logger.debug(f"  Layer '{spec.name}' (type={t}) passed through unchanged.")
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def optimize_model(
    model: Any,
    config: Optional[OptimizerConfig] = None,
) -> OptimizedModel:
    """
    Optimize a machine learning model for accelerated inference using OpenCL.

    This is the **one-liner entry point** for the entire optimizer:

        optimized = optimize_model(model)
        output    = optimized.predict(input_data)

    Args:
        model  : A Keras Sequential/Functional model, a dict spec, or any
                 supported model format. The model should already be trained.
        config : Optional OptimizerConfig to control device selection,
                 fusion passes, and verbosity.

    Returns:
        An OptimizedModel ready for inference.

    Raises:
        RuntimeError : If no layers could be extracted and fallback is disabled.
    """
    cfg = config or OptimizerConfig()

    logger.info("=" * 55)
    logger.info("  OpenCL AI Inference Optimizer — v1.0.0")
    logger.info("=" * 55)

    # 1. Select device
    device = get_device(
        platform_index = cfg.platform_index,
        device_index   = cfg.device_index,
        prefer_gpu     = cfg.prefer_gpu,
    )

    if cfg.verbose:
        info = device.device_info()
        if info.get("available"):
            logger.info(
                f"Device: {info.get('device')} "
                f"({info.get('type')}, {info.get('compute_units')} CUs)"
            )
        else:
            logger.info("Device: CPU (NumPy fallback — PyOpenCL not available)")

    # 2. Parse model → LayerSpecs
    parser = ModelParser()
    specs  = parser.parse(model)

    if not specs:
        logger.warning("No layers parsed. Returning a pass-through model.")
        return OptimizedModel(layers=[], device=device, original_model=model)

    logger.info(f"Parsed {len(specs)} layer(s) from model.")

    # 3. Graph optimizations
    if cfg.enable_fusion:
        graph_opt = GraphOptimizer()
        specs = graph_opt.optimize(specs)

    # 4. Build optimized layers
    optimized_layers: List[BaseLayer] = []
    skipped = 0
    for spec in specs:
        layer = _build_layer(spec, device)
        if layer is not None:
            optimized_layers.append(layer)
        else:
            skipped += 1

    logger.info(
        f"Built {len(optimized_layers)} optimized layer(s). "
        f"({skipped} skipped / removed by graph optimizer)"
    )

    # 5. Wrap into OptimizedModel
    opt_model = OptimizedModel(
        layers         = optimized_layers,
        device         = device,
        original_model = model,
        metadata       = {
            "original_type": type(model).__name__,
            "backend":       "OpenCL" if device.available else "CPU",
        },
    )

    if cfg.verbose:
        opt_model.summary()

    logger.info("Optimization complete. ✓")
    logger.info("=" * 55)

    return opt_model