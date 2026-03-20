"""
CNN Image Classifier (LeNet-style) — Production Example
=========================================================
Demonstrates the OpenCL optimizer on a real convolutional neural network:

    Architecture:
        Conv2D(1→8,  3×3, relu, pad=1) → MaxPool(2×2)
        Conv2D(8→16, 3×3, relu, pad=1) → MaxPool(2×2)
        Flatten → Dense(784→64, relu) → Dense(64→10, softmax)

This example simulates the common workflow:
  1. You have a trained model (weights loaded from disk / checkpoint)
  2. You wrap it with optimize_model() — one line
  3. You call .predict() on new data — identical interface

Input format:  NCHW — (N, Channels, Height, Width)
Tested with:   grayscale 28×28 images (MNIST-equivalent)

Run:
    python examples/cnn_classifier.py
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from optimizer import optimize_model, OptimizerConfig
from optimizer.utils.benchmark import Benchmark
from optimizer.utils.report import generate_report


# ---------------------------------------------------------------------------
# Model weights — simulates loading from a trained checkpoint
# ---------------------------------------------------------------------------

def load_pretrained_cnn(num_classes: int = 10) -> dict:
    """
    Build a LeNet-style CNN weight spec.

    In a real project you would load weights from a file:
        W = np.load("conv1_weights.npy")
    Here we use random weights scaled appropriately (simulating trained values).
    """
    np.random.seed(0)
    scale = 0.05

    W_c1 = (np.random.randn(8,  1, 3, 3) * scale).astype(np.float32)   # Conv1
    b_c1 = np.zeros(8,  dtype=np.float32)

    W_c2 = (np.random.randn(16, 8, 3, 3) * scale).astype(np.float32)   # Conv2
    b_c2 = np.zeros(16, dtype=np.float32)

    flat_size = 16 * 7 * 7   # after 2× MaxPool(2×2) on 28×28 → 7×7, 16 channels = 784

    W_d1 = (np.random.randn(flat_size, 64) * scale).astype(np.float32)  # FC1
    b_d1 = np.zeros(64, dtype=np.float32)

    W_d2 = (np.random.randn(64, num_classes) * scale).astype(np.float32)  # FC2 / output
    b_d2 = np.zeros(num_classes, dtype=np.float32)

    return {
        "layers": [
            {
                "type":    "conv2d",
                "name":    "conv1",
                "weights": [W_c1, b_c1],
                "config":  {"activation": "relu", "stride": 1, "padding": 1},
            },
            {
                "type":    "pooling",
                "name":    "pool1",
                "weights": [],
                "config":  {"mode": "max", "pool_size": 2, "strides": 2},
            },
            {
                "type":    "conv2d",
                "name":    "conv2",
                "weights": [W_c2, b_c2],
                "config":  {"activation": "relu", "stride": 1, "padding": 1},
            },
            {
                "type":    "pooling",
                "name":    "pool2",
                "weights": [],
                "config":  {"mode": "max", "pool_size": 2, "strides": 2},
            },
            {
                "type":    "flatten",
                "name":    "flatten",
                "weights": [],
                "config":  {},
            },
            {
                "type":    "dense",
                "name":    "fc1",
                "weights": [W_d1, b_d1],
                "config":  {"activation": "relu"},
            },
            {
                "type":    "dense",
                "name":    "output",
                "weights": [W_d2, b_d2],
                "config":  {"activation": "softmax"},
            },
        ]
    }


# ---------------------------------------------------------------------------
# Reference CPU model — used only to validate accuracy
# ---------------------------------------------------------------------------

class _NumpyCNN:
    """Pure-NumPy reference implementation for correctness validation only."""

    def __init__(self, spec: dict):
        self.spec = spec

    def predict(self, x: np.ndarray) -> np.ndarray:
        for layer in self.spec["layers"]:
            t   = layer["type"]
            cfg = layer["config"]
            if t == "conv2d":
                x = self._conv2d(x, *layer["weights"], cfg.get("stride", 1), cfg.get("padding", 0))
                if cfg.get("activation") == "relu":
                    x = np.maximum(0, x)
            elif t == "pooling":
                x = self._max_pool(x, cfg.get("pool_size", 2), cfg.get("strides", 2))
            elif t == "flatten":
                x = x.reshape(x.shape[0], -1)
            elif t == "dense":
                W, b = layer["weights"]
                x = x @ W + b
                if cfg.get("activation") == "relu":
                    x = np.maximum(0, x)
                elif cfg.get("activation") == "softmax":
                    e = np.exp(x - x.max(axis=-1, keepdims=True))
                    x = e / e.sum(axis=-1, keepdims=True)
        return x

    @staticmethod
    def _conv2d(inp, W, b, stride, padding):
        N, C, H, W_in = inp.shape
        Co, Ci, KH, KW = W.shape
        if padding > 0:
            inp = np.pad(inp, ((0,0),(0,0),(padding,padding),(padding,padding)))
        _, _, H2, W2 = inp.shape
        Ho = (H2 - KH) // stride + 1
        Wo = (W2 - KW) // stride + 1
        out = np.zeros((N, Co, Ho, Wo), dtype=np.float32)
        for co in range(Co):
            for h in range(Ho):
                for w in range(Wo):
                    patch = inp[:, :, h*stride:h*stride+KH, w*stride:w*stride+KW]
                    out[:, co, h, w] = np.sum(patch * W[co], axis=(1,2,3)) + b[co]
        return out

    @staticmethod
    def _max_pool(inp, ps, stride):
        N, C, H, W = inp.shape
        Ho = (H - ps) // stride + 1
        Wo = (W - ps) // stride + 1
        out = np.zeros((N, C, Ho, Wo), dtype=np.float32)
        for h in range(Ho):
            for w in range(Wo):
                out[:,:,h,w] = inp[:,:,h*stride:h*stride+ps,w*stride:w*stride+ps].max(axis=(2,3))
        return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    SEP = "=" * 62

    print(f"\n{SEP}")
    print("  CNN Image Classifier — OpenCL Optimizer Demo")
    print(f"{SEP}\n")

    NUM_CLASSES = 10
    IMG_SIZE    = 28
    BATCH_SIZE  = 8    # kept small so reference NumPy conv runs in reasonable time

    # ── Step 1: Load model weights ─────────────────────────────────────
    model_spec = load_pretrained_cnn(num_classes=NUM_CLASSES)
    print("✔  Model spec loaded — LeNet-style CNN, 7 layers.\n")

    # ── Step 2: Optimize — one call ────────────────────────────────────
    config    = OptimizerConfig(prefer_gpu=True, enable_fusion=True, verbose=True)
    optimized = optimize_model(model_spec, config=config)

    # ── Step 3: Inference ──────────────────────────────────────────────
    np.random.seed(99)
    sample_input = np.random.randn(BATCH_SIZE, 1, IMG_SIZE, IMG_SIZE).astype(np.float32)

    ocl_output = optimized.predict(sample_input)

    print("\nSample Predictions (first 4 images, class probabilities):")
    for i, row in enumerate(ocl_output[:4]):
        predicted = np.argmax(row)
        print(f"  Image {i+1}: class={predicted}  probs={np.round(row, 3)}")

    # ── Step 4: Accuracy validation vs NumPy reference ────────────────
    print("\n⏳  Running NumPy reference (may take a moment for conv loops)…")
    ref_model  = _NumpyCNN(model_spec)
    cpu_output = ref_model.predict(sample_input)

    max_diff = np.max(np.abs(cpu_output - ocl_output))
    tol      = 1e-3
    status   = "✅  PASS" if max_diff < tol else "❌  FAIL"
    print(f"\n{status}  Max absolute difference (CPU vs OpenCL): {max_diff:.6f}")
    if max_diff >= tol:
        print(f"     ⚠️  Exceeds tolerance {tol} — check kernel precision.")

    # ── Step 5: Benchmark ──────────────────────────────────────────────
    BENCH_BATCH = 16
    bench_input = np.random.randn(BENCH_BATCH, 1, IMG_SIZE, IMG_SIZE).astype(np.float32)

    bench  = Benchmark()
    result = bench.compare(
        fn_cpu = lambda: ref_model.predict(bench_input),
        fn_gpu = lambda: optimized.predict(bench_input),
        label  = f"CNN LeNet  batch={BENCH_BATCH}  28×28 grayscale",
        runs   = 8,
        warmup = 2,
    )
    bench.print_result(result)

    # ── Step 6: Save report ────────────────────────────────────────────
    report_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "reports", "cnn_report.md")
    )
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    generate_report(
        results     = [result],
        device_info = optimized.device_info,
        output_path = report_path,
        title       = "OpenCL Optimizer — CNN Benchmark Report",
    )
    print(f"\n📄  Report saved → {report_path}")
    print(f"\n{SEP}\n")


if __name__ == "__main__":
    main()
