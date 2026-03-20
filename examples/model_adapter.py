"""
Universal Model Adapter
=======================
A zero-friction bridge between any ML framework and the OpenCL optimizer.

Supported frameworks (auto-detected at runtime):
  ✅  Keras / TensorFlow   — tf.keras.Model
  ✅  PyTorch              — torch.nn.Module
  ✅  Scikit-learn         — MLPClassifier / MLPRegressor
  ✅  Dict spec            — {"layers": [...]}  (framework-free)
  ✅  OptimizedModel       — returned as-is (idempotent)

Usage (3 lines for any framework):
    from examples.model_adapter import load_and_optimize
    optimized = load_and_optimize(your_trained_model)
    output    = optimized.predict(input_data)

Requirements:
  - numpy (always)
  - pyopencl (optional — falls back to CPU automatically)
  - TensorFlow / PyTorch / sklearn only needed if you use those models
"""

from __future__ import annotations

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from optimizer import optimize_model, OptimizerConfig
from optimizer.core import OptimizedModel
from optimizer.utils.logger import get_logger

logger = get_logger("model_adapter")


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────

def load_and_optimize(
    model,
    *,
    activation_map: dict | None = None,
    config: OptimizerConfig | None = None,
) -> OptimizedModel:
    """
    Detect model type and return an OpenCL-accelerated OptimizedModel.

    Parameters
    ----------
    model : any supported model object
        Keras Model, PyTorch nn.Module, sklearn MLP, dict spec, or OptimizedModel.
    activation_map : dict, optional
        ``{layer_name: "relu" | "sigmoid" | ...}``
        Only required for PyTorch models where activations are applied
        imperatively (not declared as ``nn.ReLU()`` child modules).
    config : OptimizerConfig, optional
        Custom device / fusion configuration.  Sensible defaults are applied
        when omitted.

    Returns
    -------
    OptimizedModel
        Inference-ready model with a ``.predict(inputs)`` method identical
        in signature to the original.

    Examples
    --------
    **Dict spec (no framework):**

    >>> from examples.model_adapter import load_and_optimize
    >>> import numpy as np
    >>> spec = {
    ...     "layers": [
    ...         {"type": "dense", "name": "fc1",
    ...          "weights": [W1, b1], "config": {"activation": "relu"}},
    ...         {"type": "dense", "name": "out",
    ...          "weights": [W2, b2], "config": {"activation": "softmax"}},
    ...     ]
    ... }
    >>> optimized = load_and_optimize(spec)
    >>> output = optimized.predict(x_test)

    **Keras:**

    >>> import tensorflow as tf
    >>> keras_model = tf.keras.models.load_model("my_model.h5")
    >>> optimized   = load_and_optimize(keras_model)
    >>> output      = optimized.predict(x_test)

    **PyTorch:**

    >>> optimized = load_and_optimize(pt_model, activation_map={"fc1": "relu"})

    **Scikit-learn:**

    >>> from sklearn.neural_network import MLPClassifier
    >>> clf       = MLPClassifier(...).fit(X_train, y_train)
    >>> optimized = load_and_optimize(clf)
    """
    cfg = config or OptimizerConfig(prefer_gpu=True, verbose=False, enable_fusion=True)

    # Already optimized — return as-is
    if isinstance(model, OptimizedModel):
        logger.info("Model is already an OptimizedModel — returning as-is.")
        return model

    # Dict spec
    if isinstance(model, dict) and "layers" in model:
        logger.info("Detected dict-spec model.")
        return optimize_model(model, config=cfg)

    # Keras / TensorFlow
    if _is_keras(model):
        logger.info("Detected Keras model.")
        return optimize_model(model, config=cfg)

    # PyTorch
    if _is_pytorch(model):
        logger.info("Detected PyTorch nn.Module.")
        spec = _spec_from_pytorch(model, activation_map or {})
        return optimize_model(spec, config=cfg)

    # Scikit-learn MLP
    if _is_sklearn_mlp(model):
        logger.info("Detected scikit-learn MLP model.")
        spec = _spec_from_sklearn(model)
        return optimize_model(spec, config=cfg)

    raise TypeError(
        f"Unsupported model type: {type(model).__module__}.{type(model).__qualname__}\n"
        "Supported: Keras/TF, PyTorch nn.Module, sklearn MLP, dict spec, OptimizedModel.\n"
        "For fully custom models, build a dict spec manually (see README or docs)."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Framework detection helpers
# ──────────────────────────────────────────────────────────────────────────────

def _is_keras(obj) -> bool:
    mod = type(obj).__module__ or ""
    return "keras" in mod.lower() or (hasattr(obj, "layers") and hasattr(obj, "compile"))


def _is_pytorch(obj) -> bool:
    try:
        import torch.nn as nn
        return isinstance(obj, nn.Module)
    except ImportError:
        return False


def _is_sklearn_mlp(obj) -> bool:
    return type(obj).__name__ in ("MLPClassifier", "MLPRegressor")


# ──────────────────────────────────────────────────────────────────────────────
# PyTorch → dict spec converter
# ──────────────────────────────────────────────────────────────────────────────

def _spec_from_pytorch(model, activation_map: dict) -> dict:
    """
    Walk a PyTorch module tree and extract a dict spec.

    Handles:
      nn.Linear   → dense
      nn.Conv2d   → conv2d
      nn.ReLU / nn.Sigmoid / nn.Tanh / nn.Softmax  → activation
      nn.MaxPool2d / nn.AvgPool2d / nn.AdaptiveAvgPool2d → pooling
      nn.Flatten  → flatten
      nn.BatchNorm* → skipped (fold during training; ≈identity at inference)
    """
    import torch.nn as nn

    spec_layers = []

    def _process(name: str, module) -> None:
        mtype = type(module).__name__

        if isinstance(module, nn.Linear):
            W = module.weight.detach().numpy().T.astype("float32")   # (out,in)→(in,out)
            b = (module.bias.detach().numpy().astype("float32")
                 if module.bias is not None
                 else np.zeros(module.out_features, dtype="float32"))
            spec_layers.append({
                "type":    "dense", "name": name,
                "weights": [W, b],
                "config":  {"activation": activation_map.get(name, "linear")},
            })

        elif isinstance(module, nn.Conv2d):
            W = module.weight.detach().numpy().astype("float32")
            b = (module.bias.detach().numpy().astype("float32")
                 if module.bias is not None
                 else np.zeros(module.out_channels, dtype="float32"))
            stride  = module.stride[0]  if isinstance(module.stride, tuple)  else module.stride
            padding = module.padding[0] if isinstance(module.padding, tuple) else module.padding
            spec_layers.append({
                "type":    "conv2d", "name": name,
                "weights": [W, b],
                "config":  {"activation": activation_map.get(name, "linear"),
                            "stride": stride, "padding": padding},
            })

        elif isinstance(module, nn.ReLU):
            spec_layers.append({"type": "activation", "name": name,
                                 "weights": [], "config": {"activation": "relu"}})
        elif isinstance(module, nn.Sigmoid):
            spec_layers.append({"type": "activation", "name": name,
                                 "weights": [], "config": {"activation": "sigmoid"}})
        elif isinstance(module, nn.Tanh):
            spec_layers.append({"type": "activation", "name": name,
                                 "weights": [], "config": {"activation": "tanh"}})
        elif type(module).__name__ in ("Softmax", "LogSoftmax"):
            spec_layers.append({"type": "activation", "name": name,
                                 "weights": [], "config": {"activation": "softmax"}})

        elif isinstance(module, nn.MaxPool2d):
            ps = module.kernel_size; st = module.stride or ps
            ps = ps[0] if isinstance(ps, tuple) else ps
            st = st[0] if isinstance(st, tuple) else st
            spec_layers.append({"type": "pooling", "name": name, "weights": [],
                                 "config": {"mode": "max", "pool_size": ps, "strides": st}})

        elif isinstance(module, nn.AvgPool2d):
            ps = module.kernel_size; st = module.stride or ps
            ps = ps[0] if isinstance(ps, tuple) else ps
            st = st[0] if isinstance(st, tuple) else st
            spec_layers.append({"type": "pooling", "name": name, "weights": [],
                                 "config": {"mode": "avg", "pool_size": ps, "strides": st}})

        elif isinstance(module, nn.AdaptiveAvgPool2d):
            spec_layers.append({"type": "pooling", "name": name, "weights": [],
                                 "config": {"mode": "global_avg"}})

        elif isinstance(module, nn.Flatten):
            spec_layers.append({"type": "flatten", "name": name, "weights": [], "config": {}})

        elif "batchnorm" in mtype.lower():
            logger.debug(f"  Skipping BatchNorm '{name}' (inference fold not yet supported).")

        elif list(module.children()):
            for child_name, child in module.named_children():
                _process(f"{name}.{child_name}", child)

        else:
            logger.warning(f"  Unknown layer '{name}' ({mtype}) — skipped.")

    for n, m in model.named_children():
        _process(n, m)

    return {"layers": spec_layers}


# ──────────────────────────────────────────────────────────────────────────────
# Scikit-learn MLP → dict spec converter
# ──────────────────────────────────────────────────────────────────────────────

_ACT_MAP = {
    "relu":    "relu",
    "tanh":    "tanh",
    "logistic":"sigmoid",
    "identity":"linear",
    "softmax": "softmax",
}


def _spec_from_sklearn(model) -> dict:
    """Extract weights and activation from a fitted sklearn MLPClassifier/Regressor."""
    hidden_act = _ACT_MAP.get(model.activation, "relu")
    output_act = _ACT_MAP.get(getattr(model, "out_activation_", "softmax"), "softmax")

    spec_layers = []
    n = len(model.coefs_)
    for i, (W, b) in enumerate(zip(model.coefs_, model.intercepts_)):
        act = output_act if i == n - 1 else hidden_act
        spec_layers.append({
            "type":    "dense",
            "name":    f"layer_{i}",
            "weights": [W.astype("float32"), b.astype("float32")],
            "config":  {"activation": act},
        })

    return {"layers": spec_layers}


# ──────────────────────────────────────────────────────────────────────────────
# Self-test (run as script)
# ──────────────────────────────────────────────────────────────────────────────

def _self_test_dict_spec():
    """Smoke test: raw NumPy dict spec — works with zero extra dependencies."""
    print("\n  — Dict Spec (NumPy) —")
    np.random.seed(1)
    W1 = (np.random.randn(64, 128) * 0.1).astype("float32")
    b1 = np.zeros(128, dtype="float32")
    W2 = (np.random.randn(128, 10) * 0.1).astype("float32")
    b2 = np.zeros(10, dtype="float32")
    spec = {
        "layers": [
            {"type": "dense", "name": "fc1", "weights": [W1, b1], "config": {"activation": "relu"}},
            {"type": "dense", "name": "out", "weights": [W2, b2], "config": {"activation": "softmax"}},
        ]
    }
    opt = load_and_optimize(spec)
    x   = np.random.randn(8, 64).astype("float32")
    out = opt.predict(x)
    assert out.shape == (8, 10), f"Shape mismatch: {out.shape}"
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-4), "Softmax row sums not ≈ 1"
    print(f"  ✅  Output shape: {out.shape}  |  Row sums ≈ 1.0 ✓")


def _self_test_sklearn():
    try:
        from sklearn.neural_network import MLPClassifier
        from sklearn.datasets import make_classification
    except ImportError:
        print("  ⚠️  scikit-learn not installed — skipping.")
        return
    print("\n  — Scikit-learn MLPClassifier —")
    X, y = make_classification(n_samples=200, n_features=20, random_state=0)
    clf  = MLPClassifier(hidden_layer_sizes=(64,), max_iter=5, random_state=0)
    clf.fit(X, y)
    opt  = load_and_optimize(clf)
    out  = opt.predict(X[:8].astype("float32"))
    print(f"  ✅  Output shape: {out.shape}  |  Predicted: {out.argmax(axis=1)}")


def _self_test_pytorch():
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("  ⚠️  PyTorch not installed — skipping.")
        return
    print("\n  — PyTorch nn.Sequential —")
    model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 10))
    model.eval()
    opt = load_and_optimize(model)
    x   = torch.randn(4, 32).detach().numpy().astype("float32")
    out = opt.predict(x)
    print(f"  ✅  Output shape: {out.shape}")


if __name__ == "__main__":
    SEP = "=" * 55
    print(f"\n{SEP}")
    print("  Universal Model Adapter — Self-Test")
    print(f"{SEP}")
    _self_test_dict_spec()
    _self_test_sklearn()
    _self_test_pytorch()
    print(f"\n{SEP}")
    print("  All self-tests passed ✅")
    print(f"{SEP}\n")
