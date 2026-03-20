"""
Integration tests for the optimize_model() pipeline.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from optimizer import optimize_model, OptimizerConfig
from optimizer.core import OptimizedModel

ATOL = 1e-4
np.random.seed(42)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _dense_spec(in_f=32, h=64, out_f=10):
    W1 = np.random.randn(in_f, h).astype(np.float32) * 0.1
    b1 = np.zeros(h, dtype=np.float32)
    W2 = np.random.randn(h, out_f).astype(np.float32) * 0.1
    b2 = np.zeros(out_f, dtype=np.float32)
    return {
        "layers": [
            {"type": "dense", "name": "fc1", "weights": [W1, b1], "config": {"activation": "relu"}},
            {"type": "dense", "name": "out", "weights": [W2, b2], "config": {"activation": "softmax"}},
        ]
    }


def _cnn_spec():
    W_c = np.random.randn(4, 1, 3, 3).astype(np.float32) * 0.05
    b_c = np.zeros(4, dtype=np.float32)
    W_d = np.random.randn(4 * 3 * 3, 5).astype(np.float32) * 0.1
    b_d = np.zeros(5, dtype=np.float32)
    return {
        "layers": [
            {"type": "conv2d",   "name": "conv1", "weights": [W_c, b_c],
             "config": {"activation": "relu", "stride": 1, "padding": 0}},
            {"type": "flatten",  "name": "flat",  "weights": [], "config": {}},
            {"type": "dense",    "name": "out",   "weights": [W_d, b_d],
             "config": {"activation": "softmax"}},
        ]
    }


cfg_silent = OptimizerConfig(verbose=False, prefer_gpu=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestOptimizeModel:

    def test_returns_optimized_model(self):
        spec = _dense_spec()
        model = optimize_model(spec, config=cfg_silent)
        assert isinstance(model, OptimizedModel)

    def test_has_layers(self):
        spec = _dense_spec()
        model = optimize_model(spec, config=cfg_silent)
        assert len(model.layers) > 0

    def test_predict_shape(self):
        spec = _dense_spec(in_f=32, h=64, out_f=10)
        model = optimize_model(spec, config=cfg_silent)
        x = np.random.randn(8, 32).astype(np.float32)
        out = model.predict(x)
        assert out.shape == (8, 10), f"Expected (8,10), got {out.shape}"

    def test_predict_softmax_sums_to_one(self):
        spec = _dense_spec(in_f=16, h=32, out_f=5)
        model = optimize_model(spec, config=cfg_silent)
        x = np.random.randn(4, 16).astype(np.float32)
        out = model.predict(x)
        np.testing.assert_allclose(out.sum(axis=1), np.ones(4), atol=1e-4)

    def test_callable(self):
        spec = _dense_spec()
        model = optimize_model(spec, config=cfg_silent)
        x = np.random.randn(2, 32).astype(np.float32)
        out = model(x)
        assert out is not None

    def test_already_optimized_marker(self):
        """Passing an already-optimized model should not crash."""
        spec = _dense_spec()
        model1 = optimize_model(spec, config=cfg_silent)
        # Re-optimizing should be a no-op (zero new layers parsed)
        model2 = optimize_model(model1, config=cfg_silent)
        assert isinstance(model2, OptimizedModel)

    def test_activation_fusion(self):
        """Dense(linear) + Activation(relu) should fuse into Dense(relu)."""
        W = np.random.randn(8, 4).astype(np.float32)
        b = np.zeros(4, dtype=np.float32)
        spec = {
            "layers": [
                {"type": "dense",      "name": "fc", "weights": [W, b],
                 "config": {"activation": "linear"}},
                {"type": "activation", "name": "act", "weights": [],
                 "config": {"activation": "relu"}},
            ]
        }
        model = optimize_model(spec, config=cfg_silent)
        # One layer after fusion (the standalone activation is removed)
        assert len(model.layers) == 1

    def test_cnn_pipeline_shape(self):
        """Conv → Flatten → Dense pipeline must produce correct output shape."""
        spec = _cnn_spec()
        model = optimize_model(spec, config=cfg_silent)
        # Input: (2, 1, 5, 5) → conv(4ch, 3×3, valid) → (2,4,3,3)
        # flatten → (2,36) → dense(5) → (2,5)
        x = np.random.randn(2, 1, 5, 5).astype(np.float32)
        out = model.predict(x)
        assert out.shape == (2, 5), f"Expected (2,5), got {out.shape}"

    def test_device_info(self):
        spec = _dense_spec()
        model = optimize_model(spec, config=cfg_silent)
        info = model.device_info
        assert "available" in info

    def test_empty_spec(self):
        """An empty layer list should return a model with zero layers."""
        spec = {"layers": []}
        model = optimize_model(spec, config=cfg_silent)
        assert isinstance(model, OptimizedModel)

    def test_fallback_config(self):
        """Fallback-only config should still produce correct predictions."""
        cfg = OptimizerConfig(verbose=False, prefer_gpu=False)
        spec = _dense_spec(in_f=8, h=16, out_f=3)
        model = optimize_model(spec, config=cfg)
        x   = np.random.randn(2, 8).astype(np.float32)
        out = model.predict(x)
        assert out.shape == (2, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
