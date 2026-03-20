"""
Tests for layer abstractions (Dense, Conv2D, Activation, Pooling).
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from optimizer.layers.dense      import DenseLayer
from optimizer.layers.conv2d     import Conv2DLayer
from optimizer.layers.activation import ActivationLayer
from optimizer.layers.pooling    import PoolingLayer

ATOL = 1e-4
np.random.seed(0)


class TestDenseLayer:

    def _make(self, in_f, out_f, activation=None):
        W = np.random.randn(in_f, out_f).astype(np.float32) * 0.1
        b = np.zeros(out_f, dtype=np.float32)
        return DenseLayer(weights=W, bias=b, activation=activation, name="test_dense"), W, b

    def test_linear_forward(self):
        layer, W, b = self._make(32, 16)
        x = np.random.randn(4, 32).astype(np.float32)
        ref = x @ W + b
        np.testing.assert_allclose(layer.forward(x), ref, atol=ATOL)

    def test_relu_fused(self):
        layer, W, b = self._make(16, 8, activation="relu")
        x = np.random.randn(2, 16).astype(np.float32)
        ref = np.maximum(0, x @ W + b)
        np.testing.assert_allclose(layer.forward(x), ref, atol=ATOL)

    def test_softmax_fused(self):
        layer, W, b = self._make(16, 5, activation="softmax")
        x = np.random.randn(3, 16).astype(np.float32)
        out = layer.forward(x)
        assert out.shape == (3, 5)
        np.testing.assert_allclose(out.sum(axis=1), np.ones(3), atol=ATOL)

    def test_1d_input_promoted(self):
        """Single sample (features,) should be promoted to (1, features)."""
        layer, _, _ = self._make(8, 4)
        x = np.random.randn(1, 8).astype(np.float32)
        out = layer.forward(x)
        assert out.shape == (1, 4)

    def test_attributes(self):
        layer, _, _ = self._make(10, 5, "relu")
        assert layer.in_features  == 10
        assert layer.out_features == 5
        assert layer.activation   == "relu"

    def test_callable(self):
        layer, W, b = self._make(4, 2)
        x   = np.random.randn(1, 4).astype(np.float32)
        ref = x @ W + b
        np.testing.assert_allclose(layer(x), ref, atol=ATOL)

    def test_summary_string(self):
        layer, _, _ = self._make(8, 4, "relu")
        s = layer.summary()
        assert "DenseLayer" in s
        assert "relu" in s


class TestConv2DLayer:

    def _make(self, Co, Ci, K, stride=1, padding=0, activation=None):
        W = np.random.randn(Co, Ci, K, K).astype(np.float32) * 0.05
        b = np.zeros(Co, dtype=np.float32)
        return Conv2DLayer(
            weights=W, bias=b,
            stride=stride, padding=padding,
            activation=activation, name="test_conv"
        ), W, b

    def test_output_shape(self):
        layer, _, _ = self._make(8, 1, 3, stride=1, padding=0)
        x = np.random.randn(2, 1, 10, 10).astype(np.float32)
        out = layer.forward(x)
        assert out.shape == (2, 8, 8, 8)

    def test_same_padding_shape(self):
        """padding=1, kernel=3×3, stride=1 → same spatial dims."""
        layer, _, _ = self._make(4, 1, 3, stride=1, padding=1)
        x = np.random.randn(1, 1, 12, 12).astype(np.float32)
        out = layer.forward(x)
        assert out.shape == (1, 4, 12, 12)

    def test_relu_activation(self):
        layer, _, _ = self._make(4, 1, 3, activation="relu")
        x = np.random.randn(2, 1, 5, 5).astype(np.float32)
        out = layer.forward(x)
        assert np.all(out >= 0), "ReLU output has negative values"

    def test_output_shape_helper(self):
        layer, _, _ = self._make(8, 1, 3, stride=2, padding=1)
        H_out, W_out = layer.output_shape(16, 16)
        assert H_out == 8 and W_out == 8


class TestActivationLayer:

    def test_relu(self):
        layer = ActivationLayer("relu")
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        np.testing.assert_allclose(layer.forward(x), np.maximum(0, x), atol=ATOL)

    def test_sigmoid(self):
        layer = ActivationLayer("sigmoid")
        x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        ref = 1.0 / (1.0 + np.exp(-x))
        np.testing.assert_allclose(layer.forward(x), ref, atol=ATOL)

    def test_shape_preserved(self):
        layer = ActivationLayer("tanh")
        x = np.random.randn(4, 8, 8).astype(np.float32)
        assert layer.forward(x).shape == (4, 8, 8)

    def test_callable(self):
        layer = ActivationLayer("relu")
        x = np.array([-1.0, 1.0], dtype=np.float32)
        assert layer(x)[0] == 0.0 and layer(x)[1] == 1.0


class TestPoolingLayer:

    def test_max_pool_shape(self):
        layer = PoolingLayer(mode="max", pool_size=2, stride=2)
        x = np.random.randn(2, 4, 8, 8).astype(np.float32)
        out = layer.forward(x)
        assert out.shape == (2, 4, 4, 4)

    def test_avg_pool_shape(self):
        layer = PoolingLayer(mode="avg", pool_size=2, stride=2)
        x = np.random.randn(1, 3, 6, 6).astype(np.float32)
        out = layer.forward(x)
        assert out.shape == (1, 3, 3, 3)

    def test_global_avg_shape(self):
        layer = PoolingLayer(mode="global_avg")
        x = np.random.randn(4, 8, 7, 7).astype(np.float32)
        out = layer.forward(x)
        assert out.shape == (4, 8)

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="mode must be"):
            PoolingLayer(mode="median")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
