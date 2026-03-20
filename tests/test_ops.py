"""
Tests for individual OpenCL operations.
Verifies numerical correctness against NumPy reference.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from optimizer.backend.fallback import CPUFallback
from optimizer.ops.matmul       import MatMulOp
from optimizer.ops.conv2d       import Conv2DOp
from optimizer.ops.activations  import ActivationOp
from optimizer.ops.pooling      import PoolingOp

ATOL = 1e-4  # absolute tolerance for float32


class TestMatMul:
    """Test matrix multiplication op."""

    def setup_method(self):
        self.op = MatMulOp()

    def test_square(self):
        A = np.random.randn(64, 64).astype(np.float32)
        B = np.random.randn(64, 64).astype(np.float32)
        np.testing.assert_allclose(self.op.matmul(A, B), A @ B, atol=ATOL)

    def test_non_square(self):
        A = np.random.randn(32, 128).astype(np.float32)
        B = np.random.randn(128, 64).astype(np.float32)
        np.testing.assert_allclose(self.op.matmul(A, B), A @ B, atol=ATOL)

    def test_identity(self):
        A = np.eye(16, dtype=np.float32)
        B = np.random.randn(16, 16).astype(np.float32)
        np.testing.assert_allclose(self.op.matmul(A, B), B, atol=ATOL)

    def test_dense_forward(self):
        x = np.random.randn(8, 32).astype(np.float32)
        W = np.random.randn(32, 16).astype(np.float32)
        b = np.random.randn(16).astype(np.float32)
        ref = x @ W + b
        np.testing.assert_allclose(
            self.op.dense_forward(x, W, b), ref, atol=ATOL
        )

    def test_dense_single_sample(self):
        """1D input should be treated as batch of 1."""
        x = np.random.randn(1, 10).astype(np.float32)
        W = np.random.randn(10, 5).astype(np.float32)
        b = np.zeros(5, dtype=np.float32)
        result = self.op.dense_forward(x, W, b)
        assert result.shape == (1, 5)


class TestActivations:
    """Test all activation functions."""

    def setup_method(self):
        self.op = ActivationOp()
        np.random.seed(42)
        self.x = np.random.randn(100).astype(np.float32)

    def test_relu(self):
        ref = np.maximum(0, self.x)
        np.testing.assert_allclose(self.op.apply("relu", self.x), ref, atol=ATOL)

    def test_relu_2d(self):
        x = np.random.randn(16, 32).astype(np.float32)
        ref = np.maximum(0, x)
        np.testing.assert_allclose(self.op.apply("relu", x), ref, atol=ATOL)

    def test_sigmoid(self):
        ref = 1.0 / (1.0 + np.exp(-self.x))
        np.testing.assert_allclose(self.op.apply("sigmoid", self.x), ref, atol=ATOL)

    def test_tanh(self):
        ref = np.tanh(self.x)
        np.testing.assert_allclose(self.op.apply("tanh", self.x), ref, atol=ATOL)

    def test_softmax_rows(self):
        x = np.random.randn(8, 10).astype(np.float32)
        ref = CPUFallback.softmax(x)
        result = self.op.apply("softmax", x)
        np.testing.assert_allclose(result, ref, atol=ATOL)
        # Each row must sum to 1
        np.testing.assert_allclose(result.sum(axis=1), np.ones(8), atol=ATOL)

    def test_leaky_relu(self):
        alpha = 0.1
        ref = np.where(self.x >= 0, self.x, alpha * self.x)
        np.testing.assert_allclose(
            self.op.apply("leaky_relu", self.x, alpha=alpha), ref, atol=ATOL
        )

    def test_elu(self):
        alpha = 1.0
        ref = np.where(self.x >= 0, self.x, alpha * (np.exp(self.x) - 1))
        np.testing.assert_allclose(
            self.op.apply("elu", self.x, alpha=alpha), ref, atol=ATOL
        )

    def test_unknown_activation(self):
        with pytest.raises(ValueError, match="Unknown activation"):
            self.op.apply("swish_custom", self.x)

    def test_output_shape_preserved(self):
        """Shape must be identical to input shape."""
        shapes = [(10,), (8, 16), (4, 8, 8)]
        for s in shapes:
            x = np.random.randn(*s).astype(np.float32)
            assert self.op.apply("relu", x).shape == s


class TestConv2D:
    """Test Conv2D op."""

    def setup_method(self):
        self.op = Conv2DOp()
        np.random.seed(1)

    def test_output_shape_valid(self):
        x  = np.random.randn(2, 1, 6, 6).astype(np.float32)
        w  = np.random.randn(4, 1, 3, 3).astype(np.float32)
        b  = np.zeros(4, dtype=np.float32)
        out = self.op.conv2d(x, w, b, stride=1, padding=0)
        assert out.shape == (2, 4, 4, 4), f"Expected (2,4,4,4), got {out.shape}"

    def test_output_shape_same_padding(self):
        """stride=1, padding=1 with 3×3 kernel → same H,W."""
        x   = np.random.randn(1, 1, 8, 8).astype(np.float32)
        w   = np.random.randn(8, 1, 3, 3).astype(np.float32)
        b   = np.zeros(8, dtype=np.float32)
        out = self.op.conv2d(x, w, b, stride=1, padding=1)
        assert out.shape == (1, 8, 8, 8), f"Expected (1,8,8,8), got {out.shape}"

    def test_values_against_cpu(self):
        x  = np.random.randn(1, 1, 5, 5).astype(np.float32)
        w  = np.random.randn(2, 1, 3, 3).astype(np.float32) * 0.1
        b  = np.zeros(2, dtype=np.float32)
        ref = CPUFallback.conv2d(x, w, b, stride=1, padding=0)
        out = self.op.conv2d(x, w, b, stride=1, padding=0)
        np.testing.assert_allclose(out, ref, atol=1e-3)

    def test_strided(self):
        x   = np.random.randn(2, 3, 8, 8).astype(np.float32)
        w   = np.random.randn(4, 3, 3, 3).astype(np.float32) * 0.05
        b   = np.zeros(4, dtype=np.float32)
        ref = CPUFallback.conv2d(x, w, b, stride=2, padding=0)
        out = self.op.conv2d(x, w, b, stride=2, padding=0)
        np.testing.assert_allclose(out, ref, atol=1e-3)


class TestPooling:
    """Test pooling operations."""

    def setup_method(self):
        self.op = PoolingOp()
        np.random.seed(5)

    def test_max_pool_shape(self):
        x   = np.random.randn(2, 4, 8, 8).astype(np.float32)
        out = self.op.max_pool(x, pool_size=2, stride=2)
        assert out.shape == (2, 4, 4, 4), f"Unexpected shape: {out.shape}"

    def test_max_pool_values(self):
        x   = np.random.randn(1, 1, 4, 4).astype(np.float32)
        ref = CPUFallback.max_pool2d(x, 2, 2)
        out = self.op.max_pool(x, 2, 2)
        np.testing.assert_allclose(out, ref, atol=ATOL)

    def test_avg_pool_values(self):
        x   = np.random.randn(2, 3, 6, 6).astype(np.float32)
        ref = CPUFallback.avg_pool2d(x, 2, 2)
        out = self.op.avg_pool(x, 2, 2)
        np.testing.assert_allclose(out, ref, atol=ATOL)

    def test_global_avg_pool(self):
        x   = np.random.randn(4, 8, 7, 7).astype(np.float32)
        ref = x.reshape(4, 8, -1).mean(axis=2)
        out = self.op.global_avg_pool(x)
        assert out.shape == (4, 8)
        np.testing.assert_allclose(out, ref, atol=ATOL)

    def test_max_pool_preserves_max(self):
        """Max pooling must return the actual maximum in each window."""
        x = np.array([[[[1, 2, 3, 4],
                         [5, 6, 7, 8],
                         [9, 10, 11, 12],
                         [13, 14, 15, 16]]]], dtype=np.float32)
        out = self.op.max_pool(x, pool_size=2, stride=2)
        expected = np.array([[[[6, 8], [14, 16]]]], dtype=np.float32)
        np.testing.assert_allclose(out, expected, atol=ATOL)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
