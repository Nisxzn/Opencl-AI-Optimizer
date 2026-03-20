"""
CPU Fallback Implementations
============================
Pure NumPy versions of all inference operations. Used when:
  - PyOpenCL is not installed
  - OpenCL device is unavailable
  - An operation is not yet GPU-accelerated

All functions match the interface of their OpenCL counterparts so the
higher-level layer code can call either transparently.
"""

import numpy as np
from optimizer.utils.logger import get_logger

logger = get_logger(__name__)


class CPUFallback:
    """
    Pure-NumPy (CPU) implementations of all supported ML operations.
    Acts as a drop-in replacement for the OpenCL-backed ops.
    """

    @staticmethod
    def matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Dense matrix multiply: C = A @ B
        A : (M, K)
        B : (K, N)
        → (M, N)
        """
        return A.astype(np.float32) @ B.astype(np.float32)

    @staticmethod
    def dense_forward(
        inputs: np.ndarray,
        weights: np.ndarray,
        bias: np.ndarray,
    ) -> np.ndarray:
        """
        Fully-connected layer: out = inputs @ weights + bias
        inputs  : (batch, in_features)
        weights : (in_features, out_features)
        bias    : (out_features,)
        → (batch, out_features)
        """
        return inputs.astype(np.float32) @ weights.astype(np.float32) + bias.astype(np.float32)

    @staticmethod
    def conv2d(
        inputs: np.ndarray,
        kernel: np.ndarray,
        bias: np.ndarray,
        stride: int = 1,
        padding: int = 0,
    ) -> np.ndarray:
        """
        2D convolution (NCHW format).
        inputs : (N, C_in, H, W)
        kernel : (C_out, C_in, KH, KW)
        bias   : (C_out,)
        → (N, C_out, H_out, W_out)
        """
        N, C_in, H, W = inputs.shape
        C_out, _, KH, KW = kernel.shape

        # Pad input if needed
        if padding > 0:
            inputs = np.pad(
                inputs,
                ((0, 0), (0, 0), (padding, padding), (padding, padding)),
                mode="constant",
            )

        H_out = (H + 2 * padding - KH) // stride + 1
        W_out = (W + 2 * padding - KW) // stride + 1

        output = np.zeros((N, C_out, H_out, W_out), dtype=np.float32)

        for n in range(N):
            for c_out in range(C_out):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start = h * stride
                        w_start = w * stride
                        patch = inputs[
                            n, :, h_start : h_start + KH, w_start : w_start + KW
                        ]
                        output[n, c_out, h, w] = (
                            np.sum(patch * kernel[c_out]) + bias[c_out]
                        )
        return output

    @staticmethod
    def relu(inputs: np.ndarray) -> np.ndarray:
        """Element-wise ReLU: max(0, x)"""
        return np.maximum(0, inputs).astype(np.float32)

    @staticmethod
    def sigmoid(inputs: np.ndarray) -> np.ndarray:
        """Element-wise sigmoid: 1 / (1 + exp(-x))"""
        return (1.0 / (1.0 + np.exp(-inputs.astype(np.float32)))).astype(np.float32)

    @staticmethod
    def tanh(inputs: np.ndarray) -> np.ndarray:
        """Element-wise tanh."""
        return np.tanh(inputs.astype(np.float32))

    @staticmethod
    def softmax(inputs: np.ndarray) -> np.ndarray:
        """Softmax along last axis (numerically stable)."""
        x = inputs.astype(np.float32)
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    @staticmethod
    def max_pool2d(
        inputs: np.ndarray, pool_size: int = 2, stride: int = 2
    ) -> np.ndarray:
        """
        2D max pooling (NCHW format).
        inputs : (N, C, H, W)
        → (N, C, H_out, W_out)
        """
        N, C, H, W = inputs.shape
        H_out = (H - pool_size) // stride + 1
        W_out = (W - pool_size) // stride + 1
        output = np.zeros((N, C, H_out, W_out), dtype=np.float32)
        for h in range(H_out):
            for w in range(W_out):
                h_s, w_s = h * stride, w * stride
                output[:, :, h, w] = np.max(
                    inputs[:, :, h_s : h_s + pool_size, w_s : w_s + pool_size],
                    axis=(2, 3),
                )
        return output

    @staticmethod
    def avg_pool2d(
        inputs: np.ndarray, pool_size: int = 2, stride: int = 2
    ) -> np.ndarray:
        """
        2D average pooling (NCHW format).
        inputs : (N, C, H, W)
        → (N, C, H_out, W_out)
        """
        N, C, H, W = inputs.shape
        H_out = (H - pool_size) // stride + 1
        W_out = (W - pool_size) // stride + 1
        output = np.zeros((N, C, H_out, W_out), dtype=np.float32)
        for h in range(H_out):
            for w in range(W_out):
                h_s, w_s = h * stride, w * stride
                output[:, :, h, w] = np.mean(
                    inputs[:, :, h_s : h_s + pool_size, w_s : w_s + pool_size],
                    axis=(2, 3),
                )
        return output
