"""
MatMul / Dense Forward Operation
=================================
Wraps the 'matmul' and 'dense_forward' OpenCL kernels.
Falls back to NumPy if PyOpenCL is unavailable.
"""

import os
import numpy as np
from typing import Optional
from optimizer.backend.opencl_device import OpenCLDevice, get_device
from optimizer.backend.fallback import CPUFallback
from optimizer.utils.logger import get_logger

logger = get_logger(__name__)

_KERNEL_PATH = os.path.join(os.path.dirname(__file__), "kernels", "matmul.cl")


class MatMulOp:
    """
    Parallel matrix multiplication / dense forward pass.

    Automatically selects the OpenCL path when the device is available,
    otherwise falls back to a NumPy implementation.

    Example:
        op = MatMulOp()
        C  = op.matmul(A, B)                          # (M,K) @ (K,N) → (M,N)
        out = op.dense_forward(x, W, b)               # Dense layer
    """

    # Must match the #define in matmul.cl
    TILE_SIZE = 16 

    # Adaptive threshold: Only use OpenCL if total FLOPs or matrix size is large enough.
    # For MatMul (M,K) @ (K,N), total ops approx 2 * M * N * K.
    # Threshold found via benchmarking: roughly 300x300x300 matmul or larger is where
    # OpenCL starts to consistently win on most integrated GPUs on Windows.
    CPU_THRESHOLD_OPS = 50_000_000 

    def __init__(self, device: Optional[OpenCLDevice] = None):
        self.device = device or get_device()
        self._kernel_matmul = None
        self._kernel_dense  = None

        if self.device.available:
            self._compile()

    def _compile(self) -> None:
        """Pre-compile both kernels from the shared .cl file."""
        try:
            import pyopencl as cl
            # Only compile once per kernel path
            with open(_KERNEL_PATH) as f:
                source = f.read()
            self._kernel_matmul = self.device.compile_kernel(source, "matmul")
            self._kernel_dense  = self.device.compile_kernel(source, "dense_forward")
        except Exception as e:
            logger.warning(f"MatMulOp kernel compilation failed: {e}. Using CPU.")
            self.device.available = False

    def _should_use_opencl(self, M: int, K: int, N: int) -> bool:
        """Decide if OpenCL is worth the overhead for these dimensions."""
        if not self.device.available:
            return False
        
        # Heuristic: total floating point operations
        total_ops = 2 * M * N * K
        if total_ops < self.CPU_THRESHOLD_OPS:
            return False
            
        return True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def matmul(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Compute C = A @ B.
        A : (M, K)  B : (K, N)  → C : (M, N)
        """
        M, K = A.shape
        _, N = B.shape

        if not self._should_use_opencl(M, K, N):
            return CPUFallback.matmul(A, B)

        return self._cl_matmul(A, B)

    def dense_forward(
        self,
        inputs: np.ndarray,
        weights: np.ndarray,
        bias: np.ndarray,
        weights_buf: Optional[object] = None,
        bias_buf: Optional[object] = None,
    ) -> np.ndarray:
        """
        Dense layer: out = inputs @ weights + bias
        inputs      : (batch, in_features)
        weights     : (in_features, out_features)
        bias        : (out_features,)
        weights_buf : PRE-UPLOADED weight buffer (optional)
        bias_buf    : PRE-UPLOADED bias buffer (optional)
        """
        batch_size, in_features = inputs.shape
        _, out_features = weights.shape

        if not self._should_use_opencl(batch_size, in_features, out_features):
            return CPUFallback.dense_forward(inputs, weights, bias)

        return self._cl_dense(inputs, weights, bias, weights_buf, bias_buf)

    # ------------------------------------------------------------------
    # Internal OpenCL implementations
    # ------------------------------------------------------------------

    def _cl_matmul(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        import pyopencl as cl
        import numpy as np

        A = np.ascontiguousarray(A, dtype=np.float32)
        B = np.ascontiguousarray(B, dtype=np.float32)

        M, K = A.shape
        K2, N = B.shape
        assert K == K2, f"Shape mismatch: A is ({M},{K}), B is ({K2},{N})"

        buf_A = self.device.to_device(A)
        buf_B = self.device.to_device(B)
        buf_C = self.device.empty_buffer((M, N))

        import math
        global_size = (
            math.ceil(M / self.TILE_SIZE) * self.TILE_SIZE,
            math.ceil(N / self.TILE_SIZE) * self.TILE_SIZE,
        )
        local_size = (self.TILE_SIZE, self.TILE_SIZE)

        self._kernel_matmul(
            self.device.queue,
            global_size,
            local_size,
            buf_A, buf_B, buf_C,
            np.int32(M), np.int32(K), np.int32(N),
        )

        return self.device.from_device(buf_C, (M, N))

    def _cl_dense(
        self,
        inputs: np.ndarray,
        weights: np.ndarray,
        bias: np.ndarray,
        weights_buf: Optional[object] = None,
        bias_buf: Optional[object] = None,
    ) -> np.ndarray:
        import numpy as np
        import pyopencl as cl

        inputs  = np.ascontiguousarray(inputs,  dtype=np.float32)

        batch_size, in_features = inputs.shape
        _, out_features = weights.shape

        # Memory Optimization: Use pre-uploaded buffers if available
        buf_in  = self.device.to_device(inputs)
        buf_w   = weights_buf if weights_buf is not None else self.device.to_device(weights)
        buf_b   = bias_buf if bias_buf is not None else self.device.to_device(bias)
        buf_out = self.device.empty_buffer((batch_size, out_features))

        import math
        global_size = (
            math.ceil(batch_size / self.TILE_SIZE) * self.TILE_SIZE,
            math.ceil(out_features / self.TILE_SIZE) * self.TILE_SIZE,
        )
        local_size = (self.TILE_SIZE, self.TILE_SIZE)

        self._kernel_dense(
            self.device.queue,
            global_size,
            local_size,
            buf_in, buf_w, buf_b, buf_out,
            np.int32(batch_size),
            np.int32(in_features),
            np.int32(out_features),
        )

        return self.device.from_device(buf_out, (batch_size, out_features))

