"""
Pooling Operations
==================
Wraps OpenCL pooling kernels: Max Pool 2D, Average Pool 2D, Global Average Pool 2D.
Falls back to NumPy when OpenCL is unavailable.
"""

import os
import numpy as np
from typing import Optional, Tuple
from optimizer.backend.opencl_device import OpenCLDevice, get_device
from optimizer.backend.fallback import CPUFallback
from optimizer.utils.logger import get_logger

logger = get_logger(__name__)

_KERNEL_PATH = os.path.join(os.path.dirname(__file__), "kernels", "pooling.cl")


class PoolingOp:
    """
    2D Pooling operations (NCHW data format).

    Supported:
        - max   : Max pooling
        - avg   : Average pooling
        - global_avg : Global average pooling (spatial dimensions → 1x1)

    Example:
        pool = PoolingOp()
        out  = pool.max_pool(x, pool_size=2, stride=2)
        out  = pool.avg_pool(x, pool_size=2, stride=2)
        out  = pool.global_avg_pool(x)
    """

    def __init__(self, device: Optional[OpenCLDevice] = None):
        self.device = device or get_device()
        self._k_max  = None
        self._k_avg  = None
        self._k_gavg = None

        if self.device.available:
            self._compile()

    def _compile(self) -> None:
        try:
            with open(_KERNEL_PATH) as f:
                source = f.read()
            self._k_max  = self.device.compile_kernel(source, "max_pool2d")
            self._k_avg  = self.device.compile_kernel(source, "avg_pool2d")
            self._k_gavg = self.device.compile_kernel(source, "global_avg_pool2d")
        except Exception as e:
            logger.warning(f"PoolingOp compilation failed: {e}. Using CPU.")
            self.device.available = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def max_pool(
        self,
        inputs: np.ndarray,
        pool_size: int = 2,
        stride: int = 2,
    ) -> np.ndarray:
        """Max pooling 2D. inputs: (N, C, H, W) → (N, C, H_out, W_out)"""
        if not self.device.available or self._k_max is None:
            return CPUFallback.max_pool2d(inputs, pool_size, stride)
        return self._cl_pool("max", inputs, pool_size, pool_size, stride, stride)

    def avg_pool(
        self,
        inputs: np.ndarray,
        pool_size: int = 2,
        stride: int = 2,
    ) -> np.ndarray:
        """Average pooling 2D. inputs: (N, C, H, W) → (N, C, H_out, W_out)"""
        if not self.device.available or self._k_avg is None:
            return CPUFallback.avg_pool2d(inputs, pool_size, stride)
        return self._cl_pool("avg", inputs, pool_size, pool_size, stride, stride)

    def global_avg_pool(self, inputs: np.ndarray) -> np.ndarray:
        """Global average pooling. inputs: (N, C, H, W) → (N, C)"""
        if not self.device.available or self._k_gavg is None:
            N, C, H, W = inputs.shape
            return inputs.reshape(N, C, -1).mean(axis=2).astype(np.float32)
        return self._cl_global_avg(inputs)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _out_size(H: int, pool: int, stride: int) -> int:
        return (H - pool) // stride + 1

    def _cl_pool(
        self,
        mode: str,
        inputs: np.ndarray,
        pool_h: int,
        pool_w: int,
        stride_h: int,
        stride_w: int,
    ) -> np.ndarray:
        inputs = np.ascontiguousarray(inputs, dtype=np.float32)
        N, C, H, W = inputs.shape
        H_out = self._out_size(H, pool_h, stride_h)
        W_out = self._out_size(W, pool_w, stride_w)

        buf_in  = self.device.to_device(inputs)
        buf_out = self.device.empty_buffer((N, C, H_out, W_out))

        kernel = self._k_max if mode == "max" else self._k_avg
        kernel(
            self.device.queue,
            (N, C, H_out * W_out),
            None,
            buf_in, buf_out,
            np.int32(N), np.int32(C), np.int32(H), np.int32(W),
            np.int32(H_out), np.int32(W_out),
            np.int32(pool_h), np.int32(pool_w),
            np.int32(stride_h), np.int32(stride_w),
        )
        return self.device.from_device(buf_out, (N, C, H_out, W_out))

    def _cl_global_avg(self, inputs: np.ndarray) -> np.ndarray:
        inputs = np.ascontiguousarray(inputs, dtype=np.float32)
        N, C, H, W = inputs.shape

        buf_in  = self.device.to_device(inputs)
        buf_out = self.device.empty_buffer((N, C))

        self._k_gavg(
            self.device.queue,
            (N, C),
            None,
            buf_in, buf_out,
            np.int32(N), np.int32(C), np.int32(H), np.int32(W),
        )
        return self.device.from_device(buf_out, (N, C))
