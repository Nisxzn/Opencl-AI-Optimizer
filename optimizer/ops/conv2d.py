"""
Conv2D Operation
================
Wraps the 'conv2d' OpenCL kernel.
Falls back to the NumPy CPUFallback.conv2d when OpenCL is unavailable.
"""

import os
import numpy as np
from typing import Optional, Tuple
from optimizer.backend.opencl_device import OpenCLDevice, get_device
from optimizer.backend.fallback import CPUFallback
from optimizer.utils.logger import get_logger

logger = get_logger(__name__)

_KERNEL_PATH = os.path.join(os.path.dirname(__file__), "kernels", "conv2d.cl")


class Conv2DOp:
    """
    2D Convolution operation (NCHW data format).

    Example:
        op  = Conv2DOp()
        out = op.conv2d(input, weights, bias, stride=1, padding=1)
    """

    def __init__(self, device: Optional[OpenCLDevice] = None):
        self.device = device or get_device()
        self._kernel_conv2d = None

        if self.device.available:
            self._compile()

    def _compile(self) -> None:
        try:
            with open(_KERNEL_PATH) as f:
                source = f.read()
            self._kernel_conv2d = self.device.compile_kernel(source, "conv2d")
        except Exception as e:
            logger.warning(f"Conv2DOp compilation failed: {e}. Using CPU.")
            self.device.available = False

    def conv2d(
        self,
        inputs: np.ndarray,
        weights: np.ndarray,
        bias: np.ndarray,
        stride: int = 1,
        padding: int = 0,
    ) -> np.ndarray:
        """
        Run a 2D convolution.

        Args:
            inputs  : Input tensor (N, C_in, H, W) — float32
            weights : Filter tensor (C_out, C_in, KH, KW) — float32
            bias    : Bias vector (C_out,) — float32
            stride  : Convolution stride (default 1)
            padding : Zero-padding on each spatial side (default 0)

        Returns:
            Output tensor (N, C_out, H_out, W_out) — float32
        """
        if not self.device.available or self._kernel_conv2d is None:
            return CPUFallback.conv2d(inputs, weights, bias, stride, padding)

        return self._cl_conv2d(inputs, weights, bias, stride, padding)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _output_shape(
        H: int, W: int, KH: int, KW: int, stride: int, padding: int
    ) -> Tuple[int, int]:
        H_out = (H + 2 * padding - KH) // stride + 1
        W_out = (W + 2 * padding - KW) // stride + 1
        return H_out, W_out

    def _cl_conv2d(
        self,
        inputs: np.ndarray,
        weights: np.ndarray,
        bias: np.ndarray,
        stride: int,
        padding: int,
    ) -> np.ndarray:
        inputs  = np.ascontiguousarray(inputs,  dtype=np.float32)
        weights = np.ascontiguousarray(weights, dtype=np.float32)
        bias    = np.ascontiguousarray(bias,    dtype=np.float32)

        N, C_in, H, W = inputs.shape
        C_out, _, KH, KW = weights.shape
        H_out, W_out = self._output_shape(H, W, KH, KW, stride, padding)

        buf_in  = self.device.to_device(inputs)
        buf_w   = self.device.to_device(weights)
        buf_b   = self.device.to_device(bias)
        buf_out = self.device.empty_buffer((N, C_out, H_out, W_out))

        global_size = (N, C_out, H_out * W_out)

        self._kernel_conv2d(
            self.device.queue,
            global_size,
            None,
            buf_in, buf_w, buf_b, buf_out,
            np.int32(N),    np.int32(C_in),  np.int32(H),     np.int32(W),
            np.int32(C_out), np.int32(KH),   np.int32(KW),
            np.int32(H_out), np.int32(W_out),
            np.int32(stride), np.int32(padding),
        )

        return self.device.from_device(buf_out, (N, C_out, H_out, W_out))
