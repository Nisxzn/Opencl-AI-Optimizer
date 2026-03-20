"""
Conv2D Layer
============
Accelerated 2D Convolution layer backed by the OpenCL conv2d kernel.
Supports an optional fused activation.
"""

import numpy as np
from typing import Optional, Tuple
from optimizer.layers.base import BaseLayer
from optimizer.ops.conv2d import Conv2DOp
from optimizer.ops.activations import ActivationOp
from optimizer.backend.opencl_device import OpenCLDevice


class Conv2DLayer(BaseLayer):
    """
    Optimized Conv2D layer (NCHW data format).

    Args:
        weights    : Filter tensor (C_out, C_in, KH, KW), float32
        bias       : Bias vector  (C_out,), float32
        stride     : Convolution stride (default 1)
        padding    : Zero-padding on each spatial side (default 0)
        activation : Optional activation name, or None for linear.
        name       : Layer name.

    Input  : (N, C_in, H, W)
    Output : (N, C_out, H_out, W_out)

    Example:
        layer  = Conv2DLayer(weights=W, bias=b, stride=1, padding=1, activation='relu')
        output = layer.forward(inputs)
    """

    def __init__(
        self,
        weights: np.ndarray,
        bias: np.ndarray,
        stride: int = 1,
        padding: int = 0,
        activation: Optional[str] = None,
        name: str = "conv2d",
        device: Optional[OpenCLDevice] = None,
    ):
        super().__init__(name=name, device=device)
        self.weights    = np.ascontiguousarray(weights, dtype=np.float32)
        self.bias       = np.ascontiguousarray(bias,    dtype=np.float32)
        self.stride     = stride
        self.padding    = padding
        self.activation = activation.lower() if (activation and activation.lower() != "linear") else None

        self._conv_op = Conv2DOp(device=self.device)
        self._act_op  = ActivationOp(device=self.device) if self.activation else None

        self.out_channels, self.in_channels, self.kH, self.kW = weights.shape

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward pass: out = activation(conv2d(inputs, weights, bias))

        inputs : (N, C_in, H, W)
        returns: (N, C_out, H_out, W_out)
        """
        out = self._conv_op.conv2d(
            inputs, self.weights, self.bias, self.stride, self.padding
        )
        if self._act_op is not None:
            out = self._act_op.apply(self.activation, out)
        return out

    def output_shape(self, input_h: int, input_w: int) -> Tuple[int, int]:
        """Compute the (H_out, W_out) given input spatial dims."""
        H_out = (input_h + 2 * self.padding - self.kH) // self.stride + 1
        W_out = (input_w + 2 * self.padding - self.kW) // self.stride + 1
        return H_out, W_out

    def summary(self) -> str:
        act = self.activation or "linear"
        backend = "OpenCL" if self.device.available else "CPU"
        return (
            f"Conv2DLayer(name='{self.name}', "
            f"filters={self.out_channels}, kernel=({self.kH},{self.kW}), "
            f"stride={self.stride}, padding={self.padding}, "
            f"activation='{act}', backend={backend})"
        )
