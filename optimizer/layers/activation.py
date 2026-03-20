"""
Activation Layer
================
Standalone activation layer — useful when the activation is expressed
as a separate layer in the model graph (common in Keras functional API).
"""

import numpy as np
from optimizer.layers.base import BaseLayer
from optimizer.ops.activations import ActivationOp
from optimizer.backend.opencl_device import OpenCLDevice
from typing import Optional


class ActivationLayer(BaseLayer):
    """
    Standalone OpenCL-accelerated activation layer.

    Supported activations: relu, sigmoid, tanh, softmax, leaky_relu, elu

    Args:
        activation : Activation function name (string).
        alpha      : For leaky_relu / elu — the negative slope (default 0.01).
        name       : Layer name.

    Example:
        act = ActivationLayer('relu')
        out = act.forward(x)
    """

    def __init__(
        self,
        activation: str,
        alpha: float = 0.01,
        name: Optional[str] = None,
        device: Optional[OpenCLDevice] = None,
    ):
        super().__init__(name=name or activation, device=device)
        self.activation = activation.lower()
        self.alpha      = alpha
        self._act_op    = ActivationOp(device=self.device)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        if self.activation == "linear":
            return inputs
        return self._act_op.apply(self.activation, inputs, alpha=self.alpha)

    def summary(self) -> str:
        backend = "OpenCL" if self.device.available else "CPU"
        return (
            f"ActivationLayer(name='{self.name}', "
            f"fn='{self.activation}', alpha={self.alpha}, backend={backend})"
        )
