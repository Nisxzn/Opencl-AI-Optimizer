"""
Dense (Fully-Connected) Layer
==============================
Accelerated with the OpenCL dense_forward kernel.
Optionally fuses a bias add and an inline activation.
"""

import numpy as np
from typing import Optional
from optimizer.layers.base import BaseLayer
from optimizer.ops.matmul import MatMulOp
from optimizer.ops.activations import ActivationOp
from optimizer.backend.opencl_device import OpenCLDevice


class DenseLayer(BaseLayer):
    """
    Optimized Dense (Fully-Connected) layer.

    Args:
        weights    : Weight matrix (in_features, out_features), float32
        bias       : Bias vector  (out_features,), float32
        activation : Optional activation name ('relu', 'sigmoid', 'tanh',
                     'softmax', 'leaky_relu', 'elu') or None for linear.
        name       : Layer name for logging / summaries.

    Example:
        layer = DenseLayer(weights=W, bias=b, activation='relu')
        output = layer.forward(input_batch)   # (batch, out_features)
    """

    def __init__(
        self,
        weights: np.ndarray,
        bias: np.ndarray,
        activation: Optional[str] = None,
        name: str = "dense",
        device: Optional[OpenCLDevice] = None,
    ):
        super().__init__(name=name, device=device)
        self.weights    = np.ascontiguousarray(weights, dtype=np.float32)
        self.bias       = np.ascontiguousarray(bias,    dtype=np.float32)
        self.activation = activation.lower() if (activation and activation.lower() != "linear") else None

        # Instantiate ops (compile kernels on first creation)
        self._matmul_op  = MatMulOp(device=self.device)
        self._act_op     = ActivationOp(device=self.device) if self.activation else None

        self.in_features  = weights.shape[0]
        self.out_features = weights.shape[1]

        # Optimization: Pre-upload weights to device if using OpenCL
        self._weights_cl = None
        self._bias_cl    = None
        if self.device.available:
            self._weights_cl = self.device.to_device(self.weights)
            self._bias_cl    = self.device.to_device(self.bias)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward pass: out = activation(inputs @ weights + bias)

        inputs : (batch, in_features)
        returns: (batch, out_features)
        """
        # Ensure 2D input
        if inputs.ndim == 1:
            inputs = inputs[np.newaxis, :]

        out = self._matmul_op.dense_forward(
            inputs, self.weights, self.bias,
            weights_buf=self._weights_cl,
            bias_buf=self._bias_cl
        )

        if self._act_op is not None:
            out = self._act_op.apply(self.activation, out)

        return out

    def summary(self) -> str:
        act = self.activation or "linear"
        
        # Check matching what MatMulOp does (heuristic for 1 input batch)
        # Note: True decision is per-batch in forward()
        use_cl = self._matmul_op._should_use_opencl(1, self.in_features, self.out_features)
        backend = "OpenCL" if use_cl else "CPU (Adaptive Fallback)"
        
        return (
            f"DenseLayer(name='{self.name}', "
            f"in={self.in_features}, out={self.out_features}, "
            f"activation='{act}', backend='{backend}')"
        )
