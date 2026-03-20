"""
Activation Operations
=====================
Wraps OpenCL activation kernels: ReLU, Sigmoid, Tanh, Softmax, ELU, Leaky ReLU.
Automatically falls back to NumPy when OpenCL is unavailable.
"""

import os
import numpy as np
from typing import Optional
from optimizer.backend.opencl_device import OpenCLDevice, get_device
from optimizer.backend.fallback import CPUFallback
from optimizer.utils.logger import get_logger

logger = get_logger(__name__)

_KERNEL_PATH = os.path.join(os.path.dirname(__file__), "kernels", "activations.cl")

# Maps activation name → (kernel_function_name, needs_alpha, needs_softmax_shape)
_ACTIVATION_MAP = {
    "relu":       ("relu",       False, False),
    "sigmoid":    ("sigmoid",    False, False),
    "tanh":       ("tanh_act",   False, False),
    "leaky_relu": ("leaky_relu", True,  False),
    "elu":        ("elu",        True,  False),
    "softmax":    ("softmax_row",False, True),
}


class ActivationOp:
    """
    Element-wise and reduction activation functions.

    Supported: relu, sigmoid, tanh, leaky_relu, elu, softmax

    Example:
        act = ActivationOp()
        out = act.apply("relu", x)
        out = act.apply("softmax", logits)
    """

    def __init__(self, device: Optional[OpenCLDevice] = None):
        self.device = device or get_device()
        self._kernels = {}

        if self.device.available:
            self._compile_all()

    def _compile_all(self) -> None:
        try:
            with open(_KERNEL_PATH) as f:
                source = f.read()
            for name, (kname, _, _) in _ACTIVATION_MAP.items():
                k = self.device.compile_kernel(source, kname)
                if k:
                    self._kernels[name] = k
        except Exception as e:
            logger.warning(f"ActivationOp compilation failed: {e}. Using CPU.")
            self.device.available = False

    def apply(
        self,
        activation: str,
        inputs: np.ndarray,
        alpha: float = 0.01,
    ) -> np.ndarray:
        """
        Apply an activation function.

        Args:
            activation : One of 'relu', 'sigmoid', 'tanh', 'leaky_relu', 'elu', 'softmax'
            inputs     : Input array (any shape), float32
            alpha      : Negative slope for leaky_relu / elu (default 0.01)

        Returns:
            Activated array (same shape as inputs)
        """
        activation = activation.lower()
        if activation not in _ACTIVATION_MAP:
            raise ValueError(
                f"Unknown activation '{activation}'. "
                f"Supported: {list(_ACTIVATION_MAP.keys())}"
            )

        if not self.device.available or activation not in self._kernels:
            return self._cpu_apply(activation, inputs, alpha)

        return self._cl_apply(activation, inputs, alpha)

    # ------------------------------------------------------------------
    # CPU Fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _cpu_apply(activation: str, inputs: np.ndarray, alpha: float) -> np.ndarray:
        if activation == "relu":
            return CPUFallback.relu(inputs)
        if activation == "sigmoid":
            return CPUFallback.sigmoid(inputs)
        if activation == "tanh":
            return CPUFallback.tanh(inputs)
        if activation == "leaky_relu":
            x = inputs.astype(np.float32)
            return np.where(x > 0, x, alpha * x)
        if activation == "elu":
            x = inputs.astype(np.float32)
            return np.where(x >= 0, x, alpha * (np.exp(x) - 1))
        if activation == "softmax":
            return CPUFallback.softmax(inputs)
        raise ValueError(f"Unsupported activation: {activation}")

    # ------------------------------------------------------------------
    # OpenCL dispatch
    # ------------------------------------------------------------------

    def _cl_apply(
        self, activation: str, inputs: np.ndarray, alpha: float
    ) -> np.ndarray:
        flat = np.ascontiguousarray(inputs.flatten(), dtype=np.float32)
        _, _, is_softmax = _ACTIVATION_MAP[activation]
        kernel = self._kernels[activation]

        if is_softmax:
            return self._cl_softmax(inputs)

        n_elements = flat.size
        buf_in  = self.device.to_device(flat)
        buf_out = self.device.empty_buffer((n_elements,))

        _, needs_alpha, _ = _ACTIVATION_MAP[activation]

        if needs_alpha:
            kernel(
                self.device.queue,
                (n_elements,),
                None,
                buf_in, buf_out,
                np.float32(alpha),
                np.int32(n_elements),
            )
        else:
            kernel(
                self.device.queue,
                (n_elements,),
                None,
                buf_in, buf_out,
                np.int32(n_elements),
            )

        result_flat = self.device.from_device(buf_out, (n_elements,))
        return result_flat.reshape(inputs.shape)

    def _cl_softmax(self, inputs: np.ndarray) -> np.ndarray:
        """Row-wise softmax. inputs : (batch, n_classes)."""
        if inputs.ndim == 1:
            inputs = inputs[np.newaxis, :]  # treat as batch of 1
            squeeze = True
        else:
            squeeze = False

        batch_size, n_classes = inputs.shape
        flat = np.ascontiguousarray(inputs, dtype=np.float32)

        buf_in  = self.device.to_device(flat)
        buf_out = self.device.empty_buffer((batch_size, n_classes))

        self._kernels["softmax"](
            self.device.queue,
            (batch_size,),
            None,
            buf_in, buf_out,
            np.int32(n_classes),
            np.int32(batch_size),
        )

        result = self.device.from_device(buf_out, (batch_size, n_classes))
        return result[0] if squeeze else result
