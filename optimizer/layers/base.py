"""
Base Layer Interface
====================
All optimized layers inherit from BaseLayer. This ensures a consistent
forward() interface regardless of which backend is being used.
"""

import numpy as np
from abc import ABC, abstractmethod
from optimizer.backend.opencl_device import OpenCLDevice, get_device


class BaseLayer(ABC):
    """
    Abstract base class for all optimizer layers.

    Every layer must implement:
        forward(inputs) → np.ndarray

    Subclasses may also implement:
        load_weights(**kwargs)
        summary()
    """

    def __init__(self, name: str = "layer", device: OpenCLDevice = None):
        self.name = name
        self.device = device or get_device()

    @abstractmethod
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Run the layer's forward pass."""
        ...

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """Make layers callable like functions."""
        return self.forward(inputs)

    def summary(self) -> str:
        """Return a human-readable summary of this layer."""
        return f"{self.__class__.__name__}(name='{self.name}')"

    def __repr__(self) -> str:
        return self.summary()
