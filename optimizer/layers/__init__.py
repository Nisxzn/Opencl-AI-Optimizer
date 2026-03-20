"""
Layer Abstractions Package
High-level layer wrappers that dispatch to OpenCL ops or CPU fallback.
"""

from optimizer.layers.base import BaseLayer
from optimizer.layers.dense import DenseLayer
from optimizer.layers.conv2d import Conv2DLayer
from optimizer.layers.activation import ActivationLayer
from optimizer.layers.pooling import PoolingLayer

__all__ = [
    "BaseLayer",
    "DenseLayer",
    "Conv2DLayer",
    "ActivationLayer",
    "PoolingLayer",
]
