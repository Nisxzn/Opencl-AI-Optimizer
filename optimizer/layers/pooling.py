"""
Pooling Layer
=============
Standalone pooling layer supporting Max Pool, Average Pool,
and Global Average Pool with OpenCL acceleration.
"""

import numpy as np
from typing import Optional
from optimizer.layers.base import BaseLayer
from optimizer.ops.pooling import PoolingOp
from optimizer.backend.opencl_device import OpenCLDevice


class PoolingLayer(BaseLayer):
    """
    Optimized 2D Pooling layer (NCHW format).

    Args:
        mode       : 'max', 'avg', or 'global_avg'
        pool_size  : Spatial size of the pooling window (default 2)
        stride     : Pool stride (default 2). Ignored for global_avg.
        name       : Layer name.

    Input  : (N, C, H, W)
    Output : (N, C, H_out, W_out)  or  (N, C) for global_avg

    Example:
        pool = PoolingLayer(mode='max', pool_size=2, stride=2)
        out  = pool.forward(x)
    """

    def __init__(
        self,
        mode: str = "max",
        pool_size: int = 2,
        stride: int = 2,
        name: Optional[str] = None,
        device: Optional[OpenCLDevice] = None,
    ):
        mode = mode.lower()
        if mode not in ("max", "avg", "global_avg"):
            raise ValueError("mode must be 'max', 'avg', or 'global_avg'")

        super().__init__(name=name or f"{mode}_pool", device=device)
        self.mode      = mode
        self.pool_size = pool_size
        self.stride    = stride
        self._pool_op  = PoolingOp(device=self.device)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        if self.mode == "max":
            return self._pool_op.max_pool(inputs, self.pool_size, self.stride)
        if self.mode == "avg":
            return self._pool_op.avg_pool(inputs, self.pool_size, self.stride)
        # global_avg
        return self._pool_op.global_avg_pool(inputs)

    def summary(self) -> str:
        backend = "OpenCL" if self.device.available else "CPU"
        if self.mode == "global_avg":
            return f"PoolingLayer(name='{self.name}', mode='global_avg', backend={backend})"
        return (
            f"PoolingLayer(name='{self.name}', mode='{self.mode}', "
            f"pool_size={self.pool_size}, stride={self.stride}, backend={backend})"
        )
