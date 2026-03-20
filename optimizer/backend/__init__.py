"""
OpenCL Backend Package
Handles device discovery, context creation, and OpenCL environment setup.
"""

from optimizer.backend.opencl_device import OpenCLDevice, get_device
from optimizer.backend.fallback import CPUFallback

__all__ = ["OpenCLDevice", "get_device", "CPUFallback"]
