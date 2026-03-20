"""
OpenCL Device Manager
=====================
Handles device discovery, context creation, command queue, kernel compilation,
and memory buffer management. All OpenCL interactions go through this module.

Supports:
  - Automatic best-device selection (GPU preferred, then CPU)
  - Kernel source loading from .cl files or inline strings
  - Buffer allocation helpers
  - Graceful fallback when PyOpenCL is not installed
"""

import os
import numpy as np
from typing import Optional, Dict, Tuple
from optimizer.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Try to import PyOpenCL — if unavailable the whole module degrades gracefully
# ---------------------------------------------------------------------------
try:
    import pyopencl as cl
    import pyopencl.array as cl_array
    OPENCL_AVAILABLE = True
except ImportError:
    OPENCL_AVAILABLE = False
    logger.warning(
        "PyOpenCL not found. Install it with: pip install pyopencl\n"
        "Falling back to CPU (NumPy) implementation for all operations."
    )


class OpenCLDevice:
    """
    Manages a single OpenCL device context.

    Attributes:
        platform  : The selected OpenCL platform.
        device    : The selected OpenCL device (GPU or CPU).
        context   : The OpenCL context attached to the device.
        queue     : The command queue for issuing GPU commands.
        available : Whether OpenCL is usable on this system.
    """

    def __init__(
        self,
        platform_index: int = 0,
        device_index: int = 0,
        prefer_gpu: bool = True,
    ):
        self.available = OPENCL_AVAILABLE
        self.platform = None
        self.device = None
        self.context = None
        self.queue = None
        self._kernels: Dict[str, "cl.Kernel"] = {}          # compiled kernel cache
        self._programs: Dict[str, "cl.Program"] = {}        # compiled program cache

        if not self.available:
            return

        self._setup_device(platform_index, device_index, prefer_gpu)

    # ------------------------------------------------------------------
    # Device setup
    # ------------------------------------------------------------------

    def _setup_device(
        self, platform_index: int, device_index: int, prefer_gpu: bool
    ) -> None:
        """Discover platforms/devices and create context + queue."""
        platforms = cl.get_platforms()
        if not platforms:
            logger.error("No OpenCL platforms found on this system.")
            self.available = False
            return

        # Try to get GPU first if preferred
        selected_platform = None
        selected_device = None

        if prefer_gpu:
            for plat in platforms:
                gpu_devices = plat.get_devices(device_type=cl.device_type.GPU)
                if gpu_devices:
                    selected_platform = plat
                    selected_device = gpu_devices[0]
                    break

        # Fall back to CPU OpenCL device
        if selected_device is None:
            for plat in platforms:
                cpu_devices = plat.get_devices(device_type=cl.device_type.CPU)
                if cpu_devices:
                    selected_platform = plat
                    selected_device = cpu_devices[0]
                    break

        # Last resort: explicit index
        if selected_device is None:
            try:
                selected_platform = platforms[platform_index]
                all_devices = selected_platform.get_devices()
                selected_device = all_devices[device_index]
            except IndexError:
                logger.error("Could not select any OpenCL device.")
                self.available = False
                return

        self.platform = selected_platform
        self.device = selected_device
        self.context = cl.Context([self.device])
        self.queue = cl.CommandQueue(
            self.context,
            properties=cl.command_queue_properties.PROFILING_ENABLE,
        )
        self._buffer_cache: Dict[int, List["cl.Buffer"]] = {} 

        logger.info(
            f"OpenCL Device Selected:\n"
            f"  Platform : {self.platform.name}\n"
            f"  Device   : {self.device.name}\n"
            f"  Type     : {cl.device_type.to_string(self.device.type)}\n"
            f"  Max CUs  : {self.device.max_compute_units}\n"
            f"  Max Freq : {self.device.max_clock_frequency} MHz\n"
            f"  Glob Mem : {self.device.global_mem_size // (1024**2)} MB"
        )

    # ------------------------------------------------------------------
    # Kernel compilation
    # ------------------------------------------------------------------

    def compile_kernel(
        self,
        source: str,
        kernel_name: str,
        build_options: str = "",
    ) -> Optional["cl.Kernel"]:
        """
        Compile an OpenCL kernel from source string.

        Args:
            source       : OpenCL C kernel source code.
            kernel_name  : Name of the __kernel function to extract.
            build_options: Optional compiler flags (e.g. "-cl-fast-relaxed-math").

        Returns:
            A compiled cl.Kernel object, or None on failure.
        """
        if not self.available:
            return None

        cache_key = f"{kernel_name}_{hash(source)}"
        if cache_key in self._kernels:
            return self._kernels[cache_key]

        try:
            program = cl.Program(self.context, source).build(
                options=build_options + " -cl-fast-relaxed-math"
            )
            kernel = getattr(program, kernel_name)
            self._kernels[cache_key] = kernel
            self._programs[cache_key] = program
            logger.debug(f"Compiled kernel: '{kernel_name}'")
            return kernel
        except cl.RuntimeError as e:
            logger.error(f"Kernel compilation failed for '{kernel_name}': {e}")
            return None

    def compile_kernel_from_file(
        self,
        cl_file_path: str,
        kernel_name: str,
        build_options: str = "",
    ) -> Optional["cl.Kernel"]:
        """Load and compile a kernel from a .cl file."""
        try:
            with open(cl_file_path, "r") as f:
                source = f.read()
            return self.compile_kernel(source, kernel_name, build_options)
        except FileNotFoundError:
            logger.error(f"Kernel file not found: {cl_file_path}")
            return None

    # ------------------------------------------------------------------
    # Buffer helpers
    # ------------------------------------------------------------------

    def to_device(self, array: np.ndarray) -> Optional["cl.Buffer"]:
        """Copy a numpy array to device memory (read-only)."""
        if not self.available:
            return None
        arr = np.ascontiguousarray(array, dtype=np.float32)
        # Using COPY_HOST_PTR is fine for small/medium arrays.
        # For very large arrays, consider using memory pools.
        return cl.Buffer(
            self.context,
            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=arr,
        )

    def empty_buffer(self, shape: Tuple[int, ...]) -> Optional["cl.Buffer"]:
        """Allocate an empty write-only buffer for output, reusing cache if possible."""
        if not self.available:
            return None
        size = int(np.prod(shape)) * np.dtype(np.float32).itemsize
        
        # Check cache
        if size in self._buffer_cache and self._buffer_cache[size]:
            return self._buffer_cache[size].pop()

        return cl.Buffer(self.context, cl.mem_flags.READ_WRITE, size=size)

    def release_buffer(self, buffer: "cl.Buffer"):
        """Return a buffer to the cache."""
        if buffer is None: return
        size = buffer.size
        if size not in self._buffer_cache:
            self._buffer_cache[size] = []
        self._buffer_cache[size].append(buffer)

    def from_device(
        self, buffer: "cl.Buffer", shape: Tuple[int, ...], release: bool = True
    ) -> np.ndarray:
        """Copy data from device buffer back to a numpy array and optionally release."""
        result = np.empty(shape, dtype=np.float32)
        cl.enqueue_copy(self.queue, result, buffer)
        self.queue.finish()
        if release:
            self.release_buffer(buffer)
        return result

    # ------------------------------------------------------------------
    # Device info
    # ------------------------------------------------------------------

    def device_info(self) -> dict:
        """Return a dictionary of device properties."""
        if not self.available or self.device is None:
            return {"available": False}
        return {
            "available": True,
            "platform": self.platform.name,
            "device": self.device.name,
            "type": cl.device_type.to_string(self.device.type),
            "compute_units": self.device.max_compute_units,
            "max_freq_mhz": self.device.max_clock_frequency,
            "global_mem_mb": self.device.global_mem_size // (1024**2),
            "local_mem_kb": self.device.local_mem_size // 1024,
            "max_work_group_size": self.device.max_work_group_size,
        }

    def __repr__(self) -> str:
        if not self.available:
            return "OpenCLDevice(unavailable — PyOpenCL not installed)"
        if self.device is None:
            return "OpenCLDevice(no device found)"
        return f"OpenCLDevice(device='{self.device.name}', platform='{self.platform.name}')"


# ---------------------------------------------------------------------------
# Module-level singleton — shared across all ops
# ---------------------------------------------------------------------------
_global_device: Optional[OpenCLDevice] = None


def get_device(
    platform_index: int = 0,
    device_index: int = 0,
    prefer_gpu: bool = True,
    force_new: bool = False,
) -> OpenCLDevice:
    """
    Get (or create) the global OpenCL device singleton.

    Args:
        platform_index : Platform index if manual selection needed.
        device_index   : Device index if manual selection needed.
        prefer_gpu     : If True, try to pick a GPU device first.
        force_new      : If True, always create a fresh device object.

    Returns:
        An OpenCLDevice instance.
    """
    global _global_device
    if _global_device is None or force_new:
        _global_device = OpenCLDevice(
            platform_index=platform_index,
            device_index=device_index,
            prefer_gpu=prefer_gpu,
        )
    return _global_device
