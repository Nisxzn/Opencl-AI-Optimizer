
import time
import numpy as np
import pyopencl as cl
from optimizer.ops.matmul import MatMulOp
from optimizer.backend.opencl_device import get_device

def benchmark():
    device = get_device()
    if not device.available:
        print("OpenCL not available. Cannot benchmark.")
        return

    matmul_op = MatMulOp(device=device)
    
    # Test different sizes
    sizes = [
        (16, 16, 16),
        (64, 64, 64),
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024)
    ]
    
    print(f"{'Size (MxKxN)':<20} | {'NumPy (ms)':<15} | {'OpenCL (ms)':<15} | {'Speedup':<10}")
    print("-" * 70)
    
    for M, K, N in sizes:
        A = np.random.rand(M, K).astype(np.float32)
        B = np.random.rand(K, N).astype(np.float32)
        
        # Warmup
        _ = np.matmul(A, B)
        # We need to reach inside or use the Op's internal cl method to avoid the current (fake) "fallback" check if we want to measure just OpenCL
        # But wait, MatMulOp.matmul does exactly what we want to measure (it includes the transfer overhead right now)
        _ = matmul_op.matmul(A, B)
        
        # NumPy Timing
        start = time.perf_counter()
        for _ in range(10):
            C_numpy = np.matmul(A, B)
        numpy_time = (time.perf_counter() - start) * 100  # Average ms
        
        # OpenCL Timing
        start = time.perf_counter()
        for _ in range(10):
            C_cl = matmul_op.matmul(A, B)
        cl_time = (time.perf_counter() - start) * 100  # Average ms
        
        speedup = numpy_time / cl_time
        print(f"{f'{M}x{K}x{N}':<20} | {numpy_time:<15.4f} | {cl_time:<15.4f} | {speedup:<10.2f}x")

if __name__ == "__main__":
    benchmark()
