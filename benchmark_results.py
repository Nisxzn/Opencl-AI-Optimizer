
import time
import numpy as np
from optimizer import optimize_model, OptimizerConfig

def run_benchmark():
    # Test cases: Small (should use CPU) and Large (should use OpenCL)
    test_cases = [
        {"name": "Small (16x16)", "batch": 1,   "dim": 16,   "hidden": 16},
        {"name": "Medium (128x128)", "batch": 64,  "dim": 128,  "hidden": 256},
        {"name": "Large (1024x1024)", "batch": 128, "dim": 1024, "hidden": 1024},
    ]

    print(f"{'Test Case':<20} | {'Ops':<15} | {'Backend Used':<22} | {'Time (ms)':<10}")
    print("-" * 75)

    for case in test_cases:
        M, K, N = case['batch'], case['dim'], case['hidden']
        total_ops = 2 * M * K * N
        
        # Build spec
        W = np.random.randn(K, N).astype(np.float32)
        b = np.zeros(N, dtype=np.float32)
        spec = {
            "layers": [
                {"type": "dense", "name": "layer1", "weights": [W, b], "config": {"activation": "relu"}}
            ]
        }
        
        # Input
        x = np.random.randn(M, K).astype(np.float32)
        
        # Optimize
        config = OptimizerConfig(verbose=False)
        model = optimize_model(spec, config=config)
        
        # Warmup
        _ = model.predict(x)
        
        # Time
        start = time.perf_counter()
        for _ in range(50):
            _ = model.predict(x)
        elapsed = (time.perf_counter() - start) * 1000 / 50 # ms per call
        
        # Check backend used for the layer (via summary or internal)
        summary = model.layers[0].summary()
        backend = "OpenCL" if "OpenCL" in summary and "CPU" not in summary else "CPU"
        
        print(f"{case['name']:<20} | {total_ops:<15,} | {backend:<22} | {elapsed:<10.3f}")

if __name__ == "__main__":
    run_benchmark()
