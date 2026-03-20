"""
OpenCL AI Inference Optimizer
Package setup (legacy compatible + modern pyproject.toml backed).
"""

from setuptools import setup, find_packages
import os

# Read long description from README
_here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(_here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name             = "opencl-ai-optimizer",
    version          = "1.0.0",
    description      = "Lightweight OpenCL-accelerated inference optimizer for ML/DL models",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    author           = "OpenCL AI Optimizer Team",
    license          = "MIT",
    url              = "https://github.com/your-org/opencl-ai-optimizer",

    packages         = find_packages(exclude=["tests*", "examples*"]),
    package_data     = {
        "optimizer": ["ops/kernels/*.cl"],   # Include .cl kernel files in wheel
    },
    include_package_data = True,

    python_requires  = ">=3.8",
    install_requires = [
        "numpy>=1.21",
    ],
    extras_require   = {
        "opencl": ["pyopencl>=2022.1"],
        "keras":  ["tensorflow>=2.10"],
        "dev":    [
            "pytest>=7.0",
            "pytest-cov",
        ],
        "all": [
            "pyopencl>=2022.1",
            "tensorflow>=2.10",
            "pytest>=7.0",
        ],
    },

    classifiers = [
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Hardware",
    ],

    keywords = (
        "opencl gpu acceleration deep-learning inference optimizer "
        "neural-network cnn rnn matmul keras"
    ),

    entry_points = {
        "console_scripts": [
            "opencl-optimizer-benchmark=examples.benchmark_suite:main",
        ],
    },
)
