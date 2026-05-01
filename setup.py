from setuptools import setup, find_packages

setup(
    name="css586_compression",
    version="0.1.0",
    description="Optimizing Zero-Shot Model Compression Pipelines for Efficient Inference",
    author="Henry Morgan",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.30.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "flake8>=6.0.0",
        ]
    },
)
