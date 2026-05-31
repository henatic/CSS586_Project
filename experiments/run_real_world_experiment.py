"""
Experiment with applying compression pipelines to a pre-trained ResNet-18 model
on the CIFAR-10 dataset.

Run from the repository root:
    python experiments/run_real_world_experiment.py
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from compression.pruning import MagnitudePruner, StructuredPruner
from compression.quantization import DynamicQuantizer, StaticQuantizer
from compression.distillation import ZeroShotDistiller
from pipeline.pipeline import CompressionPipeline
from evaluation.metrics import (
    count_parameters,
    model_size_mb,
    compute_sparsity,
    measure_latency,
)

# ---------------------------------------------------------------------------
# Dataset and Model Loading
# ---------------------------------------------------------------------------

def get_cifar10_loaders(batch_size: int = 128) -> tuple[DataLoader, DataLoader]:
    """Get CIFAR-10 train and test loaders."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    return trainloader, testloader

def get_resnet18_model() -> nn.Module:
    """Get a pre-trained ResNet-18 model."""
    model = torchvision.models.resnet18(pretrained=True)
    # Adjust for CIFAR-10
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model

def get_resnet18_modules_to_fuse() -> list[list[str]]:
    """Return module names to fuse for ResNet-18 static quantization."""
    modules_to_fuse = [
        ["conv1", "bn1"],
    ]

    for layer_idx in range(1, 5):
        layer_name = f"layer{layer_idx}"
        for block_idx in range(2):
            block_prefix = f"{layer_name}.{block_idx}"
            modules_to_fuse.append([f"{block_prefix}.conv1", f"{block_prefix}.bn1"])
            modules_to_fuse.append([f"{block_prefix}.conv2", f"{block_prefix}.bn2"])

        # Fuse downsample path for the first block in layers 2-4
        if layer_idx in (2, 3, 4):
            modules_to_fuse.append([f"{layer_name}.0.downsample.0", f"{layer_name}.0.downsample.1"])

    return modules_to_fuse

# ---------------------------------------------------------------------------
# Evaluation Helpers
# ---------------------------------------------------------------------------

def accuracy_fn(model: nn.Module, test_loader: DataLoader) -> float:
    """Computes model accuracy on the test set."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def _latency_fn(model: nn.Module) -> dict[str, float]:
    """Measures model latency."""
    example = torch.randn(1, 3, 32, 32)
    return measure_latency(model, example, num_warmup=5, num_runs=20)

# ---------------------------------------------------------------------------
# Main Experiment
# ---------------------------------------------------------------------------

def main():
    """Run the real-world experiment."""
    # Load data and model
    train_loader, test_loader = get_cifar10_loaders()
    model = get_resnet18_model()

    # Define evaluation functions
    eval_fns = {
        "params": count_parameters,
        "size_mb": model_size_mb,
        "sparsity": compute_sparsity,
        "accuracy": lambda m: accuracy_fn(m, test_loader),
        "latency": _latency_fn,
    }

    resnet_modules_to_fuse = get_resnet18_modules_to_fuse()

    # Define pipeline configurations
    pipeline_configs: list[tuple[str, list]] = [
        ("Baseline (no compression)", []),
        ("Dynamic Quantization", [DynamicQuantizer()]),
        (
            "Static Quantization",
            [
                StaticQuantizer(
                    calibration_loader=train_loader,
                    modules_to_fuse=resnet_modules_to_fuse,
                )
            ],
        ),
        ("Magnitude Pruning (50%)", [MagnitudePruner(sparsity=0.5)]),
        (
            "Prune -> Static Quantize",
            [
                MagnitudePruner(sparsity=0.5),
                StaticQuantizer(
                    calibration_loader=train_loader,
                    modules_to_fuse=resnet_modules_to_fuse,
                ),
            ],
        ),
    ]

    # Run pipelines and collect results
    results = []
    for name, stages in pipeline_configs:
        print(f"Running pipeline: {name}...")
        # Use a fresh model for each pipeline
        current_model = get_resnet18_model()

        if not stages:
            # Baseline case
            baseline_metrics = {
                name: fn(current_model) for name, fn in eval_fns.items()
            }
            results.append((name, baseline_metrics))
            continue

        pipeline = CompressionPipeline(stages=stages, eval_fns=eval_fns)
        try:
            _, report = pipeline.run(current_model)
        except RuntimeError as exc:
            print(f"Skipping pipeline '{name}': {exc}")
            continue

        final_metrics = report["stages"][-1]["metrics_after"]
        results.append((name, final_metrics))

    output_path = "results/real_world_experiment_results.txt"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("Real-World Compression Pipeline Comparison (ResNet-18 on CIFAR-10)\n")
        f.write("-" * 80 + "\n")
        f.write(
            f"{'Pipeline':<30} | {'Size (MB)':>10} | {'Sparsity (%)':>12} | {'Accuracy':>10} | {'Latency (ms)':>15}\n"
        )
        f.write("-" * 80 + "\n")
        for name, metrics in results:
            size = metrics["size_mb"]
            sparsity = metrics["sparsity"] * 100
            acc = metrics["accuracy"]
            latency = metrics["latency"]["mean_ms"]
            f.write(
                f"{name:<30} | {size:>10.2f} | {sparsity:>11.2f} % | {acc:>9.4f} | {latency:>14.2f} ms\n"
            )
        f.write("=" * 80 + "\n")

    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
