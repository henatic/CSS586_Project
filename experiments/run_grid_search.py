"""
Automated pipeline search using grid search to find the optimal compression
pipeline for a ResNet-18 model on CIFAR-10.

Run from the repository root:
    python experiments/run_grid_search.py
"""

from __future__ import annotations

import sys
import os
import itertools
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from compression.pruning import MagnitudePruner, StructuredPruner
from compression.quantization import DynamicQuantizer, StaticQuantizer
from pipeline.pipeline import CompressionPipeline
from evaluation.metrics import (
    count_parameters,
    model_size_mb,
    compute_sparsity,
    measure_latency,
)

# ---------------------------------------------------------------------------
# Dataset and Model Loading (similar to run_real_world_experiment.py)
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
    """Get a pre-trained ResNet-18 model, adjusted for CIFAR-10."""
    model = torchvision.models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model

def get_resnet18_modules_to_fuse() -> list[list[str]]:
    """Return module names to fuse for ResNet-18 static quantization."""
    modules_to_fuse = [["conv1", "bn1"]]
    for layer_idx in range(1, 5):
        layer_name = f"layer{layer_idx}"
        for block_idx in range(2):
            block_prefix = f"{layer_name}.{block_idx}"
            modules_to_fuse.append([f"{block_prefix}.conv1", f"{block_prefix}.bn1"])
            modules_to_fuse.append([f"{block_prefix}.conv2", f"{block_prefix}.bn2"])
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
# Grid Search Experiment
# ---------------------------------------------------------------------------

def main():
    """Run the grid search experiment."""
    train_loader, test_loader = get_cifar10_loaders()
    
    eval_fns = {
        "params": count_parameters,
        "size_mb": model_size_mb,
        "sparsity": compute_sparsity,
        "accuracy": lambda m: accuracy_fn(m, test_loader),
        "latency": _latency_fn,
    }

    # Define the search space for compression techniques
    search_space = {
        "pruning": [
            None,
            MagnitudePruner(sparsity=0.25),
            MagnitudePruner(sparsity=0.5),
            StructuredPruner(sparsity=0.25),
        ],
        "quantization": [
            None,
            DynamicQuantizer(),
            StaticQuantizer(
                calibration_loader=train_loader,
                modules_to_fuse=get_resnet18_modules_to_fuse(),
            ),
        ],
    }

    # Generate all possible pipeline orderings from the search space
    components = list(search_space.keys())
    pipelines = []
    for r in range(1, len(components) + 1):
        for p in itertools.permutations(components, r):
            pipelines.append(p)

    # Prepare CSV for results
    output_path = "results/grid_search_results.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Pipeline", "Size (MB)", "Sparsity (%)", "Accuracy", "Latency (ms)"
        ])

        # Evaluate baseline
        print("Evaluating Baseline...")
        baseline_model = get_resnet18_model()
        baseline_metrics = {name: fn(baseline_model) for name, fn in eval_fns.items()}
        
        # The compute_sparsity function returns a dictionary, so we extract the global sparsity value.
        baseline_sparsity = baseline_metrics['sparsity']
        if isinstance(baseline_sparsity, dict):
            baseline_sparsity = baseline_sparsity.get('global_sparsity', 0.0)

        writer.writerow([
            "Baseline",
            f"{baseline_metrics['size_mb']:.2f}",
            f"{baseline_sparsity * 100:.2f}",
            f"{baseline_metrics['accuracy']:.4f}",
            f"{baseline_metrics['latency']['mean_ms']:.2f}",
        ])

        # Run grid search
        for pipeline_keys in pipelines:
            options = [search_space[key] for key in pipeline_keys]
            for stages_tuple in itertools.product(*options):
                stages = [s for s in stages_tuple if s is not None]
                if not stages:
                    continue

                name = " -> ".join(s.__class__.__name__ for s in stages)
                print(f"Running pipeline: {name}...")

                current_model = get_resnet18_model()
                pipeline = CompressionPipeline(stages=stages, eval_fns=eval_fns)

                try:
                    _, report = pipeline.run(current_model)
                    final_metrics = report["stages"][-1]["metrics_after"]
                    writer.writerow([
                        name,
                        f"{final_metrics['size_mb']:.2f}",
                        f"{final_metrics['sparsity'] * 100:.2f}",
                        f"{final_metrics['accuracy']:.4f}",
                        f"{final_metrics['latency']['mean_ms']:.2f}",
                    ])
                except Exception as e:
                    print(f"Skipping pipeline '{name}' due to error: {e}")

    print(f"Grid search complete. Results saved to {output_path}")

if __name__ == "__main__":
    main()
