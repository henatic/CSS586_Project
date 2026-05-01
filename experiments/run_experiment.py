"""Example experiment: compare different pipeline orderings on a small MLP.

Run from the repository root:
    python experiments/run_experiment.py

The script evaluates several pipeline configurations (orderings of
quantization and pruning) on a randomly-initialised model and prints a
formatted summary table.  No training data or GPU is required.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn as nn

from compression.pruning import MagnitudePruner, StructuredPruner
from compression.quantization import DynamicQuantizer
from pipeline.pipeline import CompressionPipeline
from evaluation.metrics import (
    count_parameters,
    model_size_mb,
    compute_sparsity,
    measure_latency,
)


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class BenchmarkMLP(nn.Module):
    """A small MLP that serves as a stand-in for a real task model."""

    def __init__(
        self,
        in_features: int = 128,
        hidden: int = 256,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

EVAL_FNS = {
    "params": count_parameters,
    "size_mb": model_size_mb,
    "sparsity": compute_sparsity,
}


def _latency_fn(model: nn.Module) -> dict[str, float]:
    example = torch.randn(1, 128)
    return measure_latency(model, example, num_warmup=5, num_runs=20)


# ---------------------------------------------------------------------------
# Pipeline configurations to compare
# ---------------------------------------------------------------------------

PIPELINE_CONFIGS: list[tuple[str, list]] = [
    (
        "Baseline (no compression)",
        [],
    ),
    (
        "Dynamic Quantization only",
        [DynamicQuantizer()],
    ),
    (
        "Magnitude Pruning (50 %) only",
        [MagnitudePruner(sparsity=0.5)],
    ),
    (
        "Prune → Quantize",
        [MagnitudePruner(sparsity=0.5), DynamicQuantizer()],
    ),
    (
        "Quantize → Prune",
        [DynamicQuantizer(), MagnitudePruner(sparsity=0.5)],
    ),
    (
        "Structured Prune → Magnitude Prune → Quantize",
        [StructuredPruner(sparsity=0.2), MagnitudePruner(sparsity=0.4), DynamicQuantizer()],
    ),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiments() -> None:
    torch.manual_seed(42)
    original_model = BenchmarkMLP()

    print("\n" + "=" * 72)
    print(" Zero-Shot Model Compression Pipeline Comparison")
    print("=" * 72)

    results = []

    for name, stages in PIPELINE_CONFIGS:
        if not stages:
            # Baseline – no compression.
            model = original_model
            size = model_size_mb(model)
            params = count_parameters(model)
            sparsity = compute_sparsity(model)["global_sparsity"]
            latency = _latency_fn(model)
            results.append(
                {
                    "name": name,
                    "size_mb": size,
                    "total_params": params["total"],
                    "sparsity": sparsity,
                    "mean_ms": latency["mean_ms"],
                    "std_ms": latency["std_ms"],
                    "duration_s": 0.0,
                }
            )
        else:
            pipeline = CompressionPipeline(
                stages=stages,
                eval_fns={
                    "size_mb": model_size_mb,
                    "params": count_parameters,
                    "sparsity": compute_sparsity,
                },
            )
            compressed, report = pipeline.run(original_model)
            after = report["stages"][-1]["metrics_after"]
            latency = _latency_fn(compressed)
            results.append(
                {
                    "name": name,
                    "size_mb": after["size_mb"],
                    "total_params": after["params"]["total"],
                    "sparsity": after["sparsity"]["global_sparsity"],
                    "mean_ms": latency["mean_ms"],
                    "std_ms": latency["std_ms"],
                    "duration_s": report["total_duration_s"],
                }
            )

    # Print results table.
    col_w = 44
    header = (
        f"{'Configuration':<{col_w}}  {'Size(MB)':>9}  {'Sparsity':>9}"
        f"  {'Latency(ms)':>11}  {'Compress(s)':>11}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['name']:<{col_w}}  {r['size_mb']:>9.3f}"
            f"  {r['sparsity']:>9.2%}"
            f"  {r['mean_ms']:>8.2f}±{r['std_ms']:.2f}"
            f"  {r['duration_s']:>11.3f}"
        )
    print("=" * len(header))
    print()


if __name__ == "__main__":
    run_experiments()
