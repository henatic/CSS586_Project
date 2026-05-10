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
        [
            StructuredPruner(sparsity=0.25),
            MagnitudePruner(sparsity=0.5),
            DynamicQuantizer(),
        ],
    ),
]


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------


def main() -> None:
    """Run all pipeline configurations and print a summary."""
    print("=" * 80)
    print("Running Compression Pipeline Comparison")
    print("=" * 80)

    # We use a new model for each pipeline to ensure a fair comparison
    results = []
    for name, stages in PIPELINE_CONFIGS:
        print(f"\nRunning pipeline: {name}...")
        model = BenchmarkMLP()
        eval_fns = {**EVAL_FNS, "latency": _latency_fn}

        if not stages:
            # Baseline case
            baseline_metrics = {name: fn(model) for name, fn in eval_fns.items()}
            results.append((name, baseline_metrics))
            continue

        pipeline = CompressionPipeline(stages=stages, eval_fns=eval_fns)
        _, report = pipeline.run(model)

        # For the final report, we care about the metrics *after* the last stage
        final_metrics = report["stages"][-1]["metrics_after"]
        results.append((name, final_metrics))

    _print_summary_table(results)


def _print_summary_table(results: list[tuple[str, dict]]) -> None:
    """Print a formatted table of results."""
    # Header
    print("\n" + "=" * 80)
    print("Summary of Results")
    print("-" * 80)
    print(
        f"{'Pipeline':<45} | {'Size (MB)':>10} | {'Sparsity (%)':>12} | {'Latency (ms)':>15}"
    )
    print("-" * 80)

    # Rows
    for name, metrics in results:
        size = metrics["size_mb"]
        sparsity = metrics["sparsity"] * 100
        latency = metrics["latency"]["mean_ms"]
        print(f"{name:<45} | {size:>10.2f} | {sparsity:>11.2f} % | {latency:>14.2f} ms")

    print("=" * 80)


if __name__ == "__main__":
    main()
