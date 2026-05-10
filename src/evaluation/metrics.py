"""Evaluation utilities for compression experiments.

Functions
---------
count_parameters
    Count the total and non-zero parameter counts of a model.
model_size_mb
    Estimate the model's on-disk size in megabytes.
measure_latency
    Measure average inference latency over multiple warm-up and timed runs.
compute_sparsity
    Compute the fraction of zero weights across all (or selected) layers.
"""

from __future__ import annotations

import io
import time

import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Return ``{"total": <int>, "nonzero": <int>}`` parameter counts.

    Parameters
    ----------
    model:
        Any ``nn.Module``.

    Returns
    -------
    dict with keys ``"total"`` and ``"nonzero"``.
    """
    total = sum(p.numel() for p in model.parameters())
    nonzero = sum(
        int(p.count_nonzero().item()) for p in model.parameters()
    )
    return {"total": total, "nonzero": nonzero}


def model_size_mb(model: nn.Module) -> float:
    """Estimate the model size in megabytes by serialising to a buffer.

    Parameters
    ----------
    model:
        Any ``nn.Module``.

    Returns
    -------
    Size in megabytes (float).
    """
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    size_bytes = buffer.tell()
    return size_bytes / (1024 ** 2)


def measure_latency(
    model: nn.Module,
    example_input: torch.Tensor,
    num_warmup: int = 10,
    num_runs: int = 50,
    device: str | None = None,
) -> dict[str, float]:
    """Measure average and standard-deviation inference latency in milliseconds.

    Parameters
    ----------
    model:
        The model to profile.
    example_input:
        A single batch tensor (or tuple of tensors) to feed to the model.
    num_warmup:
        Number of warm-up forward passes (not timed).
    num_runs:
        Number of timed forward passes.
    device:
        If provided, the model and input are moved to this device first.

    Returns
    -------
    dict with keys ``"mean_ms"`` and ``"std_ms"``.
    """
    if device is not None:
        model = model.to(device)
        if isinstance(example_input, torch.Tensor):
            example_input = example_input.to(device)
        elif isinstance(example_input, (list, tuple)):
            example_input = type(example_input)(
                t.to(device) if isinstance(t, torch.Tensor) else t
                for t in example_input
            )

    model.eval()
    with torch.no_grad():
        # Warm-up.
        for _ in range(num_warmup):
            if isinstance(example_input, (list, tuple)):
                model(*example_input)
            else:
                model(example_input)

        # Timed runs.
        durations: list[float] = []
        for _ in range(num_runs):
            start = time.perf_counter()
            if isinstance(example_input, (list, tuple)):
                model(*example_input)
            else:
                model(example_input)
            durations.append((time.perf_counter() - start) * 1000)

    mean_ms = sum(durations) / len(durations)
    variance = sum((d - mean_ms) ** 2 for d in durations) / len(durations)
    std_ms = variance ** 0.5
    return {"mean_ms": mean_ms, "std_ms": std_ms}


def compute_sparsity(
    model: nn.Module,
    layer_types: tuple[type[nn.Module], ...] = (nn.Linear, nn.Conv2d),
) -> float:
    """Compute global sparsity over all weight parameters in *layer_types*.

    Parameters
    ----------
    model:
        The model to evaluate.
    layer_types:
        A tuple of layer types to include in the sparsity calculation.

    Returns
    -------
    Global sparsity as a float in [0, 1].
    """
    total_params = 0
    total_zeros = 0
    for module in model.modules():
        if isinstance(module, layer_types):
            total_params += module.weight.numel()
            total_zeros += int(torch.sum(module.weight == 0).item())
    return total_zeros / total_params if total_params > 0 else 0.0
