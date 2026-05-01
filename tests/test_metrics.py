"""Tests for evaluation metrics."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn as nn
import pytest

from evaluation.metrics import (
    count_parameters,
    model_size_mb,
    measure_latency,
    compute_sparsity,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def simple_linear() -> nn.Module:
    return nn.Linear(8, 4)


@pytest.fixture()
def mlp() -> nn.Module:
    return nn.Sequential(
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 4),
    )


# ---------------------------------------------------------------------------
# count_parameters
# ---------------------------------------------------------------------------

class TestCountParameters:
    def test_returns_dict_with_keys(self, simple_linear):
        result = count_parameters(simple_linear)
        assert "total" in result
        assert "nonzero" in result

    def test_total_matches_manual_count(self, simple_linear):
        expected = sum(p.numel() for p in simple_linear.parameters())
        assert count_parameters(simple_linear)["total"] == expected

    def test_nonzero_le_total(self, mlp):
        result = count_parameters(mlp)
        assert result["nonzero"] <= result["total"]

    def test_zeroed_weights_reduce_nonzero(self, simple_linear):
        simple_linear.weight.data.zero_()
        result = count_parameters(simple_linear)
        # bias is still non-zero; total zeroed includes weight params
        assert result["nonzero"] < result["total"]


# ---------------------------------------------------------------------------
# model_size_mb
# ---------------------------------------------------------------------------

class TestModelSizeMb:
    def test_returns_positive_float(self, mlp):
        size = model_size_mb(mlp)
        assert isinstance(size, float)
        assert size > 0.0

    def test_larger_model_is_bigger(self):
        small = nn.Linear(4, 4)
        large = nn.Linear(256, 256)
        assert model_size_mb(large) > model_size_mb(small)


# ---------------------------------------------------------------------------
# measure_latency
# ---------------------------------------------------------------------------

class TestMeasureLatency:
    def test_returns_dict_with_mean_and_std(self, mlp):
        x = torch.randn(1, 16)
        result = measure_latency(mlp, x, num_warmup=2, num_runs=5)
        assert "mean_ms" in result
        assert "std_ms" in result

    def test_mean_is_positive(self, mlp):
        x = torch.randn(1, 16)
        result = measure_latency(mlp, x, num_warmup=1, num_runs=3)
        assert result["mean_ms"] > 0.0

    def test_std_is_non_negative(self, mlp):
        x = torch.randn(1, 16)
        result = measure_latency(mlp, x, num_warmup=1, num_runs=3)
        assert result["std_ms"] >= 0.0


# ---------------------------------------------------------------------------
# compute_sparsity
# ---------------------------------------------------------------------------

class TestComputeSparsity:
    def test_dense_model_has_low_sparsity(self, mlp):
        result = compute_sparsity(mlp)
        assert result["global_sparsity"] < 0.01

    def test_zeroed_linear_has_high_sparsity(self, simple_linear):
        simple_linear.weight.data.zero_()
        result = compute_sparsity(simple_linear, layer_types=(nn.Linear,))
        # Weight is all zeros; sparsity should be close to 1.
        assert result["global_sparsity"] > 0.9

    def test_returns_layer_sparsities_dict(self, mlp):
        result = compute_sparsity(mlp)
        assert "layer_sparsities" in result
        assert isinstance(result["layer_sparsities"], dict)

    def test_no_matching_layers_returns_zero(self):
        model = nn.Sequential(nn.ReLU(), nn.Sigmoid())
        result = compute_sparsity(model, layer_types=(nn.Linear,))
        assert result["global_sparsity"] == 0.0
