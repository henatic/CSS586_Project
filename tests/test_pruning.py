"""Tests for MagnitudePruner and StructuredPruner."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn as nn
import pytest

from compression.pruning import MagnitudePruner, StructuredPruner


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class SmallCNN(nn.Module):
    """Tiny CNN for fast testing."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.conv(x)))
        return self.fc(x.flatten(1))


@pytest.fixture()
def cnn() -> SmallCNN:
    return SmallCNN()


@pytest.fixture()
def mlp() -> nn.Module:
    return nn.Sequential(
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Linear(16, 4),
    )


# ---------------------------------------------------------------------------
# MagnitudePruner tests
# ---------------------------------------------------------------------------

class TestMagnitudePruner:
    def test_returns_module_and_metadata(self, mlp):
        pmodel, meta = MagnitudePruner(sparsity=0.5).compress(mlp)
        assert isinstance(pmodel, nn.Module)
        assert meta["technique"] == "magnitude_pruning"
        assert meta["config"]["sparsity"] == 0.5

    def test_original_model_unchanged(self, mlp):
        w0 = list(mlp.parameters())[0].clone()
        MagnitudePruner(sparsity=0.5).compress(mlp)
        assert torch.allclose(list(mlp.parameters())[0], w0), (
            "MagnitudePruner must not mutate the original model."
        )

    def test_sparsity_achieved(self, mlp):
        pmodel, _ = MagnitudePruner(sparsity=0.5).compress(mlp)
        total = sum(p.numel() for p in pmodel.parameters() if p.requires_grad)
        zeros = sum(
            int((p == 0).sum()) for p in pmodel.parameters() if p.requires_grad
        )
        assert zeros / total >= 0.45, (
            f"Expected ≥45 % zeros, got {zeros / total:.2%}"
        )

    def test_invalid_sparsity_raises(self):
        with pytest.raises(ValueError):
            MagnitudePruner(sparsity=1.0)

    def test_zero_sparsity_leaves_model_intact(self, mlp):
        pmodel, _ = MagnitudePruner(sparsity=0.0).compress(mlp)
        total = sum(p.numel() for p in pmodel.parameters())
        zeros = sum(int((p == 0).sum()) for p in pmodel.parameters())
        assert zeros == 0, "0 % sparsity should leave all weights non-zero."

    def test_cnn_runs_after_pruning(self, cnn):
        pmodel, _ = MagnitudePruner(sparsity=0.3).compress(cnn)
        x = torch.randn(2, 1, 8, 8)
        out = pmodel(x)
        assert out.shape == (2, 2)


# ---------------------------------------------------------------------------
# StructuredPruner tests
# ---------------------------------------------------------------------------

class TestStructuredPruner:
    def test_returns_module_and_metadata(self, mlp):
        pmodel, meta = StructuredPruner(sparsity=0.2).compress(mlp)
        assert isinstance(pmodel, nn.Module)
        assert meta["technique"] == "structured_pruning"

    def test_original_model_unchanged(self, mlp):
        w0 = list(mlp.parameters())[0].clone()
        StructuredPruner(sparsity=0.2).compress(mlp)
        assert torch.allclose(list(mlp.parameters())[0], w0)

    def test_invalid_sparsity_raises(self):
        with pytest.raises(ValueError):
            StructuredPruner(sparsity=-0.1)

    def test_cnn_inference_after_structured_pruning(self, cnn):
        pmodel, _ = StructuredPruner(sparsity=0.25).compress(cnn)
        x = torch.randn(2, 1, 8, 8)
        out = pmodel(x)
        assert out.shape == (2, 2)
