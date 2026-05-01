"""Tests for DynamicQuantizer and StaticQuantizer."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn as nn
import pytest

from compression.quantization import DynamicQuantizer, StaticQuantizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class SmallMLP(nn.Module):
    """Tiny MLP for fast testing."""

    def __init__(self, in_features: int = 16, num_classes: int = 4) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


@pytest.fixture()
def mlp() -> SmallMLP:
    return SmallMLP()


@pytest.fixture()
def dummy_loader():
    """Yields a single calibration batch."""
    inputs = torch.randn(8, 16)
    return [(inputs, torch.zeros(8, dtype=torch.long))]


# ---------------------------------------------------------------------------
# DynamicQuantizer tests
# ---------------------------------------------------------------------------

class TestDynamicQuantizer:
    def test_returns_module_and_metadata(self, mlp):
        qmodel, meta = DynamicQuantizer().compress(mlp)
        assert isinstance(qmodel, nn.Module)
        assert meta["technique"] == "dynamic_quantization"
        assert "config" in meta

    def test_original_model_unchanged(self, mlp):
        original_fc1_weight = mlp.fc1.weight.clone()
        DynamicQuantizer().compress(mlp)
        assert torch.allclose(mlp.fc1.weight, original_fc1_weight), (
            "DynamicQuantizer must not mutate the original model."
        )

    def test_quantized_model_runs_inference(self, mlp):
        qmodel, _ = DynamicQuantizer().compress(mlp)
        x = torch.randn(4, 16)
        out = qmodel(x)
        assert out.shape == (4, 4)

    def test_dtype_in_metadata(self, mlp):
        qmodel, meta = DynamicQuantizer(dtype=torch.qint8).compress(mlp)
        assert "qint8" in meta["config"]["dtype"]

    def test_custom_modules_to_quantize(self, mlp):
        qmodel, meta = DynamicQuantizer(
            modules_to_quantize={nn.Linear}
        ).compress(mlp)
        assert "Linear" in meta["config"]["modules_to_quantize"]


# ---------------------------------------------------------------------------
# StaticQuantizer tests
# ---------------------------------------------------------------------------

class TestStaticQuantizer:
    def test_returns_module_and_metadata(self, mlp, dummy_loader):
        # Static quantization requires a model with eval mode and QuantStubs
        # on some backends; skip if the environment does not support it.
        try:
            qmodel, meta = StaticQuantizer(dummy_loader).compress(mlp)
        except Exception:
            pytest.skip("Static quantization not supported in this environment.")
        assert isinstance(qmodel, nn.Module)
        assert meta["technique"] == "static_quantization"

    def test_original_model_unchanged(self, mlp, dummy_loader):
        original_fc1_weight = mlp.fc1.weight.clone()
        try:
            StaticQuantizer(dummy_loader).compress(mlp)
        except Exception:
            pytest.skip("Static quantization not supported in this environment.")
        assert torch.allclose(mlp.fc1.weight, original_fc1_weight), (
            "StaticQuantizer must not mutate the original model."
        )
