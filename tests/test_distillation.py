"""Tests for ZeroShotDistiller."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn as nn
import pytest

from compression.distillation import ZeroShotDistiller


# ---------------------------------------------------------------------------
# Tiny models for fast iteration
# ---------------------------------------------------------------------------

class TinyTeacher(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 4)
        self.bn = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.fc(x))


class TinyStudent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestZeroShotDistiller:
    def _make_distiller(self, student: nn.Module) -> ZeroShotDistiller:
        return ZeroShotDistiller(
            student=student,
            num_synthesis_steps=2,
            num_distillation_steps=3,
            batch_size=4,
            input_shape=(8,),
            device="cpu",
        )

    def test_returns_module_and_metadata(self):
        teacher = TinyTeacher()
        student = TinyStudent()
        distiller = self._make_distiller(student)
        result_model, meta = distiller.compress(teacher)
        assert isinstance(result_model, nn.Module)
        assert meta["technique"] == "zero_shot_distillation"
        assert "config" in meta

    def test_teacher_weights_unchanged(self):
        teacher = TinyTeacher()
        w0 = teacher.fc.weight.clone()
        student = TinyStudent()
        distiller = self._make_distiller(student)
        distiller.compress(teacher)
        assert torch.allclose(teacher.fc.weight, w0), (
            "Teacher weights must not be modified during distillation."
        )

    def test_student_is_returned_in_eval_mode(self):
        teacher = TinyTeacher()
        student = TinyStudent()
        distiller = self._make_distiller(student)
        result_model, _ = distiller.compress(teacher)
        assert not result_model.training, "Student should be in eval mode after distillation."

    def test_student_can_run_inference(self):
        teacher = TinyTeacher()
        student = TinyStudent()
        distiller = self._make_distiller(student)
        result_model, _ = distiller.compress(teacher)
        x = torch.randn(2, 8)
        out = result_model(x)
        assert out.shape == (2, 4)

    def test_metadata_config_keys(self):
        teacher = TinyTeacher()
        student = TinyStudent()
        distiller = self._make_distiller(student)
        _, meta = distiller.compress(teacher)
        cfg = meta["config"]
        for key in [
            "num_synthesis_steps",
            "num_distillation_steps",
            "synthesis_lr",
            "distillation_lr",
            "batch_size",
            "temperature",
            "input_shape",
            "device",
        ]:
            assert key in cfg, f"Missing config key: {key}"


# ---------------------------------------------------------------------------
# Unit tests for the ZeroShotDistiller.
# ---------------------------------------------------------------------------

class TeacherModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16 * 32 * 32, 10)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.flatten(x)
        return self.fc(x)

class StudentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(8 * 32 * 32, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.flatten(x)
        return self.fc(x)

@pytest.fixture
def teacher_model():
    return TeacherModel()

@pytest.fixture
def student_model():
    return StudentModel()

def test_zeroshotdistiller_initialization():
    """Test that ZeroShotDistiller initializes correctly."""
    distiller = ZeroShotDistiller(
        teacher_model=TeacherModel(),
        student_model=StudentModel(),
        input_shape=(1, 3, 32, 32),
        num_batches=1,
        num_epochs=1,
    )
    assert isinstance(distiller, ZeroShotDistiller)
    assert distiller.num_epochs == 1

def test_zeroshotdistiller_compress_model(teacher_model, student_model):
    """Test that the model can be compressed without errors."""
    distiller = ZeroShotDistiller(
        teacher_model=teacher_model,
        student_model=student_model,
        input_shape=(1, 3, 32, 32),
        num_batches=1,
        num_epochs=1,
    )
    # The distiller returns the student model, so we pass it in as a dummy
    compressed_model, _ = distiller.compress(student_model)
    assert compressed_model is not None
    assert isinstance(compressed_model, nn.Module)
    # Check if student model's parameters have been updated (they should have changed)
    initial_params = [p.clone() for p in student_model.parameters()]
    distiller.compress(student_model)
    final_params = list(student_model.parameters())

    params_changed = any(not torch.equal(initial, final) for initial, final in zip(initial_params, final_params))
    assert params_changed, "Student model parameters did not change after distillation."

def test_generate_synthetic_data(teacher_model):
    """Test the synthetic data generation."""
    distiller = ZeroShotDistiller(
        teacher_model=teacher_model,
        student_model=StudentModel(),
        input_shape=(1, 3, 32, 32),
        num_batches=1,
        num_epochs=1,
    )
    synthetic_data, _ = distiller._generate_synthetic_data(batch_size=16)
    assert synthetic_data.shape == (16, 3, 32, 32)

def test_distillation_without_bn_layers(student_model):
    """Test that distillation raises an error if teacher has no BN layers."""
    # A teacher model without any BatchNorm layers
    teacher_without_bn = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(16 * 32 * 32, 10)
    )
    with pytest.raises(ValueError, match="Teacher model must have at least one BatchNorm2d layer"):
        ZeroShotDistiller(
            teacher_model=teacher_without_bn,
            student_model=student_model,
            input_shape=(1, 3, 32, 32),
        )
