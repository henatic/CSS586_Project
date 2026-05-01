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
