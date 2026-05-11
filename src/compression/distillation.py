"""Zero-shot knowledge distillation compressor.

Traditional knowledge distillation requires labelled data to transfer soft
logits from a teacher to a student.  The zero-shot variant synthesises
*pseudo-inputs* from the teacher's batch-normalisation (BN) statistics
(mean / variance) and uses those to drive the distillation loss – no real
training samples are needed.

This module implements the approach described in:
    Haroush et al., "The Knowledge Within: Methods for Data-Free Model
    Compression", CVPR 2020.

The :class:`ZeroShotDistiller` performs a lightweight optimisation loop
(gradient-based input synthesis) to produce inputs that activate the teacher
strongly, then trains the student to mimic the teacher's soft outputs on those
synthetic inputs.

Parameters
----------
student:
    The (smaller / compressed) student model to train.
num_synthesis_steps:
    Number of gradient steps used to synthesise each pseudo-input batch.
num_distillation_steps:
    Number of gradient steps used to update the student on synthetic data.
synthesis_lr:
    Learning rate for the pseudo-input synthesis optimisation.
distillation_lr:
    Learning rate for the student optimisation.
batch_size:
    Number of synthetic samples generated per distillation step.
temperature:
    Softmax temperature used when computing the KL-divergence loss.
device:
    Torch device string (e.g. ``"cpu"`` or ``"cuda"``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseCompressor


class ZeroShotDistiller(BaseCompressor):
    """Data-free knowledge distillation via synthetic input generation.

    The teacher model's weights are **frozen** throughout.  The student model
    is optimised to match the teacher's soft predictions on teacher-synthesised
    pseudo-inputs.

    Parameters
    ----------
    student:
        Pre-built student ``nn.Module``.  Can be a smaller architecture or a
        compressed version of the teacher.
    num_synthesis_steps:
        Gradient steps per batch used to craft pseudo-inputs from the teacher's
        BN statistics.
    num_distillation_steps:
        Number of distillation update steps for the student.
    synthesis_lr:
        Learning rate for pseudo-input optimisation.
    distillation_lr:
        Learning rate for student parameter updates.
    batch_size:
        Synthetic batch size.
    temperature:
        Softmax temperature (τ > 1 produces softer distributions).
    input_shape:
        Shape of a single input sample, e.g. ``(3, 32, 32)`` for CIFAR images.
    device:
        Device to run optimisation on.
    """

    def __init__(
        self,
        student: nn.Module,
        num_synthesis_steps: int = 500,
        num_distillation_steps: int = 1000,
        synthesis_lr: float = 0.05,
        distillation_lr: float = 1e-3,
        batch_size: int = 128,
        temperature: float = 3.0,
        input_shape: tuple[int, ...] = (3, 32, 32),
        device: str = "cpu",
    ) -> None:
        self.student = student
        self.num_synthesis_steps = num_synthesis_steps
        self.num_distillation_steps = num_distillation_steps
        self.synthesis_lr = synthesis_lr
        self.distillation_lr = distillation_lr
        self.batch_size = batch_size
        self.temperature = temperature
        self.input_shape = input_shape
        self.device = device

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compress(self, model: nn.Module) -> tuple[nn.Module, dict]:
        """Run zero-shot distillation from *model* (teacher) to student.

        Returns the trained student model and metadata.
        """
        teacher = self._copy_model(model).to(self.device)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False

        student = self.student.to(self.device)
        student.train()

        # Optimizers
        student_optimizer = torch.optim.Adam(student.parameters(), lr=self.distillation_lr)

        # Collect BN statistics from the teacher
        bn_stats = self._collect_bn_stats(teacher)

        for _ in range(self.num_distillation_steps):
            # 1. Synthesize a batch of pseudo-inputs from the teacher
            inputs = self._synthesize_inputs(teacher, bn_stats)

            # 2. Get soft targets from the teacher
            with torch.no_grad():
                teacher_logits = teacher(inputs)

            # 3. Train the student on the synthetic batch
            student_optimizer.zero_grad()
            student_logits = student(inputs)
            loss = self._distillation_loss(student_logits, teacher_logits)
            loss.backward()
            student_optimizer.step()

        metadata = {
            "technique": "zero_shot_distillation",
            "config": {
                "num_synthesis_steps": self.num_synthesis_steps,
                "num_distillation_steps": self.num_distillation_steps,
                "synthesis_lr": self.synthesis_lr,
                "distillation_lr": self.distillation_lr,
                "batch_size": self.batch_size,
                "temperature": self.temperature,
            },
        }
        return student.cpu(), metadata

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _collect_bn_stats(self, teacher: nn.Module) -> list[dict]:
        """Extract running mean and variance from all BN layers."""
        bn_stats = []
        for module in teacher.modules():
            if isinstance(module, nn.BatchNorm2d):
                bn_stats.append({
                    "mean": module.running_mean,
                    "var": module.running_var,
                })
        return bn_stats

    def _synthesize_inputs(self, teacher: nn.Module, bn_stats: list[dict]) -> torch.Tensor:
        """Generate a batch of pseudo-inputs via gradient-based optimization."""
        inputs = torch.randn(
            (self.batch_size, *self.input_shape),
            device=self.device,
            requires_grad=True,
        )
        input_optimizer = torch.optim.Adam([inputs], lr=self.synthesis_lr)

        # This is a list of hooks to be populated
        bn_outputs = []
        hooks = []

        def hook_fn(module, input, output):
            bn_outputs.append(output)

        for module in teacher.modules():
            if isinstance(module, nn.BatchNorm2d):
                hooks.append(module.register_forward_hook(hook_fn))

        for _ in range(self.num_synthesis_steps):
            input_optimizer.zero_grad()
            bn_outputs.clear()
            
            teacher(inputs)  # Forward pass to populate hooks via hook_fn

            # The loss is the sum of distances between current BN stats and target stats
            loss = 0
            for i, bn_output in enumerate(bn_outputs):
                mean = bn_output.mean(dim=[0, 2, 3])
                var = bn_output.var(dim=[0, 2, 3], unbiased=False)
                loss += F.mse_loss(mean, bn_stats[i]["mean"]) + F.mse_loss(var, bn_stats[i]["var"])

            loss.backward()
            input_optimizer.step()

        for hook in hooks:
            hook.remove()

        return inputs.detach()

    def _distillation_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        """Compute KL-divergence between soft student and teacher predictions."""
        soft_teacher = F.log_softmax(teacher_logits / self.temperature, dim=1)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        return F.kl_div(soft_student, soft_teacher, log_target=True, reduction='batchmean') * (self.temperature ** 2)
