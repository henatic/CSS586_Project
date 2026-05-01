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
        """Distil knowledge from *model* (teacher) into ``self.student``.

        Parameters
        ----------
        model:
            The teacher model.  Its weights are not modified.

        Returns
        -------
        student:
            The distilled student model (on ``self.device``).
        metadata:
            Technique metadata dictionary.
        """
        teacher = model.to(self.device)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad_(False)

        student = self.student.to(self.device)
        student.train()
        student_optimizer = torch.optim.Adam(
            student.parameters(), lr=self.distillation_lr
        )

        for step in range(self.num_distillation_steps):
            # 1. Synthesise a pseudo-input batch guided by teacher activations.
            pseudo_inputs = self._synthesise_inputs(teacher)

            # 2. Compute teacher soft targets.
            with torch.no_grad():
                teacher_logits = teacher(pseudo_inputs)
                soft_targets = F.softmax(
                    teacher_logits / self.temperature, dim=-1
                )

            # 3. Update student to match soft targets.
            student_logits = student(pseudo_inputs)
            student_log_probs = F.log_softmax(
                student_logits / self.temperature, dim=-1
            )
            loss = F.kl_div(
                student_log_probs,
                soft_targets,
                reduction="batchmean",
            ) * (self.temperature ** 2)

            student_optimizer.zero_grad()
            loss.backward()
            student_optimizer.step()

        student.eval()
        metadata = {
            "technique": "zero_shot_distillation",
            "config": {
                "num_synthesis_steps": self.num_synthesis_steps,
                "num_distillation_steps": self.num_distillation_steps,
                "synthesis_lr": self.synthesis_lr,
                "distillation_lr": self.distillation_lr,
                "batch_size": self.batch_size,
                "temperature": self.temperature,
                "input_shape": list(self.input_shape),
                "device": self.device,
            },
        }
        return student, metadata

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _synthesise_inputs(self, teacher: nn.Module) -> torch.Tensor:
        """Generate a batch of pseudo-inputs that maximise teacher BN alignment.

        The inputs are initialised from a standard normal distribution and
        iteratively updated via gradient ascent to minimise the discrepancy
        between their running statistics and the teacher's stored BN mean /
        variance.
        """
        pseudo = torch.randn(
            self.batch_size, *self.input_shape,
            device=self.device,
            requires_grad=True,
        )
        optimizer = torch.optim.Adam([pseudo], lr=self.synthesis_lr)

        for _ in range(self.num_synthesis_steps):
            optimizer.zero_grad()
            teacher(pseudo)  # side-effect: BN hooks fire
            loss = self._bn_alignment_loss(teacher, pseudo)
            loss.backward()
            optimizer.step()

        return pseudo.detach()

    @staticmethod
    def _bn_alignment_loss(
        teacher: nn.Module, inputs: torch.Tensor
    ) -> torch.Tensor:
        """Compute a loss that encourages *inputs* to match BN statistics.

        For each BatchNorm layer we measure the discrepancy between:
        - the running mean/var stored in the teacher, and
        - the empirical mean/var of the current mini-batch activations.
        """
        losses: list[torch.Tensor] = []

        def _hook(module: nn.Module, _inp, output: torch.Tensor) -> None:
            if not isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                return
            # Compute mean and variance over all dimensions except the channel dim.
            channel_dim = 1
            reduce_dims = tuple(d for d in range(output.dim()) if d != channel_dim)
            batch_mean = output.mean(dim=reduce_dims)
            batch_var = output.var(dim=reduce_dims, unbiased=False)
            # BN running statistics.
            running_mean = module.running_mean.detach()  # type: ignore[attr-defined]
            running_var = module.running_var.detach()  # type: ignore[attr-defined]
            losses.append(
                (batch_mean - running_mean).pow(2).mean()
                + (batch_var - running_var).pow(2).mean()
            )

        handles = [
            m.register_forward_hook(_hook)
            for m in teacher.modules()
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
        ]
        try:
            teacher(inputs)
        finally:
            for h in handles:
                h.remove()

        if losses:
            return torch.stack(losses).sum()
        # Fallback when no BN layers: minimise negative mean activation.
        with torch.enable_grad():
            out = teacher(inputs)
        return -out.mean()
