"""Post-training quantization compressors.

Two strategies are provided:

* :class:`DynamicQuantizer` – weights are quantized to INT8 at save-time; 
  activations are quantized dynamically at run-time.  No calibration data
  required (zero-shot).

* :class:`StaticQuantizer` – both weights *and* activations are quantized
  to INT8.  A small calibration dataset is required to gather activation
  statistics, but **no gradient updates** are performed (still zero-shot in
  the training sense).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.quantization as tq

from .base import BaseCompressor


# Modules that support dynamic quantization out-of-the-box.
_DYNAMIC_QMODULES = {nn.Linear, nn.LSTM, nn.LSTMCell, nn.GRU, nn.GRUCell}


class DynamicQuantizer(BaseCompressor):
    """Apply dynamic (weight-only) INT8 quantization to linear layers.

    No calibration data is needed – this is fully zero-shot.

    Parameters
    ----------
    dtype:
        Target quantization dtype.  Defaults to ``torch.qint8``.
    modules_to_quantize:
        Set of module types to quantize.  Defaults to all supported linear
        and recurrent layer types.
    """

    def __init__(
        self,
        dtype: torch.dtype = torch.qint8,
        modules_to_quantize: set[type[nn.Module]] | None = None,
    ) -> None:
        self.dtype = dtype
        self.modules_to_quantize = modules_to_quantize or _DYNAMIC_QMODULES

    def compress(self, model: nn.Module) -> tuple[nn.Module, dict]:
        """Quantize *model* dynamically and return (quantized_model, metadata)."""
        model_copy = self._copy_model(model)
        quantized = tq.quantize_dynamic(
            model_copy,
            qconfig_spec=self.modules_to_quantize,
            dtype=self.dtype,
        )
        metadata = {
            "technique": "dynamic_quantization",
            "config": {
                "dtype": str(self.dtype),
                "modules_to_quantize": [m.__name__ for m in self.modules_to_quantize],
            },
        }
        return quantized, metadata


class StaticQuantizer(BaseCompressor):
    """Apply static (weight + activation) INT8 post-training quantization.

    A calibration dataloader is required to collect activation statistics.
    No backward pass / gradient update is performed (zero-shot w.r.t. training).

    Parameters
    ----------
    calibration_loader:
        An iterable of ``(inputs, ...)`` batches.  Only ``inputs`` is fed to
        the model during calibration.
    backend:
        PyTorch quantization backend.  ``"fbgemm"`` for x86; ``"qnnpack"``
        for ARM / mobile.
    num_calibration_batches:
        Maximum number of batches to use for calibration.
    """

    def __init__(
        self,
        calibration_loader,
        backend: str = "fbgemm",
        num_calibration_batches: int = 32,
    ) -> None:
        self.calibration_loader = calibration_loader
        self.backend = backend
        self.num_calibration_batches = num_calibration_batches

    def compress(self, model: nn.Module) -> tuple[nn.Module, dict]:
        """Calibrate and statically quantize *model*."""
        model_copy = self._copy_model(model)
        model_copy.eval()

        model_copy.qconfig = tq.get_default_qconfig(self.backend)
        tq.prepare(model_copy, inplace=True)

        # Calibration – no gradients needed.
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.calibration_loader):
                if batch_idx >= self.num_calibration_batches:
                    break
                inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
                model_copy(inputs)

        tq.convert(model_copy, inplace=True)

        metadata = {
            "technique": "static_quantization",
            "config": {
                "backend": self.backend,
                "num_calibration_batches": self.num_calibration_batches,
            },
        }
        return model_copy, metadata
