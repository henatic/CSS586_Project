"""Base class for all compression techniques."""

import abc
import copy
import torch.nn as nn


class BaseCompressor(abc.ABC):
    """Abstract interface that every compression technique must implement.

    Each subclass applies a zero-shot (or near zero-shot) transformation to
    a PyTorch ``nn.Module`` and returns the compressed model together with a
    dictionary of metadata describing what was done.
    """

    @abc.abstractmethod
    def compress(self, model: nn.Module) -> tuple[nn.Module, dict]:
        """Apply the compression technique to *model*.

        Parameters
        ----------
        model:
            The PyTorch model to compress.  The original model should **not**
            be mutated; implementations must work on a deep copy.

        Returns
        -------
        compressed_model:
            The compressed ``nn.Module``.
        metadata:
            A plain dictionary with at least the key ``"technique"`` set to a
            human-readable string and ``"config"`` set to the compressor's
            hyper-parameter dict.
        """

    def _copy_model(self, model: nn.Module) -> nn.Module:
        """Return a deep copy of *model* so the original is never mutated."""
        return copy.deepcopy(model)

    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.__class__.__name__}()"
