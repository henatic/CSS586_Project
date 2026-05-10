"""Weight pruning compressors.

Two strategies are provided:

* :class:`MagnitudePruner` – prunes individual weights whose absolute value
  falls below a sparsity-derived threshold (unstructured pruning).  Zero-shot:
  no training data required.

* :class:`StructuredPruner` – prunes entire output channels / neurons in
  ``Linear`` and ``Conv2d`` layers ranked by their L2-norm.  Zero-shot.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from .base import BaseCompressor


class MagnitudePruner(BaseCompressor):
    """Unstructured magnitude-based weight pruning.

    Zeros out the fraction *sparsity* of weights with the smallest absolute
    value, across all targeted layer types.

    Parameters
    ----------
    sparsity:
        Fraction of weights to prune, in [0, 1).  E.g. ``0.5`` removes the
        lowest 50 % of weights by magnitude.
    layer_types:
        Tuple of ``nn.Module`` subclasses to prune.  Defaults to
        ``(nn.Linear, nn.Conv2d)``.
    make_permanent:
        If ``True`` the pruning mask is applied permanently (the mask buffer
        is removed).  Defaults to ``True``.
    """

    def __init__(
        self,
        sparsity: float = 0.5,
        layer_types: tuple[type[nn.Module], ...] = (nn.Linear, nn.Conv2d),
        make_permanent: bool = True,
    ) -> None:
        if not 0.0 <= sparsity < 1.0:
            raise ValueError(f"sparsity must be in [0, 1); got {sparsity}")
        self.sparsity = sparsity
        self.layer_types = layer_types
        self.make_permanent = make_permanent

    def compress(self, model: nn.Module) -> tuple[nn.Module, dict]:
        """Prune *model* by magnitude and return (pruned_model, metadata)."""
        model_copy = self._copy_model(model)

        parameters_to_prune = [
            (module, "weight")
            for module in model_copy.modules()
            if isinstance(module, self.layer_types)
        ]

        if parameters_to_prune:
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=self.sparsity,
            )
            if self.make_permanent:
                for module, param_name in parameters_to_prune:
                    prune.remove(module, param_name)

        metadata = {
            "technique": "magnitude_pruning",
            "config": {
                "sparsity": self.sparsity,
                "layer_types": [t.__name__ for t in self.layer_types],
                "make_permanent": self.make_permanent,
            },
        }
        return model_copy, metadata


class StructuredPruner(BaseCompressor):
    """Structured pruning: removes output channels/neurons with lowest L2 norm.

    For ``nn.Linear`` layers the output neurons (rows of the weight matrix)
    are ranked; for ``nn.Conv2d`` the output channels are ranked.

    Parameters
    ----------
    sparsity:
        Fraction of channels / neurons to remove, in [0, 1).
    layer_types:
        Layer types to prune.  Defaults to ``(nn.Linear, nn.Conv2d)``.
    """

    def __init__(
        self,
        sparsity: float = 0.2,
        layer_types: tuple[type[nn.Module], ...] = (nn.Linear, nn.Conv2d),
    ) -> None:
        if not 0.0 <= sparsity < 1.0:
            raise ValueError(f"sparsity must be in [0, 1); got {sparsity}")
        self.sparsity = sparsity
        self.layer_types = layer_types

    def compress(self, model: nn.Module) -> tuple[nn.Module, dict]:
        """Prune *model* with structured L2-norm pruning."""
        model_copy = self._copy_model(model)

        for module in model_copy.modules():
            if isinstance(module, self.layer_types):
                prune.ln_structured(
                    module,
                    name="weight",
                    amount=self.sparsity,
                    n=2,  # L2 norm
                    dim=0,  # Prune output channels/neurons
                )
                prune.remove(module, "weight")

        metadata = {
            "technique": "structured_pruning",
            "config": {
                "sparsity": self.sparsity,
                "layer_types": [t.__name__ for t in self.layer_types],
            },
        }
        return model_copy, metadata
