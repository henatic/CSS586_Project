"""Pipeline orchestration for chaining multiple compression techniques.

The :class:`CompressionPipeline` accepts an ordered list of
:class:`~compression.base.BaseCompressor` instances and applies them
sequentially to a model, recording metrics before and after each stage.

Example
-------
>>> from compression import DynamicQuantizer, MagnitudePruner
>>> from pipeline import CompressionPipeline
>>> from evaluation import model_size_mb, count_parameters
>>>
>>> pipeline = CompressionPipeline(
...     stages=[MagnitudePruner(sparsity=0.5), DynamicQuantizer()],
...     eval_fns={"size_mb": model_size_mb, "params": count_parameters},
... )
>>> compressed_model, report = pipeline.run(my_model)
>>> print(report)
"""

from __future__ import annotations

import time
from typing import Any, Callable

import torch.nn as nn

from compression.base import BaseCompressor


class CompressionPipeline:
    """Sequential compression pipeline.

    Parameters
    ----------
    stages:
        Ordered list of :class:`~compression.base.BaseCompressor` instances.
        They are applied left-to-right.
    eval_fns:
        Optional mapping of metric name → callable.  Each callable receives
        a single ``nn.Module`` argument and returns a scalar or dict.
        Metrics are recorded *before* the first stage and *after* each stage.
    """

    def __init__(
        self,
        stages: list[BaseCompressor],
        eval_fns: dict[str, Callable[[nn.Module], Any]] | None = None,
    ) -> None:
        if not stages:
            raise ValueError("Pipeline must contain at least one stage.")
        self.stages = stages
        self.eval_fns = eval_fns or {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, model: nn.Module) -> tuple[nn.Module, dict]:
        """Run all compression stages sequentially on *model*.

        Parameters
        ----------
        model:
            The original (uncompressed) ``nn.Module``.

        Returns
        -------
        compressed_model:
            Model after all stages have been applied.
        report:
            Dictionary with keys:

            * ``"stages"`` – list of per-stage result dicts, each containing:
              - ``"stage_index"``
              - ``"technique"`` (from compressor metadata)
              - ``"config"`` (from compressor metadata)
              - ``"duration_s"`` (wall-clock time in seconds)
              - ``"metrics_before"`` (eval_fns evaluated before this stage)
              - ``"metrics_after"``  (eval_fns evaluated after this stage)
            * ``"total_duration_s"`` – total wall-clock seconds
            * ``"metrics_baseline"`` – metrics on the original model
        """
        report: dict[str, Any] = {
            "stages": [],
            "total_duration_s": 0.0,
            "metrics_baseline": self._evaluate(model),
        }

        current_model = model
        pipeline_start = time.perf_counter()

        for idx, compressor in enumerate(self.stages):
            metrics_before = self._evaluate(current_model)

            stage_start = time.perf_counter()
            current_model, meta = compressor.compress(current_model)
            stage_duration = time.perf_counter() - stage_start

            metrics_after = self._evaluate(current_model)

            report["stages"].append(
                {
                    "stage_index": idx,
                    "technique": meta.get("technique", "unknown"),
                    "config": meta.get("config", {}),
                    "duration_s": stage_duration,
                    "metrics_before": metrics_before,
                    "metrics_after": metrics_after,
                }
            )

        report["total_duration_s"] = time.perf_counter() - pipeline_start
        return current_model, report

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _evaluate(self, model: nn.Module) -> dict[str, Any]:
        """Run all evaluation functions on *model*."""
        if not self.eval_fns:
            return {}
        return {name: fn(model) for name, fn in self.eval_fns.items()}
