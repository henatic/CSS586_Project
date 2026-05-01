"""Tests for CompressionPipeline."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn as nn
import pytest

from compression.pruning import MagnitudePruner, StructuredPruner
from compression.quantization import DynamicQuantizer
from pipeline.pipeline import CompressionPipeline
from evaluation.metrics import count_parameters, model_size_mb, compute_sparsity


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_mlp(in_features: int = 16, num_classes: int = 4) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_features, 32),
        nn.ReLU(),
        nn.Linear(32, num_classes),
    )


EVAL_FNS = {
    "params": count_parameters,
    "size_mb": model_size_mb,
    "sparsity": compute_sparsity,
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCompressionPipeline:
    def test_single_stage_runs(self):
        model = make_mlp()
        pipeline = CompressionPipeline(
            stages=[MagnitudePruner(sparsity=0.3)],
            eval_fns=EVAL_FNS,
        )
        compressed, report = pipeline.run(model)
        assert isinstance(compressed, nn.Module)
        assert "stages" in report
        assert len(report["stages"]) == 1

    def test_two_stage_pipeline(self):
        model = make_mlp()
        pipeline = CompressionPipeline(
            stages=[MagnitudePruner(sparsity=0.3), DynamicQuantizer()],
            eval_fns=EVAL_FNS,
        )
        compressed, report = pipeline.run(model)
        assert len(report["stages"]) == 2
        assert report["stages"][0]["technique"] == "magnitude_pruning"
        assert report["stages"][1]["technique"] == "dynamic_quantization"

    def test_report_has_baseline_metrics(self):
        model = make_mlp()
        pipeline = CompressionPipeline(
            stages=[MagnitudePruner(sparsity=0.5)],
            eval_fns=EVAL_FNS,
        )
        _, report = pipeline.run(model)
        assert "metrics_baseline" in report
        assert "params" in report["metrics_baseline"]

    def test_report_records_duration(self):
        model = make_mlp()
        pipeline = CompressionPipeline(stages=[MagnitudePruner(sparsity=0.4)])
        _, report = pipeline.run(model)
        assert report["total_duration_s"] > 0
        assert report["stages"][0]["duration_s"] > 0

    def test_empty_stages_raises(self):
        with pytest.raises(ValueError):
            CompressionPipeline(stages=[])

    def test_original_model_not_mutated(self):
        model = make_mlp()
        w0 = list(model.parameters())[0].clone()
        pipeline = CompressionPipeline(stages=[MagnitudePruner(sparsity=0.5)])
        pipeline.run(model)
        assert torch.allclose(list(model.parameters())[0], w0), (
            "Pipeline must not mutate the original model."
        )

    def test_metrics_before_and_after_in_stage(self):
        model = make_mlp()
        pipeline = CompressionPipeline(
            stages=[MagnitudePruner(sparsity=0.5)],
            eval_fns={"params": count_parameters},
        )
        _, report = pipeline.run(model)
        stage = report["stages"][0]
        assert "metrics_before" in stage
        assert "metrics_after" in stage

    def test_three_stage_pipeline_structured_then_prune_then_quantize(self):
        model = make_mlp()
        pipeline = CompressionPipeline(
            stages=[
                StructuredPruner(sparsity=0.1),
                MagnitudePruner(sparsity=0.3),
                DynamicQuantizer(),
            ],
            eval_fns={"size_mb": model_size_mb},
        )
        compressed, report = pipeline.run(model)
        assert len(report["stages"]) == 3
        x = torch.randn(4, 16)
        out = compressed(x)
        assert out.shape == (4, 4)
