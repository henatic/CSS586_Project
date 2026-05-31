# CSS586 Project – Optimizing Zero-Shot Model Compression Pipelines

**CSS 586: Deep Learning**

The goal of this project is to identify the best _combination and ordering_ of
model-compression techniques that maximise inference efficiency while
requiring **zero (or near-zero) training data**.

---

## Problem Statement

Modern deep-learning models are large and slow to deploy on edge hardware.
_Model compression_ (quantization, pruning, knowledge distillation, …) can
shrink these models dramatically, but each technique makes different
trade-offs and their interactions are not well understood.

This project systematically explores:

1. Which individual techniques give the largest size / latency reduction.
2. How different _orderings_ of techniques in a pipeline affect the
   final compressed model.
3. Whether a zero-shot (no-training-data) pipeline can match or approach
   the quality of fine-tuning-based compression.

---

## Requirements

| Dependency     | Version  | Purpose                        |
| -------------- | -------- | ------------------------------ |
| `torch`        | ≥ 2.0.0  | Core deep learning framework   |
| `torchvision`  | ≥ 0.15.0 | Pre-trained model zoo          |
| `transformers` | ≥ 4.30.0 | HuggingFace transformer models |
| `numpy`        | ≥ 1.24.0 | Numerical utilities            |
| `scipy`        | ≥ 1.10.0 | Statistical analysis           |
| `tqdm`         | ≥ 4.65.0 | Progress bars                  |
| `pytest`       | ≥ 7.3.0  | Unit testing                   |
| `pytest-cov`   | ≥ 4.1.0  | Test coverage                  |
| `flake8`       | ≥ 6.0.0  | Linting                        |

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## Project Layout

```
CSS586_Project/
├── requirements.txt          # Python dependencies
├── setup.py                  # Package installation
├── src/
│   ├── compression/
│   │   ├── base.py           # Abstract BaseCompressor interface
│   │   ├── quantization.py   # DynamicQuantizer, StaticQuantizer
│   │   ├── pruning.py        # MagnitudePruner, StructuredPruner
│   │   └── distillation.py   # ZeroShotDistiller
│   ├── pipeline/
│   │   └── pipeline.py       # CompressionPipeline orchestrator
│   └── evaluation/
│       └── metrics.py        # count_parameters, model_size_mb,
│                             #   measure_latency, compute_sparsity
├── tests/
│   ├── test_quantization.py
│   ├── test_pruning.py
│   ├── test_distillation.py
│   ├── test_pipeline.py
│   └── test_metrics.py
└── experiments/
    └── run_experiment.py     # Pipeline comparison experiment
```

---

## Compression Techniques

### 1 · Post-Training Quantization (`compression/quantization.py`)

Reduces weight precision from FP32 to INT8 without any gradient updates.

| Class              | Data needed                       | Notes                                                   |
| ------------------ | --------------------------------- | ------------------------------------------------------- |
| `DynamicQuantizer` | None (zero-shot)                  | Weights quantized at save-time; activations at run-time |
| `StaticQuantizer`  | Small calibration set (no labels) | Both weights and activations quantized                  |

### 2 · Weight Pruning (`compression/pruning.py`)

Zeros out weights whose contribution is lowest; zero-shot.

| Class              | Strategy        | Notes                                                 |
| ------------------ | --------------- | ----------------------------------------------------- |
| `MagnitudePruner`  | Unstructured L1 | Global magnitude threshold across all targeted layers |
| `StructuredPruner` | Structured L2   | Removes entire output channels / neurons              |

### 3 · Zero-Shot Knowledge Distillation (`compression/distillation.py`)

Synthesises pseudo-inputs from the teacher model's batch-normalisation
statistics and trains a student model to mimic the teacher's soft logits.
No real data required.

---

## Pipeline Orchestration (`pipeline/pipeline.py`)

The `CompressionPipeline` chains an ordered list of compressors and records
metrics at every stage:

```python
from compression import MagnitudePruner, DynamicQuantizer
from pipeline import CompressionPipeline
from evaluation import model_size_mb, count_parameters

pipeline = CompressionPipeline(
    stages=[MagnitudePruner(sparsity=0.5), DynamicQuantizer()],
    eval_fns={"size_mb": model_size_mb, "params": count_parameters},
)
compressed_model, report = pipeline.run(my_model)
```

`report` contains per-stage metrics (before/after), technique metadata, and
total wall-clock time, making it easy to compare pipeline orderings.

---

## Running the Experiment

```bash
python experiments/run_experiment.py
```

Example output:

```
========================================================================
 Zero-Shot Model Compression Pipeline Comparison
========================================================================
Configuration                                 Size(MB)   Sparsity   Latency(ms)  Compress(s)
------------------------------------------------------------------------
Baseline (no compression)                        0.793      0.00%    ...
Dynamic Quantization only                        0.212      0.00%    ...
Magnitude Pruning (50 %) only                    0.793     50.00%    ...
Prune → Quantize                                 0.212     50.00%    ...
Quantize → Prune                                 0.212      0.00%    ...
Structured Prune → Magnitude Prune → Quantize    0.212     40.00%    ...
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Development Status

- [x] **Core Infrastructure**
  - [x] Abstract `BaseCompressor` interface.
  - [x] `CompressionPipeline` for chaining techniques.
  - [x] Evaluation metrics for size, sparsity, and latency.
- [x] **Compression Techniques Implemented**
  - [x] `DynamicQuantizer` (post-training dynamic quantization).
  - [x] `StaticQuantizer` (post-training static quantization).
  - [x] `MagnitudePruner` (unstructured weight pruning).
  - [x] `StructuredPruner` (structured weight pruning).
  - [x] `ZeroShotDistiller` (data-free knowledge distillation).
- [x] **Initial Experiments**
  - [x] `run_experiment.py` script to compare pipeline orderings.
  - [x] Successfully ran comparisons for pruning and dynamic quantization.
  - [x] Expanded `run_experiment.py` to include `StaticQuantizer` and `ZeroShotDistiller`.
- [ ] **Pending Tasks**
  - [ ] Write comprehensive unit tests for all compression techniques in `tests/`.
  - [ ] Apply the compression pipeline to a real-world pre-trained model (e.g., from `torchvision` or `transformers`).
  - [ ] Analyze the impact of compression on a downstream task's accuracy (requires a validation dataset).

---

## Development Plan

- [x] Define requirements and project structure
- [x] Implement `DynamicQuantizer` (zero-shot PTQ)
- [x] Implement `StaticQuantizer` (calibration-based PTQ)
- [x] Implement `MagnitudePruner` (unstructured pruning)
- [x] Implement `StructuredPruner` (channel/neuron pruning)
- [x] Implement `ZeroShotDistiller` (data-free KD)
- [x] Implement `CompressionPipeline` orchestrator
- [x] Implement evaluation metrics (size, latency, sparsity, param count)
- [x] Write unit tests for all modules
- [x] Example experiment comparing pipeline orderings
- [x] Integrate pre-trained vision / NLP models (ResNet, BERT)
- [ ] Add accuracy evaluation on standard benchmarks (CIFAR-10, GLUE)
- [ ] Automated pipeline search (grid search / Bayesian optimisation)
- [ ] GPU support and profiling
- [ ] Final report and analysis

---

## License

MIT – see [LICENSE](LICENSE).
