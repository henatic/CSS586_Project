"""
GPU support and profiling for compression pipelines.
"""

from __future__ import annotations

import sys
import os
import csv
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from datasets import load_dataset

from compression.pruning import MagnitudePruner
from compression.quantization import DynamicQuantizer
from pipeline.pipeline import CompressionPipeline
from evaluation.metrics import model_size_mb, compute_sparsity, measure_latency

# ---------------------------------------------------------------------------
# GPU Check
# ---------------------------------------------------------------------------

def is_gpu_available():
    """Check if a GPU is available and print its name."""
    if torch.cuda.is_available():
        print(f"GPU found: {torch.cuda.get_device_name(0)}")
        return True
    print("No GPU found, running on CPU.")
    return False

# ---------------------------------------------------------------------------
# Dataset and Model Loading
# ---------------------------------------------------------------------------

def get_glue_mrpc_dataloaders(batch_size: int = 32):
    """Load GLUE MRPC dataset and prepare DataLoaders."""
    raw_datasets = load_dataset("glue", "mrpc")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def tokenize_function(examples):
        return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=True, batch_size=batch_size, collate_fn=data_collator
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], batch_size=batch_size, collate_fn=data_collator
    )
    return train_dataloader, eval_dataloader, tokenizer

def get_bert_model(device):
    """Load a pre-trained BERT model and move it to the specified device."""
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2
    )
    return model.to(device)

# ---------------------------------------------------------------------------
# Evaluation Helper
# ---------------------------------------------------------------------------

def evaluate_accuracy(model, eval_dataloader, device):
    """Evaluate model accuracy on the GLUE MRPC validation set."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == batch["labels"]).sum().item()
            total += batch["labels"].size(0)
    return correct / total

def _latency_fn(model: torch.nn.Module, tokenizer, device) -> dict[str, float]:
    """Measures model latency for a single sample on the specified device."""
    inputs = tokenizer("hello world", return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Use torch.cuda.Event for accurate GPU timing
    if device.type == 'cuda':
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model(**inputs)
        
        # Timing runs
        times = []
        with torch.no_grad():
            for _ in range(20):
                start_event.record()
                _ = model(**inputs)
                end_event.record()
                torch.cuda.synchronize()
                times.append(start_event.elapsed_time(end_event))
    else:
        import time
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model(**inputs)
        
        # Timing runs
        times = []
        with torch.no_grad():
            for _ in range(20):
                start = time.time()
                _ = model(**inputs)
                times.append((time.time() - start) * 1000)

    import statistics
    return {
        "mean_ms": statistics.mean(times),
        "std_ms": statistics.stdev(times) if len(times) > 1 else 0.0,
    }

# ---------------------------------------------------------------------------
# Main Experiment
# ---------------------------------------------------------------------------

def main():
    """Run the GLUE MRPC compression experiment with GPU support."""
    use_gpu = is_gpu_available()
    device = torch.device("cuda" if use_gpu else "cpu")

    train_loader, eval_loader, tokenizer = get_glue_mrpc_dataloaders()

    eval_fns = {
        "size_mb": model_size_mb,
        "sparsity": compute_sparsity,
        "accuracy": lambda m: evaluate_accuracy(m, eval_loader, device),
        "latency": lambda m: _latency_fn(m, tokenizer, device),
    }

    pipeline_configs = [
        ("Baseline", []),
        ("Dynamic Quantization", [DynamicQuantizer()]),
        ("Magnitude Pruning (50%)", [MagnitudePruner(sparsity=0.5)]),
        ("Prune -> Quantize", [MagnitudePruner(sparsity=0.5), DynamicQuantizer()]),
    ]

    output_path = "results/gpu_profiling_results.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Pipeline", "Device", "Size (MB)", "Sparsity (%)", "Accuracy", "Latency (ms)"])

        for name, stages in pipeline_configs:
            print(f"Running pipeline: {name} on {device}...")
            model = get_bert_model(device)

            if not stages:
                sparsity_result = compute_sparsity(model)
                sparsity_value = sparsity_result.get("global_sparsity", 0.0) if isinstance(sparsity_result, dict) else sparsity_result
                
                baseline_metrics = {
                    "size_mb": model_size_mb(model),
                    "sparsity": sparsity_value,
                    "accuracy": evaluate_accuracy(model, eval_loader, device),
                    "latency": _latency_fn(model, tokenizer, device)["mean_ms"],
                }
                writer.writerow([
                    name,
                    str(device),
                    f"{baseline_metrics['size_mb']:.2f}",
                    f"{baseline_metrics['sparsity'] * 100:.2f}",
                    f"{baseline_metrics['accuracy']:.4f}",
                    f"{baseline_metrics['latency']:.2f}",
                ])
                continue

            pipeline = CompressionPipeline(stages=stages, eval_fns=eval_fns)
            try:
                compressed_model, report = pipeline.run(model)
                final_metrics = report["stages"][-1]["metrics_after"]
                writer.writerow([
                    name,
                    str(device),
                    f"{final_metrics['size_mb']:.2f}",
                    f"{final_metrics['sparsity'] * 100:.2f}",
                    f"{final_metrics['accuracy']:.4f}",
                    f"{final_metrics['latency']['mean_ms']:.2f}",
                ])
            except Exception as e:
                print(f"Skipping pipeline '{name}' due to error: {e}")

    print(f"GPU profiling complete. Results saved to {output_path}")

if __name__ == "__main__":
    main()
