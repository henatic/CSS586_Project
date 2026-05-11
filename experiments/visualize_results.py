"""Parse the experiment results and generate visualizations."""

import pandas as pd
import matplotlib.pyplot as plt
import re
import os


def parse_results(file_path: str) -> pd.DataFrame:
    """Parse the results.txt file into a pandas DataFrame."""
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    data = []
    # Regex to capture the pipeline name and the three metric values
    pattern = re.compile(
        r"^(.*?)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*%\s*\|\s*([\d.]+)\s*ms"
    )

    for line in lines:
        match = pattern.match(line.strip())
        if match:
            pipeline = match.group(1).strip()
            size = float(match.group(2))
            sparsity = float(match.group(3))
            latency = float(match.group(4))
            data.append([pipeline, size, sparsity, latency])

    df = pd.DataFrame(
        data, columns=["Pipeline", "Size (MB)", "Sparsity (%)", "Latency (ms)"]
    )
    return df


def plot_results(df: pd.DataFrame, output_dir: str) -> None:
    """Generate and save bar plots for the results."""
    os.makedirs(output_dir, exist_ok=True)

    # --- Plot 1: Model Size ---
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 8))
    pipelines = df["Pipeline"]
    size = df["Size (MB)"]

    bars = ax.barh(pipelines, size, color="skyblue")
    ax.set_xlabel("Model Size (MB)")
    ax.set_title("Model Size Comparison Across Compression Pipelines")
    ax.invert_yaxis()  # To match the order in the text file
    ax.bar_label(bars, fmt="%.2f MB", padding=3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_size_comparison.png"))
    plt.close(fig)

    # --- Plot 2: Latency ---
    fig, ax = plt.subplots(figsize=(12, 8))
    latency = df["Latency (ms)"]

    bars = ax.barh(pipelines, latency, color="salmon")
    ax.set_xlabel("Latency (ms)")
    ax.set_title("Inference Latency Comparison")
    ax.invert_yaxis()
    ax.bar_label(bars, fmt="%.2f ms", padding=3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "latency_comparison.png"))
    plt.close(fig)

    print(f"Visualizations saved to '{output_dir}'")


def main():
    """Main function to run the script."""
    results_file = "results/experiment_results.txt"
    output_dir = "results"
    
    df = parse_results(results_file)
    if not df.empty:
        plot_results(df, output_dir)
    else:
        print("No data parsed from results file. Cannot generate plots.")


if __name__ == "__main__":
    main()
