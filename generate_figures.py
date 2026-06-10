"""
Generate visualizations from the experiment results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid")

# Output directory for figures
output_dir = "/home/NETID/hdmorgan/CSS586/acmart-primary"
os.makedirs(output_dir, exist_ok=True)

# Read the result files
real_world_results = "/home/NETID/hdmorgan/CSS586/results/real_world_experiment_results1.txt"
glue_results = "/home/NETID/hdmorgan/CSS586/results/glue_mrpc_results.csv"
gpu_results = "/home/NETID/hdmorgan/CSS586/results/gpu_profiling_results.csv"

# ===== REAL-WORLD EXPERIMENT (ResNet-18 on CIFAR-10) =====
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('ResNet-18 Compression Results on CIFAR-10', fontsize=16, fontweight='bold')

# Parse real_world_experiment_results
with open(real_world_results, 'r') as f:
    lines = f.readlines()

pipelines_resnet = []
sizes_resnet = []
sparsities_resnet = []
accuracies_resnet = []
latencies_resnet = []

for line in lines:
    if '|' in line and 'Pipeline' not in line and '=' not in line and '-' not in line:
        parts = [p.strip() for p in line.split('|')]
        if len(parts) >= 5:
            try:
                pipeline_name = parts[0]
                size = float(parts[1])
                sparsity = float(parts[2].replace('%', ''))
                accuracy = float(parts[3])
                latency = float(parts[4].replace('ms', ''))
                
                pipelines_resnet.append(pipeline_name)
                sizes_resnet.append(size)
                sparsities_resnet.append(sparsity)
                accuracies_resnet.append(accuracy)
                latencies_resnet.append(latency)
            except (ValueError, IndexError):
                pass

# Plot 1: Model Size
axes[0, 0].bar(range(len(pipelines_resnet)), sizes_resnet, color='steelblue')
axes[0, 0].set_xticks(range(len(pipelines_resnet)))
axes[0, 0].set_xticklabels(pipelines_resnet, rotation=45, ha='right')
axes[0, 0].set_ylabel('Model Size (MB)', fontweight='bold')
axes[0, 0].set_title('Model Size Comparison')
axes[0, 0].grid(axis='y', alpha=0.3)

# Plot 2: Sparsity
axes[0, 1].bar(range(len(pipelines_resnet)), sparsities_resnet, color='coral')
axes[0, 1].set_xticks(range(len(pipelines_resnet)))
axes[0, 1].set_xticklabels(pipelines_resnet, rotation=45, ha='right')
axes[0, 1].set_ylabel('Sparsity (%)', fontweight='bold')
axes[0, 1].set_title('Sparsity Comparison')
axes[0, 1].grid(axis='y', alpha=0.3)

# Plot 3: Accuracy
axes[1, 0].bar(range(len(pipelines_resnet)), accuracies_resnet, color='mediumseagreen')
axes[1, 0].set_xticks(range(len(pipelines_resnet)))
axes[1, 0].set_xticklabels(pipelines_resnet, rotation=45, ha='right')
axes[1, 0].set_ylabel('Accuracy', fontweight='bold')
axes[1, 0].set_title('Accuracy Comparison')
axes[1, 0].set_ylim([0.88, 1.0])
axes[1, 0].grid(axis='y', alpha=0.3)

# Plot 4: Latency
axes[1, 1].bar(range(len(pipelines_resnet)), latencies_resnet, color='mediumpurple')
axes[1, 1].set_xticks(range(len(pipelines_resnet)))
axes[1, 1].set_xticklabels(pipelines_resnet, rotation=45, ha='right')
axes[1, 1].set_ylabel('Latency (ms)', fontweight='bold')
axes[1, 1].set_title('Latency Comparison')
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'resnet_results.png'), dpi=300, bbox_inches='tight')
print("Saved resnet_results.png")
plt.close()

# ===== GLUE EXPERIMENT (BERT on MRPC) =====
try:
    glue_df = pd.read_csv(glue_results)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('BERT Compression Results on GLUE MRPC', fontsize=16, fontweight='bold')
    
    # Plot 1: Model Size
    axes[0, 0].bar(range(len(glue_df)), glue_df['Size (MB)'], color='steelblue')
    axes[0, 0].set_xticks(range(len(glue_df)))
    axes[0, 0].set_xticklabels(glue_df['Pipeline'], rotation=45, ha='right')
    axes[0, 0].set_ylabel('Model Size (MB)', fontweight='bold')
    axes[0, 0].set_title('Model Size Comparison')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Plot 2: Sparsity
    axes[0, 1].bar(range(len(glue_df)), glue_df['Sparsity (%)'], color='coral')
    axes[0, 1].set_xticks(range(len(glue_df)))
    axes[0, 1].set_xticklabels(glue_df['Pipeline'], rotation=45, ha='right')
    axes[0, 1].set_ylabel('Sparsity (%)', fontweight='bold')
    axes[0, 1].set_title('Sparsity Comparison')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Plot 3: Accuracy
    axes[1, 0].bar(range(len(glue_df)), glue_df['Accuracy'], color='mediumseagreen')
    axes[1, 0].set_xticks(range(len(glue_df)))
    axes[1, 0].set_xticklabels(glue_df['Pipeline'], rotation=45, ha='right')
    axes[1, 0].set_ylabel('Accuracy', fontweight='bold')
    axes[1, 0].set_title('Accuracy Comparison')
    axes[1, 0].set_ylim([0.80, 0.90])
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Plot 4: Latency
    axes[1, 1].bar(range(len(glue_df)), glue_df['Latency (ms)'], color='mediumpurple')
    axes[1, 1].set_xticks(range(len(glue_df)))
    axes[1, 1].set_xticklabels(glue_df['Pipeline'], rotation=45, ha='right')
    axes[1, 1].set_ylabel('Latency (ms)', fontweight='bold')
    axes[1, 1].set_title('Latency Comparison')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bert_results.png'), dpi=300, bbox_inches='tight')
    print("Saved bert_results.png")
    plt.close()
except Exception as e:
    print(f"Error generating BERT plots: {e}")

# ===== GPU PROFILING COMPARISON =====
try:
    gpu_df = pd.read_csv(gpu_results)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('GPU vs CPU Performance Comparison', fontsize=16, fontweight='bold')
    
    # Group by device
    cpu_data = gpu_df[gpu_df['Device'] == 'cpu']
    gpu_data = gpu_df[gpu_df['Device'] == 'cuda:0']
    
    # Plot 1: Latency Comparison
    x = range(len(cpu_data))
    width = 0.35
    
    axes[0].bar([i - width/2 for i in x], cpu_data['Latency (ms)'], width, label='CPU', color='steelblue')
    axes[0].bar([i + width/2 for i in x], gpu_data['Latency (ms)'], width, label='GPU', color='coral')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(cpu_data['Pipeline'], rotation=45, ha='right')
    axes[0].set_ylabel('Latency (ms)', fontweight='bold')
    axes[0].set_title('Latency: CPU vs GPU')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # Plot 2: Accuracy Comparison
    axes[1].bar([i - width/2 for i in x], cpu_data['Accuracy'], width, label='CPU', color='mediumseagreen')
    axes[1].bar([i + width/2 for i in x], gpu_data['Accuracy'], width, label='GPU', color='mediumpurple')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(cpu_data['Pipeline'], rotation=45, ha='right')
    axes[1].set_ylabel('Accuracy', fontweight='bold')
    axes[1].set_title('Accuracy: CPU vs GPU')
    axes[1].legend()
    axes[1].set_ylim([0.80, 0.90])
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gpu_cpu_comparison.png'), dpi=300, bbox_inches='tight')
    print("Saved gpu_cpu_comparison.png")
    plt.close()
except Exception as e:
    print(f"Error generating GPU comparison plots: {e}")

print("All visualizations generated successfully!")
