#!/usr/bin/env python3
"""
Visualize evaluation results
Creates plots for ESS, beta, accuracy over time
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def plot_solver_statistics(results_path: str, output_dir: str = "plots"):
    """
    Plot solver statistics from evaluation results

    Args:
        results_path: Path to results JSON
        output_dir: Directory to save plots
    """
    print(f"Loading results from {results_path}")

    with open(results_path) as f:
        data = json.load(f)

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Get solver stats
    if 'solver_stats' not in data:
        print("No solver statistics found in results")
        return

    stats = data['solver_stats']

    # 1. Plot ESS over time
    if stats.get('ess_history'):
        plt.figure(figsize=(10, 6))
        plt.plot(stats['ess_history'], linewidth=2)
        plt.axhline(y=np.mean(stats['n_alive_history']) * 0.33,
                   color='r', linestyle='--', label='Resample threshold (0.33*N)')
        plt.xlabel('Step', fontsize=12)
        plt.ylabel('Effective Sample Size (ESS)', fontsize=12)
        plt.title('ESS over Time', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/ess_history.png", dpi=300)
        print(f"Saved: {output_dir}/ess_history.png")
        plt.close()

    # 2. Plot Beta (annealing) over time
    if stats.get('beta_history'):
        plt.figure(figsize=(10, 6))
        plt.plot(stats['beta_history'], linewidth=2, color='orange')
        plt.xlabel('Step', fontsize=12)
        plt.ylabel('Beta (Annealing Parameter)', fontsize=12)
        plt.title('Annealing Schedule', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/beta_history.png", dpi=300)
        print(f"Saved: {output_dir}/beta_history.png")
        plt.close()

    # 3. Plot N_alive over time
    if stats.get('n_alive_history'):
        plt.figure(figsize=(10, 6))
        plt.plot(stats['n_alive_history'], linewidth=2, color='green')
        plt.xlabel('Step', fontsize=12)
        plt.ylabel('Number of Alive Particles', fontsize=12)
        plt.title('Particle Count over Time', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/n_alive_history.png", dpi=300)
        print(f"Saved: {output_dir}/n_alive_history.png")
        plt.close()

    # 4. Combined plot
    if all(stats.get(k) for k in ['ess_history', 'beta_history', 'n_alive_history']):
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        # ESS
        axes[0].plot(stats['ess_history'], linewidth=2)
        axes[0].set_ylabel('ESS', fontsize=11)
        axes[0].set_title('Effective Sample Size', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # Beta
        axes[1].plot(stats['beta_history'], linewidth=2, color='orange')
        axes[1].set_ylabel('Beta', fontsize=11)
        axes[1].set_title('Annealing Parameter', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        # N_alive
        axes[2].plot(stats['n_alive_history'], linewidth=2, color='green')
        axes[2].set_ylabel('N_alive', fontsize=11)
        axes[2].set_xlabel('Step', fontsize=11)
        axes[2].set_title('Alive Particles', fontsize=12, fontweight='bold')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/combined_stats.png", dpi=300)
        print(f"Saved: {output_dir}/combined_stats.png")
        plt.close()


def plot_accuracy_by_difficulty(results_path: str, output_dir: str = "plots"):
    """Plot accuracy breakdown by difficulty level"""

    with open(results_path) as f:
        data = json.load(f)

    if 'results' not in data:
        print("No detailed results found")
        return

    # Group by difficulty
    difficulty_stats = {}
    for result in data['results']:
        level = result.get('level', 'Unknown')
        if level not in difficulty_stats:
            difficulty_stats[level] = {'correct': 0, 'total': 0}

        difficulty_stats[level]['total'] += 1
        if result['correct']:
            difficulty_stats[level]['correct'] += 1

    if not difficulty_stats:
        print("No difficulty information found")
        return

    # Compute accuracies
    levels = sorted(difficulty_stats.keys())
    accuracies = [difficulty_stats[l]['correct'] / difficulty_stats[l]['total']
                  for l in levels]
    counts = [difficulty_stats[l]['total'] for l in levels]

    # Plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(levels)), accuracies, color='steelblue', alpha=0.8)

    # Add counts on top
    for i, (bar, count) in enumerate(zip(bars, counts)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'n={count}',
                ha='center', va='bottom', fontsize=10)

    plt.xlabel('Difficulty Level', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Accuracy by Difficulty Level', fontsize=14, fontweight='bold')
    plt.xticks(range(len(levels)), levels, rotation=45)
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/accuracy_by_difficulty.png", dpi=300)
    print(f"Saved: {output_dir}/accuracy_by_difficulty.png")
    plt.close()


def plot_pass_at_k(results_path: str, output_dir: str = "plots"):
    """Plot pass@k metrics"""

    with open(results_path) as f:
        data = json.load(f)

    if 'pass_at_k' not in data:
        print("No pass@k metrics found")
        return

    pass_at_k = data['pass_at_k']

    # Extract k values and accuracies
    k_values = []
    accuracies = []
    for key, value in sorted(pass_at_k.items()):
        k = int(key.split('@')[1])
        k_values.append(k)
        accuracies.append(value)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, accuracies, marker='o', linewidth=2, markersize=8)
    plt.xlabel('k (number of samples)', fontsize=12)
    plt.ylabel('Pass@k Accuracy', fontsize=12)
    plt.title('Pass@k Performance', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pass_at_k.png", dpi=300)
    print(f"Saved: {output_dir}/pass_at_k.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize Persistent SMC results")
    parser.add_argument(
        "results_path",
        type=str,
        help="Path to results JSON file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="plots",
        help="Directory to save plots"
    )

    args = parser.parse_args()

    if not Path(args.results_path).exists():
        print(f"Error: Results file not found: {args.results_path}")
        return

    print("=" * 60)
    print("Visualizing Persistent SMC Results")
    print("=" * 60)
    print(f"Input: {args.results_path}")
    print(f"Output directory: {args.output_dir}")
    print()

    # Create plots
    try:
        plot_solver_statistics(args.results_path, args.output_dir)
        plot_accuracy_by_difficulty(args.results_path, args.output_dir)
        plot_pass_at_k(args.results_path, args.output_dir)

        print()
        print("=" * 60)
        print("Visualization complete!")
        print(f"Plots saved in: {args.output_dir}/")
        print("=" * 60)

    except Exception as e:
        print(f"Error creating plots: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
