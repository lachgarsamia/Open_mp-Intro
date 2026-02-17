#!/usr/bin/env python3
"""
Plot Jacobi method performance results.
Generates speedup and efficiency plots for Exercise 5.
"""

import matplotlib.pyplot as plt
import pandas as pd
import os

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '..', 'data')
plots_dir = os.path.join(script_dir, '..', 'plots')

os.makedirs(plots_dir, exist_ok=True)

def plot_scaling():
    """Plot speedup and efficiency for thread scaling."""
    csv_file = os.path.join(data_dir, 'jacobi_scaling.csv')
    
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found. Run the experiments first.")
        return
    
    # Read data
    df = pd.read_csv(csv_file)
    
    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Execution Time
    ax1 = axes[0]
    ax1.plot(df['threads'], df['sequential_time'], 'o--', label='Sequential', linewidth=2, markersize=8, alpha=0.7)
    ax1.plot(df['threads'], df['parallel_time'], 's-', label='Parallel', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Threads', fontsize=12)
    ax1.set_ylabel('Execution Time (s)', fontsize=12)
    ax1.set_title('Jacobi Method: Execution Time', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(df['threads'])
    
    # Plot 2: Speedup
    ax2 = axes[1]
    ax2.plot(df['threads'], df['speedup'], 'o-', label='Achieved', linewidth=2, markersize=8, color='green')
    ax2.plot(df['threads'], df['threads'], '--', label='Ideal', alpha=0.5, color='gray')
    ax2.set_xlabel('Number of Threads', fontsize=12)
    ax2.set_ylabel('Speedup', fontsize=12)
    ax2.set_title('Jacobi Method: Speedup', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(df['threads'])
    
    # Plot 3: Efficiency
    ax3 = axes[2]
    ax3.plot(df['threads'], df['efficiency'], 'o-', linewidth=2, markersize=8, color='orange')
    ax3.axhline(y=100, linestyle='--', alpha=0.5, color='gray', label='Ideal (100%)')
    ax3.set_xlabel('Number of Threads', fontsize=12)
    ax3.set_ylabel('Efficiency (%)', fontsize=12)
    ax3.set_title('Jacobi Method: Efficiency', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(df['threads'])
    ax3.set_ylim(0, 120)
    
    plt.tight_layout()
    output_file = os.path.join(plots_dir, 'jacobi_scaling.png')
    plt.savefig(output_file, dpi=150)
    print(f"Saved: {output_file}")
    plt.close()

def plot_comparison():
    """Create a comparison bar chart."""
    csv_file = os.path.join(data_dir, 'jacobi_scaling.csv')
    
    if not os.path.exists(csv_file):
        return
    
    df = pd.read_csv(csv_file)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = range(len(df))
    width = 0.35
    
    bars1 = ax.bar([i - width/2 for i in x], df['sequential_time'], width, 
                   label='Sequential', color='steelblue', alpha=0.8)
    bars2 = ax.bar([i + width/2 for i in x], df['parallel_time'], width,
                   label='Parallel', color='coral', alpha=0.8)
    
    ax.set_xlabel('Number of Threads', fontsize=12)
    ax.set_ylabel('Execution Time (s)', fontsize=12)
    ax.set_title('Jacobi Method: Sequential vs Parallel Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(df['threads'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add speedup annotations
    for i, (seq, par, sp) in enumerate(zip(df['sequential_time'], df['parallel_time'], df['speedup'])):
        ax.annotate(f'{sp:.2f}x', xy=(i, max(seq, par)),
                   xytext=(0, 5), textcoords='offset points',
                   ha='center', fontsize=9)
    
    plt.tight_layout()
    output_file = os.path.join(plots_dir, 'jacobi_comparison.png')
    plt.savefig(output_file, dpi=150)
    print(f"Saved: {output_file}")
    plt.close()

def main():
    print("Generating Jacobi Method Plots...")
    print("=" * 50)
    
    plot_scaling()
    plot_comparison()
    
    print("\nDone!")

if __name__ == '__main__':
    main()
