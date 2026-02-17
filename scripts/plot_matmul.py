#!/usr/bin/env python3
"""
Plot matrix multiplication performance results.
Generates speedup and efficiency plots for Exercise 4.
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
    csv_file = os.path.join(data_dir, 'matmul_scaling.csv')
    
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found. Run the experiments first.")
        return
    
    # Read data
    df = pd.read_csv(csv_file)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Speedup
    ax1.plot(df['threads'], df['speedup_basic'], 'o-', label='Basic Parallel', linewidth=2, markersize=8)
    ax1.plot(df['threads'], df['speedup_collapse'], 's-', label='Collapse(2)', linewidth=2, markersize=8)
    ax1.plot(df['threads'], df['threads'], '--', label='Ideal', alpha=0.5, color='gray')
    ax1.set_xlabel('Number of Threads', fontsize=12)
    ax1.set_ylabel('Speedup', fontsize=12)
    ax1.set_title('Matrix Multiplication Speedup', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(df['threads'])
    
    # Plot 2: Efficiency
    ax2.plot(df['threads'], df['efficiency_basic'], 'o-', label='Basic Parallel', linewidth=2, markersize=8)
    ax2.plot(df['threads'], df['efficiency_collapse'], 's-', label='Collapse(2)', linewidth=2, markersize=8)
    ax2.axhline(y=100, linestyle='--', alpha=0.5, color='gray', label='Ideal (100%)')
    ax2.set_xlabel('Number of Threads', fontsize=12)
    ax2.set_ylabel('Efficiency (%)', fontsize=12)
    ax2.set_title('Matrix Multiplication Efficiency', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(df['threads'])
    ax2.set_ylim(0, 120)
    
    plt.tight_layout()
    output_file = os.path.join(plots_dir, 'matmul_scaling.png')
    plt.savefig(output_file, dpi=150)
    print(f"Saved: {output_file}")
    plt.close()

def plot_schedule():
    """Plot scheduling strategy comparison."""
    csv_file = os.path.join(data_dir, 'matmul_schedule.csv')
    
    if not os.path.exists(csv_file):
        print(f"Warning: {csv_file} not found. Skipping schedule plot.")
        return
    
    try:
        df = pd.read_csv(csv_file)
    except:
        print(f"Warning: Could not parse {csv_file}. Skipping schedule plot.")
        return
    
    if df.empty:
        return
    
    # Pivot data for plotting
    pivot = df.pivot(index='chunk', columns='schedule', values='time')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = range(len(pivot.index))
    width = 0.25
    
    if 'static' in pivot.columns:
        ax.bar([i - width for i in x], pivot['static'], width, label='STATIC')
    if 'dynamic' in pivot.columns:
        ax.bar(x, pivot['dynamic'], width, label='DYNAMIC')
    if 'guided' in pivot.columns:
        ax.bar([i + width for i in x], pivot['guided'], width, label='GUIDED')
    
    ax.set_xlabel('Chunk Size', fontsize=12)
    ax.set_ylabel('Execution Time (s)', fontsize=12)
    ax.set_title('Matrix Multiplication: Scheduling Strategies Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_file = os.path.join(plots_dir, 'matmul_schedule.png')
    plt.savefig(output_file, dpi=150)
    print(f"Saved: {output_file}")
    plt.close()

def main():
    print("Generating Matrix Multiplication Plots...")
    print("=" * 50)
    
    plot_scaling()
    plot_schedule()
    
    print("\nDone!")

if __name__ == '__main__':
    main()
