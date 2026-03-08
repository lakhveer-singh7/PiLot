#!/usr/bin/env python3
"""
plot_all.py — Generate all 5 plots (mirrors the MATLAB scripts exactly).

  (a) 3 × Accuracy vs Time (one per dataset)
  (b) 1 × Inference Latency grouped bar chart
  (c) 1 × Memory Consumption grouped bar chart

Prerequisites:
  1. Run experiments:  bash run_all_local.sh
  2. Parse results:    python3 parse_results.py
  3. Then run this:    python3 plotting/plot_all.py

Output PNGs saved to results/
"""

import os
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for WSL
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
CSV_DIR = os.path.join(ROOT_DIR, 'results', 'csv')
OUT_DIR = os.path.join(ROOT_DIR, 'results')

DATASETS = ['Cricket_X', 'ECG5000', 'FaceAll']

# ── Colour palette (matches MATLAB scripts) ──
BLUE = '#3366CC'
RED  = '#D94020'


def read_csv(path):
    """Read CSV into list of dicts."""
    with open(path, 'r') as f:
        return list(csv.DictReader(f))


# ═════════════════════════════════════════════════════════════════════════════
#  (a)  Accuracy vs Time — one figure per dataset
# ═════════════════════════════════════════════════════════════════════════════

def plot_accuracy_vs_time():
    print('[1/3] Generating Accuracy vs Time plots …')
    for ds in DATASETS:
        cent_file = os.path.join(CSV_DIR, f'accuracy_vs_time_{ds}_centralized.csv')
        dist_file = os.path.join(CSV_DIR, f'accuracy_vs_time_{ds}_distributed.csv')

        fig, ax = plt.subplots(figsize=(8, 5.5))

        if os.path.exists(cent_file):
            rows = read_csv(cent_file)
            t = [float(r['time_s']) for r in rows]
            acc = [float(r['test_acc']) for r in rows]
            ax.plot(t, acc, '-o', color=BLUE, linewidth=2,
                    markersize=3.5, markerfacecolor=BLUE, label='Centralized')

        if os.path.exists(dist_file):
            rows = read_csv(dist_file)
            t = [float(r['time_s']) for r in rows]
            acc = [float(r['test_acc']) for r in rows]
            ax.plot(t, acc, '-s', color=RED, linewidth=2,
                    markersize=3.5, markerfacecolor=RED,
                    label='Distributed (7 devices, 64 MHz)')

        ax.set_xlabel('Training Time (seconds)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Test Accuracy (%)', fontsize=13, fontweight='bold')
        ax.set_title(f'Accuracy vs Time — {ds}', fontsize=15, fontweight='bold')
        ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=11)

        out = os.path.join(OUT_DIR, f'accuracy_vs_time_{ds}.png')
        fig.tight_layout()
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print(f'  Saved: {out}')


# ═════════════════════════════════════════════════════════════════════════════
#  (b)  Inference Latency — grouped bar chart
# ═════════════════════════════════════════════════════════════════════════════

def plot_inference_latency():
    print('[2/3] Generating Inference Latency plot …')
    csv_file = os.path.join(CSV_DIR, 'inference_latency.csv')
    rows = read_csv(csv_file)

    ds_labels = [r['dataset'].replace('_', ' ') for r in rows]
    cent = np.array([float(r['centralized_ms']) for r in rows])
    dist = np.array([float(r['distributed_ms']) for r in rows])

    # Distributed bar split: computation = centralized latency, communication = remainder
    dist_comp = cent.copy()         # same CNN FLOPs
    dist_comm = dist - cent          # IPC / shared-memory overhead

    x = np.arange(len(ds_labels))
    width = 0.32

    fig, ax = plt.subplots(figsize=(10, 6))

    # Centralized: solid blue bar
    b_cent = ax.bar(x - width/2, cent, width, color=BLUE,
                    edgecolor='black', linewidth=0.5,
                    label='Centralized (Computation)')

    # Distributed: stacked bar -- computation (orange) + communication (dark red)
    ORANGE = '#FF9933'
    DARK_RED = '#BF2606'
    b_comp = ax.bar(x + width/2, dist_comp, width, color=ORANGE,
                    edgecolor='black', linewidth=0.5,
                    label='Distributed - Computation')
    b_comm = ax.bar(x + width/2, dist_comm, width, bottom=dist_comp,
                    color=DARK_RED, edgecolor='black', linewidth=0.5,
                    label='Distributed - Communication (IPC)')

    # Dashed white line at the cut between computation and communication
    for i in range(len(x)):
        bx = b_comp[i].get_x()
        bw = b_comp[i].get_width()
        ax.plot([bx, bx + bw], [dist_comp[i], dist_comp[i]],
                'w--', linewidth=1.5, zorder=5)

    # --- Value labels ---
    for i in range(len(x)):
        # Centralized total on top
        ax.text(x[i] - width/2, cent[i] + 0.2, f'{cent[i]:.2f}',
                ha='center', va='bottom', fontsize=9.5, fontweight='bold')
        # Distributed computation (inside lower section)
        ax.text(x[i] + width/2, dist_comp[i]/2, f'{dist_comp[i]:.2f}',
                ha='center', va='center', fontsize=8.5, fontweight='bold', color='white')
        # Distributed communication (inside upper section)
        ax.text(x[i] + width/2, dist_comp[i] + dist_comm[i]/2, f'{dist_comm[i]:.2f}',
                ha='center', va='center', fontsize=8.5, fontweight='bold', color='white')
        # Distributed total on top
        ax.text(x[i] + width/2, dist[i] + 0.2, f'{dist[i]:.2f}',
                ha='center', va='bottom', fontsize=9.5, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(ds_labels, fontsize=12)
    ax.set_xlabel('Dataset', fontsize=13, fontweight='bold')
    ax.set_ylabel('Inference Latency (ms/sample)', fontsize=13, fontweight='bold')
    ax.set_title('Inference Latency: Centralized vs Distributed',
                 fontsize=15, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax.grid(True, axis='y', alpha=0.3)
    ax.tick_params(labelsize=11)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, 'inference_latency.png')
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f'  Saved: {out}')


# ═════════════════════════════════════════════════════════════════════════════
#  (c)  Memory Consumption — grouped bar chart with error bars
# ═════════════════════════════════════════════════════════════════════════════

def plot_memory_consumption():
    print('[3/3] Generating Memory Consumption plot …')
    csv_file = os.path.join(CSV_DIR, 'memory_consumption.csv')
    rows = read_csv(csv_file)

    ds_labels = [r['dataset'].replace('_', ' ') for r in rows]
    cent = [float(r['centralized_kb']) for r in rows]
    dist_avg = [float(r['distributed_avg_kb']) for r in rows]
    dist_std = [float(r['distributed_std_kb']) for r in rows]

    x = np.arange(len(ds_labels))
    width = 0.32

    fig, ax = plt.subplots(figsize=(9, 5.5))
    b1 = ax.bar(x - width/2, cent, width, color=BLUE,
                label='Centralized (Single Device)')
    b2 = ax.bar(x + width/2, dist_avg, width, color=RED,
                yerr=dist_std, capsize=8, error_kw={'linewidth': 1.5},
                label='Distributed (Avg per Device ± Std)')

    # Value labels
    for bar in b1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 30,
                f'{h:.0f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold')
    for bar, std in zip(b2, dist_std):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + std + 8,
                f'{h:.1f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(ds_labels, fontsize=12)
    ax.set_xlabel('Dataset', fontsize=13, fontweight='bold')
    ax.set_ylabel('Memory Consumption (KB)', fontsize=13, fontweight='bold')
    ax.set_title('Memory Consumption: Centralized vs Distributed',
                 fontsize=15, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    ax.grid(True, axis='y', alpha=0.3)
    ax.tick_params(labelsize=11)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, 'memory_consumption.png')
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f'  Saved: {out}')


# ═════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    print('=== PiLot Results Plotting (matplotlib) ===\n')
    plot_accuracy_vs_time()
    print()
    plot_inference_latency()
    print()
    plot_memory_consumption()
    print(f'\n=== All 5 plots generated! ===')
    print(f'Output: {OUT_DIR}/')
    print('  - accuracy_vs_time_Cricket_X.png')
    print('  - accuracy_vs_time_ECG5000.png')
    print('  - accuracy_vs_time_FaceAll.png')
    print('  - inference_latency.png')
    print('  - memory_consumption.png')
