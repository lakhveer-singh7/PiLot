#!/usr/bin/env python3
"""
plot_all.py — Generate all 6 plots (mirrors the MATLAB scripts exactly).

  (a)  3 × Accuracy vs Time (one per dataset)
  (b1) 1 × Inference Latency grouped bar chart
  (b2) 1 × Per-Device Inference Latency (stacked computation + communication)
  (c)  1 × Memory Consumption grouped bar chart

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
    print('[1/4] Generating Accuracy vs Time plots …')
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
    print('[2a/4] Generating Inference Latency plot …')
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
#  (b2) Per-Device Inference Latency — stacked bars (Computation + Communication)
# ═════════════════════════════════════════════════════════════════════════════

def plot_per_device_latency():
    """Per-device stacked bar: computation (FLOPs-based) + communication (IPC).

    Architecture: Head → L0_W0,L0_W1 → L1_W0,L1_W1,L1_W2 → Tail
    Computation delay = FLOPs / 64 MHz (proc_delay_flops simulation).
    Communication = (pipeline_latency − critical-path computation) / 3 hops,
    attributed to the receiving device at each IPC boundary.
    """
    print('[2b/4] Generating Per-Device Inference Latency plot …')
    csv_file = os.path.join(CSV_DIR, 'inference_latency.csv')
    rows = read_csv(csv_file)

    # ── Dataset properties (input_length, num_classes) ──
    DS_PROPS = {
        'Cricket_X': (300, 12),
        'ECG5000':   (140,  5),
        'FaceAll':   (131, 14),
    }

    # ── CNN constants ──
    PROC_HZ = 64_000_000
    L0_in, L0_out, L0_k, L0_s, L0_p = 1, 16, 5, 1, 2
    L1_in, L1_out_ch, L1_k, L1_s, L1_p = 32, 16, 5, 2, 2
    TAIL_IN = 96   # 48 ch × 2 (dual pool)

    device_names = ['Head', 'L0_W0', 'L0_W1', 'L1_W0', 'L1_W1', 'L1_W2', 'Tail']
    N_DEV = 7
    ORANGE   = '#FF9933'
    DARK_RED = '#BF2606'

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, r in enumerate(rows):
        ds = r['dataset']
        pipe_dist = float(r['distributed_ms'])
        il, nc = DS_PROPS[ds]

        # Output lengths
        L0_olen = il                                          # stride=1, same pad
        L1_olen = (il + 2 * L1_p - L1_k) // L1_s + 1

        # Forward FLOPs
        L0_flops = 2 * L0_out * L0_in * L0_k * L0_olen
        L1_flops = 2 * L1_out_ch * L1_in * L1_k * L1_olen
        tail_flops = 2 * TAIL_IN * nc

        # Computation delays (ms)
        head_c  = 0.0
        L0_c    = L0_flops / PROC_HZ * 1000
        L1_c    = L1_flops / PROC_HZ * 1000
        tail_c  = tail_flops / PROC_HZ * 1000

        crit_comp  = head_c + L0_c + L1_c + tail_c
        total_comm = max(0.0, pipe_dist - crit_comp)
        comm_hop   = total_comm / 3.0

        comp = np.array([head_c, L0_c, L0_c, L1_c, L1_c, L1_c, tail_c])
        comm = np.array([0.0, comm_hop, comm_hop, comm_hop, comm_hop, comm_hop, comm_hop])

        ax = axes[idx]
        x = np.arange(N_DEV)
        width = 0.55

        b_comp = ax.bar(x, comp, width, color=ORANGE,
                        edgecolor='black', linewidth=0.5,
                        label='Computation')
        b_comm = ax.bar(x, comm, width, bottom=comp, color=DARK_RED,
                        edgecolor='black', linewidth=0.5,
                        label='Communication (IPC)')

        # White dashed line at cut
        for i in range(N_DEV):
            if comp[i] > 0 and comm[i] > 0:
                bx_ = b_comp[i].get_x()
                bw_ = b_comp[i].get_width()
                ax.plot([bx_, bx_ + bw_], [comp[i], comp[i]],
                        'w--', linewidth=1.5, zorder=5)

        # Value labels
        for i in range(N_DEV):
            total = comp[i] + comm[i]
            # total on top
            ax.text(x[i], total + 0.08, f'{total:.2f}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
            # comp inside bottom (if tall enough)
            if comp[i] > 0.25:
                ax.text(x[i], comp[i] / 2, f'{comp[i]:.2f}',
                        ha='center', va='center', fontsize=7.5,
                        fontweight='bold', color='white')
            # comm inside top (if tall enough)
            if comm[i] > 0.25:
                ax.text(x[i], comp[i] + comm[i] / 2, f'{comm[i]:.2f}',
                        ha='center', va='center', fontsize=7.5,
                        fontweight='bold', color='white')

        ax.set_xticks(x)
        ax.set_xticklabels(device_names, fontsize=9, rotation=30, ha='right')
        ax.set_xlabel('Device', fontsize=11, fontweight='bold')
        ax.set_ylabel('Inference Latency (ms)', fontsize=11, fontweight='bold')
        ax.set_title(ds.replace('_', ' '), fontsize=13, fontweight='bold')
        ax.legend(loc='upper left', fontsize=8.5, framealpha=0.9)
        ax.grid(True, axis='y', alpha=0.3)
        ax.tick_params(labelsize=9)

    fig.suptitle('Per-Device Inference Latency (Distributed Model)',
                 fontsize=15, fontweight='bold', y=1.01)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, 'per_device_latency.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out}')


# ═════════════════════════════════════════════════════════════════════════════
#  (c)  Memory Consumption — grouped bar chart with error bars
# ═════════════════════════════════════════════════════════════════════════════

def plot_memory_consumption():
    print('[3/4] Generating Memory Consumption plot …')
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
    plot_per_device_latency()
    print()
    plot_memory_consumption()
    print(f'\n=== All 6 plots generated! ===')
    print(f'Output: {OUT_DIR}/')
    print('  - accuracy_vs_time_Cricket_X.png')
    print('  - accuracy_vs_time_ECG5000.png')
    print('  - accuracy_vs_time_FaceAll.png')
    print('  - inference_latency.png')
    print('  - per_device_latency.png')
    print('  - memory_consumption.png')
