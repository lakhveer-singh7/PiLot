#!/usr/bin/env python3
"""
plot_multi_device.py — Generate comparison plots for 7, 9, 12 device distributed configs.

Plots:
  (a) 3 × Accuracy vs Time (one per dataset, 3 curves per plot)
  (b) 3 × Inference Latency (one per dataset, 6 bars: comp + comm × 3 configs)
  (c) 3 × Memory Consumption (one per dataset, 9 bars: weight/opt/buf × 3 configs)

Prerequisites: python3 parse_multi_device_results.py
Output PNGs → results/multi_device/
"""

import os, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = SCRIPT_DIR  # script lives in repo root
CSV_DIR = os.path.join(ROOT_DIR, 'results', 'csv_multi')
OUT_DIR = os.path.join(ROOT_DIR, 'results', 'multi_device')
os.makedirs(OUT_DIR, exist_ok=True)

# Colour palette
COLORS_7  = '#3366CC'
COLORS_9  = '#FF9933'
COLORS_12 = '#33AA55'
DEV_COLORS = {7: COLORS_7, 9: COLORS_9, 12: COLORS_12}
DEV_LABELS = {7: '7 Devices', 9: '9 Devices', 12: '12 Devices'}

COMP_COLOR = '#FF9933'   # orange
COMM_COLOR = '#BF2606'   # dark red

WEIGHT_COLOR = '#3366CC'
OPT_COLOR    = '#FF9933'
BUF_COLOR    = '#33AA55'

DATASETS = ['Cricket_X', 'ECG5000', 'FaceAll']
NDEVS = [7, 9, 12]


def read_csv_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


# ═══════════════════════════════════════════════════════════════
#  (a) Accuracy vs Time — 3 plots
# ═══════════════════════════════════════════════════════════════
def plot_accuracy_vs_time():
    print('[1/3] Generating Accuracy vs Time plots …')
    for ds in DATASETS:
        csv_path = os.path.join(CSV_DIR, f'accuracy_vs_time_{ds}.csv')
        rows = read_csv_rows(csv_path)

        fig, ax = plt.subplots(figsize=(10, 6))
        for ndev in NDEVS:
            subset = [r for r in rows if int(r['num_devices']) == ndev]
            if not subset:
                continue
            t = [float(r['timespan_s']) for r in subset]
            acc = [float(r['test_acc']) for r in subset]
            ax.plot(t, acc, '-o', color=DEV_COLORS[ndev], label=DEV_LABELS[ndev],
                    markersize=3, linewidth=1.5, alpha=0.85)

        ax.set_xlabel('Training Time (s)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Test Accuracy (%)', fontsize=13, fontweight='bold')
        ax.set_title(f'Accuracy vs Time — {ds.replace("_", " ")}  (Distributed)',
                     fontsize=15, fontweight='bold')
        ax.legend(fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=11)

        fig.tight_layout()
        out = os.path.join(OUT_DIR, f'accuracy_vs_time_{ds}.png')
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print(f'  Saved: {out}')


# ═══════════════════════════════════════════════════════════════
#  (b) Inference Latency — 3 plots (6 bars each)
# ═══════════════════════════════════════════════════════════════
def plot_inference_latency():
    print('[2/3] Generating Inference Latency plots …')
    csv_path = os.path.join(CSV_DIR, 'inference_latency.csv')
    rows = read_csv_rows(csv_path)

    for ds in DATASETS:
        subset = [r for r in rows if r['dataset'] == ds]
        if not subset:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        x_positions = []
        x_labels = []
        bar_width = 0.35

        for idx, ndev in enumerate(NDEVS):
            r = next((r for r in subset if int(r['num_devices']) == ndev), None)
            if not r:
                continue
            comp = float(r['computation_ms'])
            comm = float(r['communication_ms'])
            base_x = idx * 2.5  # spacing between groups

            # Computation bar
            x_comp = base_x
            b1 = ax.bar(x_comp, comp, bar_width, color=COMP_COLOR,
                        edgecolor='black', linewidth=0.5)
            ax.text(x_comp, comp + 0.15, f'{comp:.2f}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

            # Communication bar
            x_comm = base_x + bar_width + 0.05
            b2 = ax.bar(x_comm, comm, bar_width, color=COMM_COLOR,
                        edgecolor='black', linewidth=0.5)
            ax.text(x_comm, comm + 0.15, f'{comm:.2f}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

            # Group label
            group_center = (x_comp + x_comm) / 2
            x_positions.append(group_center)
            x_labels.append(f'{ndev} Devices')

        # Legend manually
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=COMP_COLOR, edgecolor='black', label='Computation'),
            Patch(facecolor=COMM_COLOR, edgecolor='black', label='Communication (IPC)'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)

        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, fontsize=12)
        ax.set_xlabel('Configuration', fontsize=13, fontweight='bold')
        ax.set_ylabel('Inference Latency (ms)', fontsize=13, fontweight='bold')
        ax.set_title(f'Inference Latency — {ds.replace("_", " ")}  (Distributed)',
                     fontsize=15, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
        ax.tick_params(labelsize=11)

        fig.tight_layout()
        out = os.path.join(OUT_DIR, f'inference_latency_{ds}.png')
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print(f'  Saved: {out}')


# ═══════════════════════════════════════════════════════════════
#  (c) Memory Consumption — 3 plots (9 bars each: 3 groups × 3 types)
# ═══════════════════════════════════════════════════════════════
def plot_memory_consumption():
    print('[3/3] Generating Memory Consumption plots …')
    csv_path = os.path.join(CSV_DIR, 'memory_consumption.csv')
    rows = read_csv_rows(csv_path)

    for ds in DATASETS:
        subset = [r for r in rows if r['dataset'] == ds]
        if not subset:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        bar_width = 0.22
        x_positions = []
        x_labels = []

        for idx, ndev in enumerate(NDEVS):
            r = next((r for r in subset if int(r['num_devices']) == ndev), None)
            if not r:
                continue
            w_kb = float(r['weight_avg_kb'])
            o_kb = float(r['optimizer_avg_kb'])
            b_kb = float(r['buffer_avg_kb'])
            w_std = float(r['weight_std_kb'])
            o_std = float(r['optimizer_std_kb'])
            b_std = float(r['buffer_std_kb'])

            base_x = idx * 3.0
            x_w = base_x - bar_width
            x_o = base_x
            x_b = base_x + bar_width

            ax.bar(x_w, w_kb, bar_width, color=WEIGHT_COLOR,
                   edgecolor='black', linewidth=0.5, yerr=w_std, capsize=4)
            ax.text(x_w, w_kb + w_std + 0.3, f'{w_kb:.1f}', ha='center', va='bottom',
                    fontsize=8.5, fontweight='bold')

            ax.bar(x_o, o_kb, bar_width, color=OPT_COLOR,
                   edgecolor='black', linewidth=0.5, yerr=o_std, capsize=4)
            ax.text(x_o, o_kb + o_std + 0.3, f'{o_kb:.1f}', ha='center', va='bottom',
                    fontsize=8.5, fontweight='bold')

            ax.bar(x_b, b_kb, bar_width, color=BUF_COLOR,
                   edgecolor='black', linewidth=0.5, yerr=b_std, capsize=4)
            ax.text(x_b, b_kb + b_std + 0.3, f'{b_kb:.1f}', ha='center', va='bottom',
                    fontsize=8.5, fontweight='bold')

            x_positions.append(base_x)
            x_labels.append(f'{ndev} Devices')

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=WEIGHT_COLOR, edgecolor='black', label='Weights (avg)'),
            Patch(facecolor=OPT_COLOR, edgecolor='black', label='Optimizer (avg)'),
            Patch(facecolor=BUF_COLOR, edgecolor='black', label='Buffers (avg)'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)

        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, fontsize=12)
        ax.set_xlabel('Configuration', fontsize=13, fontweight='bold')
        ax.set_ylabel('Avg Memory per Device (KB)', fontsize=13, fontweight='bold')
        ax.set_title(f'Memory Consumption — {ds.replace("_", " ")}  (Distributed)',
                     fontsize=15, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
        ax.tick_params(labelsize=11)

        fig.tight_layout()
        out = os.path.join(OUT_DIR, f'memory_consumption_{ds}.png')
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print(f'  Saved: {out}')


# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('=== PiLot Multi-Device Comparison Plots ===\n')
    plot_accuracy_vs_time()
    print()
    plot_inference_latency()
    print()
    plot_memory_consumption()
    print(f'\n=== All 9 plots generated! ===')
    print(f'Output: {OUT_DIR}/')
