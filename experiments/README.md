# Experiment Pipeline — PiLot vs RockNet Comparison

## Overview

This folder contains everything needed to run experiments comparing:
- **Centralized PiLot** — single-device CNN training (no constraints)
- **Distributed PiLot** — pipeline-parallel CNN across N devices (64 MHz, 256 KB RAM/device)
- **Distributed RockNet** — distributed MiniRocket across N devices

### Datasets (5)
| Dataset | Time Series Length | Classes | Train | Test |
|---------|-------------------|---------|-------|------|
| Coffee | 286 | 2 | 28 | 28 |
| Cricket_X | 300 | 12 | 348 | 522 |
| ECG5000 | 140 | 5 | 500 | 4500 |
| ElectricDevices | 96 | 7 | 500 | 2202 |
| FaceAll | 131 | 14 | 560 | 1690 |

### Part 1 — Model Comparison (all 5 datasets)
- Centralized PiLot vs Distributed PiLot (N=7) vs Distributed RockNet (N=7)
- Plots: Accuracy vs Time, Inference Latency (bar), Memory Consumption (bar)

### Part 2 — Scaling Analysis (all 5 datasets)
- Distributed PiLot: N=7 vs N=8 vs N=10
- Plots: Accuracy vs Time, Inference Latency (bar), Memory Consumption (bar with std dev)

---

## Directory Structure

```
experiments/
├── README.md                          # This file
├── generate_all_configs.py            # Generate all config JSONs
├── run_all_experiments.sh             # Master run script
├── extract_results.py                 # Parse logs → CSVs
├── csv_to_tsv.py                      # Convert CSV datasets to TSV (for RockNet)
├── configs/                           # Generated configs (auto-created)
│   ├── part1/
│   │   ├── centralized_Coffee/model_config.json
│   │   ├── distributed_N7_Coffee/model_config.json
│   │   └── ...
│   └── part2/
│       ├── distributed_N7_Cricket_X/model_config.json
│       ├── distributed_N8_Cricket_X/model_config.json
│       └── distributed_N10_Cricket_X/model_config.json
├── results/                           # Raw experiment logs (auto-created)
│   ├── part1/
│   │   ├── centralized_Coffee/pilot_centralized.log
│   │   ├── distributed_N7_Coffee/device_06_tail.log
│   │   └── rocknet_N7_Coffee/rocknet.log
│   └── part2/
│       ├── distributed_N8_Coffee/device_07_tail.log
│       └── distributed_N10_Coffee/device_09_tail.log
├── csv_results/                       # Extracted CSVs for MATLAB (auto-created)
│   ├── part1/
│   │   ├── accuracy_vs_time/
│   │   ├── inference_latency_summary.csv
│   │   └── memory_consumption_summary.csv
│   └── part2/
│       ├── accuracy_vs_time/
│       ├── inference_latency_<dataset>.csv
│       └── memory_consumption_<dataset>.csv
└── matlab/                            # MATLAB plotting scripts
    ├── run_all_plots.m                # Master plot runner
    ├── figures/                       # Output figures (auto-created)
    ├── part1_accuracy_vs_time/
    │   └── plot_accuracy_vs_time.m
    ├── part1_inference_latency/
    │   └── plot_inference_latency.m
    ├── part1_memory_consumption/
    │   └── plot_memory_consumption.m
    ├── part2_accuracy_vs_time/
    │   └── plot_accuracy_vs_time.m
    ├── part2_inference_latency/
    │   └── plot_inference_latency.m
    └── part2_memory_consumption/
        └── plot_memory_consumption.m
```

---

## How to Run (Step by Step)

### Prerequisites
- Linux system (Ubuntu 18.04+ or similar)
- GCC, CMake (for building C code)
- Python 3.7+ with: `numpy`, `pandas`, `sympy`, `jinja2`
- MATLAB (for plotting)

### Step 1: Pull the repo
```bash
git pull origin main
cd /path/to/Final_Results
```

### Step 2: Install Python dependencies
```bash
pip3 install numpy pandas sympy jinja2
```

### Step 3: Set UCR_DATA_ROOT (if datasets are not in Final_Results/)
```bash
# Datasets should be at $UCR_DATA_ROOT/<DatasetName>/<DatasetName>_TRAIN
# Default: parent of experiments/ directory (i.e., Final_Results/)
export UCR_DATA_ROOT="/path/to/Final_Results"
```

### Step 4: Run ALL experiments
```bash
cd experiments
chmod +x run_all_experiments.sh
./run_all_experiments.sh
```

Or run individual parts:
```bash
./run_all_experiments.sh --part1      # Centralized + Distributed PiLot
./run_all_experiments.sh --rocknet    # RockNet Distributed
./run_all_experiments.sh --part2      # Distributed N=8,10 (N=7 reuses Part 1)
```

### Step 5: Extract results to CSV
```bash
python3 extract_results.py
```

### Step 6: Generate MATLAB plots
```matlab
% In MATLAB, navigate to experiments/matlab/
cd('experiments/matlab');
run_all_plots;
```

Or run individual plot scripts:
```matlab
cd('part1_accuracy_vs_time');
plot_accuracy_vs_time;
```

---

## Configuration Details

### Early Stopping
All models use **patience=30**: if test accuracy doesn't improve for 30 consecutive epochs, training stops.

### PiLot Architecture (all configs)
| Layer | Type | Output Shape | Params |
|-------|------|-------------|--------|
| 0 | Conv1D(1→32, k=5, s=1, p=2) + GroupNorm + LeakyReLU | 32×L | 192 |
| 1 | Conv1D(32→48, k=5, s=2, p=2) + GroupNorm + LeakyReLU | 48×L/2 | 7,728 |
| 2 | DualPool(GAP+GMP) → Dropout(0.25) → FC(96→C) | C | 96×C+C |

### Device Configurations
| N | Head | L0 Workers | L1 Workers | Tail | Ch/Device L0 | Ch/Device L1 |
|---|------|-----------|-----------|------|-------------|-------------|
| 7 | 1 | 2 (16 ch) | 3 (16 ch) | 1 | 16 | 16 |
| 8 | 1 | 2 (16 ch) | 4 (12 ch) | 1 | 16 | 12 |
| 10 | 1 | 2 (16 ch) | 6 (8 ch) | 1 | 16 | 8 |

### Constraints
| Parameter | Centralized | Distributed |
|-----------|-------------|-------------|
| Processor | No constraint | 64 MHz (Cortex-M4F) |
| RAM/device | No constraint | 256 KB |
| Flash/device | No constraint | 1 MB |

---

## Plot Descriptions

### Part 1 Plots
1. **Accuracy vs Time** (5 plots, one per dataset): Line plot with 3 curves — Centralized PiLot, Distributed PiLot (N=7), Distributed RockNet (N=7). X-axis: elapsed time (s), Y-axis: test accuracy (%).

2. **Inference Latency** (1 plot): Bar chart with 3 bars per dataset. Centralized PiLot (computation only), Distributed PiLot (stacked: computation + communication), Distributed RockNet (total).

3. **Memory Consumption** (1 plot): Bar chart with 3 bars per dataset. Centralized PiLot (total), Distributed PiLot (avg per device), Distributed RockNet (runtime RSS).

### Part 2 Plots
1. **Accuracy vs Time** (5 plots, one per dataset): Line plot with 3 curves — N=7, N=8, N=10. X-axis: elapsed time (s), Y-axis: test accuracy (%).

2. **Inference Latency** (5 plots, one per dataset): Bar chart with 6 bars — 2 bars per config (computation, communication) for N=7, N=8, N=10.

3. **Memory Consumption** (5 plots, one per dataset): Bar chart with 9 bars + error bars — 3 bars per config (weights avg, optimizer avg, buffer avg) for N=7, N=8, N=10. Error bars show standard deviation across devices.
