# PiLot — Experiment Setup

## Overview
Compares **Centralized** vs **Distributed** PiLot CNN on 3 UCR time-series datasets.

| Setting | Centralized | Distributed |
|---------|-------------|-------------|
| Devices | 1 (single process) | 7 (1 Head + 2 L0 + 3 L1 + 1 Tail) |
| Conv Layers | 2 | 2 |
| Processor | No constraint | 64 MHz (Cortex-M4 simulation) |
| RAM/device | No constraint | 256 KB |
| Early Stop | 30 epochs patience | 30 epochs patience |
| Max Epochs | 300 | 300 |

## Datasets

| Dataset | Length | Classes |
|---------|--------|---------|
| Cricket_X | 300 | 12 |
| ECG5000 | 140 | 5 |
| FaceAll | 131 | 14 |

## Quick Start (on the target device)

### 1. Clone & setup
```bash
git clone https://github.com/SSKR-collab/PiLot.git
cd PiLot
```

### 2. Place datasets
Datasets should be in the **parent directory** of PiLot/ with this structure:
```
<parent>/
  Cricket_X/Cricket_X_TRAIN
  Cricket_X/Cricket_X_TEST
  ECG5000/ECG5000_TRAIN
  ECG5000/ECG5000_TEST
  FaceAll/FaceAll_TRAIN
  FaceAll/FaceAll_TEST
  PiLot/           ← this repo
```

Or set `UCR_DATA_ROOT` to point to the directory containing dataset folders:
```bash
export UCR_DATA_ROOT=/path/to/datasets
```

### 3. Run all experiments
```bash
# Run everything (Centralized + Distributed × 3 datasets)
bash run_all.sh
```

Or run individually:
```bash
# Single dataset, single model
bash run_centralized.sh Cricket_X
bash run_distributed.sh ECG5000
```

### 4. Parse results
```bash
python3 parse_results.py
```
This creates CSV files in `results/csv/` for MATLAB plotting.

### 5. Generate plots (MATLAB)
```matlab
cd plotting
plot_all
```
Or from command line:
```bash
matlab -nodisplay -r "cd('plotting'); plot_all; exit"
```

## Output

### Plots generated (in `results/`):
- **accuracy_vs_time_Cricket_X.png** — Accuracy vs Time comparison
- **accuracy_vs_time_ECG5000.png** — Accuracy vs Time comparison
- **accuracy_vs_time_FaceAll.png** — Accuracy vs Time comparison
- **inference_latency.png** — Bar chart: computation (centralized) vs computation+communication (distributed)
- **memory_consumption.png** — Bar chart: single device (centralized) vs avg/device (distributed)

### Log files (in `results/<dataset>/<model>/`):
- Centralized: `pilot_centralized.log`
- Distributed: `device_00_head.log`, `device_01_worker_L0_W0.log`, ..., `device_06_tail.log`

## Architecture
```
Conv Layer 0: Conv1D(1→32, k=5, s=1, p=2) → GroupNorm(8) → ReLU
Conv Layer 1: Conv1D(32→48, k=5, s=2, p=2) → GroupNorm(8) → ReLU
Classifier:   DualPool(GAP+GMP) → Dropout(0.2) → FC(96→classes)

Distributed device mapping (7 devices):
  Device 0: Head (data feeder)
  Device 1-2: Layer 0 workers (16 channels each)
  Device 3-5: Layer 1 workers (16 channels each)
  Device 6: Tail (classifier)
```

## Requirements
- GCC/Clang with C11 support
- CMake ≥ 3.20
- Python 3 (for config generation and log parsing)
- MATLAB (for plotting)
