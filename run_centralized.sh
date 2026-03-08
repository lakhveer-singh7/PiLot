#!/bin/bash
# ============================================================
#  Run PiLot Centralized for a specific dataset
# ============================================================
#  Usage:
#    ./run_centralized.sh Cricket_X
#    ./run_centralized.sh ECG5000
#    ./run_centralized.sh FaceAll
# ============================================================

set -euo pipefail

DATASET="${1:?Usage: $0 <dataset_name> (Cricket_X | ECG5000 | FaceAll)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/PiLot_Centralized"
BUILD_DIR="$PROJECT_DIR/build"
BIN="$BUILD_DIR/pilot_centralized"
CONFIG="$PROJECT_DIR/configs/config_${DATASET}.json"
LOG_DIR="$SCRIPT_DIR/results/${DATASET}/centralized"

# Validate config exists
if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: Config not found: $CONFIG"
    echo "Run: python3 generate_all_configs.py first"
    exit 1
fi

# Set UCR data root (datasets/ inside the repo)
export UCR_DATA_ROOT="${UCR_DATA_ROOT:-$SCRIPT_DIR/datasets}"

# ---- Build ----
echo "=== Building PiLot Centralized ==="
mkdir -p "$BUILD_DIR"
cmake -S "$PROJECT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release > /dev/null
cmake --build "$BUILD_DIR" -j"$(nproc)" 2>&1 | tail -5
echo ""

if [[ ! -x "$BIN" ]]; then
    echo "ERROR: Build failed — $BIN not found"
    exit 1
fi

# ---- Run ----
mkdir -p "$LOG_DIR"
echo "=== Running PiLot Centralized on $DATASET ==="
echo "Config : $CONFIG"
echo "Data   : $UCR_DATA_ROOT/$DATASET/"
echo "Log dir: $LOG_DIR"
echo "==========================================="

"$BIN" --config="$CONFIG" --dataset="$DATASET" --log-dir="$LOG_DIR"

echo ""
echo "=== Centralized $DATASET Complete ==="
echo "Logs: $LOG_DIR/pilot_centralized.log"
