#!/bin/bash
# ============================================================
#  PiLot Centralized — Build & Run
# ============================================================
#  Usage:
#    ./run.sh                            # defaults
#    ./run.sh --dataset=ECG5000          # override dataset
#    ./run.sh --epochs=100 --debug       # extra flags
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
BIN="$BUILD_DIR/pilot_centralized"
CONFIG="$SCRIPT_DIR/configs/model_config.json"
LOG_DIR="$SCRIPT_DIR/logs"

# ---- Build ----
echo "=== Building PiLot Centralized ==="
mkdir -p "$BUILD_DIR"
cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Debug > /dev/null
cmake --build "$BUILD_DIR" -j"$(nproc)" 2>&1 | tail -5
echo ""

if [[ ! -x "$BIN" ]]; then
    echo "ERROR: Build failed — $BIN not found"
    exit 1
fi

# ---- Run ----
echo "=== Running PiLot Centralized ==="
echo "Config : $CONFIG"
echo "Log dir: $LOG_DIR"
echo "Extra  : $*"
echo "==========================================="
echo ""

"$BIN" --config="$CONFIG" --log-dir="$LOG_DIR" "$@"

echo ""
echo "=== Done ==="
echo "Logs: $LOG_DIR/pilot_centralized.log"
