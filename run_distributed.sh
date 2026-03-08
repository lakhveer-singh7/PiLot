#!/bin/bash
# ============================================================
#  Run PiLot Distributed for a specific dataset
# ============================================================
#  Usage:
#    ./run_distributed.sh Cricket_X
#    ./run_distributed.sh ECG5000
#    ./run_distributed.sh FaceAll
#
#  Pipeline: 1 Head + 2 L0-workers + 3 L1-workers + 1 Tail = 7 devices
#  Constraints: 256 KB RAM/device, 64 MHz processor simulation
# ============================================================

set -euo pipefail

DATASET="${1:?Usage: $0 <dataset_name> (Cricket_X | ECG5000 | FaceAll)}"

# Dataset → num_classes mapping
declare -A DS_CLASSES=( ["Cricket_X"]=12 ["ECG5000"]=5 ["FaceAll"]=14 )
NUM_CLASSES="${DS_CLASSES[$DATASET]:-}"
if [[ -z "$NUM_CLASSES" ]]; then
    echo "ERROR: Unknown dataset '$DATASET'. Must be: Cricket_X, ECG5000, or FaceAll"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/PiLot_Distributed"
BUILD_DIR="$PROJECT_DIR/build"
BIN="$BUILD_DIR/device"
CONFIG="$PROJECT_DIR/configs/config_${DATASET}.json"
LOG_DIR="$SCRIPT_DIR/results/${DATASET}/distributed"

# Set UCR data root (datasets/ inside the repo)
export UCR_DATA_ROOT="${UCR_DATA_ROOT:-$SCRIPT_DIR/datasets}"

# Validate config exists
if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: Config not found: $CONFIG"
    echo "Run: python3 generate_all_configs.py first"
    exit 1
fi

# ---- Build ----
echo "=== Building PiLot Distributed ==="
mkdir -p "$BUILD_DIR"
cmake -S "$PROJECT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release > /dev/null
cmake --build "$BUILD_DIR" -j"$(nproc)" 2>&1 | tail -5
echo ""

if [[ ! -x "$BIN" ]]; then
    echo "ERROR: Build failed — $BIN not found"
    exit 1
fi

# ---- Cleanup shared memory from prior runs ----
cleanup() {
    echo ""
    echo "Cleaning up IPC resources..."
    rm -f /dev/shm/sem.ipc_sem_L* /dev/shm/ipc_tensor_L* 2>/dev/null
    pkill -P $$ 2>/dev/null || true
    wait 2>/dev/null || true
    echo "Cleanup complete."
}
trap cleanup EXIT INT TERM
cleanup > /dev/null 2>&1

# ---- Create log directory ----
mkdir -p "$LOG_DIR"

# ---- Print banner ----
echo "=== Running PiLot Distributed on $DATASET ==="
echo "Config  : $CONFIG"
echo "Dataset : $DATASET ($NUM_CLASSES classes)"
echo "Data    : $UCR_DATA_ROOT/$DATASET/"
echo "Log dir : $LOG_DIR"
echo "Devices : 7 (1 Head + 2 L0 + 3 L1 + 1 Tail)"
echo "Constraints: 256 KB RAM/device, 64 MHz processor"
echo "==========================================="
echo ""

# Common args: processor constraint enabled (-p), memory limit 256KB
COMMON_ARGS="-p --mem-limit=262144"

# ---- Launch devices ----
echo "[0] Head"
"$BIN" --config="$CONFIG" --id=0 --role=head --dataset="$DATASET" \
       --log-dir="$LOG_DIR" $COMMON_ARGS &
HEAD_PID=$!
sleep 1

echo "[1] Layer-0 Worker 0/2"
"$BIN" --config="$CONFIG" --id=1 --role=worker --layer-id=0 --worker-id=0 \
       --num-workers=2 --log-dir="$LOG_DIR" $COMMON_ARGS &
sleep 0.3

echo "[2] Layer-0 Worker 1/2"
"$BIN" --config="$CONFIG" --id=2 --role=worker --layer-id=0 --worker-id=1 \
       --num-workers=2 --log-dir="$LOG_DIR" $COMMON_ARGS &
sleep 0.5

echo "[3] Layer-1 Worker 0/3"
"$BIN" --config="$CONFIG" --id=3 --role=worker --layer-id=1 --worker-id=0 \
       --num-workers=3 --log-dir="$LOG_DIR" $COMMON_ARGS &
sleep 0.3

echo "[4] Layer-1 Worker 1/3"
"$BIN" --config="$CONFIG" --id=4 --role=worker --layer-id=1 --worker-id=1 \
       --num-workers=3 --log-dir="$LOG_DIR" $COMMON_ARGS &
sleep 0.3

echo "[5] Layer-1 Worker 2/3"
"$BIN" --config="$CONFIG" --id=5 --role=worker --layer-id=1 --worker-id=2 \
       --num-workers=3 --log-dir="$LOG_DIR" $COMMON_ARGS &
sleep 0.5

echo "[6] Tail"
"$BIN" --config="$CONFIG" --id=6 --role=tail --classes="$NUM_CLASSES" \
       --log-dir="$LOG_DIR" $COMMON_ARGS &
TAIL_PID=$!
sleep 1

echo ""
echo "All 7 devices launched. Logs → $LOG_DIR/"
echo "Tailing tail log (Ctrl+C to stop)..."
echo "==========================================="
echo ""

# ---- Monitor ----
tail -f "$LOG_DIR/device_06_tail.log" 2>/dev/null &
TAIL_LOG_PID=$!

wait $HEAD_PID
HEAD_EXIT=$?
kill $TAIL_LOG_PID 2>/dev/null || true

echo ""
echo "=== Head exited (code $HEAD_EXIT) — waiting for pipeline to drain ==="
wait 2>/dev/null || true

echo ""
echo "=== Distributed $DATASET Complete ==="
echo "Logs:"
ls -1 "$LOG_DIR/"*.log 2>/dev/null | sed 's/^/  /'
