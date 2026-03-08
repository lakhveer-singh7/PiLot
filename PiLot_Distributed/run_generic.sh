#!/bin/bash
# ============================================================
#  PiLot Distributed — Generic Build & Run
#  Reads device topology from model_config.json automatically.
# ============================================================
#  Usage:
#    ./run.sh                                        # defaults
#    ./run.sh --config=configs/config_9dev.json      # 9-device config
#    ./run.sh --dataset=ECG5000 --config=...         # override dataset
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$SCRIPT_DIR/build"
BIN="$BUILD_DIR/device"
CONFIG="$SCRIPT_DIR/configs/model_config.json"
DATASET=""
LOG_DIR="$SCRIPT_DIR/logs"
EXTRA_ARGS=()

# Set UCR data root (datasets/ inside the repo)
export UCR_DATA_ROOT="${UCR_DATA_ROOT:-$REPO_ROOT/datasets}"

# ---- Parse optional overrides ----
for arg in "$@"; do
    case "$arg" in
        --config=*) CONFIG="${arg#--config=}" ;;
        --dataset=*) DATASET="${arg#--dataset=}" ;;
        --log-dir=*) LOG_DIR="${arg#--log-dir=}" ;;
        *)          EXTRA_ARGS+=("$arg") ;;
    esac
done

# ---- Build ----
echo "=== Building PiLot Distributed ==="
mkdir -p "$BUILD_DIR"
cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Debug > /dev/null
cmake --build "$BUILD_DIR" -j"$(nproc)" 2>&1 | tail -5
echo ""

if [[ ! -x "$BIN" ]]; then
    echo "ERROR: Build failed — $BIN not found"
    exit 1
fi

# ---- Extract topology from JSON config ----
# Read num_devices for each conv layer + num_classes from config
L0_WORKERS=$(python3 -c "
import json, sys
with open('$CONFIG') as f: cfg = json.load(f)
layers = cfg['layers']
print(layers[0]['num_devices'])
")
L1_WORKERS=$(python3 -c "
import json, sys
with open('$CONFIG') as f: cfg = json.load(f)
layers = cfg['layers']
print(layers[1]['num_devices'])
")
NUM_CLASSES=$(python3 -c "
import json, sys
with open('$CONFIG') as f: cfg = json.load(f)
print(cfg['global']['num_classes'])
")
DS_FROM_CFG=$(python3 -c "
import json, sys
with open('$CONFIG') as f: cfg = json.load(f)
print(cfg['global'].get('dataset', 'Cricket_X'))
")

TOTAL_WORKERS=$((L0_WORKERS + L1_WORKERS))
TOTAL_DEVICES=$((TOTAL_WORKERS + 2))  # + Head + Tail
TAIL_ID=$((TOTAL_DEVICES - 1))

# Use dataset from CLI if provided, else from config
if [[ -z "$DATASET" ]]; then
    DATASET="$DS_FROM_CFG"
fi

# ---- Cleanup shared memory from prior runs ----
cleanup() {
    echo ""
    echo "Cleaning up IPC resources..."
    rm -f /dev/shm/sem.ipc_sem_L* /dev/shm/ipc_tensor_L* 2>/dev/null
    rm -f /tmp/ipc_early_stop 2>/dev/null
    pkill -P $$ 2>/dev/null || true
    wait 2>/dev/null || true
    echo "Cleanup complete."
}
trap cleanup EXIT INT TERM
cleanup > /dev/null 2>&1

# ---- Create log directory ----
mkdir -p "$LOG_DIR"

# ---- Print banner ----
echo "=== Running PiLot Distributed Pipeline ==="
echo "Config  : $CONFIG"
echo "Dataset : $DATASET"
echo "Classes : $NUM_CLASSES"
echo "Log dir : $LOG_DIR"
echo "Architecture: $TOTAL_DEVICES devices (1 Head + $L0_WORKERS L0 + $L1_WORKERS L1 + 1 Tail)"
echo "==========================================="
echo ""

# ---- Launch devices ----
DEVICE_ID=0

# Head
echo "[$DEVICE_ID] Head"
"$BIN" --config="$CONFIG" --id=$DEVICE_ID --role=head --dataset="$DATASET" \
       --log-dir="$LOG_DIR" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" &
HEAD_PID=$!
DEVICE_ID=$((DEVICE_ID + 1))
sleep 1

# Layer 0 workers
for ((w=0; w<L0_WORKERS; w++)); do
    echo "[$DEVICE_ID] Layer-0 Worker $w/$L0_WORKERS"
    "$BIN" --config="$CONFIG" --id=$DEVICE_ID --role=worker \
           --layer-id=0 --worker-id=$w --num-workers=$L0_WORKERS \
           --log-dir="$LOG_DIR" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" &
    DEVICE_ID=$((DEVICE_ID + 1))
    sleep 0.3
done
sleep 0.5

# Layer 1 workers
for ((w=0; w<L1_WORKERS; w++)); do
    echo "[$DEVICE_ID] Layer-1 Worker $w/$L1_WORKERS"
    "$BIN" --config="$CONFIG" --id=$DEVICE_ID --role=worker \
           --layer-id=1 --worker-id=$w --num-workers=$L1_WORKERS \
           --log-dir="$LOG_DIR" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" &
    DEVICE_ID=$((DEVICE_ID + 1))
    sleep 0.3
done
sleep 0.5

# Tail
echo "[$DEVICE_ID] Tail"
"$BIN" --config="$CONFIG" --id=$DEVICE_ID --role=tail --classes=$NUM_CLASSES \
       --log-dir="$LOG_DIR" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" &
TAIL_PID=$!
sleep 1

echo ""
echo "All $TOTAL_DEVICES devices launched.  Logs → $LOG_DIR/"
echo "Tailing head log (Ctrl+C to stop)..."
echo "==========================================="
echo ""

# ---- Monitor ----
tail -f "$LOG_DIR/device_00_head.log" 2>/dev/null &
TAIL_LOG_PID=$!

wait $HEAD_PID
HEAD_EXIT=$?
kill $TAIL_LOG_PID 2>/dev/null || true

echo ""
echo "=== Head exited (code $HEAD_EXIT) — waiting for pipeline to drain ==="
wait 2>/dev/null || true

echo ""
echo "=== Done ==="
echo "Logs:"
ls -1 "$LOG_DIR/"*.log 2>/dev/null | sed 's/^/  /'
echo ""
echo "Quick check:"
echo "  grep 'Accuracy\|Loss\|EPOCH' $LOG_DIR/device_*_tail.log"
