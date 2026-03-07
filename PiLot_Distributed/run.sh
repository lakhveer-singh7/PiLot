#!/bin/bash
# ============================================================
#  PiLot Distributed — Build & Run
# ============================================================
#  Usage:
#    ./run.sh                                        # defaults (Cricket_X)
#    ./run.sh --config=configs/model_config_ecg5000.json  # different config
#    ./run.sh --epochs=100                           # override epochs
#
#  Pipeline:  1 Head  +  2 L0-workers  +  3 L1-workers  +  1 Tail  = 7 devices
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
BIN="$BUILD_DIR/device"
CONFIG="$SCRIPT_DIR/configs/model_config_cricket_x.json"
DATASET="Cricket_X"
LOG_DIR="$SCRIPT_DIR/logs"
EXTRA_ARGS=()

# ---- Parse optional overrides ----
for arg in "$@"; do
    case "$arg" in
        --config=*) CONFIG="${arg#--config=}" ;;
        --dataset=*) DATASET="${arg#--dataset=}" ;;
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
echo "=== Running PiLot Distributed Pipeline ==="
echo "Config  : $CONFIG"
echo "Dataset : $DATASET"
echo "Log dir : $LOG_DIR"
echo "Architecture: 7 devices (1 Head + 2 L0 + 3 L1 + 1 Tail)"
echo "==========================================="
echo ""

# ---- Launch devices ----
echo "[0] Head"
"$BIN" --config="$CONFIG" --id=0 --role=head --dataset="$DATASET" \
       --log-dir="$LOG_DIR" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" &
HEAD_PID=$!
sleep 1

echo "[1] Layer-0 Worker 0/2"
"$BIN" --config="$CONFIG" --id=1 --role=worker --layer-id=0 --worker-id=0 \
       --num-workers=2 --log-dir="$LOG_DIR" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" &
sleep 0.3

echo "[2] Layer-0 Worker 1/2"
"$BIN" --config="$CONFIG" --id=2 --role=worker --layer-id=0 --worker-id=1 \
       --num-workers=2 --log-dir="$LOG_DIR" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" &
sleep 0.5

echo "[3] Layer-1 Worker 0/3"
"$BIN" --config="$CONFIG" --id=3 --role=worker --layer-id=1 --worker-id=0 \
       --num-workers=3 --log-dir="$LOG_DIR" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" &
sleep 0.3

echo "[4] Layer-1 Worker 1/3"
"$BIN" --config="$CONFIG" --id=4 --role=worker --layer-id=1 --worker-id=1 \
       --num-workers=3 --log-dir="$LOG_DIR" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" &
sleep 0.3

echo "[5] Layer-1 Worker 2/3"
"$BIN" --config="$CONFIG" --id=5 --role=worker --layer-id=1 --worker-id=2 \
       --num-workers=3 --log-dir="$LOG_DIR" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" &
sleep 0.5

echo "[6] Tail"
"$BIN" --config="$CONFIG" --id=6 --role=tail --classes=12 \
       --log-dir="$LOG_DIR" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" &
TAIL_PID=$!
sleep 1

echo ""
echo "All 7 devices launched.  Logs → $LOG_DIR/"
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
echo "  grep 'Accuracy\|Loss\|EPOCH' $LOG_DIR/device_06_tail.log"
