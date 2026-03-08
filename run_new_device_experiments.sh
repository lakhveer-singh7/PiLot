#!/bin/bash
# ============================================================
#  run_new_device_experiments.sh — Run 9-dev and 12-dev experiments
#  (6 experiments: 2 configs × 3 datasets)
#  7-dev results already exist in results/<DS>/distributed/
# ============================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIST_DIR="$ROOT/PiLot_Distributed"
RESULTS_DIR="$ROOT/results"
export UCR_DATA_ROOT="${UCR_DATA_ROOT:-$ROOT/datasets}"

DATASETS=("Cricket_X" "ECG5000" "FaceAll")
DEVS=(9 12)

# Build once
echo "=== Building PiLot Distributed ==="
cd "$DIST_DIR"
mkdir -p build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release > /dev/null
cmake --build build -j$(nproc) 2>&1 | tail -3
BIN="$DIST_DIR/build/device"
cd "$ROOT"
echo ""

# Symlink 7-device results
for DS in "${DATASETS[@]}"; do
    TARGET="$RESULTS_DIR/${DS}/distributed_7dev"
    if [[ ! -d "$TARGET" ]]; then
        if [[ -d "$RESULTS_DIR/${DS}/distributed" ]]; then
            ln -sfn "$RESULTS_DIR/${DS}/distributed" "$TARGET"
            echo "Linked: $TARGET → distributed/ (existing 7-dev results)"
        fi
    fi
done
echo ""

EXPERIMENT=0
TOTAL=6

for NDEV in "${DEVS[@]}"; do
    for DS in "${DATASETS[@]}"; do
        EXPERIMENT=$((EXPERIMENT + 1))
        CONFIG="$DIST_DIR/configs/config_${NDEV}dev_${DS}.json"
        LOG_DIR="$RESULTS_DIR/${DS}/distributed_${NDEV}dev"
        
        mkdir -p "$LOG_DIR"
        
        echo "====================================================================="
        echo "  [$EXPERIMENT/$TOTAL]  ${NDEV}-device × $DS"
        echo "  Config : $CONFIG"
        echo "  Logs   : $LOG_DIR"
        echo "  Started: $(date)"
        echo "====================================================================="
        
        # Clean IPC before each experiment
        rm -f /dev/shm/sem.ipc_sem_L* /dev/shm/ipc_tensor_L* /tmp/ipc_early_stop 2>/dev/null || true
        sleep 1
        
        # Extract topology from JSON
        L0_W=$(python3 -c "import json; c=json.load(open('$CONFIG')); print(c['layers'][0]['num_devices'])")
        L1_W=$(python3 -c "import json; c=json.load(open('$CONFIG')); print(c['layers'][1]['num_devices'])")
        NC=$(python3 -c "import json; c=json.load(open('$CONFIG')); print(c['global']['num_classes'])")
        TOTAL_DEV=$((1 + L0_W + L1_W + 1))
        TAIL_ID=$((TOTAL_DEV - 1))
        
        echo "  Devices: $TOTAL_DEV (1 Head + $L0_W L0 + $L1_W L1 + 1 Tail)"
        
        # Launch head
        "$BIN" --config="$CONFIG" --id=0 --role=head --dataset="$DS" \
               --log-dir="$LOG_DIR" -p --mem-limit=262144 &
        HEAD_PID=$!
        sleep 1
        
        # Launch L0 workers
        DID=1
        for ((w=0; w<L0_W; w++)); do
            "$BIN" --config="$CONFIG" --id=$DID --role=worker \
                   --layer-id=0 --worker-id=$w --num-workers=$L0_W \
                   --log-dir="$LOG_DIR" -p --mem-limit=262144 &
            DID=$((DID + 1))
            sleep 0.3
        done
        sleep 0.5
        
        # Launch L1 workers
        for ((w=0; w<L1_W; w++)); do
            "$BIN" --config="$CONFIG" --id=$DID --role=worker \
                   --layer-id=1 --worker-id=$w --num-workers=$L1_W \
                   --log-dir="$LOG_DIR" -p --mem-limit=262144 &
            DID=$((DID + 1))
            sleep 0.3
        done
        sleep 0.5
        
        # Launch tail
        "$BIN" --config="$CONFIG" --id=$DID --role=tail --classes=$NC \
               --log-dir="$LOG_DIR" -p --mem-limit=262144 &
        
        # Wait for head to finish (head drives the pipeline)
        wait $HEAD_PID || true
        sleep 3
        
        # Kill any remaining processes from this experiment
        pkill -P $$ 2>/dev/null || true
        wait 2>/dev/null || true
        sleep 2
        
        # Clean IPC
        rm -f /dev/shm/sem.ipc_sem_L* /dev/shm/ipc_tensor_L* /tmp/ipc_early_stop 2>/dev/null || true
        
        echo "  Finished: $(date)"
        
        # Quick result check
        TAIL_LOG=$(ls "$LOG_DIR"/device_*_tail.log 2>/dev/null | head -1)
        if [[ -n "$TAIL_LOG" ]]; then
            BEST=$(grep "METRICS" "$TAIL_LOG" | grep -oP 'Test_Acc=\K[0-9.]+' | sort -n | tail -1)
            echo "  Best Test Accuracy: ${BEST:-N/A}%"
        fi
        echo ""
    done
done

echo "====================================================================="
echo "  All $TOTAL experiments complete!  $(date)"
echo "====================================================================="
echo "Results:"
for DS in "${DATASETS[@]}"; do
    for NDEV in 7 9 12; do
        DIR="$RESULTS_DIR/${DS}/distributed_${NDEV}dev"
        if [[ -d "$DIR" ]] || [[ -L "$DIR" ]]; then
            TAIL_LOG=$(ls "$DIR"/device_*_tail.log 2>/dev/null | head -1)
            if [[ -n "$TAIL_LOG" ]]; then
                BEST=$(grep "METRICS" "$TAIL_LOG" | grep -oP 'Test_Acc=\K[0-9.]+' | sort -n | tail -1)
                printf "  %-12s %2d-dev: %s%%\n" "$DS" "$NDEV" "${BEST:-N/A}"
            fi
        fi
    done
done
