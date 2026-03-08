#!/bin/bash
# ============================================================
#  run_all_multi_device.sh — Run all 9 distributed experiments
#  (3 device configs × 3 datasets)
# ============================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIST_DIR="$ROOT/PiLot_Distributed"
RUN_SCRIPT="$DIST_DIR/run_generic.sh"
RESULTS_DIR="$ROOT/results"
COMMON_ARGS="-p --mem-limit=262144"

chmod +x "$RUN_SCRIPT"

DATASETS=("Cricket_X" "ECG5000" "FaceAll")
DEVS=(7 9 12)

echo "=== PiLot Multi-Device Experiment Runner ==="
echo "Configs : 7, 9, 12 devices"
echo "Datasets: Cricket_X, ECG5000, FaceAll"
echo "Total   : 9 experiments"
echo "============================================="
echo ""

EXPERIMENT=0
TOTAL=9

for NDEV in "${DEVS[@]}"; do
    for DS in "${DATASETS[@]}"; do
        EXPERIMENT=$((EXPERIMENT + 1))
        CONFIG="$DIST_DIR/configs/config_${NDEV}dev_${DS}.json"
        LOG_DIR="$RESULTS_DIR/${DS}/distributed_${NDEV}dev"
        
        mkdir -p "$LOG_DIR"
        
        echo ""
        echo "====================================================================="
        echo "  [$EXPERIMENT/$TOTAL]  ${NDEV}-device × $DS"
        echo "  Config : $CONFIG"
        echo "  Logs   : $LOG_DIR"
        echo "  Started: $(date)"
        echo "====================================================================="
        
        # Clean IPC before each experiment
        rm -f /dev/shm/sem.ipc_sem_L* /dev/shm/ipc_tensor_L* 2>/dev/null || true
        rm -f /tmp/ipc_early_stop 2>/dev/null || true
        
        # Run — redirect output to a run log
        cd "$DIST_DIR"
        bash run_generic.sh \
            --config="$CONFIG" \
            --dataset="$DS" \
            --log-dir="$LOG_DIR" \
            $COMMON_ARGS \
            > "$LOG_DIR/run_output.log" 2>&1 || {
                echo "  WARNING: Experiment $DS ${NDEV}dev may have failed (exit code $?)"
            }
        cd "$ROOT"
        
        echo "  Finished: $(date)"
        
        # Quick result check
        TAIL_LOG=$(ls "$LOG_DIR"/device_*_tail.log 2>/dev/null | head -1)
        if [[ -n "$TAIL_LOG" ]]; then
            BEST=$(grep "METRICS" "$TAIL_LOG" | grep -oP 'Test_Acc=\K[0-9.]+' | sort -n | tail -1)
            echo "  Best Test Accuracy: ${BEST:-N/A}%"
        fi
    done
done

echo ""
echo "====================================================================="
echo "  All $TOTAL experiments complete!  $(date)"
echo "====================================================================="
echo ""
echo "Results in: $RESULTS_DIR/"
ls -d "$RESULTS_DIR"/*/distributed_*dev 2>/dev/null | sed 's/^/  /'
