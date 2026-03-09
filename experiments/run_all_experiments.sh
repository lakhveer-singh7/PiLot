#!/bin/bash
# ============================================================
#  Master Run Script — All Experiments
# ============================================================
#  This script runs ALL experiments for Part 1 and Part 2.
#  It generates configs, builds binaries, and runs training.
#
#  Usage:
#    ./run_all_experiments.sh              # Run everything
#    ./run_all_experiments.sh --part1      # Only Part 1
#    ./run_all_experiments.sh --part2      # Only Part 2
#    ./run_all_experiments.sh --rocknet    # Only RockNet
#
#  Environment:
#    UCR_DATA_ROOT  — path to folder containing dataset subfolders
#                     (default: parent of this experiments/ directory)
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PILOT_DIR="$(dirname "$SCRIPT_DIR")"          # PiLot/ (parent of experiments/)
ROOT_DIR="$(dirname "$PILOT_DIR")"            # Final_Results/ (parent of PiLot/)
ROCKNET_DIR="$ROOT_DIR/RockNet-main"
RESULTS_DIR="$SCRIPT_DIR/results"
CONFIGS_DIR="$SCRIPT_DIR/configs"

# UCR dataset root — datasets are at $UCR_DATA_ROOT/<DatasetName>/<DatasetName>_TRAIN
export UCR_DATA_ROOT="${UCR_DATA_ROOT:-$ROOT_DIR}"

DATASETS=("Coffee" "Cricket_X" "ECG5000" "ElectricDevices" "FaceAll")

RUN_PART1=true
RUN_PART2=true
RUN_ROCKNET=true

for arg in "$@"; do
    case "$arg" in
        --part1)   RUN_PART1=true; RUN_PART2=false; RUN_ROCKNET=false ;;
        --part2)   RUN_PART1=false; RUN_PART2=true; RUN_ROCKNET=false ;;
        --rocknet) RUN_PART1=false; RUN_PART2=false; RUN_ROCKNET=true ;;
        --all)     RUN_PART1=true; RUN_PART2=true; RUN_ROCKNET=true ;;
    esac
done

echo "============================================================"
echo "  Master Experiment Runner"
echo "============================================================"
echo "  UCR_DATA_ROOT: $UCR_DATA_ROOT"
echo "  PILOT_DIR:     $PILOT_DIR"
echo "  ROCKNET_DIR:   $ROCKNET_DIR"
echo "  RESULTS_DIR:   $RESULTS_DIR"
echo "  Run Part 1:    $RUN_PART1"
echo "  Run Part 2:    $RUN_PART2"
echo "  Run RockNet:   $RUN_ROCKNET"
echo "============================================================"
echo ""

# ---- Step 0: Generate all configs ----
echo "=== Generating all configs ==="
cd "$SCRIPT_DIR"
python3 generate_all_configs.py
echo ""

# ---- Step 1: Build PiLot Centralized ----
echo "=== Building PiLot Centralized ==="
cd "$PILOT_DIR/PiLot_Centralized"
mkdir -p build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release > /dev/null 2>&1
cmake --build build -j"$(nproc)" 2>&1 | tail -3
CENT_BIN="$PILOT_DIR/PiLot_Centralized/build/pilot_centralized"
echo ""

# ---- Step 2: Build PiLot Distributed ----
echo "=== Building PiLot Distributed ==="
cd "$PILOT_DIR/PiLot_Distributed"
mkdir -p build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release > /dev/null 2>&1
cmake --build build -j"$(nproc)" 2>&1 | tail -3
DIST_BIN="$PILOT_DIR/PiLot_Distributed/build/device"
echo ""

# ============================================================
#  Helper: Run PiLot Centralized
# ============================================================
run_centralized() {
    local dataset="$1"
    local config="$2"
    local log_dir="$3"

    mkdir -p "$log_dir"
    echo "  Running Centralized PiLot on $dataset..."
    echo "  Config: $config"
    echo "  Logs:   $log_dir"

    "$CENT_BIN" --config="$config" --log-dir="$log_dir" --dataset="$dataset" \
        2>&1 | tail -5

    echo "  Done: $log_dir/pilot_centralized.log"
    echo ""
}

# ============================================================
#  Helper: Run PiLot Distributed (dynamic N devices)
# ============================================================
run_distributed() {
    local dataset="$1"
    local config="$2"
    local log_dir="$3"
    local n_total="$4"  # Total number of devices (Head + workers + Tail)

    mkdir -p "$log_dir"

    # Read layer configuration from JSON
    local l0_ndev=$(python3 -c "import json; c=json.load(open('$config')); print(c['layers'][0]['num_devices'])")
    local l1_ndev=$(python3 -c "import json; c=json.load(open('$config')); print(c['layers'][1]['num_devices'])")

    echo "  Running Distributed PiLot (N=$n_total) on $dataset..."
    echo "  Config: $config"
    echo "  L0 workers: $l0_ndev, L1 workers: $l1_ndev, FC: 1 tail, Head: 1"
    echo "  Logs:   $log_dir"

    # Cleanup IPC from prior runs
    rm -f /dev/shm/sem.ipc_sem_L* /dev/shm/ipc_tensor_L* 2>/dev/null || true

    local pids=()
    local device_id=0

    # Head (device 0)
    "$DIST_BIN" --config="$config" --id=$device_id --role=head \
        --dataset="$dataset" --log-dir="$log_dir" -p &
    pids+=($!)
    device_id=$((device_id + 1))
    sleep 1

    # Layer-0 workers
    for ((w=0; w<l0_ndev; w++)); do
        "$DIST_BIN" --config="$config" --id=$device_id --role=worker \
            --layer-id=0 --worker-id=$w --num-workers=$l0_ndev \
            --log-dir="$log_dir" -p &
        pids+=($!)
        device_id=$((device_id + 1))
        sleep 0.3
    done
    sleep 0.5

    # Layer-1 workers
    for ((w=0; w<l1_ndev; w++)); do
        "$DIST_BIN" --config="$config" --id=$device_id --role=worker \
            --layer-id=1 --worker-id=$w --num-workers=$l1_ndev \
            --log-dir="$log_dir" -p &
        pids+=($!)
        device_id=$((device_id + 1))
        sleep 0.3
    done
    sleep 0.5

    # Tail (last device)
    "$DIST_BIN" --config="$config" --id=$device_id --role=tail \
        --log-dir="$log_dir" -p &
    pids+=($!)

    echo "  Launched $((device_id + 1)) devices. Waiting..."

    # Wait for head to finish (it controls the epoch loop)
    wait "${pids[0]}" 2>/dev/null || true

    # Kill remaining processes
    for pid in "${pids[@]:1}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true

    # Cleanup IPC
    rm -f /dev/shm/sem.ipc_sem_L* /dev/shm/ipc_tensor_L* 2>/dev/null || true

    echo "  Done: $log_dir/"
    echo ""
}

# ============================================================
#  PART 1: Centralized + Distributed (N=7) for all 5 datasets
#          + RockNet Distributed (N=7)
# ============================================================
if $RUN_PART1; then
    echo ""
    echo "################################################################"
    echo "  PART 1: Centralized PiLot + Distributed PiLot (N=7)"
    echo "################################################################"
    echo ""

    for ds in "${DATASETS[@]}"; do
        echo "-------- Dataset: $ds --------"

        # Centralized PiLot
        run_centralized "$ds" \
            "$CONFIGS_DIR/part1/centralized_${ds}/model_config.json" \
            "$RESULTS_DIR/part1/centralized_${ds}"

        # Distributed PiLot N=7
        run_distributed "$ds" \
            "$CONFIGS_DIR/part1/distributed_N7_${ds}/model_config.json" \
            "$RESULTS_DIR/part1/distributed_N7_${ds}" \
            7
    done
fi

# ============================================================
#  RockNet Distributed (N=7) for all 5 datasets
# ============================================================
if $RUN_ROCKNET; then
    echo ""
    echo "################################################################"
    echo "  RockNet Distributed (N=7) for all 5 datasets"
    echo "################################################################"
    echo ""

    ROCKNET_SIM_DIR="$ROCKNET_DIR/c_src/distributed_sim"

    for ds in "${DATASETS[@]}"; do
        echo "-------- RockNet: $ds --------"

        # Step 1: Prepare TSV data for RockNet (it expects TSV in ~/datasets/)
        ROCKNET_DS_DIR="$HOME/datasets/$ds"
        mkdir -p "$ROCKNET_DS_DIR"

        # Convert CSV to TSV (RockNet expects tab-separated: label\tval1\tval2\t...)
        python3 "$SCRIPT_DIR/csv_to_tsv.py" \
            "$UCR_DATA_ROOT/$ds/${ds}_TRAIN" \
            "$ROCKNET_DS_DIR/${ds}_TRAIN.tsv"
        python3 "$SCRIPT_DIR/csv_to_tsv.py" \
            "$UCR_DATA_ROOT/$ds/${ds}_TEST" \
            "$ROCKNET_DS_DIR/${ds}_TEST.tsv"

        # Step 2: Generate RockNet config for this dataset (N=7)
        cd "$ROCKNET_DIR"
        python3 generate_distributed_config.py "$ds" 7

        # Step 3: Build
        cd "$ROCKNET_SIM_DIR"
        gcc -O2 -o "distributed_rocknet_${ds}" \
            main.c conv.c linear_classifier.c dynamic_tree_quantization.c \
            -lm -lrt -lpthread

        # Step 4: Run and capture output
        ROCKNET_LOG="$RESULTS_DIR/part1/rocknet_N7_${ds}/rocknet.log"
        mkdir -p "$(dirname "$ROCKNET_LOG")"
        echo "  Running RockNet Distributed (N=7) on $ds..."
        "./distributed_rocknet_${ds}" 2>&1 | tee "$ROCKNET_LOG"
        echo "  Done: $ROCKNET_LOG"
        echo ""
    done
fi

# ============================================================
#  PART 2: Distributed PiLot N=8, N=10 for all 5 datasets
#          (N=7 reuses Part 1 results)
# ============================================================
if $RUN_PART2; then
    echo ""
    echo "################################################################"
    echo "  PART 2: Distributed PiLot N=8 and N=10 for all 5 datasets"
    echo "################################################################"
    echo ""

    for ds in "${DATASETS[@]}"; do
        echo "-------- Dataset: $ds (N=8) --------"

        # Generate config for N=8 for this dataset
        N8_CONFIG="$CONFIGS_DIR/part2/distributed_N8_${ds}/model_config.json"
        N8_L1_NDEV=$(python3 -c "import json; c=json.load(open('$N8_CONFIG')); print(c['layers'][1]['num_devices'])")
        N8_TOTAL=$((1 + 2 + N8_L1_NDEV + 1))

        run_distributed "$ds" \
            "$N8_CONFIG" \
            "$RESULTS_DIR/part2/distributed_N8_${ds}" \
            "$N8_TOTAL"

        echo "-------- Dataset: $ds (N=10) --------"

        N10_CONFIG="$CONFIGS_DIR/part2/distributed_N10_${ds}/model_config.json"
        N10_L1_NDEV=$(python3 -c "import json; c=json.load(open('$N10_CONFIG')); print(c['layers'][1]['num_devices'])")
        N10_TOTAL=$((1 + 2 + N10_L1_NDEV + 1))

        run_distributed "$ds" \
            "$N10_CONFIG" \
            "$RESULTS_DIR/part2/distributed_N10_${ds}" \
            "$N10_TOTAL"
    done
fi

echo ""
echo "============================================================"
echo "  All experiments complete!"
echo "  Results: $RESULTS_DIR/"
echo ""
echo "  Next: Run extract_results.py to generate CSVs for MATLAB"
echo "    cd $SCRIPT_DIR && python3 extract_results.py"
echo "============================================================"
