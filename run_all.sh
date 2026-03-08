#!/bin/bash
# ============================================================
#  PiLot — Master Run Script
#  Runs ALL experiments: Centralized & Distributed × 3 Datasets
# ============================================================
#
#  Usage:
#    ./run_all.sh
#
#  Prerequisites:
#    - CMake, GCC/Clang installed
#    - Datasets in parent directory (or set UCR_DATA_ROOT)
#    - Structure: <UCR_DATA_ROOT>/<DatasetName>/<DatasetName>_TRAIN
#
#  After completion:
#    python3 parse_results.py       # Extract metrics to CSV
#    cd plotting && matlab -r plot_all  # Generate plots
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Ensure UCR_DATA_ROOT is set (default: datasets/ inside the repo)
export UCR_DATA_ROOT="${UCR_DATA_ROOT:-$SCRIPT_DIR/datasets}"

DATASETS=("Cricket_X" "ECG5000" "FaceAll")

echo "=============================================="
echo "  PiLot — Full Experiment Suite"
echo "=============================================="
echo "  Data Root  : $UCR_DATA_ROOT"
echo "  Datasets   : ${DATASETS[*]}"
echo "  Models     : Centralized, Distributed (7 devices)"
echo "  Constraints: Distributed: 256KB RAM/device, 64MHz"
echo "             : Centralized: No constraints"
echo "  Early Stop : 30 epochs patience"
echo "  Max Epochs : 300"
echo "=============================================="
echo ""

# Verify datasets exist
for DS in "${DATASETS[@]}"; do
    TRAIN_FILE="$UCR_DATA_ROOT/$DS/${DS}_TRAIN"
    if [[ ! -f "$TRAIN_FILE" ]]; then
        echo "ERROR: Dataset file not found: $TRAIN_FILE"
        echo "Set UCR_DATA_ROOT to the directory containing dataset folders."
        exit 1
    fi
done
echo "All datasets verified."
echo ""

# ===== Generate configs =====
echo "=== Generating config files ==="
python3 generate_all_configs.py
echo ""

# ===== Build both projects =====
echo "=== Building Centralized ==="
mkdir -p PiLot_Centralized/build
cmake -S PiLot_Centralized -B PiLot_Centralized/build -DCMAKE_BUILD_TYPE=Release > /dev/null
cmake --build PiLot_Centralized/build -j"$(nproc)" 2>&1 | tail -3
echo ""

echo "=== Building Distributed ==="
mkdir -p PiLot_Distributed/build
cmake -S PiLot_Distributed -B PiLot_Distributed/build -DCMAKE_BUILD_TYPE=Release > /dev/null
cmake --build PiLot_Distributed/build -j"$(nproc)" 2>&1 | tail -3
echo ""

# ===== Run experiments =====
TOTAL_START=$(date +%s)

for DS in "${DATASETS[@]}"; do
    echo ""
    echo "######################################################"
    echo "  Dataset: $DS"
    echo "######################################################"
    
    # --- Centralized ---
    echo ""
    echo ">>> Running Centralized on $DS ..."
    START=$(date +%s)
    bash run_centralized.sh "$DS" 2>&1 | tee "results/${DS}/centralized_stdout.log"
    END=$(date +%s)
    echo ">>> Centralized $DS completed in $((END - START)) seconds"
    
    # --- Distributed ---
    echo ""
    echo ">>> Running Distributed on $DS ..."
    START=$(date +%s)
    bash run_distributed.sh "$DS" 2>&1 | tee "results/${DS}/distributed_stdout.log"
    END=$(date +%s)
    echo ">>> Distributed $DS completed in $((END - START)) seconds"
done

TOTAL_END=$(date +%s)

echo ""
echo "=============================================="
echo "  ALL EXPERIMENTS COMPLETE"
echo "  Total time: $((TOTAL_END - TOTAL_START)) seconds"
echo "=============================================="
echo ""

# ===== Parse results =====
echo "=== Parsing results to CSV ==="
python3 parse_results.py

echo ""
echo "=============================================="
echo "  NEXT STEPS:"
echo "  1. Open MATLAB"
echo "  2. cd to PiLot/plotting/"
echo "  3. Run: plot_all"
echo ""
echo "  Or from command line:"
echo "  matlab -nodisplay -r \"cd('plotting'); plot_all; exit\""
echo "=============================================="
