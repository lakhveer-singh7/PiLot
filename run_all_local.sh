#!/bin/bash
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== START: $(date) ==="

for ds in Cricket_X ECG5000 FaceAll; do
    echo ""
    echo "========================================="
    echo "--- $ds Centralized: $(date) ---"
    echo "========================================="
    bash run_centralized.sh "$ds" 2>&1

    echo ""
    echo "========================================="
    echo "--- $ds Distributed: $(date) ---"
    echo "========================================="
    # Clean IPC before each distributed run
    rm -f /dev/shm/sem.ipc_sem_L* /dev/shm/ipc_tensor_L* /tmp/ipc_early_stop 2>/dev/null
    bash run_distributed.sh "$ds" 2>&1
done

echo ""
echo "=== ALL 6 EXPERIMENTS COMPLETE: $(date) ==="
