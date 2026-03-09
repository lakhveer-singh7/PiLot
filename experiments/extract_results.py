#!/usr/bin/env python3
"""
Extract experiment results from logs → CSV files for MATLAB plotting.

Parses:
  - PiLot Centralized logs:  "Epoch X | Train Acc Y% | Test Acc Z% ... | Infer I ms | Ts"
  - PiLot Distributed logs:  "[METRICS] ... Epoch=X | ... Test_Acc=Z% | Infer_Latency=I ms | Memory=M KB"
  - RockNet Distributed logs: "[METRICS] ... Epoch=X | ... Test_Acc=Z% | Infer_Latency=I ms | Memory=M KB | Devices=N"

Outputs CSV files organized for easy MATLAB reading.
"""

import os
import re
import json
import csv
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"
CONFIGS_DIR = SCRIPT_DIR / "configs"
CSV_DIR = SCRIPT_DIR / "csv_results"

DATASETS = ["Coffee", "Cricket_X", "ECG5000", "ElectricDevices", "FaceAll"]


def parse_centralized_log(log_path):
    """Parse PiLot Centralized log file.
    Format: [SEC.MSEC] INFO: Epoch  X | Train Acc Y% | Test Acc Z% (n/m) | Loss L | LR lr | Infer Ims | Ts
    """
    rows = []
    pattern = re.compile(
        r'Epoch\s+(\d+)\s*\|\s*Train Acc\s+([\d.]+)%\s*\|\s*Test Acc\s+([\d.]+)%\s*\((\d+)/(\d+)\)\s*\|\s*'
        r'Loss\s+([\d.]+)\s*\|\s*LR\s+([\d.eE+-]+)\s*\|\s*Infer\s+([\d.]+)ms\s*\|\s*([\d.]+)s'
    )
    if not os.path.exists(log_path):
        print(f"  WARNING: Log not found: {log_path}")
        return rows

    with open(log_path, 'r') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                rows.append({
                    'epoch': int(m.group(1)),
                    'train_acc': float(m.group(2)),
                    'test_acc': float(m.group(3)),
                    'correct': int(m.group(4)),
                    'total': int(m.group(5)),
                    'loss': float(m.group(6)),
                    'lr': float(m.group(7)),
                    'infer_latency_ms': float(m.group(8)),
                    'elapsed_s': float(m.group(9)),
                })
    return rows


def parse_distributed_log(log_path):
    """Parse PiLot Distributed tail log or RockNet log.
    Format: [METRICS] Timestamp=... | Timespan=Xs | Epoch=N | Train_Acc=Y% | Test_Acc=Z% | Infer_Latency=Ims | Memory=MKB [| Devices=D]
    """
    rows = []
    pattern = re.compile(
        r'\[METRICS\].*?Timespan=([\d.]+)s\s*\|\s*Epoch=(\d+)\s*\|\s*'
        r'Train_Acc=([\d.]+)%\s*\|\s*Test_Acc=([\d.]+)%\s*\|\s*'
        r'Infer_Latency=([\d.]+)ms\s*\|\s*Memory=(\d+)KB'
    )
    if not os.path.exists(log_path):
        print(f"  WARNING: Log not found: {log_path}")
        return rows

    with open(log_path, 'r') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                rows.append({
                    'elapsed_s': float(m.group(1)),
                    'epoch': int(m.group(2)),
                    'train_acc': float(m.group(3)),
                    'test_acc': float(m.group(4)),
                    'infer_latency_ms': float(m.group(5)),
                    'memory_kb': int(m.group(6)),
                })
    return rows


def compute_memory_from_config(config_path):
    """Compute per-device memory from config JSON (for distributed).
    Returns dict: {device_role: memory_bytes, ...} and average.
    """
    if not os.path.exists(config_path):
        return {}

    with open(config_path) as f:
        cfg = json.load(f)

    device_memories = []
    for layer in cfg['layers']:
        mem = layer.get('memory_per_device_bytes', layer.get('memory_bytes', 0))
        n_dev = layer.get('num_devices', 1)
        for _ in range(n_dev):
            device_memories.append(mem)

    return {
        'device_memories': device_memories,
        'total_memory': sum(device_memories),
        'avg_memory': sum(device_memories) / len(device_memories) if device_memories else 0,
        'max_memory': max(device_memories) if device_memories else 0,
        'num_devices': len(device_memories),
    }


def compute_ops_from_config(config_path):
    """Compute per-device ops from config JSON.
    Returns list of ops per device and derived latency estimates.
    """
    if not os.path.exists(config_path):
        return {}

    with open(config_path) as f:
        cfg = json.load(f)

    device_ops = []
    layer_ops = []
    for layer in cfg['layers']:
        ops = layer.get('ops_per_device', layer.get('ops', 0))
        n_dev = layer.get('num_devices', 1)
        layer_ops.append({'ops_per_device': ops, 'num_devices': n_dev, 'type': layer['type']})
        for _ in range(n_dev):
            device_ops.append(ops)

    total_ops = sum(device_ops)

    # Computation latency at 64 MHz (distributed): pipeline bottleneck
    # Each device processes at 64 MHz, 1 FLOP/cycle
    clock_hz = 64_000_000
    max_ops_per_device = max(device_ops) if device_ops else 0
    pipeline_latency_s = max_ops_per_device / clock_hz  # Bottleneck device
    sequential_latency_s = total_ops / clock_hz

    return {
        'device_ops': device_ops,
        'layer_ops': layer_ops,
        'total_ops': total_ops,
        'max_ops': max_ops_per_device,
        'computation_latency_ms': pipeline_latency_s * 1000,
        'sequential_latency_ms': sequential_latency_s * 1000,
    }


def write_csv(rows, path, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Written: {path} ({len(rows)} rows)")


def extract_part1():
    """Extract Part 1 results: Centralized + Distributed N=7 + RockNet N=7"""
    print("\n=== Extracting Part 1 Results ===")

    for ds in DATASETS:
        print(f"\n  Dataset: {ds}")

        # ---- Centralized PiLot ----
        cent_log = RESULTS_DIR / "part1" / f"centralized_{ds}" / "pilot_centralized.log"
        cent_rows = parse_centralized_log(str(cent_log))
        if cent_rows:
            write_csv(cent_rows,
                      str(CSV_DIR / "part1" / f"accuracy_vs_time" / f"centralized_{ds}.csv"),
                      ['epoch', 'train_acc', 'test_acc', 'loss', 'lr', 'infer_latency_ms', 'elapsed_s'])
        else:
            print(f"    No centralized results for {ds}")

        # ---- Distributed PiLot N=7 ----
        dist_log = RESULTS_DIR / "part1" / f"distributed_N7_{ds}" / "device_06_tail.log"
        dist_rows = parse_distributed_log(str(dist_log))
        if dist_rows:
            write_csv(dist_rows,
                      str(CSV_DIR / "part1" / f"accuracy_vs_time" / f"distributed_N7_{ds}.csv"),
                      ['epoch', 'train_acc', 'test_acc', 'infer_latency_ms', 'memory_kb', 'elapsed_s'])
        else:
            print(f"    No distributed results for {ds}")

        # ---- RockNet Distributed N=7 ----
        rock_log = RESULTS_DIR / "part1" / f"rocknet_N7_{ds}" / "rocknet.log"
        rock_rows = parse_distributed_log(str(rock_log))
        if rock_rows:
            write_csv(rock_rows,
                      str(CSV_DIR / "part1" / f"accuracy_vs_time" / f"rocknet_N7_{ds}.csv"),
                      ['epoch', 'train_acc', 'test_acc', 'infer_latency_ms', 'memory_kb', 'elapsed_s'])
        else:
            print(f"    No RockNet results for {ds}")

    # ---- Inference Latency Summary (for bar plots) ----
    print("\n  Generating inference latency summary...")
    latency_rows = []
    for ds in DATASETS:
        # Centralized: last epoch infer_latency_ms (computation only)
        cent_log = RESULTS_DIR / "part1" / f"centralized_{ds}" / "pilot_centralized.log"
        cent_rows = parse_centralized_log(str(cent_log))
        cent_latency = cent_rows[-1]['infer_latency_ms'] if cent_rows else 0.0

        # Centralized ops from config (for theoretical latency)
        cent_cfg = CONFIGS_DIR / "part1" / f"centralized_{ds}" / "model_config.json"
        cent_ops = compute_ops_from_config(str(cent_cfg))

        # Distributed: last epoch infer_latency_ms (includes communication)
        dist_log = RESULTS_DIR / "part1" / f"distributed_N7_{ds}" / "device_06_tail.log"
        dist_rows = parse_distributed_log(str(dist_log))
        dist_latency = dist_rows[-1]['infer_latency_ms'] if dist_rows else 0.0

        # Distributed ops from config
        dist_cfg = CONFIGS_DIR / "part1" / f"distributed_N7_{ds}" / "model_config.json"
        dist_ops = compute_ops_from_config(str(dist_cfg))

        # For distributed, computation latency = pipeline bottleneck latency
        dist_comp_latency = dist_ops.get('computation_latency_ms', 0.0)
        dist_comm_latency = max(0.0, dist_latency - dist_comp_latency)

        # RockNet
        rock_log = RESULTS_DIR / "part1" / f"rocknet_N7_{ds}" / "rocknet.log"
        rock_rows = parse_distributed_log(str(rock_log))
        rock_latency = rock_rows[-1]['infer_latency_ms'] if rock_rows else 0.0

        latency_rows.append({
            'dataset': ds,
            'centralized_computation_ms': cent_latency,
            'distributed_total_ms': dist_latency,
            'distributed_computation_ms': dist_comp_latency,
            'distributed_communication_ms': dist_comm_latency,
            'rocknet_total_ms': rock_latency,
        })

    write_csv(latency_rows,
              str(CSV_DIR / "part1" / "inference_latency_summary.csv"),
              ['dataset', 'centralized_computation_ms', 'distributed_total_ms',
               'distributed_computation_ms', 'distributed_communication_ms', 'rocknet_total_ms'])

    # ---- Memory Consumption Summary ----
    print("\n  Generating memory consumption summary...")
    memory_rows = []
    for ds in DATASETS:
        # Centralized: total memory from config
        cent_cfg = CONFIGS_DIR / "part1" / f"centralized_{ds}" / "model_config.json"
        cent_mem = compute_memory_from_config(str(cent_cfg))

        # Distributed N=7: average device memory from config
        dist_cfg = CONFIGS_DIR / "part1" / f"distributed_N7_{ds}" / "model_config.json"
        dist_mem = compute_memory_from_config(str(dist_cfg))

        # Runtime memory from logs
        dist_log = RESULTS_DIR / "part1" / f"distributed_N7_{ds}" / "device_06_tail.log"
        dist_rows = parse_distributed_log(str(dist_log))
        dist_runtime_kb = dist_rows[-1]['memory_kb'] if dist_rows else 0

        rock_log = RESULTS_DIR / "part1" / f"rocknet_N7_{ds}" / "rocknet.log"
        rock_rows = parse_distributed_log(str(rock_log))
        rock_runtime_kb = rock_rows[-1]['memory_kb'] if rock_rows else 0

        memory_rows.append({
            'dataset': ds,
            'centralized_total_bytes': cent_mem.get('total_memory', 0),
            'centralized_avg_bytes': cent_mem.get('avg_memory', 0),
            'distributed_total_bytes': dist_mem.get('total_memory', 0),
            'distributed_avg_bytes': dist_mem.get('avg_memory', 0),
            'distributed_max_bytes': dist_mem.get('max_memory', 0),
            'distributed_num_devices': dist_mem.get('num_devices', 0),
            'distributed_runtime_kb': dist_runtime_kb,
            'rocknet_runtime_kb': rock_runtime_kb,
        })

    write_csv(memory_rows,
              str(CSV_DIR / "part1" / "memory_consumption_summary.csv"),
              ['dataset', 'centralized_total_bytes', 'centralized_avg_bytes',
               'distributed_total_bytes', 'distributed_avg_bytes', 'distributed_max_bytes',
               'distributed_num_devices', 'distributed_runtime_kb', 'rocknet_runtime_kb'])


def extract_part2():
    """Extract Part 2 results: Distributed N=7, N=8, N=10 for all datasets"""
    print("\n=== Extracting Part 2 Results ===")

    for ds in DATASETS:
        print(f"\n  Dataset: {ds}")
        for N in [7, 8, 10]:
            if N == 7:
                # Reuse Part 1 results
                log_path = RESULTS_DIR / "part1" / f"distributed_N7_{ds}" / "device_06_tail.log"
                cfg_path = CONFIGS_DIR / "part1" / f"distributed_N7_{ds}" / "model_config.json"
            else:
                cfg_path = CONFIGS_DIR / "part2" / f"distributed_N{N}_{ds}" / "model_config.json"
                # N=8: 1 head + 2 L0 + 4 L1 + 1 tail = device 07 is tail
                # N=10: 1 head + 2 L0 + 6 L1 + 1 tail = device 09 is tail
                tail_id = N - 1
                log_path = RESULTS_DIR / "part2" / f"distributed_N{N}_{ds}" / f"device_{tail_id:02d}_tail.log"

            rows = parse_distributed_log(str(log_path))
            if rows:
                write_csv(rows,
                          str(CSV_DIR / "part2" / f"accuracy_vs_time" / f"distributed_N{N}_{ds}.csv"),
                          ['epoch', 'train_acc', 'test_acc', 'infer_latency_ms', 'memory_kb', 'elapsed_s'])
            else:
                print(f"    No results for N={N} on {ds}")

    # ---- Inference Latency Summary for Part 2 ----
    print("\n  Generating Part 2 inference latency summary...")
    for ds in DATASETS:
        latency_rows = []
        for N in [7, 8, 10]:
            if N == 7:
                cfg_path = CONFIGS_DIR / "part1" / f"distributed_N7_{ds}" / "model_config.json"
                log_path = RESULTS_DIR / "part1" / f"distributed_N7_{ds}" / "device_06_tail.log"
            else:
                cfg_path = CONFIGS_DIR / "part2" / f"distributed_N{N}_{ds}" / "model_config.json"
                tail_id = N - 1
                log_path = RESULTS_DIR / "part2" / f"distributed_N{N}_{ds}" / f"device_{tail_id:02d}_tail.log"

            rows = parse_distributed_log(str(log_path))
            total_latency = rows[-1]['infer_latency_ms'] if rows else 0.0

            ops = compute_ops_from_config(str(cfg_path))
            comp_latency = ops.get('computation_latency_ms', 0.0)
            comm_latency = max(0.0, total_latency - comp_latency)

            latency_rows.append({
                'N': N,
                'total_latency_ms': total_latency,
                'computation_ms': comp_latency,
                'communication_ms': comm_latency,
            })

        write_csv(latency_rows,
                  str(CSV_DIR / "part2" / f"inference_latency_{ds}.csv"),
                  ['N', 'total_latency_ms', 'computation_ms', 'communication_ms'])

    # ---- Memory Consumption Summary for Part 2 ----
    print("\n  Generating Part 2 memory consumption summary...")
    for ds in DATASETS:
        memory_rows = []
        for N in [7, 8, 10]:
            if N == 7:
                cfg_path = CONFIGS_DIR / "part1" / f"distributed_N7_{ds}" / "model_config.json"
            else:
                cfg_path = CONFIGS_DIR / "part2" / f"distributed_N{N}_{ds}" / "model_config.json"

            mem = compute_memory_from_config(str(cfg_path))
            ops = compute_ops_from_config(str(cfg_path))

            # Break down memory into weights, optimizer, and buffer (activations)
            if os.path.exists(str(cfg_path)):
                with open(str(cfg_path)) as f:
                    cfg = json.load(f)

                weight_mems = []
                optim_mems = []
                buffer_mems = []
                for layer in cfg['layers']:
                    n_dev = layer.get('num_devices', 1)
                    total_mem = layer.get('memory_per_device_bytes', layer.get('memory_bytes', 0))

                    if layer['type'] == 'conv1d':
                        cpd = layer.get('channels_per_device', layer['out_channels'])
                        in_ch = layer['in_channels']
                        k = layer['kernel_size']
                        # Weights + bias + GN params
                        w_floats = in_ch * k * cpd + cpd + 2 * cpd
                        w_bytes = w_floats * 4
                        # Optimizer (2x params for m,v)
                        o_bytes = 2 * w_bytes
                        # Buffer = total - weights - optimizer
                        b_bytes = total_mem - w_bytes - o_bytes
                    else:  # fc
                        fc_in = layer['in_features']
                        fc_out = layer['out_features']
                        w_floats = fc_in * fc_out + fc_out
                        w_bytes = w_floats * 4
                        o_bytes = 2 * w_bytes
                        b_bytes = total_mem - w_bytes - o_bytes

                    for _ in range(n_dev):
                        weight_mems.append(w_bytes)
                        optim_mems.append(o_bytes)
                        buffer_mems.append(b_bytes)

                avg_weight = sum(weight_mems) / len(weight_mems) if weight_mems else 0
                avg_optim = sum(optim_mems) / len(optim_mems) if optim_mems else 0
                avg_buffer = sum(buffer_mems) / len(buffer_mems) if buffer_mems else 0
                std_weight = (sum((x - avg_weight)**2 for x in weight_mems) / len(weight_mems))**0.5 if weight_mems else 0
                std_optim = (sum((x - avg_optim)**2 for x in optim_mems) / len(optim_mems))**0.5 if optim_mems else 0
                std_buffer = (sum((x - avg_buffer)**2 for x in buffer_mems) / len(buffer_mems))**0.5 if buffer_mems else 0
            else:
                avg_weight = avg_optim = avg_buffer = 0
                std_weight = std_optim = std_buffer = 0

            memory_rows.append({
                'N': N,
                'avg_weight_bytes': avg_weight,
                'std_weight_bytes': std_weight,
                'avg_optimizer_bytes': avg_optim,
                'std_optimizer_bytes': std_optim,
                'avg_buffer_bytes': avg_buffer,
                'std_buffer_bytes': std_buffer,
                'total_avg_bytes': mem.get('avg_memory', 0),
            })

        write_csv(memory_rows,
                  str(CSV_DIR / "part2" / f"memory_consumption_{ds}.csv"),
                  ['N', 'avg_weight_bytes', 'std_weight_bytes', 'avg_optimizer_bytes',
                   'std_optimizer_bytes', 'avg_buffer_bytes', 'std_buffer_bytes', 'total_avg_bytes'])


def main():
    print("=" * 60)
    print("  Extracting Results → CSV for MATLAB")
    print("=" * 60)

    extract_part1()
    extract_part2()

    print("\n" + "=" * 60)
    print(f"  All CSVs written to: {CSV_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
