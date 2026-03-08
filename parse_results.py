#!/usr/bin/env python3
"""
parse_results.py — Extract metrics from PiLot logs into CSV files for MATLAB plotting.

Reads log files from results/<dataset>/{centralized,distributed}/ and produces:
  results/csv/accuracy_vs_time_<dataset>_centralized.csv
  results/csv/accuracy_vs_time_<dataset>_distributed.csv
  results/csv/inference_latency.csv
  results/csv/memory_consumption.csv

Usage:
  python3 parse_results.py
"""

import os
import re
import csv
import glob
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
CSV_DIR = os.path.join(RESULTS_DIR, "csv")

DATASETS = ["Cricket_X", "ECG5000", "FaceAll"]


def parse_centralized_log(log_path):
    """Parse centralized pilot_centralized.log for [METRICS] lines.
    
    Returns list of dicts with keys: epoch, timespan, train_acc, test_acc, infer_latency, memory_kb
    """
    metrics = []
    pattern = re.compile(
        r'\[METRICS\]\s*Epoch=(\d+)\s*\|\s*Timespan=([\d.]+)s\s*\|\s*'
        r'Train_Acc=([\d.]+)%\s*\|\s*Test_Acc=([\d.]+)%\s*\|\s*'
        r'Infer_Latency=([\d.]+)ms\s*\|\s*Memory=(\d+)KB'
    )
    
    if not os.path.exists(log_path):
        print(f"  WARNING: Log not found: {log_path}")
        return metrics
    
    with open(log_path, 'r') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                metrics.append({
                    'epoch': int(m.group(1)),
                    'timespan': float(m.group(2)),
                    'train_acc': float(m.group(3)),
                    'test_acc': float(m.group(4)),
                    'infer_latency': float(m.group(5)),
                    'memory_kb': int(m.group(6)),
                })
    
    print(f"  Centralized: {len(metrics)} epochs parsed from {log_path}")
    return metrics


def parse_distributed_tail_log(log_path):
    """Parse distributed tail log (device_06_tail.log) for [METRICS] lines.
    
    Returns list of dicts with keys: epoch, timespan, train_acc, test_acc, infer_latency, memory_kb
    """
    metrics = []
    pattern = re.compile(
        r'\[METRICS\]\s*Timestamp=.*?\|\s*Timespan=([\d.]+)s\s*\|\s*'
        r'Epoch=(\d+)\s*\|\s*Train_Acc=([\d.]+)%\s*\|\s*'
        r'Test_Acc=([\d.]+)%\s*\|\s*Infer_Latency=([\d.]+)ms\s*\|\s*Memory=(\d+)KB'
    )
    
    if not os.path.exists(log_path):
        print(f"  WARNING: Log not found: {log_path}")
        return metrics
    
    with open(log_path, 'r') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                metrics.append({
                    'epoch': int(m.group(2)),
                    'timespan': float(m.group(1)),
                    'train_acc': float(m.group(3)),
                    'test_acc': float(m.group(4)),
                    'infer_latency': float(m.group(5)),
                    'memory_kb': int(m.group(6)),
                })
    
    print(f"  Distributed tail: {len(metrics)} epochs parsed from {log_path}")
    return metrics


def parse_distributed_head_log(log_path):
    """Parse distributed head log (device_00_head.log) for [HEAD_METRICS] lines.
    
    Returns list of dicts with keys: epoch, pipeline_infer_latency
    """
    metrics = []
    pattern = re.compile(
        r'\[HEAD_METRICS\]\s*Epoch=(\d+)\s*\|\s*'
        r'Pipeline_Infer_Latency=([\d.]+)ms'
    )
    
    if not os.path.exists(log_path):
        print(f"  WARNING: Head log not found: {log_path}")
        return metrics
    
    with open(log_path, 'r') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                metrics.append({
                    'epoch': int(m.group(1)),
                    'pipeline_infer_latency': float(m.group(2)),
                })
    
    print(f"  Distributed head: {len(metrics)} epochs parsed from {log_path}")
    return metrics


def parse_device_memory(log_dir):
    """Parse all device logs for peak memory usage.
    
    From print_memory_usage(): "Memory usage: current=%zu bytes, peak=%zu bytes, limit=%zu bytes"
    Returns dict: device_id -> peak_bytes
    """
    device_peak = {}
    pattern = re.compile(r'Memory usage:.*peak=(\d+) bytes')
    
    for log_file in sorted(glob.glob(os.path.join(log_dir, "device_*.log"))):
        # Extract device ID from filename
        basename = os.path.basename(log_file)
        id_match = re.search(r'device_(\d+)', basename)
        if not id_match:
            continue
        dev_id = int(id_match.group(1))
        
        peak = 0
        with open(log_file, 'r') as f:
            for line in f:
                m = pattern.search(line)
                if m:
                    p = int(m.group(1))
                    if p > peak:
                        peak = p
        
        if peak > 0:
            device_peak[dev_id] = peak
    
    if device_peak:
        print(f"  Device memory: {len(device_peak)} devices, "
              f"avg={sum(device_peak.values())/len(device_peak)/1024:.1f} KB, "
              f"max={max(device_peak.values())/1024:.1f} KB")
    return device_peak


def write_accuracy_csv(metrics, csv_path):
    """Write accuracy vs time CSV."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'time_s', 'train_acc', 'test_acc'])
        for m in metrics:
            writer.writerow([m['epoch'], m['timespan'], m['train_acc'], m['test_acc']])
    print(f"  Written: {csv_path}")


def main():
    os.makedirs(CSV_DIR, exist_ok=True)
    
    # Per-dataset inference latency and memory data
    inference_data = []   # list of dicts: dataset, cent_infer, dist_infer
    memory_data = []      # list of dicts: dataset, cent_mem_kb, dist_avg_kb, dist_std_kb
    
    for ds in DATASETS:
        print(f"\n=== Processing {ds} ===")
        
        cent_log_dir = os.path.join(RESULTS_DIR, ds, "centralized")
        dist_log_dir = os.path.join(RESULTS_DIR, ds, "distributed")
        
        # --- Centralized ---
        cent_log = os.path.join(cent_log_dir, "pilot_centralized.log")
        cent_metrics = parse_centralized_log(cent_log)
        
        if cent_metrics:
            csv_path = os.path.join(CSV_DIR, f"accuracy_vs_time_{ds}_centralized.csv")
            write_accuracy_csv(cent_metrics, csv_path)
            
            # Use last epoch's metrics for summary
            last = cent_metrics[-1]
            cent_infer = last['infer_latency']
            cent_mem_kb = last['memory_kb']
        else:
            cent_infer = 0.0
            cent_mem_kb = 0
        
        # --- Distributed ---
        tail_log = os.path.join(dist_log_dir, "device_06_tail.log")
        head_log = os.path.join(dist_log_dir, "device_00_head.log")
        
        dist_metrics = parse_distributed_tail_log(tail_log)
        head_metrics = parse_distributed_head_log(head_log)
        device_memory = parse_device_memory(dist_log_dir)
        
        if dist_metrics:
            csv_path = os.path.join(CSV_DIR, f"accuracy_vs_time_{ds}_distributed.csv")
            write_accuracy_csv(dist_metrics, csv_path)
            
            # Use last epoch's tail inference latency (computation only at tail)
            last_tail = dist_metrics[-1]
            tail_infer = last_tail['infer_latency']
        else:
            tail_infer = 0.0
        
        # Pipeline latency from head (includes all computation + communication)
        if head_metrics:
            # Use last epoch's pipeline latency
            dist_pipeline_infer = head_metrics[-1]['pipeline_infer_latency']
        else:
            dist_pipeline_infer = tail_infer  # fallback
        
        # Device memory: average and std across all devices
        if device_memory:
            mem_values = list(device_memory.values())
            avg_mem_kb = sum(mem_values) / len(mem_values) / 1024.0  # bytes to KB
            import statistics
            std_mem_kb = statistics.stdev(mem_values) / 1024.0 if len(mem_values) > 1 else 0.0
        else:
            avg_mem_kb = 0.0
            std_mem_kb = 0.0
        
        inference_data.append({
            'dataset': ds,
            'centralized_ms': cent_infer,
            'distributed_ms': dist_pipeline_infer,
        })
        
        memory_data.append({
            'dataset': ds,
            'centralized_kb': cent_mem_kb,
            'distributed_avg_kb': avg_mem_kb,
            'distributed_std_kb': std_mem_kb,
        })
    
    # --- Write summary CSVs ---
    print("\n=== Writing summary CSVs ===")
    
    # Inference latency
    inf_path = os.path.join(CSV_DIR, "inference_latency.csv")
    with open(inf_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'centralized_ms', 'distributed_ms'])
        for d in inference_data:
            writer.writerow([d['dataset'], d['centralized_ms'], d['distributed_ms']])
    print(f"  Written: {inf_path}")
    
    # Memory consumption
    mem_path = os.path.join(CSV_DIR, "memory_consumption.csv")
    with open(mem_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'centralized_kb', 'distributed_avg_kb', 'distributed_std_kb'])
        for d in memory_data:
            writer.writerow([d['dataset'], d['centralized_kb'],
                           f"{d['distributed_avg_kb']:.2f}", f"{d['distributed_std_kb']:.2f}"])
    print(f"  Written: {mem_path}")
    
    print("\n=== Done! CSV files ready for MATLAB plotting ===")
    print(f"Results in: {CSV_DIR}/")


if __name__ == "__main__":
    main()
