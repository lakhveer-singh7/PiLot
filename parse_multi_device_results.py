#!/usr/bin/env python3
"""
parse_multi_device_results.py — Parse log files from 7, 9, 12 device experiments
and produce CSV files for comparison plotting.

Reads from: results/<dataset>/distributed_<N>dev/device_*_tail.log
                                                  /device_*_head.log
                                                  /device_*_worker_*.log

Outputs:
  results/csv_multi/accuracy_vs_time_<dataset>.csv
  results/csv_multi/inference_latency.csv
  results/csv_multi/memory_per_device.csv
"""

import os, re, csv, statistics, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, 'results')
OUT_DIR = os.path.join(RESULTS_DIR, 'csv_multi')
os.makedirs(OUT_DIR, exist_ok=True)

DATASETS = ['Cricket_X', 'ECG5000', 'FaceAll']
NDEVS = [7, 9, 12]

# ── Architecture definitions ──
DEVICE_CONFIGS = {
    7:  {'L0_workers': 2, 'L0_ch': 16,  'L1_workers': 3, 'L1_ch': 16},
    9:  {'L0_workers': 3, 'L0_ch':  8,  'L1_workers': 4, 'L1_ch':  8},
    12: {'L0_workers': 4, 'L0_ch':  8,  'L1_workers': 6, 'L1_ch':  8},
}
DATASETS_PROPS = {
    'Cricket_X': (300, 12),
    'ECG5000':   (140,  5),
    'FaceAll':   (131, 14),
}
PROC_HZ = 64_000_000
K = 5
L0_S, L0_P = 1, 2
L1_S, L1_P = 2, 2


def log_dir(ds, ndev):
    """Return log directory for a dataset × device-count experiment."""
    d = os.path.join(RESULTS_DIR, ds, f'distributed_{ndev}dev')
    if os.path.islink(d):
        d = os.path.realpath(d)
    return d


def find_log(directory, role_pattern):
    """Find the first matching log file."""
    if not os.path.isdir(directory):
        return None
    for f in sorted(os.listdir(directory)):
        if re.search(role_pattern, f):
            return os.path.join(directory, f)
    return None


def parse_tail_metrics(log_path):
    """Parse [METRICS] lines from tail log → list of dicts."""
    rows = []
    if not log_path or not os.path.isfile(log_path):
        return rows
    with open(log_path) as f:
        for line in f:
            m = re.search(r'\[METRICS\].*Timespan=([0-9.]+)s.*Epoch=(\d+).*'
                          r'Train_Acc=([0-9.]+)%.*Test_Acc=([0-9.]+)%.*'
                          r'Infer_Latency=([0-9.]+)ms.*Memory=(\d+)KB', line)
            if m:
                rows.append({
                    'timespan': float(m.group(1)),
                    'epoch': int(m.group(2)),
                    'train_acc': float(m.group(3)),
                    'test_acc': float(m.group(4)),
                    'infer_latency_ms': float(m.group(5)),
                    'memory_kb': int(m.group(6)),
                })
    return rows


def parse_head_pipeline_latency(log_path):
    """Parse [HEAD_METRICS] → list of pipeline latency per epoch."""
    vals = []
    if not log_path or not os.path.isfile(log_path):
        return vals
    with open(log_path) as f:
        for line in f:
            m = re.search(r'Pipeline_Infer_Latency=([0-9.]+)ms', line)
            if m:
                vals.append(float(m.group(1)))
    return vals


def parse_device_memory(directory, ndev):
    """Parse RSS/Memory from all device logs → dict of device_id→memory_kb."""
    mem = {}
    if not os.path.isdir(directory):
        return mem
    for f in sorted(os.listdir(directory)):
        if not f.endswith('.log') or not f.startswith('device_'):
            continue
        path = os.path.join(directory, f)
        dev_mem = 0
        with open(path) as fh:
            for line in fh:
                # Head and tail logs have Memory=NNNkb in METRICS
                m = re.search(r'Memory=(\d+)KB', line)
                if m:
                    dev_mem = max(dev_mem, int(m.group(1)))
                # Worker logs report peak memory
                m2 = re.search(r'peak.*?(\d+)\s*KB', line, re.IGNORECASE)
                if m2:
                    dev_mem = max(dev_mem, int(m2.group(1)))
        if dev_mem > 0:
            mem[f] = dev_mem
    return mem


# ══════════════════════════════════════════════════════════════
#  (a) Accuracy vs Time CSV — one per dataset
# ══════════════════════════════════════════════════════════════
def gen_accuracy_vs_time_csvs():
    print('[1/3] Generating accuracy_vs_time CSVs ...')
    for ds in DATASETS:
        rows_out = []
        for ndev in NDEVS:
            d = log_dir(ds, ndev)
            tail_log = find_log(d, r'tail\.log$')
            metrics = parse_tail_metrics(tail_log)
            for r in metrics:
                rows_out.append({
                    'num_devices': ndev,
                    'epoch': r['epoch'],
                    'timespan_s': r['timespan'],
                    'train_acc': r['train_acc'],
                    'test_acc': r['test_acc'],
                })
        csv_path = os.path.join(OUT_DIR, f'accuracy_vs_time_{ds}.csv')
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['num_devices','epoch','timespan_s','train_acc','test_acc'])
            w.writeheader()
            w.writerows(rows_out)
        print(f'  {csv_path}  ({len(rows_out)} rows)')


# ══════════════════════════════════════════════════════════════
#  (b) Inference Latency CSV — per dataset × config
# ══════════════════════════════════════════════════════════════
def gen_inference_latency_csv():
    print('[2/3] Generating inference_latency CSV ...')
    rows_out = []
    for ds in DATASETS:
        il, nc = DATASETS_PROPS[ds]
        for ndev in NDEVS:
            dc = DEVICE_CONFIGS[ndev]
            L0_total = dc['L0_workers'] * dc['L0_ch']
            L1_total = dc['L1_workers'] * dc['L1_ch']
            L0_olen = (il + 2*L0_P - K) // L0_S + 1
            L1_olen = (L0_olen + 2*L1_P - K) // L1_S + 1
            tail_in = L1_total * 2

            # Computation per device (forward only, in ms)
            L0_flops = 2 * dc['L0_ch'] * 1 * K * L0_olen
            L1_flops = 2 * dc['L1_ch'] * L0_total * K * L1_olen
            tail_flops = 2 * tail_in * nc
            L0_comp_ms = L0_flops / PROC_HZ * 1000
            L1_comp_ms = L1_flops / PROC_HZ * 1000
            tail_comp_ms = tail_flops / PROC_HZ * 1000
            total_comp_ms = L0_comp_ms + L1_comp_ms + tail_comp_ms

            # Pipeline latency from head log
            d = log_dir(ds, ndev)
            head_log = find_log(d, r'head\.log$')
            plats = parse_head_pipeline_latency(head_log)
            if plats:
                # Use average of last 10 epochs (stable)
                avg_pipeline = statistics.mean(plats[-10:]) if len(plats) >= 10 else statistics.mean(plats)
            else:
                avg_pipeline = total_comp_ms  # fallback

            total_comm_ms = max(0.0, avg_pipeline - total_comp_ms)

            rows_out.append({
                'dataset': ds,
                'num_devices': ndev,
                'pipeline_latency_ms': round(avg_pipeline, 3),
                'computation_ms': round(total_comp_ms, 3),
                'communication_ms': round(total_comm_ms, 3),
            })

    csv_path = os.path.join(OUT_DIR, 'inference_latency.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['dataset','num_devices','pipeline_latency_ms',
                                           'computation_ms','communication_ms'])
        w.writeheader()
        w.writerows(rows_out)
    print(f'  {csv_path}  ({len(rows_out)} rows)')
    for r in rows_out:
        print(f"    {r['dataset']:>12s}  {r['num_devices']:>2d}-dev  pipe={r['pipeline_latency_ms']:.2f}  "
              f"comp={r['computation_ms']:.2f}  comm={r['communication_ms']:.2f}")


# ══════════════════════════════════════════════════════════════
#  (c) Memory Consumption CSV — weight/optimizer/buffer avg per config
# ══════════════════════════════════════════════════════════════
def gen_memory_csv():
    """Compute per-device weight, optimizer, buffer memory from architecture."""
    print('[3/3] Generating memory_consumption CSV ...')
    SIZEOF_FLOAT = 4
    rows_out = []
    for ds in DATASETS:
        il, nc = DATASETS_PROPS[ds]
        for ndev in NDEVS:
            dc = DEVICE_CONFIGS[ndev]
            L0_total = dc['L0_workers'] * dc['L0_ch']
            L1_total = dc['L1_workers'] * dc['L1_ch']
            L0_olen = (il + 2*L0_P - K) // L0_S + 1
            L1_olen = (L0_olen + 2*L1_P - K) // L1_S + 1
            tail_in = L1_total * 2

            # Per-device memory breakdown (bytes)
            devices = []

            # Head: no weights/optimizer, just buffers (input sample + augmented)
            head_buf = (1 * il) * SIZEOF_FLOAT * 2  # raw + augmented
            devices.append({'role': 'Head', 'weights': 0, 'optimizer': 0, 'buffers': head_buf})

            # L0 workers
            for w in range(dc['L0_workers']):
                w_bytes = (1 * dc['L0_ch'] * K + dc['L0_ch']) * SIZEOF_FLOAT  # weights + bias
                gn_bytes = 2 * dc['L0_ch'] * SIZEOF_FLOAT                      # GroupNorm
                w_total = w_bytes + gn_bytes
                opt_bytes = w_total * 2                                         # Adam m + v
                buf_bytes = (1 * il + dc['L0_ch'] * L0_olen) * SIZEOF_FLOAT    # input + output activations
                devices.append({'role': f'L0_W{w}', 'weights': w_total, 'optimizer': opt_bytes, 'buffers': buf_bytes})

            # L1 workers
            for w in range(dc['L1_workers']):
                w_bytes = (L0_total * dc['L1_ch'] * K + dc['L1_ch']) * SIZEOF_FLOAT
                gn_bytes = 2 * dc['L1_ch'] * SIZEOF_FLOAT
                w_total = w_bytes + gn_bytes
                opt_bytes = w_total * 2
                buf_bytes = (L0_total * L0_olen + dc['L1_ch'] * L1_olen) * SIZEOF_FLOAT
                devices.append({'role': f'L1_W{w}', 'weights': w_total, 'optimizer': opt_bytes, 'buffers': buf_bytes})

            # Tail: FC weights + pooling buffer
            fc_w = (tail_in * nc + nc) * SIZEOF_FLOAT
            fc_opt = fc_w * 2
            tail_buf = (L1_total * L1_olen + tail_in + nc) * SIZEOF_FLOAT
            devices.append({'role': 'Tail', 'weights': fc_w, 'optimizer': fc_opt, 'buffers': tail_buf})

            # Averages across all devices (in KB)
            n = len(devices)
            avg_w = sum(d['weights'] for d in devices) / n / 1024
            avg_o = sum(d['optimizer'] for d in devices) / n / 1024
            avg_b = sum(d['buffers'] for d in devices) / n / 1024
            std_w = statistics.stdev([d['weights']/1024 for d in devices]) if n > 1 else 0
            std_o = statistics.stdev([d['optimizer']/1024 for d in devices]) if n > 1 else 0
            std_b = statistics.stdev([d['buffers']/1024 for d in devices]) if n > 1 else 0

            rows_out.append({
                'dataset': ds,
                'num_devices': ndev,
                'weight_avg_kb': round(avg_w, 2),
                'weight_std_kb': round(std_w, 2),
                'optimizer_avg_kb': round(avg_o, 2),
                'optimizer_std_kb': round(std_o, 2),
                'buffer_avg_kb': round(avg_b, 2),
                'buffer_std_kb': round(std_b, 2),
                'total_avg_kb': round(avg_w + avg_o + avg_b, 2),
            })

    csv_path = os.path.join(OUT_DIR, 'memory_consumption.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['dataset','num_devices',
                                           'weight_avg_kb','weight_std_kb',
                                           'optimizer_avg_kb','optimizer_std_kb',
                                           'buffer_avg_kb','buffer_std_kb','total_avg_kb'])
        w.writeheader()
        w.writerows(rows_out)
    print(f'  {csv_path}  ({len(rows_out)} rows)')
    for r in rows_out:
        print(f"    {r['dataset']:>12s}  {r['num_devices']:>2d}-dev  "
              f"W={r['weight_avg_kb']:.1f}KB  O={r['optimizer_avg_kb']:.1f}KB  "
              f"B={r['buffer_avg_kb']:.1f}KB  Total={r['total_avg_kb']:.1f}KB")


if __name__ == '__main__':
    print('=== Parsing Multi-Device Results ===\n')
    gen_accuracy_vs_time_csvs()
    print()
    gen_inference_latency_csv()
    print()
    gen_memory_csv()
    print('\n=== Done ===')
