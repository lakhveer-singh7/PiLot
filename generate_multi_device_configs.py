#!/usr/bin/env python3
"""
generate_multi_device_configs.py — Generate model_config JSON files for
3 device configurations × 3 datasets = 9 configs.

Device configurations (from architecture diagrams):
  7-dev:  L0: 2 workers × 16ch = 32ch,  L1: 3 workers × 16ch = 48ch,  tail_in = 96
  9-dev:  L0: 3 workers ×  8ch = 24ch,  L1: 4 workers ×  8ch = 32ch,  tail_in = 64
 12-dev:  L0: 4 workers ×  8ch = 32ch,  L1: 6 workers ×  8ch = 48ch,  tail_in = 96

Datasets:
  Cricket_X:  input_length=300, num_classes=12
  ECG5000:    input_length=140, num_classes=5
  FaceAll:    input_length=131, num_classes=14
"""

import json, os

SIZEOF_FLOAT = 4

DEVICE_CONFIGS = {
    7:  {'L0_workers': 2, 'L0_ch': 16,  'L1_workers': 3, 'L1_ch': 16},
    9:  {'L0_workers': 3, 'L0_ch':  8,  'L1_workers': 4, 'L1_ch':  8},
    12: {'L0_workers': 4, 'L0_ch':  8,  'L1_workers': 6, 'L1_ch':  8},
}

DATASETS = {
    'Cricket_X': {'input_length': 300, 'num_classes': 12},
    'ECG5000':   {'input_length': 140, 'num_classes':  5},
    'FaceAll':   {'input_length': 131, 'num_classes': 14},
}

KERNEL_SIZE = 5
L0_STRIDE, L0_PADDING = 1, 2
L1_STRIDE, L1_PADDING = 2, 2

EPOCHS = 300
LEARNING_RATE = 0.01
MEMORY_LIMIT = 262144
FLASH_MEMORY = 1048576


def conv_output_length(il, k, s, p):
    return (il + 2 * p - k) // s + 1


def conv_mem(in_ch, cpd, k, ilen, olen):
    w = in_ch * k * cpd
    b = cpd
    gn = 2 * cpd
    params = w + b + gn
    acts = in_ch * ilen + cpd * olen + cpd * olen  # input + output + gn
    return (params + params + 2 * params + acts) * SIZEOF_FLOAT  # params + grad + adam + acts


def conv_ops(in_ch, cpd, k, olen):
    return cpd * olen * in_ch * k * 2 + cpd * olen + cpd * olen * 5 + cpd * olen


def fc_mem(fc_in, fc_out, last_ch, last_len):
    pool_buf = last_ch * last_len
    w = fc_in * fc_out
    b = fc_out
    params = w + b
    return (pool_buf + fc_in + params + params + 2 * params + fc_out) * SIZEOF_FLOAT


def fc_ops(fc_in, fc_out, last_ch, last_len):
    return last_ch * last_len * 2 + fc_in + fc_in * fc_out * 2 + fc_out + fc_out * 3


def make_config(n_dev, ds_name):
    dc = DEVICE_CONFIGS[n_dev]
    ds = DATASETS[ds_name]
    il = ds['input_length']
    nc = ds['num_classes']

    L0_total = dc['L0_workers'] * dc['L0_ch']
    L1_total = dc['L1_workers'] * dc['L1_ch']
    L0_olen = conv_output_length(il, KERNEL_SIZE, L0_STRIDE, L0_PADDING)
    L1_olen = conv_output_length(L0_olen, KERNEL_SIZE, L1_STRIDE, L1_PADDING)
    tail_in = L1_total * 2

    return {
        "model": {
            "name": f"nRF52840_CNN_{ds_name}_{n_dev}dev",
            "version": "2.0"
        },
        "global": {
            "dataset": ds_name,
            "epochs": EPOCHS,
            "num_classes": nc,
            "input_length": il,
            "memory_limit_bytes": MEMORY_LIMIT,
            "flash_memory_bytes": FLASH_MEMORY,
            "learning_rate": LEARNING_RATE
        },
        "layers": [
            {
                "id": 0,
                "type": "conv1d",
                "in_channels": 1,
                "out_channels": L0_total,
                "kernel_size": KERNEL_SIZE,
                "stride": L0_STRIDE,
                "padding": L0_PADDING,
                "input_length": il,
                "output_length": L0_olen,
                "channels_per_device": dc['L0_ch'],
                "num_devices": dc['L0_workers'],
                "memory_per_device_bytes": conv_mem(1, dc['L0_ch'], KERNEL_SIZE, il, L0_olen),
                "ops_per_device": conv_ops(1, dc['L0_ch'], KERNEL_SIZE, L0_olen)
            },
            {
                "id": 1,
                "type": "conv1d",
                "in_channels": L0_total,
                "out_channels": L1_total,
                "kernel_size": KERNEL_SIZE,
                "stride": L1_STRIDE,
                "padding": L1_PADDING,
                "input_length": L0_olen,
                "output_length": L1_olen,
                "channels_per_device": dc['L1_ch'],
                "num_devices": dc['L1_workers'],
                "memory_per_device_bytes": conv_mem(L0_total, dc['L1_ch'], KERNEL_SIZE, L0_olen, L1_olen),
                "ops_per_device": conv_ops(L0_total, dc['L1_ch'], KERNEL_SIZE, L1_olen)
            },
            {
                "id": 2,
                "type": "fc",
                "input_length": L1_olen,
                "pooling": ["avg", "max"],
                "in_features": tail_in,
                "out_features": nc,
                "num_devices": 1,
                "memory_per_device_bytes": fc_mem(tail_in, nc, L1_total, L1_olen),
                "ops_per_device": fc_ops(tail_in, nc, L1_total, L1_olen)
            }
        ]
    }


if __name__ == '__main__':
    root = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(root, 'PiLot_Distributed', 'configs')
    os.makedirs(out_dir, exist_ok=True)

    for n_dev in [7, 9, 12]:
        for ds in ['Cricket_X', 'ECG5000', 'FaceAll']:
            cfg = make_config(n_dev, ds)
            fname = f'config_{n_dev}dev_{ds}.json'
            path = os.path.join(out_dir, fname)
            with open(path, 'w') as f:
                json.dump(cfg, f, indent=2)
            dc = DEVICE_CONFIGS[n_dev]
            L0_total = dc['L0_workers'] * dc['L0_ch']
            L1_total = dc['L1_workers'] * dc['L1_ch']
            print(f'  {fname:40s}  L0:{dc["L0_workers"]}×{dc["L0_ch"]}={L0_total:2d}ch  '
                  f'L1:{dc["L1_workers"]}×{dc["L1_ch"]}={L1_total:2d}ch')

    print(f'\nAll 9 config files written to {out_dir}/')
