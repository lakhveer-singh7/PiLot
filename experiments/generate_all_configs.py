#!/usr/bin/env python3
"""
Generate all PiLot configs for Part 1 and Part 2 experiments.

Part 1: 5 datasets × {Centralized, Distributed N=7}
Part 2: Cricket_X × {Distributed N=7, N=8, N=10}

Each config goes into experiments/configs/<experiment_name>/model_config.json
"""

import json
import os
import sys
import math
from pathlib import Path

# =====================================================================
#  Dataset definitions: name → (input_length, num_classes, max_train, max_test)
# =====================================================================
DATASETS = {
    "Coffee":          (286,  2,   28,    28),
    "Cricket_X":       (300, 12,  348,   522),
    "ECG5000":         (140,  5,  500,  4500),
    "ElectricDevices": (96,   7,  500,  2202),
    "FaceAll":         (131, 14,  560,  1690),
}

# =====================================================================
#  Common hyperparameters
# =====================================================================
COMMON_HYPER = {
    "epochs": 100,
    "learning_rate": 0.005,
    "dropout_rate": 0.25,
    "weight_decay": 0.0003,
    "grad_accum_steps": 5,
    "patience": 30,             # Stop if no improvement for 30 epochs
    "grad_clip_max": 5.0,
    "warmup_epochs": 3,
    "eta_min": 1e-5,
}

# Memory and flash constraints for distributed
MEMORY_LIMIT = 262144    # 256 KB
FLASH_MEMORY = 1048576   # 1 MB

SIZEOF_FLOAT = 4


# =====================================================================
#  Helper functions (from generate_config.py)
# =====================================================================
def conv_output_length(input_length, kernel_size, stride, padding):
    return (input_length + 2 * padding - kernel_size) // stride + 1


def conv_memory_per_device(in_ch, cpd, k, in_len, out_len):
    weights = in_ch * k * cpd
    bias = cpd
    gn_params = 2 * cpd
    params = weights + bias + gn_params
    input_act = in_ch * in_len
    output_act = cpd * out_len
    gn_act = cpd * out_len
    grad = params
    adam_states = 2 * params
    total_floats = params + input_act + output_act + gn_act + grad + adam_states
    return total_floats * SIZEOF_FLOAT


def conv_ops_per_device(in_ch, cpd, k, out_len):
    conv_mac = cpd * out_len * in_ch * k * 2
    bias_add = cpd * out_len
    gn_ops = cpd * out_len * 5
    relu_ops = cpd * out_len
    return conv_mac + bias_add + gn_ops + relu_ops


def fc_memory_per_device(fc_in, fc_out, last_out_ch, input_length):
    pool_input = last_out_ch * input_length
    pooled_vec = fc_in
    fc_weights = fc_in * fc_out
    fc_bias = fc_out
    fc_output = fc_out
    params = fc_weights + fc_bias
    grad = params
    adam_states = 2 * params
    total_floats = pool_input + pooled_vec + params + fc_output + grad + adam_states
    return total_floats * SIZEOF_FLOAT


def fc_ops_per_device(fc_in, fc_out, last_out_ch, input_length):
    gap_ops = last_out_ch * input_length
    gmp_ops = last_out_ch * input_length
    dropout_ops = fc_in
    fc_mac = fc_in * fc_out * 2
    bias_add = fc_out
    softmax_ops = fc_out * 3
    return gap_ops + gmp_ops + dropout_ops + fc_mac + bias_add + softmax_ops


def build_config(dataset, input_length, num_classes, max_train, max_test,
                 layers_spec, centralized=False):
    """Build a config dictionary from layer specifications."""
    layers = []
    in_ch = 1
    cur_length = input_length

    for idx, spec in enumerate(layers_spec):
        ltype = spec["type"]
        if ltype == "conv1d":
            out_ch = spec["out_channels"]
            k = spec.get("kernel_size", 5)
            s = spec.get("stride", 1)
            p = spec.get("padding", k // 2)
            out_len = conv_output_length(cur_length, k, s, p)

            if centralized:
                cpd = out_ch
                n_dev = 1
            else:
                cpd = spec["channels_per_device"]
                n_dev = spec.get("num_devices", out_ch // cpd)

            mem = conv_memory_per_device(in_ch, cpd, k, cur_length, out_len)
            ops = conv_ops_per_device(in_ch, cpd, k, out_len)

            layer = {
                "id": idx,
                "type": "conv1d",
                "in_channels": in_ch,
                "out_channels": out_ch,
                "kernel_size": k,
                "stride": s,
                "padding": p,
                "input_length": cur_length,
                "output_length": out_len,
            }
            if centralized:
                layer["memory_bytes"] = mem
                layer["ops"] = ops
            else:
                layer["channels_per_device"] = cpd
                layer["num_devices"] = n_dev
                layer["memory_per_device_bytes"] = mem
                layer["ops_per_device"] = ops
            layers.append(layer)
            in_ch = out_ch
            cur_length = out_len

        elif ltype == "fc":
            fc_in = layers[-1]["out_channels"] * 2  # dual pooling
            fc_out = num_classes
            last_out_ch = layers[-1]["out_channels"]
            mem = fc_memory_per_device(fc_in, fc_out, last_out_ch, cur_length)
            ops = fc_ops_per_device(fc_in, fc_out, last_out_ch, cur_length)

            layer = {
                "id": idx,
                "type": "fc",
                "input_length": cur_length,
                "pooling": ["avg", "max"],
                "in_features": fc_in,
                "out_features": fc_out,
            }
            if centralized:
                layer["memory_bytes"] = mem
                layer["ops"] = ops
            else:
                layer["num_devices"] = 1
                layer["memory_per_device_bytes"] = mem
                layer["ops_per_device"] = ops
            layers.append(layer)

    cfg = {
        "model": {
            "name": f"nRF52840_UniformCNN_{dataset}",
            "version": "2.0",
        },
        "global": {
            "dataset": dataset,
            "epochs": COMMON_HYPER["epochs"],
            "num_classes": num_classes,
            "input_length": input_length,
            "memory_limit_bytes": 0 if centralized else MEMORY_LIMIT,
            "flash_memory_bytes": 0 if centralized else FLASH_MEMORY,
            "learning_rate": COMMON_HYPER["learning_rate"],
            "dropout_rate": COMMON_HYPER["dropout_rate"],
            "weight_decay": COMMON_HYPER["weight_decay"],
            "grad_accum_steps": COMMON_HYPER["grad_accum_steps"],
            "patience": COMMON_HYPER["patience"],
            "grad_clip_max": COMMON_HYPER["grad_clip_max"],
            "warmup_epochs": COMMON_HYPER["warmup_epochs"],
            "eta_min": COMMON_HYPER["eta_min"],
            "max_train_samples": max_train,
            "max_test_samples": max_test,
        },
        "layers": layers,
    }
    return cfg


def write_config(cfg, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  Written: {path}")


# =====================================================================
#  Layer specifications for different N configurations
# =====================================================================

def get_distributed_layers_N7():
    """N=7 total: Head(1) + 2 L0 workers + 3 L1 workers + 1 Tail = 7"""
    return [
        {"type": "conv1d", "out_channels": 32, "kernel_size": 5, "stride": 1,
         "padding": 2, "channels_per_device": 16, "num_devices": 2},
        {"type": "conv1d", "out_channels": 48, "kernel_size": 5, "stride": 2,
         "padding": 2, "channels_per_device": 16, "num_devices": 3},
        {"type": "fc"},
    ]


def get_distributed_layers_N8():
    """N=8 total: Head(1) + 2 L0 workers + 4 L1 workers + 1 Tail = 8"""
    return [
        {"type": "conv1d", "out_channels": 32, "kernel_size": 5, "stride": 1,
         "padding": 2, "channels_per_device": 16, "num_devices": 2},
        {"type": "conv1d", "out_channels": 48, "kernel_size": 5, "stride": 2,
         "padding": 2, "channels_per_device": 12, "num_devices": 4},
        {"type": "fc"},
    ]


def get_distributed_layers_N10():
    """N=10 total: Head(1) + 2 L0 workers + 6 L1 workers + 1 Tail = 10"""
    return [
        {"type": "conv1d", "out_channels": 32, "kernel_size": 5, "stride": 1,
         "padding": 2, "channels_per_device": 16, "num_devices": 2},
        {"type": "conv1d", "out_channels": 48, "kernel_size": 5, "stride": 2,
         "padding": 2, "channels_per_device": 8, "num_devices": 6},
        {"type": "fc"},
    ]


def get_centralized_layers():
    """Same architecture but single device."""
    return [
        {"type": "conv1d", "out_channels": 32, "kernel_size": 5, "stride": 1,
         "padding": 2, "channels_per_device": 32},
        {"type": "conv1d", "out_channels": 48, "kernel_size": 5, "stride": 2,
         "padding": 2, "channels_per_device": 48},
        {"type": "fc"},
    ]


def main():
    root = Path(__file__).parent
    configs_dir = root / "configs"

    print("=" * 60)
    print("  Generating ALL experiment configs")
    print("=" * 60)

    # =================================================================
    #  PART 1: All 5 datasets × {Centralized, Distributed N=7}
    # =================================================================
    print("\n--- PART 1: Centralized + Distributed (N=7) for all datasets ---")
    for ds_name, (input_len, num_cls, max_train, max_test) in DATASETS.items():
        print(f"\n  Dataset: {ds_name} (len={input_len}, classes={num_cls})")

        # Centralized
        cfg = build_config(ds_name, input_len, num_cls, max_train, max_test,
                           get_centralized_layers(), centralized=True)
        write_config(cfg, str(configs_dir / "part1" / f"centralized_{ds_name}" / "model_config.json"))

        # Distributed N=7
        cfg = build_config(ds_name, input_len, num_cls, max_train, max_test,
                           get_distributed_layers_N7(), centralized=False)
        write_config(cfg, str(configs_dir / "part1" / f"distributed_N7_{ds_name}" / "model_config.json"))

    # =================================================================
    #  PART 2: All 5 datasets × {N=7, N=8, N=10} (N=7 reuses Part 1 results)
    # =================================================================
    print("\n--- PART 2: Distributed N=7,8,10 for all datasets ---")
    for ds_name, (input_len, num_cls, max_train, max_test) in DATASETS.items():
        print(f"\n  Dataset: {ds_name} (Part 2)")

        # N=7 — config is same as Part 1, still generate for completeness
        cfg = build_config(ds_name, input_len, num_cls, max_train, max_test,
                           get_distributed_layers_N7(), centralized=False)
        write_config(cfg, str(configs_dir / "part2" / f"distributed_N7_{ds_name}" / "model_config.json"))

        # N=8
        cfg = build_config(ds_name, input_len, num_cls, max_train, max_test,
                           get_distributed_layers_N8(), centralized=False)
        write_config(cfg, str(configs_dir / "part2" / f"distributed_N8_{ds_name}" / "model_config.json"))

        # N=10
        cfg = build_config(ds_name, input_len, num_cls, max_train, max_test,
                           get_distributed_layers_N10(), centralized=False)
        write_config(cfg, str(configs_dir / "part2" / f"distributed_N10_{ds_name}" / "model_config.json"))

    print("\n" + "=" * 60)
    print("  All configs generated!")
    print("=" * 60)

    # Print summary
    print("\nConfig summary:")
    for p in sorted(configs_dir.rglob("model_config.json")):
        rel = p.relative_to(configs_dir)
        with open(p) as f:
            c = json.load(f)
        n_dev = sum(l.get("num_devices", 1) for l in c["layers"])
        total_dev = n_dev + 1 if c["global"]["memory_limit_bytes"] > 0 else 1
        ds = c["global"]["dataset"]
        mem = c["global"]["memory_limit_bytes"]
        print(f"  {str(rel):<55} {ds:>17}  devices={total_dev}  mem_limit={mem}")


if __name__ == "__main__":
    main()
