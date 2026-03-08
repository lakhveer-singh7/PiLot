#!/usr/bin/env python3
"""
Generate all config files for all datasets and both models.
Creates dataset-specific configs in each project's configs/ directory.

Datasets: Cricket_X (300, 12 classes), ECG5000 (140, 5 classes), FaceAll (131, 14 classes)
Architecture: 2 Conv layers + FC (7 distributed devices: 1 Head + 2 L0 + 3 L1 + 1 Tail)
"""

import json
import os
import math

SIZEOF_FLOAT = 4

DATASETS = {
    "Cricket_X": (300, 12),
    "ECG5000":   (140, 5),
    "FaceAll":   (131, 14),
}

EPOCHS = 300          # High max; early stopping at patience=30 will terminate earlier
LEARNING_RATE = 0.01
MEMORY_LIMIT = 262144   # 256 KB per device (nRF52840 SRAM)
FLASH_MEMORY = 1048576  # 1 MB flash per device

# 2 conv layers + 1 FC = 7 distributed devices (1 Head + 2 L0 + 3 L1 + 1 Tail)
MODEL_STRUCTURE = [
    {"type": "conv1d", "out_channels": 32, "kernel_size": 5, "stride": 1, "padding": 2, "channels_per_device": 16},
    {"type": "conv1d", "out_channels": 48, "kernel_size": 5, "stride": 2, "padding": 2, "channels_per_device": 16},
    {"type": "fc", "pooling": ["avg", "max"], "num_devices": 1},
]


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


def build_config(dataset, input_length, num_classes, centralized=False):
    layers = []
    in_ch = 1
    cur_length = input_length

    for idx, raw in enumerate(MODEL_STRUCTURE):
        ltype = raw["type"]
        if ltype == "conv1d":
            out_ch = raw["out_channels"]
            k = raw.get("kernel_size", 5)
            s = raw.get("stride", 1)
            p = raw.get("padding", k // 2)
            out_len = conv_output_length(cur_length, k, s, p)

            if centralized:
                cpd = out_ch
                n_dev = 1
            else:
                cpd = raw.get("channels_per_device", out_ch)
                n_dev = max(1, out_ch // cpd)

            mem = conv_memory_per_device(in_ch, cpd, k, cur_length, out_len)
            ops = conv_ops_per_device(in_ch, cpd, k, out_len)

            layer = {
                "id": idx, "type": "conv1d",
                "in_channels": in_ch, "out_channels": out_ch,
                "kernel_size": k, "stride": s, "padding": p,
                "input_length": cur_length, "output_length": out_len,
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
            fc_in = layers[-1]["out_channels"] * 2
            fc_out = raw.get("out_features", num_classes)
            pool = raw.get("pooling", ["avg", "max"])
            last_out_ch = layers[-1]["out_channels"]
            mem = fc_memory_per_device(fc_in, fc_out, last_out_ch, cur_length)
            ops = fc_ops_per_device(fc_in, fc_out, last_out_ch, cur_length)

            layer = {"id": idx, "type": "fc", "input_length": cur_length,
                     "pooling": pool, "in_features": fc_in, "out_features": fc_out}
            if centralized:
                layer["memory_bytes"] = mem
                layer["ops"] = ops
            else:
                layer["num_devices"] = 1
                layer["memory_per_device_bytes"] = mem
                layer["ops_per_device"] = ops
            layers.append(layer)

    return {
        "model": {"name": f"nRF52840_UniformCNN_{dataset}", "version": "2.0"},
        "global": {
            "dataset": dataset, "epochs": EPOCHS,
            "num_classes": num_classes, "input_length": input_length,
            "memory_limit_bytes": 0 if centralized else MEMORY_LIMIT,
            "flash_memory_bytes": 0 if centralized else FLASH_MEMORY,
            "learning_rate": LEARNING_RATE,
        },
        "layers": layers,
    }


def write_config(config, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Written: {path}")


def main():
    root = os.path.dirname(os.path.abspath(__file__))

    for ds_name, (input_length, num_classes) in DATASETS.items():
        print(f"\n--- {ds_name} (length={input_length}, classes={num_classes}) ---")

        # Centralized config
        cent_cfg = build_config(ds_name, input_length, num_classes, centralized=True)
        cent_path = os.path.join(root, "PiLot_Centralized", "configs", f"config_{ds_name}.json")
        write_config(cent_cfg, cent_path)

        # Distributed config
        dist_cfg = build_config(ds_name, input_length, num_classes, centralized=False)
        dist_path = os.path.join(root, "PiLot_Distributed", "configs", f"config_{ds_name}.json")
        write_config(dist_cfg, dist_path)

        # Print device count summary for distributed
        total_devices = 1  # head
        for layer in dist_cfg["layers"]:
            if layer["type"] == "conv1d":
                total_devices += layer.get("num_devices", 1)
            elif layer["type"] == "fc":
                total_devices += 1  # tail
        print(f"  Distributed: {total_devices} devices total")

    print("\nAll configs generated successfully!")


if __name__ == "__main__":
    main()
