#!/usr/bin/env python3
"""
generate_config.py — PiLot CNN Model Config Generator

Generates JSON config files for both PiLot_Centralized and PiLot_Distributed.
Each project keeps exactly ONE config (model_config.json), overwritten each run.

╔═══════════════════════════════════════════════════════════════════╗
║  HOW TO USE:                                                     ║
║  1. Edit the MODEL STRUCTURE section below to define your CNN.   ║
║  2. Run: python generate_config.py                               ║
║  3. Both projects get configs/model_config.json (overwritten).   ║
║                                                                  ║
║  You can also override dataset/epochs/lr from the command line:  ║
║    python generate_config.py --dataset ECG5000 --epochs 50       ║
║    python generate_config.py --list-datasets                     ║
╚═══════════════════════════════════════════════════════════════════╝
"""

import json
import os
import math
import argparse
from pathlib import Path


# =====================================================================
#  ██  GLOBAL TRAINING SETTINGS  — edit these as needed
# =====================================================================
DATASET         = "Cricket_X"    # UCR dataset name (see UCR_DATASETS below)
EPOCHS          = 20             # number of training epochs
LEARNING_RATE   = 0.01           # base learning rate for AdamW
MEMORY_LIMIT    = 262144         # per-device RAM budget in bytes (256 KB, nRF52840 SRAM)
FLASH_MEMORY    = 1048576        # per-device flash budget in bytes (1 MB, nRF52840 flash)


# =====================================================================
#  ██  MODEL STRUCTURE  — edit this list to change the CNN architecture
# =====================================================================
#
#  Each dict is one layer.  The script auto-computes input_length,
#  output_length, in_channels, and in_features from the previous layer,
#  so you only need to specify the fields you care about.
#
#  ── Conv1D layer fields ─────────────────────────────────────────────
#    "type":                "conv1d"
#    "out_channels":        number of output filters          (required)
#    "kernel_size":         convolution kernel width           (default: 5)
#    "stride":              convolution stride                 (default: 1)
#    "padding":             zero-padding on each side          (default: kernel_size // 2)
#    "channels_per_device": filters per distributed worker     (default: out_channels)
#    "num_devices":         number of distributed workers      (default: auto = out_channels / channels_per_device)
#
#  ── FC (fully-connected) layer fields ───────────────────────────────
#    "type":                "fc"
#    "out_features":        number of output classes            (default: auto from dataset)
#    "pooling":             ["avg", "max"] for dual pooling     (default: ["avg", "max"])
#    "num_devices":         always 1 for FC                     (default: 1)
#
#  The FC layer's in_features is auto-computed as 2 × last_conv_out_channels
#  (because DualPooling concatenates GAP and GMP).
#
#  ── Example: 2-layer model (default) ────────────────────────────────
#    Layer 0: Conv1D(1→32,  k=5, s=1, p=2)  — keeps length
#    Layer 1: Conv1D(32→48, k=5, s=2, p=2)  — halves length
#    Layer 2: DualPool → FC(96→num_classes)
#
#  ── Example: 3-layer deeper model ───────────────────────────────────
#    Just add another conv1d dict before the fc dict.
#
#  ── Example: wider first layer with more devices ────────────────────
#    Change out_channels to 64 and channels_per_device to 16
#    → num_devices will auto-compute to 4
#

MODEL_STRUCTURE = [
    # ── Conv Layer 0 ──────────────────────────────────────────────
    {
        "type":                "conv1d",
        "out_channels":        32,           # output filters
        "kernel_size":         5,            # kernel width
        "stride":              1,            # stride (1 = keep length)
        "padding":             2,            # zero-pad each side
        "channels_per_device": 16,           # filters per worker
        # "num_devices":       2,            # auto = out_channels / channels_per_device
    },
    # ── Conv Layer 1 ──────────────────────────────────────────────
    {
        "type":                "conv1d",
        "out_channels":        48,           # output filters
        "kernel_size":         5,            # kernel width
        "stride":              2,            # stride (2 = halve length)
        "padding":             2,            # zero-pad each side
        "channels_per_device": 16,           # filters per worker
        # "num_devices":       3,            # auto = out_channels / channels_per_device
    },
    # ── Uncomment to add a 3rd conv layer ─────────────────────────
    # {
    #     "type":                "conv1d",
    #     "out_channels":        64,
    #     "kernel_size":         3,
    #     "stride":              2,
    #     "padding":             1,
    #     "channels_per_device": 16,
    # },
    # ── FC Layer (always last) ────────────────────────────────────
    {
        "type":                "fc",
        # out_features auto-set to num_classes; override here if needed:
        # "out_features":      12,
        "pooling":             ["avg", "max"],   # dual pooling
        "num_devices":         1,
    },
]


# =====================================================================
#  Known UCR datasets: name → (input_length, num_classes)
# =====================================================================
UCR_DATASETS = {
    "Coffee":       (286, 2),
    "ECG5000":      (140, 5),
    "Cricket_X":    (300, 12),
    "FaceAll":      (131, 14),
    "GunPoint":     (150, 2),
    "FaceFour":     (350, 4),
    "Lightning2":   (637, 2),
    "Lightning7":   (319, 7),
    "Beef":         (470, 5),
    "OliveOil":     (570, 4),
    "Car":          (577, 4),
    "Plane":        (144, 7),
    "Earthquakes":  (512, 2),
    "Wafer":        (152, 2),
    "Yoga":         (426, 2),
    "FordA":        (500, 2),
    "FordB":        (500, 2),
    "SwedishLeaf":  (128, 15),
    "Adiac":        (176, 37),
    "WordSynonyms": (270, 25),
    "TwoPatterns":  (128, 4),
    "CBF":          (128, 3),
    "Trace":        (275, 4),
    "MedicalImages":(99,  10),
    "Fish":         (463, 7),
    "OSULeaf":      (427, 6),
    "50words":      (270, 50),
}


# =====================================================================
#  Build & Validate — turns MODEL_STRUCTURE into a complete config
# =====================================================================
CONFIG_FILENAME = "model_config.json"


SIZEOF_FLOAT = 4  # bytes per float32


def conv_output_length(input_length, kernel_size, stride, padding):
    """Compute 1-D convolution output length."""
    return (input_length + 2 * padding - kernel_size) // stride + 1


def conv_memory_per_device(in_ch, cpd, k, in_len, out_len):
    """Memory consumed by one distributed worker for a conv1d layer.

    Includes: weights, bias, GroupNorm γ/β, input & output activations,
    weight gradients, and AdamW optimizer states (m, v per weight).
    """
    weights     = in_ch * k * cpd                # conv weights
    bias        = cpd                            # conv bias
    gn_params   = 2 * cpd                        # GroupNorm γ + β
    params      = weights + bias + gn_params

    input_act   = in_ch * in_len                 # input activation
    output_act  = cpd * out_len                  # output activation (post-conv)
    gn_act      = cpd * out_len                  # GroupNorm intermediate (mean/var cache)

    grad        = params                         # gradient storage (same size as params)
    adam_states = 2 * params                     # AdamW m + v

    total_floats = params + input_act + output_act + gn_act + grad + adam_states
    return total_floats * SIZEOF_FLOAT


def conv_ops_per_device(in_ch, cpd, k, out_len):
    """Number of arithmetic operations per device for a conv1d layer.

    Counts multiply-accumulate as 2 ops (1 mul + 1 add).
    Includes: convolution, bias add, GroupNorm, LeakyReLU.
    """
    conv_mac   = cpd * out_len * in_ch * k * 2   # conv multiply-adds
    bias_add   = cpd * out_len                    # bias
    gn_ops     = cpd * out_len * 5                # mean, var, normalize, scale, shift
    relu_ops   = cpd * out_len                    # LeakyReLU comparison + mul
    return conv_mac + bias_add + gn_ops + relu_ops


def fc_memory_per_device(fc_in, fc_out, last_out_ch, input_length):
    """Memory consumed by the FC/classifier device.

    Includes: pooling input, pooled vector, FC weights+bias,
    output, gradients, AdamW states, softmax buffer.
    """
    pool_input  = last_out_ch * input_length     # full feature map before pooling
    pooled_vec  = fc_in                          # after GAP+GMP concat
    fc_weights  = fc_in * fc_out                 # FC weight matrix
    fc_bias     = fc_out                         # FC bias
    fc_output   = fc_out                         # logits / softmax output
    params      = fc_weights + fc_bias

    grad        = params                         # gradient storage
    adam_states = 2 * params                     # AdamW m + v

    total_floats = pool_input + pooled_vec + params + fc_output + grad + adam_states
    return total_floats * SIZEOF_FLOAT


def fc_ops_per_device(fc_in, fc_out, last_out_ch, input_length):
    """Number of arithmetic operations for the FC/classifier device.

    Includes: GAP, GMP, matmul, bias, softmax.
    """
    gap_ops     = last_out_ch * input_length      # sum + divide
    gmp_ops     = last_out_ch * input_length      # comparisons
    dropout_ops = fc_in                           # mask generation + mul
    fc_mac      = fc_in * fc_out * 2              # matmul multiply-adds
    bias_add    = fc_out                          # bias
    softmax_ops = fc_out * 3                      # exp + sum + divide
    return gap_ops + gmp_ops + dropout_ops + fc_mac + bias_add + softmax_ops


def build_config(dataset, input_length, num_classes, epochs, lr, memory_limit,
                 flash_memory, structure, centralized=False):
    """
    Take the user-defined MODEL_STRUCTURE list and resolve all
    auto-computed fields.  Returns the full JSON-ready config dict.

    centralized=True  → single device: num_devices=1, channels_per_device=out_channels,
                        memory & ops reflect the whole layer on one device.
    centralized=False → distributed: honours channels_per_device / num_devices from
                        MODEL_STRUCTURE, memory & ops are per-worker.
    """
    layers = []
    in_ch = 1                   # UCR = univariate
    cur_length = input_length

    for idx, raw in enumerate(structure):
        ltype = raw["type"]

        if ltype == "conv1d":
            out_ch   = raw["out_channels"]
            k        = raw.get("kernel_size", 5)
            s        = raw.get("stride", 1)
            p        = raw.get("padding", k // 2)
            out_len  = conv_output_length(cur_length, k, s, p)

            assert out_ch > 0,  f"Layer {idx}: out_channels must be > 0"
            assert out_len > 0, f"Layer {idx}: output_length={out_len} ≤ 0 (check kernel/stride/padding)"

            if centralized:
                # Single device handles ALL channels
                cpd   = out_ch
                n_dev = 1
            else:
                cpd   = raw.get("channels_per_device", out_ch)
                n_dev = raw.get("num_devices", max(1, out_ch // cpd))
                assert out_ch % cpd == 0, (
                    f"Layer {idx}: out_channels ({out_ch}) must be divisible by "
                    f"channels_per_device ({cpd})"
                )

            mem = conv_memory_per_device(in_ch, cpd, k, cur_length, out_len)
            ops = conv_ops_per_device(in_ch, cpd, k, out_len)

            layer = {
                "id":                  idx,
                "type":                "conv1d",
                "in_channels":         in_ch,
                "out_channels":        out_ch,
                "kernel_size":         k,
                "stride":              s,
                "padding":             p,
                "input_length":        cur_length,
                "output_length":       out_len,
            }
            if centralized:
                layer["memory_bytes"] = mem
                layer["ops"]          = ops
            else:
                layer["channels_per_device"] = cpd
                layer["num_devices"]         = n_dev
                layer["memory_per_device_bytes"] = mem
                layer["ops_per_device"]          = ops
            layers.append(layer)
            in_ch = out_ch
            cur_length = out_len

        elif ltype == "fc":
            assert len(layers) > 0, "FC layer must come after at least one conv1d"
            fc_in  = layers[-1]["out_channels"] * 2   # dual pooling
            fc_out = raw.get("out_features", num_classes)
            pool   = raw.get("pooling", ["avg", "max"])
            n_dev  = 1  # FC is always on one device
            last_out_ch = layers[-1]["out_channels"]

            mem = fc_memory_per_device(fc_in, fc_out, last_out_ch, cur_length)
            ops = fc_ops_per_device(fc_in, fc_out, last_out_ch, cur_length)

            layer = {
                "id":            idx,
                "type":          "fc",
                "input_length":  cur_length,
                "pooling":       pool,
                "in_features":   fc_in,
                "out_features":  fc_out,
            }
            if centralized:
                layer["memory_bytes"] = mem
                layer["ops"]          = ops
            else:
                layer["num_devices"]         = n_dev
                layer["memory_per_device_bytes"] = mem
                layer["ops_per_device"]          = ops
            layers.append(layer)
        else:
            raise ValueError(f"Layer {idx}: unknown type '{ltype}'")

    # Sanity: last layer must be FC
    assert layers[-1]["type"] == "fc", "Last layer must be 'fc'"

    cfg = {
        "model": {
            "name":    f"nRF52840_UniformCNN_{dataset}",
            "version": "2.0",
        },
        "global": {
            "dataset":            dataset,
            "epochs":             epochs,
            "num_classes":        num_classes,
            "input_length":       input_length,
            "memory_limit_bytes": 0 if centralized else memory_limit,
            "flash_memory_bytes": 0 if centralized else flash_memory,
            "learning_rate":      lr,
        },
        "layers": layers,
    }
    cfg["_centralized"] = centralized   # internal flag for print_summary
    return cfg


def _fmt_bytes(b):
    """Human-friendly byte size."""
    if b < 1024:
        return f"{b} B"
    elif b < 1024 * 1024:
        return f"{b / 1024:.1f} KB"
    else:
        return f"{b / (1024*1024):.2f} MB"


def _fmt_ops(n):
    """Human-friendly operation count."""
    if n < 1_000:
        return f"{n}"
    elif n < 1_000_000:
        return f"{n / 1_000:.1f} KOps"
    else:
        return f"{n / 1_000_000:.2f} MOps"


def print_summary(config, label=""):
    """Print a human-readable model summary with memory & ops."""
    g = config["global"]
    layers = config["layers"]
    is_centralized = config.get("_centralized", False)
    total_params = 0
    total_mem    = 0
    total_ops    = 0
    max_dev_mem  = 0
    max_dev_ops  = 0

    header = f"  PiLot CNN — {g['dataset']}"
    if label:
        header += f"  [{label}]"
    print(f"\n{'='*75}")
    print(header)
    print(f"  Input: 1 × {g['input_length']}   Classes: {g['num_classes']}")
    print(f"{'='*75}")

    for l in layers:
        mem = l.get("memory_per_device_bytes", l.get("memory_bytes", 0))
        ops = l.get("ops_per_device", l.get("ops", 0))
        n_dev = l.get("num_devices", 1)
        total_mem += mem * n_dev
        total_ops += ops * n_dev
        if mem > max_dev_mem:
            max_dev_mem = mem
        if ops > max_dev_ops:
            max_dev_ops = ops

        if l["type"] == "conv1d":
            w = l["out_channels"] * l["in_channels"] * l["kernel_size"]
            b = l["out_channels"]
            params = w + b
            total_params += params
            if is_centralized:
                print(f"  Layer {l['id']}: Conv1D({l['in_channels']}→{l['out_channels']}, "
                      f"k={l['kernel_size']}, s={l['stride']}, p={l['padding']}) "
                      f"→ {l['output_length']}")
                print(f"           → GroupNorm(8) → LeakyReLU(0.01)")
                print(f"           params={params}  |  "
                      f"mem={_fmt_bytes(mem)}  |  "
                      f"ops={_fmt_ops(ops)}")
            else:
                print(f"  Layer {l['id']}: Conv1D({l['in_channels']}→{l['out_channels']}, "
                      f"k={l['kernel_size']}, s={l['stride']}, p={l['padding']}) "
                      f"→ {l['output_length']}  [{n_dev} device(s)]")
                print(f"           → GroupNorm(8) → LeakyReLU(0.01)")
                print(f"           params={params}  |  "
                      f"mem/device={_fmt_bytes(mem)}  |  "
                      f"ops/device={_fmt_ops(ops)}")
        elif l["type"] == "fc":
            params = l["in_features"] * l["out_features"] + l["out_features"]
            total_params += params
            print(f"  Layer {l['id']}: DualPool(GAP+GMP) → Dropout(0.2) → "
                  f"FC({l['in_features']}→{l['out_features']})")
            if is_centralized:
                print(f"           params={params}  |  "
                      f"mem={_fmt_bytes(mem)}  |  "
                      f"ops={_fmt_ops(ops)}")
            else:
                print(f"           params={params}  |  "
                      f"mem/device={_fmt_bytes(mem)}  |  "
                      f"ops/device={_fmt_ops(ops)}")

    print(f"{'─'*75}")
    print(f"  Total parameters : {total_params}  "
          f"({total_params * SIZEOF_FLOAT / 1024:.1f} KB weights)")

    if is_centralized:
        # One device runs everything — memory is cumulative, no limit
        print(f"  Total memory     : {_fmt_bytes(total_mem)}  "
              f"(single device, all layers)")
        print(f"  Total ops        : {_fmt_ops(total_ops)}  "
              f"(single device, all layers)")
    else:
        mem_limit = g.get("memory_limit_bytes", 0)
        flash_limit = g.get("flash_memory_bytes", 0)
        print(f"  Total memory     : {_fmt_bytes(total_mem)}  "
              f"(across all devices)")
        if mem_limit:
            status = '✓ fits' if max_dev_mem <= mem_limit else '✗ EXCEEDS'
            print(f"  Peak device RAM  : {_fmt_bytes(max_dev_mem)}"
                  f"  {status} {_fmt_bytes(mem_limit)} RAM limit")
        else:
            print(f"  Peak device RAM  : {_fmt_bytes(max_dev_mem)}")
        if flash_limit:
            print(f"  Flash per device : {_fmt_bytes(flash_limit)}  "
                  f"(dataset storage)")
        print(f"  Total ops        : {_fmt_ops(total_ops)}  "
              f"(across all devices)")
        print(f"  Peak device ops  : {_fmt_ops(max_dev_ops)}")

    print(f"  Optimizer: AdamW (lr={g['learning_rate']}, wd=0.0003)")
    print(f"  LR Schedule: Cosine Annealing (T_max=60, warmup=3 epochs)")
    print(f"  Epochs: {g['epochs']}  |  Early Stopping: patience=50")
    print(f"{'='*75}\n")


def write_config(config, path):
    """Write config dict to a JSON file (strips internal keys)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Remove internal flags before writing
    out = {k: v for k, v in config.items() if not k.startswith("_")}
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  ✓ Written: {path}")


# =====================================================================
#  CLI
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Generate PiLot CNN model configs.  "
                    "Edit MODEL_STRUCTURE at the top of this file to change layers."
    )
    parser.add_argument("--dataset", type=str, default=None,
                        help=f"UCR dataset name (default: {DATASET})")
    parser.add_argument("--epochs", type=int, default=None,
                        help=f"Training epochs (default: {EPOCHS})")
    parser.add_argument("--lr", type=float, default=None,
                        help=f"Learning rate (default: {LEARNING_RATE})")
    parser.add_argument("--memory-limit", type=int, default=None,
                        help=f"Per-device RAM in bytes (default: {MEMORY_LIMIT})")
    parser.add_argument("--flash-memory", type=int, default=None,
                        help=f"Per-device flash in bytes (default: {FLASH_MEMORY})")
    parser.add_argument("--list-datasets", action="store_true",
                        help="List all known UCR datasets and exit.")
    args = parser.parse_args()

    if args.list_datasets:
        print(f"\n{'Dataset':<20} {'Length':>7} {'Classes':>8}")
        print("─" * 37)
        for name, (length, classes) in sorted(UCR_DATASETS.items()):
            print(f"  {name:<18} {length:>7} {classes:>8}")
        print()
        return

    # Resolve settings: CLI overrides → file-level constants → dataset table
    dataset      = args.dataset      or DATASET
    epochs       = args.epochs       or EPOCHS
    lr           = args.lr           or LEARNING_RATE
    memory_limit = args.memory_limit or MEMORY_LIMIT
    flash_memory = args.flash_memory if hasattr(args, 'flash_memory') and args.flash_memory else FLASH_MEMORY

    if dataset not in UCR_DATASETS:
        print(f"Warning: '{dataset}' not in known dataset list.")
        input_length = int(input("  input_length: "))
        num_classes  = int(input("  num_classes:  "))
    else:
        input_length, num_classes = UCR_DATASETS[dataset]

    # Build separate configs for each project
    dist_config = build_config(dataset, input_length, num_classes,
                               epochs, lr, memory_limit, flash_memory,
                               MODEL_STRUCTURE, centralized=False)
    cent_config = build_config(dataset, input_length, num_classes,
                               epochs, lr, memory_limit, flash_memory,
                               MODEL_STRUCTURE, centralized=True)

    print_summary(dist_config, label="Distributed")
    print_summary(cent_config, label="Centralized")

    # Write to both projects (single file, overwritten each run)
    root = Path(__file__).parent
    write_config(dist_config, str(root / "PiLot_Distributed" / "configs" / CONFIG_FILENAME))
    write_config(cent_config, str(root / "PiLot_Centralized" / "configs" / CONFIG_FILENAME))
    print(f"Config written to {CONFIG_FILENAME} in both projects (overwrites previous).\n")


if __name__ == "__main__":
    main()