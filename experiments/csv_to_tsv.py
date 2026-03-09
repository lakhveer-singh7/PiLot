#!/usr/bin/env python3
"""
Convert UCR CSV format (label,v1,v2,...) to TSV format (label\tv1\tv2\t...)
required by RockNet's generate_distributed_config.py.

Usage: python3 csv_to_tsv.py input.csv output.tsv
"""
import sys

def convert(input_path, output_path):
    with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            # Replace commas with tabs
            fout.write(line.replace(',', '\t') + '\n')
    print(f"  Converted: {input_path} -> {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 csv_to_tsv.py <input.csv> <output.tsv>")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])
