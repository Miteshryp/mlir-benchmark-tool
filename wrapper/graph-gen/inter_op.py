#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Compare perf metrics within each op_type.")
    parser.add_argument("--timings-dir", type=str, default="output/timings", help="Path to timing data")
    parser.add_argument("--metric", type=str, default="cycles", help="Metric column to analyze")
    return parser.parse_args()


args = parse_args()
TIMING_DIR = args.timing_dir
METRIC = args.metric


def load_data(op_type):
    """Load all CSVs for a specific op_type."""
    op_path = os.path.join(TIMING_DIR, op_type)
    if not os.path.isdir(op_path):
        print(f"No directory found for {op_type}")
        return pd.DataFrame()

    data = []
    for fname in os.listdir(op_path):
        if not fname.endswith(".csv"):
            continue

        csv_path = os.path.join(op_path, fname)
        df = pd.read_csv(csv_path)
        if METRIC not in df.columns:
            continue

        avg_value = df[METRIC].mean()
        data.append({"operator": os.path.splitext(fname)[0], METRIC: avg_value})

    return pd.DataFrame(data)

def plot_within_optype(op_type, df):
    """Plot comparative bar chart within one op_type."""
    df = df.sort_values(METRIC, ascending=False)
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df["operator"], df[METRIC])
    plt.title(f"Comparison of {METRIC} within '{op_type}'")
    plt.xlabel("Operator (File ID)")
    plt.ylabel(f"Average {METRIC}")
    plt.xticks(rotation=45, ha="right")

    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.show()

def main():
    op_types = [d for d in os.listdir(TIMING_DIR) if os.path.isdir(os.path.join(TIMING_DIR, d))]
    for op_type in op_types:
        df = load_data(op_type)
        if df.empty:
            continue
        print(f"\n=== {op_type} ===")
        print(df)
        plot_within_optype(op_type, df)

if __name__ == "__main__":
    main()

