#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze perf timing CSVs by op_type.")
    parser.add_argument(
        "--timings-dir",
        type=str,
        default="output/timings",
        help="Path to the root timing directory (default: ./output/timings)"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="cycles",
        help="Metric column to analyze (e.g. 'seconds', 'cycles', 'instructions')"
    )
    return parser.parse_args()

def load_data(timing_dir, metric):
    data = []
    for op_type in os.listdir(timing_dir):
        op_path = os.path.join(timing_dir, op_type)
        if not os.path.isdir(op_path):
            continue

        for fname in os.listdir(op_path):
            if not fname.endswith(".csv"):
                continue

            csv_path = os.path.join(op_path, fname)
            df = pd.read_csv(csv_path)
            if metric not in df.columns:
                continue

            avg_value = df[metric].mean()
            data.append({"op_type": op_type, "file": fname, metric: avg_value})

    return pd.DataFrame(data)

def plot_optype_aggregate(df, metric):
    agg = df.groupby("op_type")[metric].sum().sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(agg.index, agg.values)
    plt.title(f"Total {metric} by Operator Type")
    plt.xlabel("Operator Type")
    plt.ylabel(f"Total {metric}")
    plt.xticks(rotation=45, ha="right")

    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.show()

def main():
    args = parse_args()
    df = load_data(args.timing_dir, args.metric)

    if df.empty:
        print("No valid data found.")
        return

    print(f"Loaded {len(df)} CSV files from {args.timing_dir}")
    print(df.head())

    plot_optype_aggregate(df, args.metric)

if __name__ == "__main__":
    main()

