#!/usr/bin/env python3
import os
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Compare aggregated perf metrics between two benchmark outputs.")
    parser.add_argument("--output-dir1", type=str, required=True, help="Path to first benchmark output directory (contains 'timings' folder).")
    parser.add_argument("--output-dir2", type=str, required=True, help="Path to second benchmark output directory (contains 'timings' folder).")
    parser.add_argument("--metric", type=str, default="task-clock", help="Metric column to compare.")
    parser.add_argument("--labels", nargs=2, default=["Version 1", "Version 2"], help="Labels for the two datasets.")
    parser.add_argument("--graphs-dir", type=str, default="graphs", help="Folder to save generated graphs.")
    return parser.parse_args()

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def get_timings_dir(base_dir):
    """Return the timings subdirectory path if it exists."""
    timings_path = os.path.join(base_dir, "timings")
    if not os.path.isdir(timings_path):
        raise FileNotFoundError(f"Missing 'timings' directory in: {base_dir}")
    return timings_path

def get_optype_dirs(timings_dir):
    """Return list of subdirectories inside 'timings/' (like `ls -d timings/*/`)."""
    return [d for d in glob.glob(os.path.join(timings_dir, "*/")) if os.path.isdir(d)]

def load_dataset(timings_dir, metric):
    """Aggregate average metric for each operator CSV grouped by op_type."""
    data = []
    for op_path in get_optype_dirs(timings_dir):
        op_type = os.path.basename(os.path.normpath(op_path))
        for csv_path in glob.glob(os.path.join(op_path, "*.csv")):
            df = pd.read_csv(csv_path)
            if metric not in df.columns:
                continue
            avg_value = df[metric].mean()
            data.append({"op_type": op_type, metric: avg_value})
    df = pd.DataFrame(data)
    if df.empty:
        return df
    return df.groupby("op_type")[metric].sum().reset_index()

def plot_comparison(df1, df2, metric, labels, save_path):
    merged = pd.merge(df1, df2, on="op_type", suffixes=("_v1", "_v2"))
    if merged.empty:
        print("⚠️ No common op_types to compare.")
        return

    merged = merged.sort_values(by=f"{metric}_v1", ascending=False)
    x = range(len(merged))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar([i - width/2 for i in x], merged[f"{metric}_v1"], width=width, label=labels[0])
    plt.bar([i + width/2 for i in x], merged[f"{metric}_v2"], width=width, label=labels[1])

    plt.xticks(x, merged["op_type"], rotation=45, ha="right")
    plt.ylabel(f"Total {metric}")
    plt.title(f"Inter-OpType Comparison ({metric})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Saved inter-optype comparison graph → {save_path}")

def main():
    args = parse_args()
    ensure_dir(args.graphs_dir)

    tdir1 = get_timings_dir(args.output_dir1)
    tdir2 = get_timings_dir(args.output_dir2)

    df1 = load_dataset(tdir1, args.metric)
    df2 = load_dataset(tdir2, args.metric)
    if df1.empty or df2.empty:
        print("❌ Error: One or both datasets are empty or missing the requested metric.")
        return

    save_path = os.path.join(args.graphs_dir, f"inter_optype_comparison_{args.metric}.png")
    plot_comparison(df1, df2, args.metric, args.labels, save_path)

if __name__ == "__main__":
    main()

