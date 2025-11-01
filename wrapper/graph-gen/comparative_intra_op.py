import os
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Compare perf metrics within each op_type across two benchmark outputs.")
    parser.add_argument("--output-dir1", type=str, required=True, help="Path to first benchmark output directory (contains 'timings' folder).")
    parser.add_argument("--output-dir2", type=str, required=True, help="Path to second benchmark output directory (contains 'timings' folder).")
    parser.add_argument("--metric", type=str, default="task-clock", help="Metric column to compare.")
    parser.add_argument("--labels", nargs=2, default=["Version 1", "Version 2"], help="Labels for the two datasets.")
    parser.add_argument("--graphs-dir", type=str, default="graphs", help="Folder to save generated graphs.")
    return parser.parse_args()

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def get_timings_dir(base_dir):
    timings_path = os.path.join(base_dir, "timings")
    if not os.path.isdir(timings_path):
        raise FileNotFoundError(f"Missing 'timings' directory in: {base_dir}")
    return timings_path

def get_optype_dirs(timings_dir):
    return [d for d in glob.glob(os.path.join(timings_dir, "*/")) if os.path.isdir(d)]

def load_optype_data(timings_dir, op_type, metric):
    op_path = os.path.join(timings_dir, op_type)
    if not os.path.isdir(op_path):
        return pd.DataFrame()

    data = []
    for csv_path in glob.glob(os.path.join(op_path, "*.csv")):
        df = pd.read_csv(csv_path)
        if metric not in df.columns:
            continue
        avg_value = df[metric].mean()
        data.append({"operator": os.path.splitext(os.path.basename(csv_path))[0], metric: avg_value})
    return pd.DataFrame(data)

def plot_within_optype(op_type, df1, df2, metric, labels, save_path):
    merged = pd.merge(df1, df2, on="operator", suffixes=("_v1", "_v2"))
    if merged.empty:
        print(f"⚠️ No common operators in {op_type} to compare.")
        return False

    merged = merged.sort_values(by=f"{metric}_v1", ascending=False)
    x = range(len(merged))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar([i - width/2 for i in x], merged[f"{metric}_v1"], width=width, label=labels[0])
    plt.bar([i + width/2 for i in x], merged[f"{metric}_v2"], width=width, label=labels[1])

    plt.xticks(x, merged["operator"], rotation=45, ha="right")
    plt.ylabel(f"Average {metric}")
    plt.title(f"Intra-OpType Comparison for '{op_type}' ({metric})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Saved intra-op graph for '{op_type}' → {save_path}")
    return True

def main():
    args = parse_args()
    tdir1 = get_timings_dir(args.output_dir1)
    tdir2 = get_timings_dir(args.output_dir2)

    op_types_1 = [os.path.basename(os.path.normpath(p)) for p in get_optype_dirs(tdir1)]
    op_types_2 = [os.path.basename(os.path.normpath(p)) for p in get_optype_dirs(tdir2)]
    op_types = sorted(set(op_types_1) | set(op_types_2))

    for op_type in op_types:
        df1 = load_optype_data(tdir1, op_type, args.metric)
        df2 = load_optype_data(tdir2, op_type, args.metric)
        if df1.empty or df2.empty:
            continue

        output_dir = os.path.join(args.graphs_dir, op_type)
        ensure_dir(output_dir)
        save_path = os.path.join(output_dir, f"{op_type}_intra_{args.metric}.png")
        plot_within_optype(op_type, df1, df2, args.metric, args.labels, save_path)

if __name__ == "__main__":
    main()

