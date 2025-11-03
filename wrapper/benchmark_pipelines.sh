
sudo ./clean_premake.sh
sudo ./clean_past_benchmarks.sh


# Build benchmark tool
premake5 --cc=clang gmake
make

# Run baseline
sudo ./build/Debug/WrapperModule \
  --build-path="../torch-mlir/build" \
  --output-dir="$(pwd)/baseline_output" \
  --cc="/usr/bin/clang++" \
  --pipeline="$(pwd)/baseline_pipeline.json" \
alexnet_torch.mlir 



# Run O2
sudo ./build/Debug/WrapperModule \
  --build-path="../torch-mlir/build" \
  --output-dir="$(pwd)/o2_output" \
  --cc="/usr/bin/clang++" \
  --pipeline="$(pwd)/o2_pipeline.json" \
alexnet_torch.mlir 




# Run O3 (Omitted from demo)
# sudo ./build/Debug/WrapperModule \
#   --build-path="/data/anubhav/torch-mlir/build" \
#   --output-dir="$(pwd)/o3_output" \
#   --cc="/usr/bin/clang++" \
#   --pipeline="$(pwd)/o3_pipeline.json" \
# alexnet_torch.mlir 


# Generate graphs
python graph-gen/comparative_inter_op.py \
  --output-dir1 $(pwd)/baseline_output \
  --output-dir2 $(pwd)/o2_output \
  --metric cycles \
  --graphs-dir graphs/o2_comparison \
  --labels "Baseline" "Optimized"

python graph-gen/comparative_intra_op.py \
  --output-dir1 $(pwd)/baseline_output \
  --output-dir2 $(pwd)/o2_output \
  --metric cycles \
  --graphs-dir graphs/o2_comparison \
  --labels "Baseline" "Optimized"
