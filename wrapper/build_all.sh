./clean_premake.sh
sudo rm -rf output
premake5 --cc=clang gmake
make
sudo ./build/Debug/WrapperModule \
  --build-path="/data/anubhav/torch-mlir/build" \
  --output-dir="$(pwd)/output" \
  --cc="/usr/bin/clang++" \
  --pipeline="baseline_pipeline.json" \
alexnet_torch.mlir 
