./clean_premake.sh
rm -rf output
premake5 --cc=clang gmake
make
./build/Debug/WrapperModule \
  --build-path="/Volumes/mitesh-ssd/Dev/torch-mlir/build" \
  --output-dir="/Volumes/mitesh-ssd/Dev/mlir-benchmark-tool/wrapper/output" \
  --cc="/usr/bin/clang++" \
resnet_torch_dialect.mlir 
