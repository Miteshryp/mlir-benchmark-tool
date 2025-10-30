



./clean_premake.sh
rm -rf output
premake5 --cc=clang gmake
make
./build/Debug/WrapperModule \
  --build-path="/data/anubhav/torch-mlir/build" \
  --output-dir="$(pwd)/output" \
  --cc="/usr/bin/clang++" \
resnet_torch_dialect.mlir 
