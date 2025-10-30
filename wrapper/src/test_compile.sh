#!/bin/zsh

LIB_PATH=$CONDA_PREFIX/lib
echo $CONDA_PREFIX
echo $LIB_PATH


g++ \
    -I/Users/mitesh/Dev/json-cpp/include \
    -I/Volumes/mitesh-ssd/Dev/mlir-benchmark-tool/wrapper/include \
    -I/opt/homebrew/Cellar/libffi/3.5.2/include \
    -I$LIB_PATH/python3.11/site-packages/numpy/_core/include \
    $(python3-config --cflags) \
    --std=c++20 utils.cpp tensor_fuzzer.cpp command_manager.cpp wrapper.cpp \
    -L$LIB_PATH -lpython3.11 -lffi $(python3-config --ldflags)



#
# g++ \
#     -I/Users/mitesh/Dev/json-cpp/include \
#     -I/Volumes/mitesh-ssd/Dev/mlir-benchmark-tool/wrapper/include \
#     -I/opt/homebrew/Cellar/libffi/3.5.2/include \
#     -I/Users/mitesh/miniconda3/envs/torch-mlir/lib/python3.11/site-packages/numpy/_core/include \
#     $(python3-config --cflags) \
#     --std=c++20 utils.cpp tensor_fuzzer.cpp command_manager.cpp wrapper.cpp \
#     -L/Users/mitesh/miniconda3/envs/torch-mlir/lib -lpython3.11 -lffi $(python3-config --ldflags)
