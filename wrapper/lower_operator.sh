#!/bin/bash


INPUT_FILE=$1

rm -rf module_isolated
mkdir module_isolated

# Checking if the file exists
if [ ! -f $INPUT_FILE ]; then
    echo "$INPUT_FILE does not exist"
fi

# Steps
# 1. Run torch-mlir-opt with passes to lower the outlined operator to tensor-linalg operator form
torch-mlir-opt \
    -pass-pipeline="builtin.module(torch-backend-to-linalg-on-tensors-backend-pipeline)" \
    $INPUT_FILE > lower.mlir



# 2. Convert the linalg and vector dialect operations into llvm operations
mlir-opt lower.mlir \
    --canonicalize --cse \
    --one-shot-bufferize="bufferize-function-boundaries" \
    --convert-linalg-to-loops \
    --convert-scf-to-cf \
    --lower-affine \
    --expand-strided-metadata \
    --finalize-memref-to-llvm \
    --convert-arith-to-llvm \
    --convert-func-to-llvm \
    --convert-cf-to-llvm \
    --reconcile-unrealized-casts \
-o llvm_lower.mlir


# 3. Use mlir-translate to generate the ll file (Not important right now)
mlir-translate --mlir-to-llvmir llvm_lower.mlir > module.ll


  
