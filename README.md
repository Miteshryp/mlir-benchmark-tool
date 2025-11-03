# MLIR Benchmark Tool

This repository provides an automated command line tool to benchmark and compare the performance of Torch based models on an operator level granularity using Torch-MLIR.
It builds MLIR modules, runs configurable pipelines, and visualizes performance metrics and optimization effects.

---

## 📁 Important Files

```
mlir-benchmark-tool/
├── torch-mlir/                     # External submodule (Torch-MLIR, switch to cs6886 branch)
│   ├── externals/                  # Nested submodule (also switch to cs6886)
│   │   └── llvm-project/           # External dependency (also a git submodule)
│   └── build_tools/build_standalone.sh
│
├── wrapper/
│   ├── ext/perf-cpp                # External C++ library (git submodule)
│   ├── graph-gen/                  # Scripts/utilities for graph generation
│   ├── graphs/
│   │   └── o2_comparison/          # Generated comparative graphs
│   ├── include/                    # Header files for wrapper source
│   ├── src/                        # Implementation sources
│   ├── alexnet_linalg_generics.mlir
│   ├── alexnet_torch.mlir
│   ├── baseline_pipeline.json      # Pipeline configuration for baseline benchmark
│   ├── o2_pipeline.json            # Pipeline configuration for O2 benchmark
│   ├── benchmark_pipelines.sh      # Master benchmark runner
│   ├── clean_past_benchmarks.sh    # Utility to clear previous results
│   ├── clean_premake.sh            # Utility to clean Premake build artifacts
│   ├── run_structural_pass.sh      # Runs a structural MLIR transformation pass
│   ├── premake5.lua                # Premake configuration file
│   ├── requirements.txt            # Python dependencies
│
└── .gitmodules
```

---

## ⚙️ Setup Instructions

Before proceeding with the following steps, please ensure that Premake is installed on your system. This is the build tool which is used to build the wrapper module

```
sudo apt install premake5      # Linux 
brew install premake           # macOS
```


Once this is done, the following steps can be followed:

### 1. Clone the Repository with Submodules

```bash
git clone --recurse-submodules https://github.com/Miteshryp/mlir-benchmark-tool.git
cd mlir-benchmark-tool
```

If already cloned without submodules:

```bash
git submodule update --init --recursive
```

---

### 2. Switch Torch MLIR and LLVM Repo's to `cs6886` Branch

This is a custom branch created for in-tree pass builds.

```bash
cd torch-mlir
git checkout cs6886
git submodule update --init --recursive
cd externals/llvm-project
git checkout cs6886
```

The main repository can be left at the `main` branch itself:

---

### 3. Create and Activate the Conda Environment

```bash
cd wrapper
conda create -n mlir-benchmark python=3.11 -y
conda activate mlir-benchmark
pip install -r requirements.txt
```

---

### 4. Build Torch-MLIR

```bash
cd ../torch-mlir
./build_tools/build_standalone.sh
```

> ⚠️ Ensure both `torch-mlir` and its `externals` folder are switched to the correct branch **before** building.

---

### 5. Build the Wrapper Project

After Torch-MLIR is built and the conda environment is active, we can execute the pipeline using helper scripts provided in the wrapper folder:

```bash
cd ../wrapper
sudo ./benchmark_pipelines.sh
```

---

## 🚀 Running Benchmarks

### Run All Pipelines

```bash
bash benchmark_pipelines.sh
```

This executes:

* The **baseline** pipeline (`baseline_pipeline.json`)
* The **O2 optimized** pipeline (`o2_pipeline.json`)
* Stores performance data in `baseline_output` and  `o2_output` folders created after benchmark runs 
* Generates visual comparison graphs in `graphs/o2_comparison/`

### Clean Previous Results (Do this if a previous run exists)

```bash
bash clean_past_benchmarks.sh
```

### Generate Linalg.Generic based Structural Information

```bash
bash run_structural_pass.sh
```

---

## 📊 Output Summary

| Output Folder                                | Description                                    |
| -------------------------------------------- | ---------------------------------------------- |
| `graphs/o2_comparison/`                      | Visual comparison of O2 vs baseline benchmarks |
|                                              |                                                |
| `baseline_pipeline.json`, `o2_pipeline.json` | Pipeline definitions used for benchmarking     |

---

## 🧩 Developer Notes

* **Premake Project:** Configured through `premake5.lua`. Use `clean_premake.sh` to reset build artifacts.
* **Graph Generation:** Located in `graph-gen/` and outputs plots into `graphs/o2_comparison/`.
* **Ext Library:** The `wrapper/ext/` directory is an external dependency required for successful compilation.
* **Environment:** All Python-based graph utilities and benchmark management scripts depend on the Conda environment.

---

## 🧠 Troubleshooting

**Error: Missing externals or branch mismatch**

```bash
git submodule update --init --recursive
git checkout cs6886
```

**Error: Conda environment not found**

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mlir-benchmark
```

**Error: Premake not installed**

```bash
sudo apt install premake5      # Linux
brew install premake           # macOS
```

---

## 👥 Authors

* **Mitesh Sharma** — Core developer

---

## 📄 Citation

If you use this repository in academic work or reports:

```
@misc{mlir-benchmark-tool,
  author = {Mitesh Sharma},
  title  = {MLIR Benchmark Tool},
  year   = {2025},
  note   = {https://github.com/Miteshryp/mlir-benchmark-tool}
}
```

