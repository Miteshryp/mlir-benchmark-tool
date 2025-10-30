#include "nlohmann/json_fwd.hpp"
#include <Python.h>
#include <argparse/argparse.hpp>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <nlohmann/json.hpp>
#include <numpy/arrayobject.h>
#include <numpy/ndarraytypes.h>
#include <sstream>
#include <string>

#include "command_manager.h"
#include "utils.h"

namespace fs = std::filesystem;

// Type Alias
using json = nlohmann::json;

int main(int argc, char **args) {
  // Take argument as model name
  // Input: <model-mlir-file>

  std::cout << "We're entering here? " << std::endl;
  argparse::ArgumentParser program("torch-metric-collector");

  program.add_argument("-B", "--build-path")
      .help("Path to a Torch MLIR build")
      .required();

  program.add_argument("--output-dir")
      .help("Folder path to contain the output produced by the tool. "
            "Must be an absolute path")
      .default_value(fs::current_path().append(get_timestamp_string()));

  program.add_argument("--pass-logs")
      .help("Enables Pass lowering output in temp folder")
      .default_value(false);

  program.add_argument("--output-logs")
      .help("Enables output logs for each kernel run")
      .default_value(false);
  // program.add_argument("--torch-mlir-build-path")
  // .help("Path to a Torch MLIR build")
  // .required();

  program.add_argument("--cc")
      .help("Path to compiler to be used for object file generation")
      .default_value("/usr/bin/clang++");

  program.add_argument("--pipeline")
      .help("Path to pipeline specified in JSON file")
      .default_value(fs::current_path().append("pipeline.json"));

  program.add_argument("model-file")
      .help("Torch-MLIR file for the model to be benchmarked")
      .required();

  try {
    program.parse_args(argc, args);
  } catch (const std::exception &err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return 1;
  }

  // For now, the torch and llvm setup are being built into the same build, so
  // seperation at the interface level can be skipped for now
  std::string buildPath = program.get<std::string>("--build-path");
  // std::string pipelineJsonPath = program.get<std::string>("--pipeline");
  std::string pipelineJsonPath = "pipeline.json";

  std::string outputFolderPath = program.get<std::string>("--output-dir");

  std::string compiler_path = program.get<std::string>("--cc");
  std::string model_file = program.get<std::string>("model-file");

  // Setting up the Command Manager
  CommandManager::set_llvm_install_path(buildPath);
  CommandManager::set_torch_install_path(buildPath);
  CommandManager::set_compiler_executable(compiler_path);
  CommandManager::set_output_folder(outputFolderPath);
  CommandManager::set_pipeline_json_filepath(pipelineJsonPath);
  CommandManager::initialise_environment();

  std::cout << "Pipeline path: " << pipelineJsonPath << std::endl;

  // Lowering the model
  CommandManager::isolate_torch_kernels(model_file);

  // Test
  std::vector<std::string> operation_types =
      CommandManager::get_operation_types();
  for (auto s : operation_types)
    std::cout << "Operation Types: " << s << std::endl;

  // Iterate over operator types detected in the source program
  for (auto op_type : operation_types) {
    // std::string op_type = fs::path(op_relative_path).filename();
    // fs::path folder_path(fs::current_path().append("lowerings/" + op_type));
    fs::path folder_path(CommandManager::get_lowering_folder().append(op_type));

    // Get the list of outlined files
    std::vector<std::string> kernel_vec =
        CommandManager::get_file_list(folder_path);

    // Iterate over all lowered files in the current operator type folder
    // Tasks
    //  1. Extract argument metadata
    //  2. Lower to LLVM-IR
    //  3. Create executable
    //  4. Execute and Time it
    //  5. Detect the corresponding python kernel and time it
    //  6. Generate aggregate and comparative results

    for (auto &kernel_file : kernel_vec) {
      // std::cout << "This is kernel vec: " << kernel_file << std::endl;

      fs::path mlirFilePath = fs::path(folder_path).append(kernel_file);
      fs::path output_json =
          fs::path(folder_path).append(kernel_file + ".json");

      // Log file path for debugging purposes (just in case)
      fs::path log_file = fs::path(folder_path)
                              .parent_path()
                              .append(std::string("gen_log_") +
                                      mlirFilePath.filename().c_str() + ".log");

      std::cout << "Generating Metadata: " << mlirFilePath << "\n";
      CommandManager::generate_metadata_json(mlirFilePath, output_json);

      // Lower the file to .ll format
      fs::path ll_object = CommandManager::generate_ll_file(mlirFilePath);
      // fs::path llvm_dialect_filepath =
      //     CommandManager::lower_to_llvm_dialect(mlir_file_path);
      //
      // fs::path ll_object =
      //     CommandManager::compile_llvm_dialect(llvm_dialect_filepath);

      std::cout << "Starting Execution: \n";
      std::vector<std::map<std::string, double>> time_metrics =
          CommandManager::execute_with_parameters(ll_object, output_json);

      auto aggregate_metrics = [&]() -> std::vector<float> {
        for (auto &run_metrics : time_metrics) {
        }
      };

      // mlir_output, mlir_metrics =
      // CommandManager::execute_with_parameters(ll_filepath, json_filename)
      //
      // py_output, python_metrics =
      // CommandManager::execute_py_kernel(op_type)
      //
      // verifyOutputs(mlir_output, py_output)
      //
      // CommandManager::generate_comparative_results(mlir_metrics,
      // python_metrics)
    }
    // Dump the aggregated output into a file
  }

  // Lowering command follows file structure
  // lowerings/<type-of-op>/<kernel-name>.mlir
}
