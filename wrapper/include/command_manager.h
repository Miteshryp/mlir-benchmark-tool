#pragma once

#include "nlohmann/json_fwd.hpp"
#include <string>

#include "perfcpp/event_counter.h"
#include "utils.h"

namespace fs = std::filesystem;
using json = nlohmann::json;

// TODO: Use this instead of all the static bullshit you fucking doofus!!
struct CommandManagerSettings {
  std::string compiler;
  fs::path outputFolder;
  fs::path loweringFolder;
  bool enableLogFiles;

  fs::path torch_mlir_install_path;
  fs::path llvm_install_path;
  fs::path torch_opt_exec;
  fs::path mlir_opt_exec;
  fs::path llvm_lib_path;
  fs::path pipeline_json;
};

/*
 * Interface class for all interactions with the compiler passes
 */

class CommandManager {
  static std::string compiler;
  static fs::path outputFolder;
  static fs::path loweringFolder;
  static bool enableLogFiles;
  static bool enableRunLogs;
  // static perf::EventCounter perf_event_counter;
  static unsigned int perf_run_count;

  static fs::path torch_mlir_install_path;
  static fs::path llvm_install_path;
  static fs::path torch_opt_exec;
  static fs::path mlir_opt_exec;
  static fs::path llvm_lib_path;
  static fs::path pipeline_json;

  static std::vector<std::string> perf_metrics;

private:
  /*
   * Routine to read input from the executed command
   */
  static std::string exec(const std::string &cmd);
  static bool verifyParameters();

  static fs::path lower_to_llvm_dialect(const fs::path &mlirFilePath);

  static fs::path compile_llvm_dialect(const fs::path &llvm_mlir_filepath);

public:
  static void set_llvm_install_path(const fs::path &path);
  static void set_torch_install_path(const fs::path &path);
  static void set_compiler_executable(const fs::path &binary);
  static void set_perf_metrics(const std::vector<std::string> &metrics);
  static void set_pass_log_flag(bool flag);
  static void set_run_log_flag(bool flag);
  static void set_perf_sample_run_count(const unsigned int &count);

  static void set_output_folder(const fs::path &output);
  static void set_pipeline_json_filepath(const fs::path &filepath);
  static fs::path get_output_folder();
  static fs::path get_lowering_folder();

  static std::map<std::string, double>
  aggregate_metrics(std::vector<std::map<std::string, double>> &metrics);

  static void initialise_environment();
  static void isolate_torch_kernels(const std::string &filename);
  static std::vector<std::string> get_operation_types();
  static std::string extract_pipeline();

  static std::vector<std::string> get_file_list(const fs::path &folderPath);

  static void generate_metadata_json(const std::string &mlir_filepath,
                                     const std::string &json_filename,
                                     const std::string &log_filename = "");

  static fs::path generate_ll_file(const fs::path &mlirFilePath);
  /*
   * Routine used to execute a command in the terminal, and collect its
   * outputs as an array of strings We will need a delimiter to seperate
   * each output Generally commands will have delimeter as \n, but this
   * options is given just to be on the safer side
   */
  static std::vector<std::string> get_cmd_output(const std::string &cmd,
                                                 char delimiter);
  // static void execute_with_python(fs::path json_filepath,
  //                                 const std::string &op_type);

  static std::vector<std::map<std::string, double>>
  execute_with_parameters(const fs::path &ll_object_filepath,
                          const fs::path &json_filepath);
};
