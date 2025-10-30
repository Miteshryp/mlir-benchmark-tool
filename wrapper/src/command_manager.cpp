#include "command_manager.h"

// Profiling only possible on linux as of now
#ifdef __linux__
// #include <perfcpp/counter.h>
// #include <perfcpp/event_counter.h>
// #include <perfcpp/hardware_info.h>
#endif

#ifdef __APPLE__
#include <ffi/ffi.h>
#else
#include <ffi.h> // Linux is required if not MACOS (Windows does not have standard FFI library)
#endif

#include "tensor_fuzzer.h"
#include "utils.h"
// #include <Python.h>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <nlohmann/json.hpp>
// #include <numpy/arrayobject.h>
// #include <numpy/ndarraytypes.h>
#include <sstream>

namespace fs = std::filesystem;

ffi_type *create_memref_struct_type(int rank);
void destroy_memref_struct_type(ffi_type *memref_type);

perf::EventCounter CommandManager::perf_event_counter = perf::EventCounter{};
fs::path CommandManager::torch_mlir_install_path;

fs::path CommandManager::torch_opt_exec;
fs::path CommandManager::mlir_opt_exec;

fs::path CommandManager::llvm_install_path;

fs::path CommandManager::pipeline_json;
fs::path CommandManager::llvm_lib_path;

std::string CommandManager::compiler = "/usr/bin/clang++";
fs::path CommandManager::outputFolder;
fs::path CommandManager::loweringFolder;
bool CommandManager::enableLogFiles = false;
bool CommandManager::enableRunLogs = false;

std::vector<std::string> CommandManager::perf_metrics;

/*
 * Execute a command on the system's command line
 *
 * NOTE: Here, the directory from which the program was launched is taken as the
 * base directory (Current working directory)
 */
std::string CommandManager::exec(const std::string &cmd) {
  std::array<char, 128> buffer;
  std::string result;
  std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"),
                                                pclose);
  if (!pipe) {
    throw std::runtime_error("popen() failed!");
  }
  while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe.get()) !=
         nullptr) {
    result += buffer.data();
  }
  return result;
}

/*
 *
 * Initialisation Call: Create all needed directories here
 *  This method must be called before any processing command
 */
void CommandManager::initialise_environment() {
  CommandManager::verifyParameters();
  std::cout << "Creating directory: "
            << CommandManager::outputFolder.generic_string() << std::endl;
  CommandManager::exec("mkdir " +
                       CommandManager::outputFolder.generic_string());

  // CommandManager::perf_event_counter.add(
  //     {"seconds", "instructions", "cycles", "cache-misses"});
  CommandManager::perf_event_counter.add(CommandManager::perf_metrics);
}

std::map<std::string, double> CommandManager::aggregate_metrics(
    std::vector<std::map<std::string, double>> &metrics) {

  if (!metrics.size()) {
    std::cerr << "Empty metrics passed. Skipping aggregation\n";
    return std::map<std::string, double>();
  }

  std::map<std::string, double> aggregated_map;
  unsigned int total_counter = 0;

  // Collecting metric sums
  for (auto v : metrics) {
    for (std::string &perf : CommandManager::perf_metrics) {
      aggregated_map[perf] += v[perf];
    }
    total_counter++;
  }

  // Perform Avg reduction
  for (auto &p : aggregated_map) {
    p.second /= total_counter;
  }

  return aggregated_map;
}

bool CommandManager::verifyParameters() {
  assert(CommandManager::llvm_install_path.generic_string().size() > 0);
  assert(CommandManager::mlir_opt_exec.generic_string().size() > 0);
  assert(CommandManager::torch_mlir_install_path.generic_string().size() > 0);
  assert(CommandManager::torch_opt_exec.generic_string().size() > 0);
  assert(CommandManager::pipeline_json.generic_string().size() > 0);
  assert(CommandManager::outputFolder.generic_string().size() > 0);
  assert(CommandManager::compiler.size() > 0);
  assert(CommandManager::perf_metrics.size() > 0);

  // Derivative variables
  assert(CommandManager::llvm_lib_path.generic_string().size() > 0);
  assert(CommandManager::loweringFolder.generic_string().size() > 0);

  return true;
}

/*
 * Command Manager Configuration methods
 */
void CommandManager::set_pass_log_flag(bool flag) {
  CommandManager::enableLogFiles = flag;
}
void CommandManager::set_run_log_flag(bool flag) {
  CommandManager::enableRunLogs = flag;
}

void CommandManager::set_compiler_executable(const fs::path &binary) {
  CommandManager::compiler = binary.generic_string();
}

void CommandManager::set_pipeline_json_filepath(const fs::path &filepath) {
  CommandManager::pipeline_json = filepath;
}

void CommandManager::set_output_folder(const fs::path &output) {
  CommandManager::outputFolder = output;
  CommandManager::loweringFolder = fs::path(outputFolder).append("lowerings");
}

void CommandManager::set_llvm_install_path(const fs::path &path) {
  CommandManager::llvm_install_path = path;
  CommandManager::mlir_opt_exec =
      fs::path(CommandManager::llvm_install_path).append("bin/mlir-opt");
  CommandManager::llvm_lib_path = fs::path(llvm_install_path).append("lib");
}

void CommandManager::set_torch_install_path(const fs::path &path) {
  CommandManager::torch_mlir_install_path = path;
  fs::path execPath = (fs::path(CommandManager::torch_mlir_install_path));

  execPath.append("bin/torch-mlir-opt");
  CommandManager::torch_opt_exec = execPath;
}

void CommandManager::set_perf_metrics(const std::vector<std::string> &metrics) {
  CommandManager::perf_metrics = perf_metrics;
}
/*
 *
 */
fs::path CommandManager::get_output_folder() {
  return CommandManager::outputFolder;
}
fs::path CommandManager::get_lowering_folder() {
  return CommandManager::loweringFolder;
}

/*
 * Isolate all the torch operators present in the input 'mlir' file
 */
void CommandManager::isolate_torch_kernels(const std::string &filepath) {

  // Verify correct setup
  CommandManager::verifyParameters();

  // Detecting model filepath
  fs::path model_filepath = fs::current_path().append(filepath);
  fs::path log_path = fs::path(CommandManager::outputFolder)
                          .append("logs_" + get_timestamp_string());

  // Prepare lowering output folder path

  std::string model_isolation_command =
      CommandManager::torch_opt_exec.generic_string() +
      " --isolate-torch-ops=\"output-path=" +
      CommandManager::loweringFolder.generic_string() + "\" " +
      model_filepath.generic_string() + " > " +
      CommandManager::outputFolder.generic_string() + "/model_lower.log";

  std::cout << "Executing command: " << model_isolation_command.c_str()
            << std::endl;
  // Create model lowerings
  CommandManager::exec(model_isolation_command.c_str());
  std::cout << "Successfully isolated torch operators\n";
}

std::vector<std::string> CommandManager::get_operation_types() {
  // fs::path lowering_path =
  //     fs::path(CommandManager::outputFolder).append("lowerings");
  if (!fs::is_directory(CommandManager::loweringFolder)) {
    std::cerr << "Isolation seems to have failed. Can't fetch op types\n";
    return std::vector<std::string>();
  }

  std::cout << "Path for lowering: "
            << CommandManager::loweringFolder.generic_string() << '\n';
  std::string list_command =
      "ls -d " + CommandManager::loweringFolder.generic_string() + "/*/";
  std::cout << "List command: " << list_command << std::endl;
  std::vector<std::string> returnedDirPaths =
      CommandManager::get_cmd_output(list_command, '\n');

  std::vector<std::string> opNames;
  for (std::string s : returnedDirPaths) {
    std::cout << "We have it here: " << s << std::endl;
    opNames.push_back(fs::path(s).parent_path().filename().generic_string());
  }

  return opNames;
}

// Get the list of outlined files in the specified path
std::vector<std::string>
CommandManager::get_file_list(const fs::path &folderPath) {
  std::vector<std::string> kernel_vec = CommandManager::get_cmd_output(
      std::string("ls ") + folderPath.c_str() + "/*", '\n');
  return kernel_vec;
}

fs::path CommandManager::lower_to_llvm_dialect(const fs::path &mlirFilePath) {
  // 1. Lower Torch to Linalg
  fs::path linalg_path(mlirFilePath.parent_path().append(
      mlirFilePath.filename().replace_extension(".linalg.mlir").c_str()));

  std::string linalg_lowering_cmd =
      CommandManager::torch_opt_exec.generic_string() + " \
  -pass-pipeline=\"builtin.module(torch-backend-to-linalg-on-tensors-backend-pipeline)\" " +
      mlirFilePath.generic_string() + " > " + linalg_path.c_str();
  CommandManager::exec(linalg_lowering_cmd.c_str());

  std::string pass_seq = CommandManager::extract_pipeline();

  // 2. Lower Linalg to LLVM
  // Lower the file to .ll format
  fs::path llvm_mlir_filepath = mlirFilePath.parent_path().append(
      mlirFilePath.filename().replace_extension(".llvm.mlir").c_str());

  std::cout << "Extracted pipeline: " << pass_seq << std::endl;

  // std::string llvm_lowering_cmd =
  //     CommandManager::mlir_opt_exec.generic_string() + " " +
  //     linalg_path.c_str() + " \
  //           --canonicalize \
  //           --cse \
  //           --allow-unregistered-dialect \
  //           --one-shot-bufferize=\"bufferize-function-boundaries\" \
  //           --convert-linalg-to-loops \
  //           --convert-scf-to-cf \
  //           --lower-affine \
  //           --expand-strided-metadata \
  //           --finalize-memref-to-llvm \
  //           --convert-arith-to-llvm \
  //           --convert-func-to-llvm \
  //           --convert-cf-to-llvm \
  //           --reconcile-unrealized-casts \
  //       -o " +
  //     llvm_mlir_filepath.c_str();

  // Create .ll file using supplied pass pipeline
  std::string llvm_lowering_cmd =
      CommandManager::mlir_opt_exec.generic_string() + " " +
      linalg_path.c_str() + pass_seq + " -o " + llvm_mlir_filepath.c_str();

  CommandManager::get_cmd_output(llvm_lowering_cmd, '\0');
  return llvm_mlir_filepath;
}

// Compiling the LLVM Dialect file
fs::path
CommandManager::compile_llvm_dialect(const fs::path &llvm_mlir_filepath) {
  fs::path ll_filepath = fs::path(llvm_mlir_filepath).replace_extension(".ll");
  std::string cmd_string = "\
        mlir-translate --mlir-to-llvmir " +
                           llvm_mlir_filepath.generic_string() + " > " +
                           ll_filepath.generic_string();

  CommandManager::exec(cmd_string);

  return ll_filepath;
}

void CommandManager::generate_metadata_json(const std::string &mlir_filepath,
                                            const std::string &json_filename,
                                            const std::string &log_filename) {

  std::string log_file_appending =
      CommandManager::enableLogFiles || log_filename.size()
          ? " > " + log_filename
          : "";
  std::string param_gen_cmd =
      CommandManager::torch_opt_exec.generic_string() +
      " --generate-param-metadata=\"output-json=" + std::string(json_filename) +
      "\" " + mlir_filepath.c_str() + log_file_appending;

  // This will generate the output
  CommandManager::exec(param_gen_cmd);
}

/*
 * Get the output as a string vector seperated by a delimiter
 */
std::vector<std::string> CommandManager::get_cmd_output(const std::string &cmd,
                                                        char delimeter) {
  std::string ls_result = CommandManager::exec(cmd.c_str());
  std::stringstream ss(ls_result);
  std::vector<std::string> operation_types;

  std::string temp;
  while (std::getline(ss, temp, delimeter)) {
    operation_types.push_back(temp);
  }

  return operation_types;
}

fs::path CommandManager::generate_ll_file(const fs::path &mlirFilePath) {
  fs::path llvm_mlir_filepath =
      CommandManager::lower_to_llvm_dialect(mlirFilePath);
  return CommandManager::compile_llvm_dialect(llvm_mlir_filepath);
}

/*
 * Execute python module with the specified argument metadata
 *
 * TODO: Create a systematic documentation of where op_type is coming from
 *
 * NOTE: The verification part of the bonus deliverable is abandoned for
 * technical reasons. We need to find another way to verify the results of the
 * outlined operators later
 *
 * This is happening because current outlining is done at the torch operator
 * level. Due to this, the outliner has become extremely generic, which is a
 * design decision I have made to incorporate other dialects (Linalg, etc) later
 * on.
 *
 * The issue arises when we want to configure the python kernel with the same
 * configurations as that of the source model. Currently, the argument analysis
 * pass only captures dimension information about the input tensors, which are
 * general purpose in models and can be generalised/randomized (Dense, sparse,
 * zero tensors can be passed to any operator) But if we wish to execute python
 * kernel for verification, we also need to capture the EXACT configuration
 * parameter of each torch operator (Example: For conv = stride, padding, value
 * need to be captured and passed into python kernel. For matmul - none
 * ).
 *
 * But since these configuration details will vary widely across different
 * operators, capturing these from the pass will make the outliner non-generic
 * because the python kernel caller will now need to detect and pass in each
 * configuration to corresponding variable.
 *
 * This might be mitigated if we are able to generate python kernel from the
 * wrapper code, but that is MLIR->Python transformation, which is non-trivial
 * Hence, we are abandoning it now. If I get some time to try out this route, we
 * might be able to achieve this
 *
 */
// void CommandManager::execute_with_python(fs::path json_filepath,
//                                          const std::string &op_type) {
//   // Initialize Python interpreter
//   Py_Initialize();
//
//   // Add the kernel directory to Python sys.path
//   PyRun_SimpleString("import sys; sys.path.append('./kernels')");
//
//   // Import the Python module
//   PyObject *pKernelName = PyUnicode_DecodeFSDefault(
//       op_type.c_str()); // Create a unicode in python to load the file
//   PyObject *pModule =
//       PyImport_Import(pKernelName); // import the file in python runtime
//   Py_DECREF(pKernelName); // Decrement the reference count of unicode string.
//                           // This is effectively freeing the string object
//
//   if (pModule == NULL) {
//     // Module not found. This should never happen given that the kernel
//     folder
//     //    is present along with the executing binary
//     // @_TODO: Explore if this program itself should generate the required
//     kernels PyErr_Print(); fprintf(stderr, "Failed to load
//     model_runner.py\n"); return;
//   }
//
//   // Get the function from the module
//   // NOTE: All kernels made by the pass will have the function name as
//   //    "kernel_call"
//   PyObject *pFunc = PyObject_GetAttrString(pModule, "kernel_call");
//
//   if (pFunc && PyCallable_Check(pFunc)) {
//     int n = 10;
//     // @_TODO: Implement the fuzzer here
//     float *data = TensorFuzzer::generate_random_data(n);
//
//     // @_TODO: Build out the argument passing logic for tensors
//     // Build arguments (pointer + int)
//     PyObject *pArgs = PyTuple_New(2);
//     PyObject *pPtr = PyLong_FromVoidPtr((void *)data);
//     PyTuple_SetItem(pArgs, 0, pPtr);
//     PyTuple_SetItem(pArgs, 1, PyLong_FromLong(n));
//
//     // Call the Python function
//     PyObject *pResult = PyObject_CallObject(pFunc, pArgs);
//     Py_DECREF(pArgs);
//
//     if (pResult != NULL) {
//       // Convert NumPy array result → C pointer
//       PyArrayObject *np_arr = (PyArrayObject *)pResult;
//       float *result_data = (float *)PyArray_DATA(np_arr);
//       npy_intp size = PyArray_SIZE(np_arr);
//
//       printf("Received result (size %ld):\n", size);
//       for (int i = 0; i < size; i++) {
//         printf("%f ", result_data[i]);
//       }
//       printf("\n");
//
//       Py_DECREF(pResult);
//     } else {
//       PyErr_Print();
//     }
//
//     free(data);
//   } else {
//     PyErr_Print();
//     fprintf(stderr, "Function not callable\n");
//   }
//
//   Py_XDECREF(pFunc);
//   Py_DECREF(pModule);
//   Py_Finalize();
// }

std::string CommandManager::extract_pipeline() {
  json file = load_json_from_file(CommandManager::pipeline_json);
  std::vector<std::string> pass_list =
      file["pass"].template get<std::vector<std::string>>();

  std::string pass_seq = " ";
  for (const std::string &pass : pass_list) {
    pass_seq += " --" + pass + " ";
  }
  return pass_seq;
}

/*
 * Execute the specified ll-file with the specified argument metadata
 */
std::vector<std::map<std::string, double>>
CommandManager::execute_with_parameters(const fs::path &ll_object_filepath,
                                        const fs::path &json_filepath) {
  // 1. Read in JSON
  fs::path parent_path = ll_object_filepath.parent_path();
  std::cout << "Working on: " << ll_object_filepath.filename().generic_string()
            << std::endl;

  json metadata = load_json_from_file(json_filepath);

  int llvm_arg_count = 0;
  json kernel_function_json = metadata["kernel_call"].template get<json>();

  std::vector<json> arg_arr =
      kernel_function_json["args"].template get<std::vector<json>>();
  std::vector<json> return_arg_arr =
      kernel_function_json["returns"].template get<std::vector<json>>();

  // Prepare appropriate MemRef structures
  int llvm_argument_count = 0; // Calculate the total argument count here itself
  std::ofstream data_output_filestream(ll_object_filepath.generic_string() +
                                       ".output");
  std::cout << "File opened\n";

  // Storing tensor arguments as MemRef argument structures
  std::vector<MemRefArg *> argument_data;

  int log_counter = 1;

  // Parse Arguments from JSON and Generate data for arguments
  for (auto a : arg_arr) {
    JSONArgument argObject = a.template get<JSONArgument>();

    // Representing each argument using the MemRefArg structure
    MemRefArg *arg = new MemRefArg(argObject);

    // Generate random normalised data
    DataFormatInfo dataInfo;
    auto elem_count = arg->get_tensor_elem_count();
    dataInfo.setElemCount(elem_count);

    // Add generated data into argument
    float *generated_data = TensorFuzzer::generate_data(dataInfo);
    arg->setData(generated_data);

    argument_data.push_back(arg);

    // @DEBUG: Logging results
    // std::cout << "Start\n";

    if (CommandManager::enableRunLogs) {
      data_output_filestream << "Input " << log_counter++ << ": [" << "\n";
      for (int k = 0; k < elem_count; k++) {
        data_output_filestream << generated_data[k] << ", ";
      }
      data_output_filestream << "]\n\n";
      data_output_filestream.flush();
      // std::cout << "End\n";

      llvm_argument_count += 3 + arg->m_tensor_rank * 2;
    }
  }

  // Compile this file to a ".so" file
  fs::path output_filepath = fs::path(parent_path).append("kernel_call.so");
  std::string compilation_command =
      CommandManager::compiler + " \
      --std=c++20 \
      -fPIC \
      -shared \
      -o " +
      output_filepath.generic_string() + " -Wl,-rpath," +
      CommandManager::llvm_lib_path.generic_string() + " -L" +
      CommandManager::llvm_lib_path.generic_string() +
      " -lmlir_runner_utils -lmlir_c_runner_utils " +
      ll_object_filepath.generic_string();
  CommandManager::exec(compilation_command);

  //
  // Prepare and call C++ using ffi
  //

  //  Import it using dlopen
  void *fHandle = dlopen(output_filepath.c_str(), RTLD_LAZY);
  if (fHandle == NULL) {
    std::cerr << "Failed to open compiled version of "
              << fs::path(ll_object_filepath).replace_extension().filename()
              << std::endl;
    std::cerr << dlerror() << std::endl;
    return std::vector<std::map<std::string, double>>();
  }

  //  Import function handle using dlsym
  void *kHandle = dlsym(fHandle, "kernel_call");
  if (!kHandle) {
    std::cerr << "Failed to load kernel function: "
              << fs::path(ll_object_filepath).replace_extension().filename()
              << std::endl;
    dlclose(fHandle);
    return;
  }

  //  Prepare the argument type list and data array to call the function
  std::vector<ffi_type *> func_arg_types;
  std::vector<void *> func_arg_data;

  for (int i = 0; i < argument_data.size(); i++) {
    MemRefArg *curr_memarg = argument_data[i];
    // std::cout << "Argument State: " << i << std::endl;
    // curr_memarg->printState();

    // llvm base pointer
    func_arg_types.push_back(&ffi_type_pointer);
    func_arg_data.push_back(&(curr_memarg->m_desc->base_ptr));

    // llvm aligned pointer
    func_arg_types.push_back(&ffi_type_pointer);
    func_arg_data.push_back(&(curr_memarg->m_desc->aligned_ptr));

    // Offset information
    func_arg_types.push_back(&ffi_type_sint64);
    func_arg_data.push_back(&(curr_memarg->m_desc->offset));

    // OPTIMIZATION: Merge the 2 loops
    // Pass in dimension type information
    for (int j = 0; j < curr_memarg->get_tensor_rank(); j++) {
      func_arg_types.push_back(&ffi_type_sint64);
      func_arg_data.push_back(&(curr_memarg->m_desc->dimension[j]));
    }

    // Pass in stride type information
    for (int j = 0; j < curr_memarg->get_tensor_rank(); j++) {
      func_arg_types.push_back(&ffi_type_sint64);
      func_arg_data.push_back(&(curr_memarg->m_desc->strides[j]));
    }
  }

  // Preparing Return Data Type and memory alignments
  JSONArgument returnArgObject = return_arg_arr[0].template get<JSONArgument>();
  // std::cout << "Return Rank: " << returnArgObject.rank << std::endl;
  ffi_type *ret_arg_type = create_memref_struct_type(
      returnArgObject.rank); // This will be a MemRef descriptor
  // structure (most probably)

  // std::cout << "Arguments: " << func_arg_types.size() << "\t"
  //           << func_arg_data.size() << std::endl;

  // Construct argument data array
  ffi_cif calling_interface;
  ffi_status status =
      ffi_prep_cif(&calling_interface, FFI_DEFAULT_ABI, func_arg_types.size(),
                   ret_arg_type, func_arg_types.data());

  // std::cout << "Return arg prepared\n";
  // std::cout << "\nRET_TYPE set: [size=" << ret_arg_type->size
  // << "]  alignment=[" << ret_arg_type->alignment << "]";

  if (status != FFI_OK) {
    std::cerr << "Failed to prepare kernel call: " << std::endl;
    dlclose(fHandle);
    return;
  }

  MemRefArg return_arg_data(returnArgObject);
  return_arg_data.updateWithFFITemplate(ret_arg_type);

  // TODO: Time this using perf-cpp
  // CommandManager::perf_event_counter.start();
  // ffi_call(&calling_interface, FFI_FN(kHandle), return_arg.getData(),
  //          func_arg_data.data());

  // std::cout << "Printer: \n";
  // for (int i = 0; i < func_arg_data.size(); i++) {
  //   std::cout << *((uint64_t *)func_arg_data[i]) << std::endl;
  // }

  // void *returned_ptr = malloc(ret_arg_type->size);
  void *returned_ptr;
  posix_memalign(&returned_ptr, ret_arg_type->alignment, ret_arg_type->size);
  // memset(returned_ptr, 0xCC, ret_arg_type->size); // scribble to detect
  // writes

  std::vector<std::map<std::string, double>> collected_metrics;

  for (int i = 0; i < CommandManager::perf_run_count; i++) {

    perf::EventCounter perf_event_counter = perf::EventCounter{};
    perf_event_counter.add(CommandManager::perf_metrics);
    perf_event_counter.start();
    ffi_call(&calling_interface, FFI_FN(kHandle), returned_ptr,
             func_arg_data.data());
    perf_event_counter.stop();

    auto result = perf_event.result();

    // Collect Timing Metrics
    // Write individual run to csv
    std::ifstream perf_stream(ll_object_filepath.generic_string() + "." +
                              std::string(i) + ".metric");
    perf_stream << result.to_csv();
    perf_stream.close();

    // We dont need to check if the key is in the perf_metrics vector since that
    // vector is what was used to initialise the perf_counter
    std::map run_result_map(result.begin(), result.end());
    collected_metrics.push_back(run_result_map);
  }

  // std::cout << "Function called\n";
  // uint64_t *format_ptr = (uint64_t *)returned_ptr;
  // for (int i = 0; i < ret_arg_type->size / ret_arg_type->alignment; i++) {
  //   std::cout << format_ptr[i] << " - ";
  // }
  // std::cout << std::endl;

  if (CommandManager::enableRunLogs) {
    return_arg_data.extractDescFromFFIPtr(returned_ptr);
    // return_arg_data.setData((float *)(*((uint64_t *)returned_ptr)));

    // Transform structure to MemRefDescriptor

    data_output_filestream << "Output: [\n";
    for (int i = 0; i < return_arg_data.get_tensor_elem_count(); i++) {
      data_output_filestream << *(((float *)return_arg_data.getData()) +
                                  return_arg_data.m_desc->offset + i)
                             << ", ";
    }
    data_output_filestream << "\n]\n";
  }
  data_output_filestream.close();

  dlclose(fHandle);

  //  e. setup perf classes to initiate timing - DONE
  //  f. collect the timing details in a structure

  //  g. Remove the .so file, so the next file run does not conflict
  CommandManager::exec("rm " + output_filepath.generic_string());
  return collected_metrics;
}

/**
 * Dynamically creates the ffi_type_struct for an MLIR MemRef descriptor.
 * The MemRef descriptor format is: { ptr, ptr, i64, [rank x i64], [rank x i64]
 * }
 *
 * @param rank The dimension count (e.g., 0 for scalar, 4 for 4D tensor).
 * @return A pointer to the dynamically allocated ffi_type structure.
 */
ffi_type *create_memref_struct_type(int rank) {
  // Total fields: 3 fixed fields + 2 * rank fields = (3 + 2 * rank)
  int total_fields = 3 + (2 * rank);

  // Allocate memory for the ffi_type_struct and the array of its elements
  ffi_type *memref_type = (ffi_type *)malloc(sizeof(ffi_type));
  if (!memref_type) {
    std::cerr << "Failed to initialize memref_type\n";
    return NULL;
  }

  // Allocate memory for the array of field pointers, plus one NULL terminator
  ffi_type **elements =
      (ffi_type **)malloc(sizeof(ffi_type *) * (total_fields + 1));
  if (!elements) {
    std::cerr << "Failed to initialize elements\n";
    free(memref_type);
    return NULL;
  }

  int i = 0;

  // 3 Elements are always fixed
  elements[i++] = &ffi_type_pointer; // allocatedPtr
  elements[i++] = &ffi_type_pointer; // alignedPtr
  elements[i++] = &ffi_type_sint64;  // offset

  // Remaining arguments depend on the rank of the tensor

  // Pushing types for dimension arguments
  for (int j = 0; j < rank; j++) {
    elements[i++] = &ffi_type_sint64;
  }

  // Pushing types for strides argument
  for (int j = 0; j < rank; j++) {
    elements[i++] = &ffi_type_sint64;
  }

  // 5. NULL terminator is essential (Why???)
  elements[i] = NULL;

  // Initialize the ffi_type_struct
  memset(memref_type, 0, sizeof(ffi_type));
  memref_type->type = FFI_TYPE_STRUCT;
  memref_type->elements = elements;

  // Note: size and alignment fields are zero, as ffi_prep_cif will calculate
  // them.
  return memref_type;
}

/**
 * Function to clean up the dynamically created ffi_type structure.
 */
void destroy_memref_struct_type(ffi_type *memref_type) {
  if (memref_type) {
    // Free the elements array first
    if (memref_type->elements) {
      free(memref_type->elements);
    }
    // Free the ffi_type struct itself
    free(memref_type);
  }
}
