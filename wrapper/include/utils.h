#pragma once

#include "nlohmann/json.hpp"
#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>


#ifdef __APPLE__
#include <ffi/ffi.h>
#else
#include <ffi.h> // Linux is required if not MACOS (Windows does not have standard FFI library)
#endif



using json = nlohmann::json;
namespace fs = std::filesystem;

struct JSONArgument {
  std::string dtype;
  uint64_t rank;
  std::vector<uint64_t> shape;
};

/*
 * MemRef Descriptor structure
 *
 *
 * Structure to handle MemRef Representation of MLIR tensor returned in FFI
 * calls
 *
 * NOTE: This structure is not a one to one mapping of the tensor returned from
 * MLIR since our kernel outliner is generic, meaning that tensors returned can
 * be of arbitrary rank, hence MemRef Descriptor structure will also vary.
 *
 * To accomodate variable MemRef structure, use the @MemRefArg structure which
 * encapsulates this descriptor structure with extra metadata and helper
 * functions to fill in descriptor
 */
struct MemRefDescriptor {
  void *base_ptr;
  void *aligned_ptr;
  int64_t offset;

  int64_t *dimension;
  int64_t *strides;
};

/*
 * MemRefArg Structure
 *
 * The core structure to be used for MemRef Argument passing and return value
 * handling
 *
 * This wrapper encapsulates the descriptor, and provides methods for populating
 * it.
 *
 * Currently assuming data is always aligned.
 * That is,
 *    aligned_ptr = base_ptr, offset = 0
 *  (I have not yet encountered a scenario where this changes. TODO: Explore
 * cases where this may happen for our use case)
 * */
struct MemRefArg {
  int64_t m_tensor_rank;
  int64_t m_tensor_elem_count; // Total data elements stored in the tensor
  int64_t m_desc_alignment;    // Alignment of the descriptor
  int64_t m_desc_size;         // Total descriptor size in bytes

  MemRefDescriptor *m_desc;

  MemRefArg() = delete;
  MemRefArg(const int &tensor_rank);
  MemRefArg(std::initializer_list<uint64_t> dimension_list);
  MemRefArg(const JSONArgument &argument_data);

  // Methods
  void printState();

  // NOTE: This is not doing a safety check on the data pointer passed in right
  // now. This could lead security issues. Fix this later
  void setData(void *data, const uint64_t &offset = 0);
  void *getData();
  void *getDataAligned();

  void updateWithFFITemplate(const ffi_type *ffi_return_type_template);
  bool extractDescFromFFIPtr(void *ffi_returned_ptr);

  int get_tensor_elem_count();
  int64_t get_tensor_rank();
};

/*
 * JSON loader function (Required for import mapping)
 */
void from_json(const json &j, JSONArgument &a);
// void from_json(const json &j, JSONReturnArgument &a);

json load_json_from_file(const fs::path &filePath);

// Time stamp generator
std::string get_timestamp_string();
