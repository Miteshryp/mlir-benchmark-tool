#pragma once

#include "utils.h"
#include <cassert>
#include <cstdint>
#include <sstream>

using std::ostringstream;

namespace fs = std::filesystem;

void from_json(const json &j, JSONArgument &a) {
  j.at("dtype").get_to(a.dtype);
  j.at("rank").get_to(a.rank);
  j.at("shape").get_to(a.shape);
}

json load_json_from_file(const fs::path &filePath) {
  std::ifstream f(filePath);
  json obj = json::parse(f);

  return obj;
}

/* -----------------------
 * MemRefArg Definitions
 * -----------------------
 */
MemRefArg::MemRefArg(const int &tensor_rank) {
  this->m_tensor_rank = tensor_rank;
  this->m_desc = new MemRefDescriptor();
}

// NOTE: This is never used (Probably will never be used)
MemRefArg::MemRefArg(std::initializer_list<uint64_t> dimension_list) {
  assert(dimension_list.size() > 0);
  this->m_tensor_rank = dimension_list.size();
  this->m_desc = new MemRefDescriptor();

  // Allocating space for dimension and stride data based on tensor rank
  // detected
  int64_t *dimension_data =
      (int64_t *)malloc(sizeof(int64_t) * dimension_list.size());
  int64_t *stride_data =
      (int64_t *)malloc(sizeof(int64_t) * dimension_list.size());

  for (int i = 0; i < dimension_list.size(); i++) {
    dimension_data[i] = *(dimension_list.begin() + i);
  }

  int accum = 1;
  for (int i = dimension_list.size() - 1; i >= 0; i--) {
    stride_data[i] = accum;
    accum *= dimension_data[i];
  }

  this->m_tensor_elem_count = accum;

  this->m_desc->dimension = dimension_data;
  this->m_desc->strides = stride_data;
  this->m_desc->offset = 0;

  // Data has to be explicitly set
  this->m_desc->base_ptr = malloc(sizeof(float) * m_tensor_elem_count);
  this->m_desc->aligned_ptr = malloc(sizeof(float) * m_tensor_elem_count);
}

// Loading MemRefArg structure details based on JSON Argument loaded, and
// allocating required buffers
MemRefArg::MemRefArg(const JSONArgument &argument_data) {
  assert(argument_data.shape.size() > 0);

  this->m_tensor_rank = argument_data.rank;
  this->m_desc = new MemRefDescriptor();

  int64_t *dimension_data =
      (int64_t *)malloc(sizeof(int64_t) * argument_data.shape.size());
  int64_t *stride_data =
      (int64_t *)malloc(sizeof(int64_t) * argument_data.shape.size());

  for (int i = 0; i < argument_data.shape.size(); i++) {
    dimension_data[i] = *(argument_data.shape.begin() + i);
  }

  int accum = 1;
  for (int i = argument_data.shape.size() - 1; i >= 0; i--) {
    stride_data[i] = accum;
    accum *= dimension_data[i];
  }
  this->m_tensor_elem_count = accum;

  this->m_desc->dimension = dimension_data;
  this->m_desc->strides = stride_data;
  this->m_desc->offset = 0;

  // Data has to be explicitly set
  this->m_desc->base_ptr = malloc(sizeof(float) * m_tensor_elem_count);
  this->m_desc->aligned_ptr = malloc(sizeof(float) * m_tensor_elem_count);
  m_desc_size = -1;     // Unknown
  m_desc_alignment = 8; // Assuming by default
}

void MemRefArg::printState() {
  std::cout << "Tensor Rank: " << m_tensor_rank << std::endl;

  std::cout << "Descriptor: \n";
  std::cout << "\tOffset: " << m_desc->offset << std::endl;

  std::cout << "\tDimension Data: [";
  for (int i = 0; i < m_tensor_rank; i++) {
    std::cout << m_desc->dimension[i] << ',';
  }
  std::cout << "]\n";

  std::cout << "\tStride Data: [";
  for (int i = 0; i < m_tensor_rank; i++) {
    std::cout << m_desc->strides[i] << ',';
  }
  std::cout << "]\n";

  if (m_desc->base_ptr != nullptr) {
    std::cout << "Data Values: [";
    for (int i = 0; i < m_tensor_elem_count; i++) {
      std::cout << ((float *)m_desc->base_ptr)[i] << ",";
    }
  }
}

bool MemRefArg::extractDescFromFFIPtr(void *ffi_returned_ptr) {

  if (m_desc->dimension == nullptr || m_desc->strides == nullptr) {

    std::cerr << "Descriptor Extraction Failed: Dimension or Stride pointer "
                 "was empty"
              << std::endl;
    return false;
  }
  // Returned ptr =>
  //    (base_ptr = 8B)
  //    (aligned_ptr = 8B)
  //    (offset = 8B)

  // 64bit address to be added.
  int64_t *desc_ptr = ((int64_t *)ffi_returned_ptr);
  int64_t *offset_ptr = desc_ptr + 2;

  if (offset_ptr == nullptr || desc_ptr == nullptr) {
    std::cerr
        << "Descriptor Extraction Failed: Returned pointer has null values"
        << std::endl;
    return false;
  }

  this->setData((float *)(*desc_ptr), *offset_ptr);

  // Pointer to first dimension value returned
  int64_t *iterator_ptr = (desc_ptr + 3);
  for (int i = 0; i < m_tensor_rank; i++) {
    m_desc->dimension[i] = *(iterator_ptr++);
  }

  for (int i = 0; i < m_tensor_rank; i++) {
    m_desc->strides[i] = *(iterator_ptr++);
  }

  return true;
}

void MemRefArg::setData(void *data_ptr, const uint64_t &offset) {
  if (m_desc->base_ptr)
    free(m_desc->base_ptr);
  if (m_desc->aligned_ptr)
    free(m_desc->aligned_ptr);

  m_desc->base_ptr = data_ptr;
  m_desc->aligned_ptr = (void *)((char *)data_ptr + offset);
  m_desc->offset = offset;
}

void MemRefArg::updateWithFFITemplate(const ffi_type *type) {
  m_desc_alignment = type->alignment;
  m_desc_size = type->size;
}

void *MemRefArg::getData() { return m_desc->base_ptr; }

void *MemRefArg::getDataAligned() { return m_desc->aligned_ptr; }

int MemRefArg::get_tensor_elem_count() { return m_tensor_elem_count; }

int64_t MemRefArg::get_tensor_rank() { return m_tensor_rank; }

std::string get_timestamp_string() {
  auto now = std::chrono::system_clock::now();
  auto now_time_t = std::chrono::system_clock::to_time_t(now);

  // Use std::localtime for local time or std::gmtime for UTC
  std::tm *local_tm = std::localtime(&now_time_t);

  std::ostringstream oss;
  oss << std::put_time(
      local_tm, "%Y-%m-%d_%H%M%S"); // Example format: YYYY-MM-DD HH:MM:SS
  return oss.str();
}
