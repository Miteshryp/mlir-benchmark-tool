#pragma once
#include "tensor_fuzzer.h"
#include <chrono>
#include <cstdlib>
#include <iostream>

void applyTimeSeed() {

  // Returns the current no of clock ticks based on current time since last
  // epoch
  unsigned int current_clock =
      std::chrono::system_clock::now().time_since_epoch().count();

  // We use this as a psuedo-random seed
  std::srand(current_clock);
}

/*
 * -----------------------------------
 * DataFormatInfo definitions
 * -----------------------------------
 */

DataFormatInfo::DataFormatInfo(const DataProfile &profile,
                               const float &sparsity_percentage) {
  m_profile = profile;

  // NOTE: Not applicable as of now
  // if (profile == DataProfile::SPARSE) {
  //   SparsityProfile sp;
  //   sp.sparsity_percentage = sparsity_percentage;
  //   // sp.distribution_type =
  // }
}

DataFormatInfo::F32Range DataFormatInfo::getRange() {
  return this->m_range_bounds;
}

void DataFormatInfo::setRange(const float &vmin, const float &vmax) {
  m_range_bounds.min_val = vmin;
  m_range_bounds.max_val = vmax;
}

void DataFormatInfo::setElemCount(const uint64_t &elem_count) {
  m_elem_count = elem_count;
}

/*
 * -----------------------------------
 * TensorFuzzer definitions
 * -----------------------------------
 */
float *TensorFuzzer::generate_zero_data(DataFormatInfo info) {
  uint64_t elem_count = info.m_elem_count;
  float *array = (float *)calloc(elem_count, sizeof(float));
  return array;
}

float *TensorFuzzer::generate_random_data(DataFormatInfo info) {
  uint64_t elem_count = info.m_elem_count;
  float range_min = info.getRange().min_val;
  float range_max = info.getRange().max_val;

  float range_length = range_max - range_min;

  float *array = (float *)malloc(elem_count * sizeof(float));
  applyTimeSeed(); // Initialises the Psuedo Random Generator

  for (int i = 0; i < elem_count; i++) {
    float norm_scale = (float)std::rand() / RAND_MAX;
    float value = range_min + (range_length * norm_scale);
    array[i] = value;
  }
  return array;
}

/*
 * Generates f32 data with values in the range of 0 - 1
 */
float *TensorFuzzer::generate_random_data_norm(DataFormatInfo info) {
  uint64_t elem_count = info.m_elem_count;

  float *array = (float *)malloc(elem_count * sizeof(float));
  applyTimeSeed(); // Initialises the Psuedo Random Generator

  for (int i = 0; i < elem_count; i++) {
    float norm_scale = (float)std::rand() / RAND_MAX;
    array[i] = norm_scale;
  }
  return array;
}

float *TensorFuzzer::generate_test_data(DataFormatInfo info) {
  uint64_t elem_count = info.m_elem_count;

  float *array = (float *)malloc(elem_count * sizeof(float));
  applyTimeSeed(); // Initialises the Psuedo Random Generator

  for (int i = 0; i < elem_count; i++) {

    float norm_scale = 2;
    // std::cout << norm_scale << " ";
    array[i] = norm_scale;
  }
  return array;
}

// float *TensorFuzzer::generate_sparse_data(DataFormatInfo info) {}

/*
 * Interface function for fuzzer:
 * Creates data based on the specified profile
 */
float *TensorFuzzer::generate_data(DataFormatInfo dataInfo) {
  switch (dataInfo.m_profile) {
  case TEST:
    return generate_random_data(dataInfo);
  case RANDOM:
    return generate_random_data(dataInfo);
  case RANDOM_NORM:
    return generate_random_data_norm(dataInfo);
  case ZEROS:
  case SPARSE:
  default:
    return nullptr;
  }
}
