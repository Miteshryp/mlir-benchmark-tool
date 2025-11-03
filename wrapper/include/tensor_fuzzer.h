#pragma once

#include <cmath>
#include <cstdint>

enum DataProfile { TEST, RANDOM, RANDOM_NORM, ZEROS, SPARSE };
enum DataOrder { NCHW, NCWH };

/*
 * TODO: Implement Distribution detection
 *
 * @Issue: Torch operators may support different types of data format, which
 * need to be communicated in this wrapper by the metadata extraction pass. The
 * issue is that from what I see right now, this information can be extracted
 * from linalg lowering of torch operator, but I'm not sure how to extract this
 * information from the torch operator version in the torch dialect, or whether
 * or not if it's even possible.
 */
enum SparsityDist {
  STRUCTURED_ROW_WISE, // Row is selected and pruned
  STRUCTURED_COL_WISE, // Column is selected and pruned
  UNSTRUCTURED_PURE,
  UNSTRUCTURED_TILE_1D,   // 1D Tile starting from an index is selected and
                          // pruned.
  UNSTRUCTURED_TILE_2D,   // 2D Tile starting from an index is selected, and all
                          // elements in the tile are pruned
  UNSTRUCTURED_DIAGONALS, // Diagonals are selected at random and pruned
};

struct SparsityProfile {
  SparsityDist distribution_type;
  union {
    uint8_t col_block_size;
    uint8_t row_block_size =
        1; // Default: Single rows/cols at random are set to zeros
  };
  float sparsity_percentage; // Probability of selecting data unit for data
                             // pruning
};

// TODO: Complete fuzzer feature
struct DataFormatInfo {

  struct F32Range {
    float min_val;
    float max_val;
  }; // Can ignore in case of normalized data

public:
  DataProfile m_profile; // Data characteristics
  SparsityProfile
      m_sp_profile; // Sparsity characteristics (Only applies to sparse data)

  F32Range m_range_bounds;
  uint64_t m_elem_count;

  // Default setting: RANDOM_NORM profile
  DataFormatInfo(const DataProfile &profile = DataProfile::RANDOM_NORM,
                 const float &sparsity_percentage = 0.f);

  F32Range getRange();
  void setRange(const float &vmin, const float &vmax);
  void setElemCount(const uint64_t &elem_count);
  void setProfile(const DataProfile &profile);

  // TODO: Add sparsity support;
  // void setSparsityDensity(float percentage);
  // void setSparsityDist();
};

class TensorFuzzer {
  static float *generate_random_data(DataFormatInfo info);
  static float *generate_test_data(DataFormatInfo info);
  static float *generate_random_data_norm(DataFormatInfo info);
  static float *generate_zero_data(DataFormatInfo info);
  // static float *generate_sparse_data(const uint64_t &elem_count,
  // const float &sparse_percentage);

public:
  static float *generate_data(DataFormatInfo dataInfo);
};
