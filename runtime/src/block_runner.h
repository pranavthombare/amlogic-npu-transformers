#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "gemma_npu/adapter_abi.h"

struct LoadedTensor {
  std::string name;
  int dtype = GEMMA_DTYPE_F32;
  std::vector<int64_t> shape;
  std::vector<uint8_t> bytes;
};

struct TensorRef {
  std::string name;
  std::string path;
  std::string dtype;
  std::vector<int64_t> shape;
};

struct GraphManifest {
  std::string name;
  std::string nb_path;
  std::string adapter_library;
  std::vector<TensorRef> inputs;
  std::vector<TensorRef> outputs;
};

struct CompareResult {
  double max_abs = 0.0;
  double max_rel = 0.0;
  double cosine = 0.0;
};

GraphManifest load_graph_manifest(const std::string& manifest_path, const std::string& graph_name);
LoadedTensor load_npy_tensor(const std::string& name, const std::string& path);
CompareResult compare_f32(const LoadedTensor& expected, const LoadedTensor& actual);
GemmaTensor as_abi_tensor(LoadedTensor& tensor);

