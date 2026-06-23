#include "gemma_npu/adapter_abi.h"

#include <cstring>

namespace {

const GemmaTensor* find_source_tensor(
  const GemmaTensor* inputs,
  size_t input_count,
  const GemmaTensor& output
) {
  for (size_t i = 0; i < input_count; ++i) {
    if (inputs[i].dtype == output.dtype && inputs[i].byte_size == output.byte_size) {
      return &inputs[i];
    }
  }
  return nullptr;
}

}  // namespace

extern "C" int gemma_aml_init(const char* graph_name, const char* nb_path) {
  return graph_name != nullptr && nb_path != nullptr ? 0 : -1;
}

extern "C" int gemma_aml_run(
  const GemmaTensor* inputs,
  size_t input_count,
  GemmaTensor* outputs,
  size_t output_count
) {
  if (inputs == nullptr || outputs == nullptr || input_count == 0) {
    return -1;
  }

  for (size_t i = 0; i < output_count; ++i) {
    const GemmaTensor* source = find_source_tensor(inputs, input_count, outputs[i]);
    if (source == nullptr || outputs[i].data == nullptr || source->data == nullptr) {
      return -2;
    }
    std::memcpy(outputs[i].data, source->data, outputs[i].byte_size);
  }

  return 0;
}

extern "C" void gemma_aml_shutdown() {}
