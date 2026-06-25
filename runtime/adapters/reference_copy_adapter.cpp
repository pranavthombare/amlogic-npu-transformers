#include "amlogic_transformers/adapter_abi.h"

#include <cstring>

namespace {

const AmlogicTransformersTensor* find_source_tensor(
  const AmlogicTransformersTensor* inputs,
  size_t input_count,
  const AmlogicTransformersTensor& output
) {
  for (size_t i = 0; i < input_count; ++i) {
    if (inputs[i].dtype == output.dtype && inputs[i].byte_size == output.byte_size) {
      return &inputs[i];
    }
  }
  return nullptr;
}

}  // namespace

extern "C" int amlogic_transformers_init(const char* graph_name, const char* nb_path) {
  return graph_name != nullptr && nb_path != nullptr ? 0 : -1;
}

extern "C" int amlogic_transformers_run(
  const AmlogicTransformersTensor* inputs,
  size_t input_count,
  AmlogicTransformersTensor* outputs,
  size_t output_count
) {
  if (inputs == nullptr || outputs == nullptr || input_count == 0) {
    return -1;
  }

  for (size_t i = 0; i < output_count; ++i) {
    const AmlogicTransformersTensor* source = find_source_tensor(inputs, input_count, outputs[i]);
    if (source == nullptr || outputs[i].data == nullptr || source->data == nullptr) {
      return -2;
    }
    std::memcpy(outputs[i].data, source->data, outputs[i].byte_size);
  }

  return 0;
}

extern "C" void amlogic_transformers_shutdown() {}
