#include "amlogic_transformers/adapter_abi.h"

// Template for the board-specific adapter that bridges this project's stable
// ABI to Amlogic/Acuity generated code. Copy this file next to the generated
// vnn_*.c/.h artifacts and replace the TODOs with generated model calls.

extern "C" int amlogic_transformers_init(const char* graph_name, const char* nb_path) {
  (void)graph_name;
  (void)nb_path;
  // TODO: initialize OpenVX/Amlogic runtime and load the generated .nb graph.
  return -1;
}

extern "C" int amlogic_transformers_run(
  const AmlogicTransformersTensor* inputs,
  size_t input_count,
  AmlogicTransformersTensor* outputs,
  size_t output_count
) {
  (void)inputs;
  (void)input_count;
  (void)outputs;
  (void)output_count;
  // TODO: map AmlogicTransformersTensor buffers to generated Acuity input/output buffers.
  return -1;
}

extern "C" void amlogic_transformers_shutdown() {
  // TODO: release graph/runtime resources.
}

