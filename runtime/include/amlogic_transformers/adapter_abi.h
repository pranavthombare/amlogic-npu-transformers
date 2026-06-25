#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

enum AmlogicTransformersDType {
  AML_TRANSFORMERS_DTYPE_F32 = 1,
  AML_TRANSFORMERS_DTYPE_I64 = 2,
  AML_TRANSFORMERS_DTYPE_I32 = 3,
  AML_TRANSFORMERS_DTYPE_U8 = 4,
  AML_TRANSFORMERS_DTYPE_I8 = 5,
};

struct AmlogicTransformersTensor {
  const char* name;
  int dtype;
  int rank;
  int64_t shape[8];
  void* data;
  size_t byte_size;
};

typedef int (*amlogic_transformers_init_fn)(const char* graph_name, const char* nb_path);
typedef int (*amlogic_transformers_run_fn)(
  const AmlogicTransformersTensor* inputs,
  size_t input_count,
  AmlogicTransformersTensor* outputs,
  size_t output_count
);
typedef void (*amlogic_transformers_shutdown_fn)();

#ifdef __cplusplus
}
#endif
