#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

enum GemmaDType {
  GEMMA_DTYPE_F32 = 1,
  GEMMA_DTYPE_I64 = 2,
  GEMMA_DTYPE_I32 = 3,
  GEMMA_DTYPE_U8 = 4,
  GEMMA_DTYPE_I8 = 5,
};

struct GemmaTensor {
  const char* name;
  int dtype;
  int rank;
  int64_t shape[8];
  void* data;
  size_t byte_size;
};

typedef int (*gemma_aml_init_fn)(const char* graph_name, const char* nb_path);
typedef int (*gemma_aml_run_fn)(
  const GemmaTensor* inputs,
  size_t input_count,
  GemmaTensor* outputs,
  size_t output_count
);
typedef void (*gemma_aml_shutdown_fn)();

#ifdef __cplusplus
}
#endif

