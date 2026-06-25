# Implementation Notes

## Inference Boundary

This project does not use `llama.cpp`, GGUF execution, or a generic CPU LLM
runtime. The inference boundary is the Amlogic NPU SDK. CPU code is used only
for orchestration, tokenizer/generation control, unsupported operations, and
reference checks.

## First Milestone

The required first milestone is a numerically checked decoder-block run:

1. Export `google/gemma-3-270m-it` layer `0` to fixed-shape ONNX.
2. Generate `.npy` reference inputs and outputs.
3. Convert the ONNX graph through `aml_npu_sdk/acuity-toolkit/python/convert`.
4. Build a board-specific adapter implementing `amlogic_transformers_init`, `amlogic_transformers_run`, and `amlogic_transformers_shutdown`.
5. Run `amlogic-transformers-block-runner` and pass drift thresholds.

The local `make test` target does not claim NPU correctness. It uses
`runtime/adapters/reference_copy_adapter.cpp` with a tiny generated fixture to
prove the manifest format, `.npy` loading, adapter ABI, dynamic loading, and
comparison logic all work before the board-specific adapter exists.

## Adapter Contract

The adapter ABI in `runtime/include/amlogic_transformers/adapter_abi.h` is the stable
boundary between this project and Acuity-generated code. Generated SDK code can
change per conversion; the rest of the runtime should not.

The adapter receives named tensors matching the ONNX graph inputs and writes to
preallocated output buffers. It should own NPU graph initialization and cleanup.

Adapter implementations must not allocate output buffers owned by the runner.
They should copy or map NPU output into the provided `AmlogicTransformersTensor.data` memory
and return a non-zero status if any graph input or output cannot be matched.

## Failure Policy

The API must not silently fall back to a generic CPU LLM. If NPU artifacts are
missing or the adapter is absent, `/v1/chat/completions` returns `503`. If the
block harness has not been proven, it returns `501`.
