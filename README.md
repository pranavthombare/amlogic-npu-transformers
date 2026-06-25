# Amlogic NPU Transformers Inference

Amlogic NPU inference pipeline for `google/gemma-3-270m-it` on Khadas VIM3.
This is intentionally not a `llama.cpp`, GGUF, or CPU fallback stack.

The current implementation proves the first required unit of work: a
fixed-shape Gemma decoder block can be exported, converted, loaded through a
stable adapter ABI, executed by a small C++ runner, and checked against CPU
reference tensors. The full chat server stays disabled until the NPU block path
passes on real VIM3 artifacts.

## Current Status

Working locally:

- Python config and artifact manifest handling.
- Gemma decoder-block ONNX export entry point.
- Amlogic Acuity conversion command generation.
- C++ runtime harness with a stable adapter ABI.
- SDK-free reference adapter and generated fixture for CI.
- OpenAI-compatible API shell that refuses to serve until NPU artifacts exist.

Still requires the board and SDK:

- Running Acuity conversion against exported Gemma ONNX graphs.
- Implementing the real `libamlogic_transformers_adapter.so` around generated Amlogic code.
- Passing numerical checks on VIM3 NPU output.
- Enabling token generation beyond the one-block milestone.

## Target

- Board: Khadas VIM3
- OS: Khadas Ubuntu BSP with Galcore/OpenVX/NPU stack
- SDK: `aml_npu_sdk` with `acuity-toolkit/python/convert`
- Model: `google/gemma-3-270m-it`
- First graph shapes: `prefill` sequence length 64, `decode` sequence length 1

## Repo Layout

- `configs/gemma3_270m_block.yaml` - canonical v1 target config.
- `scripts/export_gemma_block.py` - exports fixed-shape decoder-block ONNX files and reference tensors.
- `scripts/convert_acuity.py` - wraps Amlogic Acuity conversion and updates artifact manifests.
- `scripts/make_runtime_fixture.py` - generates a tiny local fixture for CI/runtime smoke tests.
- `src/amlogic_transformers/` - Python package for config, manifests, export, conversion, and serving.
- `runtime/` - C++ block runner, adapter ABI, and adapter implementations.
- `.github/workflows/ci.yml` - GitHub Actions workflow running the local test path.

## Fast Local Check

This path does not need Hugging Face credentials or the Amlogic SDK. It proves
the repo builds and the runtime ABI works.

```sh
cd /home/pranavthombare/amlogic-npu-transformers
make test
```

Expected final line:

```text
hidden_states_out max_abs=0 max_rel=0 cosine=1
```

## Development Setup

```sh
cd /home/pranavthombare/amlogic-npu-transformers
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[export,serve]'
```

Gemma access is gated. Export requires an authenticated Hugging Face token:

```sh
export HF_TOKEN=...
```

The Amlogic converter requires the SDK:

```sh
export AML_NPU_SDK=/path/to/aml_npu_sdk
```

## VIM3 Milestone Commands

Export one Gemma decoder block:

```sh
python scripts/export_gemma_block.py --config configs/gemma3_270m_block.yaml
```

Convert generated ONNX graphs through Acuity:

```sh
python scripts/convert_acuity.py --config configs/gemma3_270m_block.yaml
```

Build the runtime harness without CMake:

```sh
make build-runtime
```

Or with CMake:

```sh
cmake -S runtime -B runtime/build
cmake --build runtime/build
```

Run the harness against a generated adapter:

```sh
build/local/amlogic-transformers-block-runner \
  --manifest artifacts/gemma3_270m/block0/artifact_manifest.json \
  --graph prefill \
  --adapter /path/to/libamlogic_transformers_adapter.so
```

Start the API shell:

```sh
amlogic-transformers-server
```

The chat endpoint returns `503` until converted NPU artifacts exist, and `501`
until the one-block NPU numerical check has passed.

## Publishing Notes

The repo is ready for GitHub CI. Before making it public, choose a license and
confirm no private SDK files, model weights, `.nb` files, Hugging Face tokens,
or board-specific generated binaries are committed.
