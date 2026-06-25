# VIM3 Runbook

This is the board path for proving `google/gemma-3-270m-it` inference through
the Amlogic NPU SDK.

## 1. Prepare the Board

Use Khadas Ubuntu BSP for VIM3 and confirm the Amlogic NPU stack is installed.
The runtime path assumes the Galcore/OpenVX libraries and Acuity-generated
runtime dependencies are available on the board.

Install the Python package:

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[export,serve]'
```

## 2. Export the Decoder Block

```sh
export HF_TOKEN=...
python scripts/export_gemma_block.py --config configs/gemma3_270m_block.yaml
```

Expected artifacts:

- `artifacts/gemma3_270m/block0/prefill/prefill.onnx`
- `artifacts/gemma3_270m/block0/decode/decode.onnx`
- `artifacts/gemma3_270m/block0/*/reference/*.npy`
- `artifacts/gemma3_270m/block0/artifact_manifest.json`

## 3. Convert with Acuity

```sh
export AML_NPU_SDK=/path/to/aml_npu_sdk
python scripts/convert_acuity.py --config configs/gemma3_270m_block.yaml
```

The script writes the Acuity command for each graph into the artifact manifest.
If conversion succeeds, the manifest should also contain `.nb` paths and the
generated `libnn_*.so` path discovered under the SDK output directory.

## 4. Implement the Amlogic Adapter

Use `runtime/adapters/acuity_shim_template.cpp` as the starting point and expose
the ABI in `runtime/include/amlogic_transformers/adapter_abi.h`:

- `amlogic_transformers_init`
- `amlogic_transformers_run`
- `amlogic_transformers_shutdown`

The adapter should initialize the generated graph, copy named runner inputs into
the SDK input tensors, run inference, and copy SDK outputs into the preallocated
runner output buffers.

## 5. Check Numerical Drift

```sh
make build-runtime
build/local/amlogic-transformers-block-runner \
  --manifest artifacts/gemma3_270m/block0/artifact_manifest.json \
  --graph prefill \
  --adapter /path/to/libamlogic_transformers_adapter.so
```

Repeat for `decode`.

Passing this check is the gate for enabling generation and the OpenAI-compatible
server path.
