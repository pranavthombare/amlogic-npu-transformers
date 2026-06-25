# VIM3 Runbook

This is the board path for proving `google/gemma-3-270m-it` inference through
the Amlogic NPU SDK.

Do not run the Acuity converter on the VIM3. The converter is distributed as an
x86-64 tool and belongs on an x86 development host or CI worker. The VIM3 should
only receive generated `.nb` files and generated C/runtime files, then run the
native ARM64 adapter and block runner against `/dev/galcore`.

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

## 2. Export and Convert Off-Board

Run this on an x86 host with the Amlogic SDK and Hugging Face access:

```sh
export HF_TOKEN=...
python scripts/export_gemma_block.py --config configs/gemma3_270m_block.yaml
export AML_NPU_SDK=/path/to/aml_npu_sdk
python scripts/convert_acuity.py --config configs/gemma3_270m_block.yaml
```

Copy `artifacts/gemma3_270m/block0` and Acuity-generated source/runtime files
to the VIM3 after conversion.

## 3. Implement the Amlogic Adapter on VIM3

Use `runtime/adapters/acuity_shim_template.cpp` as the starting point and expose
the ABI in `runtime/include/amlogic_transformers/adapter_abi.h`:

- `amlogic_transformers_init`
- `amlogic_transformers_run`
- `amlogic_transformers_shutdown`

The adapter should initialize the generated graph, copy named runner inputs into
the SDK input tensors, run inference, and copy SDK outputs into the preallocated
runner output buffers.

## 4. Check Numerical Drift on VIM3

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
