# Amlogic NPU Transformers

Transformer inference tooling for Amlogic NPU targets, starting with
`google/gemma-3-270m-it` on Khadas VIM3.

This is not a `llama.cpp`, GGUF, or CPU fallback stack. The runtime boundary is
the Amlogic NPU SDK.

Conversion is an off-board host step. VIM3 runtime stays native ARM64 and talks
to the NPU through the Amlogic stack.

## Status

- Fixed-shape Gemma decoder-block export path.
- Amlogic Acuity conversion wrapper.
- C++ runtime harness with a stable adapter ABI.
- Local reference adapter for SDK-free CI checks.
- OpenAI-compatible API shell gated behind real NPU artifacts.

## Check

```sh
make test
```

Expected runtime check:

```text
hidden_states_out max_abs=0 max_rel=0 cosine=1
```

## VIM3 Path

```sh
export HF_TOKEN=...
export AML_NPU_SDK=/path/to/aml_npu_sdk
python scripts/export_gemma_block.py --config configs/gemma3_270m_block.yaml
python scripts/convert_acuity.py --config configs/gemma3_270m_block.yaml
make build-runtime
```

See `docs/vim3_runbook.md` for the board runbook.
