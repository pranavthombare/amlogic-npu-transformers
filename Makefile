PYTHON ?= python3
CXX ?= g++
BUILD_DIR ?= build/local

RUNTIME_BIN := $(BUILD_DIR)/gemma-npu-block-runner
REFERENCE_ADAPTER := $(BUILD_DIR)/libgemma_reference_adapter.so
FIXTURE_DIR := $(BUILD_DIR)/fixture

.PHONY: help build-runtime fixture test test-python test-runtime clean

help:
	@printf '%s\n' \
		'Targets:' \
		'  make test          Run Python tests and local C++ runtime smoke test' \
		'  make build-runtime Build the C++ runner and local reference adapter' \
		'  make fixture       Generate a tiny artifact manifest and .npy tensors' \
		'  make clean         Remove local build output'

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(RUNTIME_BIN): runtime/src/main.cpp runtime/src/block_runner.cpp runtime/src/block_runner.h runtime/include/gemma_npu/adapter_abi.h | $(BUILD_DIR)
	$(CXX) -std=c++17 -Wall -Wextra -Wpedantic -Iruntime/include -Iruntime/src runtime/src/main.cpp runtime/src/block_runner.cpp -ldl -o $(RUNTIME_BIN)

$(REFERENCE_ADAPTER): runtime/adapters/reference_copy_adapter.cpp runtime/include/gemma_npu/adapter_abi.h | $(BUILD_DIR)
	$(CXX) -std=c++17 -Wall -Wextra -Wpedantic -fPIC -shared -Iruntime/include runtime/adapters/reference_copy_adapter.cpp -o $(REFERENCE_ADAPTER)

build-runtime: $(RUNTIME_BIN) $(REFERENCE_ADAPTER)

fixture: build-runtime
	PYTHONPATH=src $(PYTHON) scripts/make_runtime_fixture.py --output $(FIXTURE_DIR) --adapter ../libgemma_reference_adapter.so

test-python:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v

test-runtime: fixture
	$(RUNTIME_BIN) --manifest $(FIXTURE_DIR)/artifact_manifest.json --graph prefill --atol 0 --rtol 0 --min-cosine 0.999999

test: test-python test-runtime

clean:
	rm -rf build
