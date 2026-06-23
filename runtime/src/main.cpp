#include "block_runner.h"

#include <dlfcn.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Args {
  std::string manifest;
  std::string graph = "prefill";
  std::string adapter;
  double atol = 0.08;
  double rtol = 0.08;
  double min_cosine = 0.98;
};

void usage(const char* program) {
  std::cerr
    << "usage: " << program
    << " --manifest artifact_manifest.json --graph prefill"
    << " [--adapter libgemma_aml_adapter.so]"
    << " [--atol 0.08] [--rtol 0.08] [--min-cosine 0.98]\n";
}

Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    auto require_value = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error(std::string("missing value for ") + name);
      }
      return argv[++i];
    };
    if (key == "--manifest") {
      args.manifest = require_value("--manifest");
    } else if (key == "--graph") {
      args.graph = require_value("--graph");
    } else if (key == "--adapter") {
      args.adapter = require_value("--adapter");
    } else if (key == "--atol") {
      args.atol = std::stod(require_value("--atol"));
    } else if (key == "--rtol") {
      args.rtol = std::stod(require_value("--rtol"));
    } else if (key == "--min-cosine") {
      args.min_cosine = std::stod(require_value("--min-cosine"));
    } else if (key == "--help" || key == "-h") {
      usage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + key);
    }
  }
  if (args.manifest.empty()) {
    throw std::runtime_error("--manifest is required");
  }
  return args;
}

template <typename T>
T load_symbol(void* handle, const char* name) {
  dlerror();
  void* symbol = dlsym(handle, name);
  const char* error = dlerror();
  if (error != nullptr || symbol == nullptr) {
    throw std::runtime_error(std::string("missing adapter symbol: ") + name);
  }
  return reinterpret_cast<T>(symbol);
}

std::string manifest_root_from_path(const std::string& manifest_path) {
  const auto pos = manifest_path.find_last_of('/');
  return pos == std::string::npos ? "." : manifest_path.substr(0, pos);
}

std::string join_path(const std::string& root, const std::string& path) {
  if (path.empty() || path[0] == '/') {
    return path;
  }
  return root + "/" + path;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = parse_args(argc, argv);
    GraphManifest graph = load_graph_manifest(args.manifest, args.graph);
    if (!args.adapter.empty()) {
      graph.adapter_library = args.adapter;
    }
    if (graph.adapter_library.empty()) {
      throw std::runtime_error("adapter library is missing; pass --adapter or update manifest");
    }

    const std::string manifest_root = manifest_root_from_path(args.manifest);

    std::vector<LoadedTensor> input_storage;
    std::vector<GemmaTensor> input_tensors;
    for (const auto& ref : graph.inputs) {
      input_storage.push_back(load_npy_tensor(ref.name, join_path(manifest_root, ref.path)));
    }
    for (auto& tensor : input_storage) {
      input_tensors.push_back(as_abi_tensor(tensor));
    }

    std::vector<LoadedTensor> expected_outputs;
    std::vector<LoadedTensor> actual_outputs;
    std::vector<GemmaTensor> output_tensors;
    for (const auto& ref : graph.outputs) {
      auto expected = load_npy_tensor(ref.name, join_path(manifest_root, ref.path));
      LoadedTensor actual = expected;
      std::fill(actual.bytes.begin(), actual.bytes.end(), 0);
      expected_outputs.push_back(std::move(expected));
      actual_outputs.push_back(std::move(actual));
    }
    for (auto& tensor : actual_outputs) {
      output_tensors.push_back(as_abi_tensor(tensor));
    }

    void* handle = dlopen(graph.adapter_library.c_str(), RTLD_NOW);
    if (handle == nullptr) {
      throw std::runtime_error(std::string("failed to load adapter: ") + dlerror());
    }
    auto init = load_symbol<gemma_aml_init_fn>(handle, "gemma_aml_init");
    auto run = load_symbol<gemma_aml_run_fn>(handle, "gemma_aml_run");
    auto shutdown = load_symbol<gemma_aml_shutdown_fn>(handle, "gemma_aml_shutdown");

    if (init(graph.name.c_str(), graph.nb_path.c_str()) != 0) {
      throw std::runtime_error("gemma_aml_init failed");
    }
    if (run(input_tensors.data(), input_tensors.size(), output_tensors.data(), output_tensors.size()) != 0) {
      shutdown();
      throw std::runtime_error("gemma_aml_run failed");
    }
    shutdown();
    dlclose(handle);

    bool ok = true;
    for (size_t i = 0; i < expected_outputs.size(); ++i) {
      const auto result = compare_f32(expected_outputs[i], actual_outputs[i]);
      std::cout << expected_outputs[i].name
                << " max_abs=" << result.max_abs
                << " max_rel=" << result.max_rel
                << " cosine=" << result.cosine << "\n";
      if (result.max_abs > args.atol || result.max_rel > args.rtol || result.cosine < args.min_cosine) {
        ok = false;
      }
    }
    return ok ? 0 : 2;
  } catch (const std::exception& exc) {
    std::cerr << "error: " << exc.what() << "\n";
    usage(argv[0]);
    return 1;
  }
}
