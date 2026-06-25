#include "block_runner.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace {

std::string read_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("failed to open " + path);
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

std::string dirname_of(const std::string& path) {
  const auto pos = path.find_last_of('/');
  if (pos == std::string::npos) {
    return ".";
  }
  return path.substr(0, pos);
}

std::string join_path(const std::string& root, const std::string& path) {
  if (path.empty()) {
    return path;
  }
  if (path[0] == '/') {
    return path;
  }
  return root + "/" + path;
}

std::string extract_string(const std::string& object, const std::string& key, bool required = true) {
  const std::regex pattern("\"" + key + "\"\\s*:\\s*(null|\"([^\"]*)\")");
  std::smatch match;
  if (!std::regex_search(object, match, pattern)) {
    if (required) {
      throw std::runtime_error("missing JSON key: " + key);
    }
    return "";
  }
  if (match[1] == "null") {
    return "";
  }
  return match[2];
}

std::vector<int64_t> extract_shape(const std::string& object) {
  const std::regex pattern("\"shape\"\\s*:\\s*\\[([^\\]]*)\\]");
  std::smatch match;
  if (!std::regex_search(object, match, pattern)) {
    throw std::runtime_error("missing tensor shape");
  }
  std::vector<int64_t> shape;
  std::stringstream ss(match[1]);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty()) {
      shape.push_back(std::stoll(item));
    }
  }
  return shape;
}

std::vector<std::string> extract_objects_from_array(const std::string& text, const std::string& key) {
  const auto key_pos = text.find("\"" + key + "\"");
  if (key_pos == std::string::npos) {
    return {};
  }
  const auto array_start = text.find('[', key_pos);
  if (array_start == std::string::npos) {
    return {};
  }

  std::vector<std::string> objects;
  int array_depth = 0;
  int object_depth = 0;
  bool in_string = false;
  bool escaped = false;
  size_t object_start = std::string::npos;

  for (size_t i = array_start; i < text.size(); ++i) {
    const char c = text[i];
    if (escaped) {
      escaped = false;
      continue;
    }
    if (c == '\\' && in_string) {
      escaped = true;
      continue;
    }
    if (c == '"') {
      in_string = !in_string;
      continue;
    }
    if (in_string) {
      continue;
    }
    if (c == '[') {
      ++array_depth;
    } else if (c == ']') {
      --array_depth;
      if (array_depth == 0) {
        break;
      }
    } else if (c == '{') {
      if (object_depth == 0) {
        object_start = i;
      }
      ++object_depth;
    } else if (c == '}') {
      --object_depth;
      if (object_depth == 0 && object_start != std::string::npos) {
        objects.push_back(text.substr(object_start, i - object_start + 1));
        object_start = std::string::npos;
      }
    }
  }
  return objects;
}

std::vector<TensorRef> extract_tensor_refs(const std::string& graph_object, const std::string& key) {
  std::vector<TensorRef> refs;
  for (const auto& object : extract_objects_from_array(graph_object, key)) {
    refs.push_back(TensorRef{
      extract_string(object, "name"),
      extract_string(object, "path"),
      extract_string(object, "dtype"),
      extract_shape(object),
    });
  }
  return refs;
}

int dtype_from_npy_descr(const std::string& descr) {
  if (descr == "<f4" || descr == "|f4") {
    return AML_TRANSFORMERS_DTYPE_F32;
  }
  if (descr == "<i8" || descr == "|i8") {
    return AML_TRANSFORMERS_DTYPE_I64;
  }
  if (descr == "<i4" || descr == "|i4") {
    return AML_TRANSFORMERS_DTYPE_I32;
  }
  if (descr == "|u1") {
    return AML_TRANSFORMERS_DTYPE_U8;
  }
  if (descr == "|i1") {
    return AML_TRANSFORMERS_DTYPE_I8;
  }
  throw std::runtime_error("unsupported npy dtype descriptor: " + descr);
}

size_t dtype_size(int dtype) {
  switch (dtype) {
    case AML_TRANSFORMERS_DTYPE_F32:
    case AML_TRANSFORMERS_DTYPE_I32:
      return 4;
    case AML_TRANSFORMERS_DTYPE_I64:
      return 8;
    case AML_TRANSFORMERS_DTYPE_U8:
    case AML_TRANSFORMERS_DTYPE_I8:
      return 1;
    default:
      throw std::runtime_error("unsupported dtype");
  }
}

std::vector<int64_t> parse_npy_shape(const std::string& header) {
  const auto open = header.find('(');
  const auto close = header.find(')', open);
  if (open == std::string::npos || close == std::string::npos) {
    throw std::runtime_error("invalid npy header shape");
  }
  std::vector<int64_t> shape;
  std::stringstream ss(header.substr(open + 1, close - open - 1));
  std::string item;
  while (std::getline(ss, item, ',')) {
    item.erase(std::remove_if(item.begin(), item.end(), ::isspace), item.end());
    if (!item.empty()) {
      shape.push_back(std::stoll(item));
    }
  }
  return shape;
}

std::string parse_npy_descr(const std::string& header) {
  const std::regex pattern("'descr'\\s*:\\s*'([^']+)'");
  std::smatch match;
  if (!std::regex_search(header, match, pattern)) {
    throw std::runtime_error("invalid npy header dtype");
  }
  return match[1];
}

}  // namespace

GraphManifest load_graph_manifest(const std::string& manifest_path, const std::string& graph_name) {
  const auto text = read_file(manifest_path);
  const auto root = dirname_of(manifest_path);

  for (const auto& graph_object : extract_objects_from_array(text, "graphs")) {
    const auto name = extract_string(graph_object, "name");
    if (name != graph_name) {
      continue;
    }
    auto nb_path = extract_string(graph_object, "nb_path", false);
    auto adapter_library = extract_string(graph_object, "adapter_library", false);
    return GraphManifest{
      name,
      join_path(root, nb_path),
      join_path(root, adapter_library),
      extract_tensor_refs(graph_object, "reference_inputs"),
      extract_tensor_refs(graph_object, "reference_outputs"),
    };
  }

  throw std::runtime_error("graph not found in manifest: " + graph_name);
}

LoadedTensor load_npy_tensor(const std::string& name, const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("failed to open npy tensor " + path);
  }

  char magic[6];
  in.read(magic, sizeof(magic));
  if (std::strncmp(magic, "\x93NUMPY", 6) != 0) {
    throw std::runtime_error("not an npy file: " + path);
  }
  uint8_t major = 0;
  uint8_t minor = 0;
  in.read(reinterpret_cast<char*>(&major), 1);
  in.read(reinterpret_cast<char*>(&minor), 1);

  uint32_t header_len = 0;
  if (major == 1) {
    uint16_t small_len = 0;
    in.read(reinterpret_cast<char*>(&small_len), 2);
    header_len = small_len;
  } else if (major == 2 || major == 3) {
    in.read(reinterpret_cast<char*>(&header_len), 4);
  } else {
    throw std::runtime_error("unsupported npy version in " + path);
  }

  std::string header(header_len, '\0');
  in.read(header.data(), header_len);
  const auto dtype = dtype_from_npy_descr(parse_npy_descr(header));
  const auto shape = parse_npy_shape(header);
  if (header.find("'fortran_order': True") != std::string::npos) {
    throw std::runtime_error("fortran-order npy tensors are not supported: " + path);
  }

  size_t elements = 1;
  for (const auto dim : shape) {
    elements *= static_cast<size_t>(dim);
  }
  std::vector<uint8_t> bytes(elements * dtype_size(dtype));
  in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  if (!in) {
    throw std::runtime_error("truncated npy tensor: " + path);
  }
  return LoadedTensor{name, dtype, shape, std::move(bytes)};
}

CompareResult compare_f32(const LoadedTensor& expected, const LoadedTensor& actual) {
  if (expected.dtype != AML_TRANSFORMERS_DTYPE_F32 || actual.dtype != AML_TRANSFORMERS_DTYPE_F32) {
    throw std::runtime_error("compare_f32 requires float32 tensors");
  }
  if (expected.bytes.size() != actual.bytes.size()) {
    throw std::runtime_error("tensor byte sizes differ");
  }

  const auto* a = reinterpret_cast<const float*>(expected.bytes.data());
  const auto* b = reinterpret_cast<const float*>(actual.bytes.data());
  const size_t count = expected.bytes.size() / sizeof(float);
  double max_abs = 0.0;
  double max_rel = 0.0;
  double dot = 0.0;
  double norm_a = 0.0;
  double norm_b = 0.0;

  for (size_t i = 0; i < count; ++i) {
    const double da = a[i];
    const double db = b[i];
    const double abs_err = std::abs(da - db);
    const double rel_err = abs_err / std::max(std::abs(da), 1e-9);
    max_abs = std::max(max_abs, abs_err);
    max_rel = std::max(max_rel, rel_err);
    dot += da * db;
    norm_a += da * da;
    norm_b += db * db;
  }

  const double denom = std::sqrt(norm_a) * std::sqrt(norm_b);
  return CompareResult{max_abs, max_rel, denom > 0.0 ? dot / denom : 0.0};
}

AmlogicTransformersTensor as_abi_tensor(LoadedTensor& tensor) {
  if (tensor.shape.size() > 8) {
    throw std::runtime_error("rank > 8 is not supported by adapter ABI");
  }
  AmlogicTransformersTensor abi{};
  abi.name = tensor.name.c_str();
  abi.dtype = tensor.dtype;
  abi.rank = static_cast<int>(tensor.shape.size());
  for (size_t i = 0; i < tensor.shape.size(); ++i) {
    abi.shape[i] = tensor.shape[i];
  }
  abi.data = tensor.bytes.data();
  abi.byte_size = tensor.bytes.size();
  return abi;
}
