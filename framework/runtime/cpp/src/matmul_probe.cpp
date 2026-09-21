#include "qwen35_dflash/matmul_probe.hpp"
#include "qwen35_dflash/sha256.hpp"
#include <acl/acl.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace qwen35::dflash {
namespace {
void Require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}
void Check(aclError status, const char* operation) {
  Require(status == ACL_SUCCESS, std::string(operation) + " failed: " + std::to_string(status));
}
std::string Quote(const std::string& text) {
  std::string result = "\"";
  for (unsigned char c : text) {
    if (c == '"' || c == '\\') { result += '\\'; result += static_cast<char>(c); }
    else if (c < 32) {
      const char* digits = "0123456789abcdef";
      result += "\\u00"; result += digits[c >> 4]; result += digits[c & 15];
    } else result += static_cast<char>(c);
  }
  return result + '"';
}
struct Buffer {
  void* device = nullptr;
  void* host = nullptr;
  aclDataBuffer* data = nullptr;
  aclmdlDataset* dataset = nullptr;
  std::size_t bytes = 0;
  void Allocate(std::size_t size) {
    bytes = size;
    Check(aclrtMalloc(&device, size, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc");
    Check(aclrtMallocHost(&host, size), "aclrtMallocHost");
    data = aclCreateDataBuffer(device, size);
    dataset = aclmdlCreateDataset();
    Require(data && dataset, "failed to create probe dataset/buffer");
    Check(aclmdlAddDatasetBuffer(dataset, data), "aclmdlAddDatasetBuffer");
  }
};
struct Session {
  bool initialized = false, device_set = false, loaded = false;
  int device_id = 0;
  std::uint32_t model_id = 0;
  aclrtContext context = nullptr;
  aclrtStream stream = nullptr;
  aclmdlDesc* desc = nullptr;
  Buffer input, output;
  ~Session() { Close(); }
  std::size_t Close() noexcept {
    std::size_t errors = 0;
    auto check = [&](aclError status) { if (status != ACL_SUCCESS) ++errors; };
    if (context) check(aclrtSetCurrentContext(context));
    if (stream) check(aclrtSynchronizeStream(stream));
    for (auto* buffer : {&input, &output}) {
      if (buffer->dataset) { check(aclmdlDestroyDataset(buffer->dataset)); buffer->dataset = nullptr; }
      if (buffer->data) { check(aclDestroyDataBuffer(buffer->data)); buffer->data = nullptr; }
      if (buffer->device) { check(aclrtFree(buffer->device)); buffer->device = nullptr; }
      if (buffer->host) { check(aclrtFreeHost(buffer->host)); buffer->host = nullptr; }
    }
    if (desc) { check(aclmdlDestroyDesc(desc)); desc = nullptr; }
    if (loaded) { check(aclmdlUnload(model_id)); loaded = false; }
    if (stream) { check(aclrtDestroyStream(stream)); stream = nullptr; }
    if (context) { check(aclrtDestroyContext(context)); context = nullptr; }
    if (device_set) { check(aclrtResetDevice(device_id)); device_set = false; }
    if (initialized) { check(aclFinalize()); initialized = false; }
    return errors;
  }
  void Load(const std::filesystem::path& path, int device) {
    device_id = device;
    Check(aclInit(nullptr), "aclInit"); initialized = true;
    Check(aclrtSetDevice(device), "aclrtSetDevice"); device_set = true;
    Check(aclrtCreateContext(&context, device), "aclrtCreateContext");
    Check(aclrtSetCurrentContext(context), "aclrtSetCurrentContext");
    Check(aclrtCreateStream(&stream), "aclrtCreateStream");
    Check(aclmdlLoadFromFile(path.c_str(), &model_id), "aclmdlLoadFromFile"); loaded = true;
    desc = aclmdlCreateDesc(); Require(desc != nullptr, "aclmdlCreateDesc returned null");
    Check(aclmdlGetDesc(desc, model_id), "aclmdlGetDesc");
  }
  void ValidateIo(std::size_t rows, std::size_t k, std::size_t n) {
    Require(aclmdlGetNumInputs(desc) == 1 && aclmdlGetNumOutputs(desc) == 1,
            "probe OM must have exactly one input and one output");
    for (bool out : {false, true}) {
      aclmdlIODims dims{};
      Check(out ? aclmdlGetOutputDims(desc, 0, &dims) : aclmdlGetInputDims(desc, 0, &dims), "aclmdlGetIODims");
      const auto dtype = out ? aclmdlGetOutputDataType(desc, 0) : aclmdlGetInputDataType(desc, 0);
      const auto bytes = out ? aclmdlGetOutputSizeByIndex(desc, 0) : aclmdlGetInputSizeByIndex(desc, 0);
      const auto columns = out ? n : k;
      Require(dtype == ACL_FLOAT16 && dims.dimCount == 2 && dims.dims[0] == static_cast<std::int64_t>(rows) &&
              dims.dims[1] == static_cast<std::int64_t>(columns) && bytes == rows * columns * 2,
              std::string("probe OM ") + (out ? "output" : "input") + " ABI must be FP16 [M,K/N] with exact byte count");
    }
    input.Allocate(rows * k * 2);
    output.Allocate(rows * n * 2);
  }
  void Execute(const char* values) {
    std::memcpy(input.host, values, input.bytes);
    Check(aclrtMemcpyAsync(input.device, input.bytes, input.host, input.bytes, ACL_MEMCPY_HOST_TO_DEVICE, stream), "aclrtMemcpyAsync(H2D)");
    Check(aclmdlExecuteAsync(model_id, input.dataset, output.dataset, stream), "aclmdlExecuteAsync");
    Check(aclrtMemcpyAsync(output.host, output.bytes, output.device, output.bytes, ACL_MEMCPY_DEVICE_TO_HOST, stream), "aclrtMemcpyAsync(D2H)");
    Check(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream");
  }
};
}

int RunMatmulProbe(int argc, char** argv) {
  std::map<std::string, std::string> options;
  for (int i = 2; i < argc; i += 2) {
    const std::string name(argv[i]);
    Require(i + 1 < argc && name.rfind("--", 0) == 0 && options.emplace(name.substr(2), argv[i + 1]).second,
            "probe expects distinct --name value pairs");
  }
  auto take = [&](const char* key) {
    const auto it = options.find(key);
    Require(it != options.end(), std::string("missing probe option --") + key);
    const auto value = it->second; options.erase(it); return value;
  };
  auto number = [&](const char* key, std::size_t min, std::size_t max) {
    const auto text = take(key);
    Require(!text.empty() && std::all_of(text.begin(), text.end(), [](char c) { return c >= '0' && c <= '9'; }), "invalid probe integer");
    const auto value = std::stoull(text);
    Require(value >= min && value <= max, std::string("probe value out of range: ") + key);
    return static_cast<std::size_t>(value);
  };
  const std::filesystem::path model = take("model"), inputs = take("input"), directory = take("output-dir");
  const auto model_hash = take("model-sha256"), input_hash = take("input-sha256");
  const auto rows = number("rows", 16, 64), k = number("k", 1, 12800), n = number("n", 1, 65535);
  const auto repetitions = number("repetitions", 1, 1000), device = number("device-id", 0, 65535);
  Require(options.empty() && (rows == 16 || rows == 64), "unknown probe option or unsupported M gear");
  Require(Sha256File(model) == model_hash && Sha256File(inputs) == input_hash, "probe model/input hash differs");
  const std::size_t input_bytes = rows * k * 2;
  Require(std::filesystem::file_size(inputs) == input_bytes * 4, "probe input file must contain exactly four FP16 vectors");
  Require(std::filesystem::create_directory(directory), "probe output directory must be new");
  std::vector<char> values(input_bytes * 4);
  std::ifstream in(inputs, std::ios::binary);
  Require(bool(in.read(values.data(), static_cast<std::streamsize>(values.size()))), "cannot read probe inputs");
  Require(Sha256(std::string_view(values.data(), values.size())) == input_hash, "probe inputs changed while reading");
  std::string status = "FAIL", phase = "load", error;
  std::size_t calls = 0;
  try {
    Session session;
    session.Load(model, static_cast<int>(device));
    session.ValidateIo(rows, k, n);
    phase = "execute";
    std::ofstream out(directory / "outputs.bin", std::ios::binary);
    Require(bool(out), "cannot open probe outputs");
    for (std::size_t vector = 0; vector < 4; ++vector) {
      for (std::size_t repeat = 0; repeat < repetitions; ++repeat) {
        session.Execute(values.data() + vector * input_bytes);
        out.write(static_cast<const char*>(session.output.host), static_cast<std::streamsize>(session.output.bytes));
        Require(bool(out), "cannot save probe outputs");
        ++calls;
      }
    }
    out.close(); Require(bool(out), "cannot flush probe outputs");
    phase = "cleanup";
    Require(session.Close() == 0, "probe ACL cleanup failed");
    Require(Sha256File(model) == model_hash && Sha256File(inputs) == input_hash, "probe sources changed during execution");
    status = "PASS"; phase = "complete";
  } catch (const std::exception& exception) {
    error = exception.what();
  }
  const bool fake = std::string(QWEN35_DFLASH_RUNNER_VERSION).find("fake-acl") != std::string::npos;
  std::ofstream report(directory / "execution.json");
  report << "{\"schema_version\":1,\"runtime\":\"AscendCL C++ matmul probe\",\"runner_version\":"
         << Quote(QWEN35_DFLASH_RUNNER_VERSION) << ",\"fake_acl\":" << (fake ? "true" : "false")
         << ",\"cpu_fallback\":false,\"status\":" << Quote(status) << ",\"phase\":" << Quote(phase)
         << ",\"error\":" << Quote(error) << ",\"device_id\":" << device
         << ",\"model_sha256\":" << Quote(model_hash) << ",\"input_sha256\":" << Quote(input_hash)
         << ",\"rows\":" << rows << ",\"k\":" << k << ",\"n\":" << n
         << ",\"vectors\":4,\"repetitions\":" << repetitions << ",\"calls\":" << calls
         << ",\"io_validated\":" << (phase == "load" ? "false" : "true")
         << ",\"output_sha256\":" << Quote(status == "PASS" ? Sha256File(directory / "outputs.bin") : "") << "}\n";
  report.close(); Require(bool(report), "cannot save probe execution report");
  if (!error.empty()) std::cerr << "[matmul-probe] " << error << '\n';
  return status == "PASS" ? 0 : 1;
}
}  // namespace qwen35::dflash
