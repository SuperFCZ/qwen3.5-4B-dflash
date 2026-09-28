#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnn_d_flash_group_quant_linear.h"

namespace {
constexpr int64_t kM = 16;
constexpr int64_t kK = 256;
constexpr int64_t kN = 64;
constexpr int kRepetitions = 2;
constexpr size_t kGuard = 512;  // retain device pointer alignment after the prefix
constexpr uint8_t kCanary = 0xA5;

template <typename T>
void Check(T result, const char *action)
{
    if (result != ACL_SUCCESS) {
        const char *recent = aclGetRecentErrMsg();
        throw std::runtime_error(std::string(action) + " failed: " + std::to_string(result) +
                                 "; aclGetRecentErrMsg: " + (recent != nullptr ? recent : "<none>"));
    }
}

std::vector<uint8_t> ReadBytes(const std::string &path, size_t count)
{
    std::vector<uint8_t> result(count);
    std::ifstream file(path, std::ios::binary);
    if (!file.read(reinterpret_cast<char *>(result.data()), count) ||
        file.peek() != std::char_traits<char>::eof()) {
        throw std::runtime_error("input file has wrong size: " + path);
    }
    return result;
}

void WriteBytes(const std::string &path, const std::vector<uint8_t> &bytes)
{
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
    if (!file) throw std::runtime_error("cannot write: " + path);
}

struct GuardedBuffer {
    void *allocation = nullptr;
    void *data = nullptr;
    size_t bytes = 0;

    void Release()
    {
        if (allocation != nullptr) aclrtFree(allocation);
        allocation = data = nullptr;
        bytes = 0;
    }

    void Allocate(size_t size)
    {
        Release();
        bytes = size;
        Check(aclrtMalloc(&allocation, bytes + 2 * kGuard, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc");
        data = static_cast<uint8_t *>(allocation) + kGuard;
        Check(aclrtMemset(allocation, bytes + 2 * kGuard, kCanary, bytes + 2 * kGuard), "init guards");
    }

    void CopyIn(const std::vector<uint8_t> &host)
    {
        if (host.size() != bytes) throw std::runtime_error("copy input size differs");
        Check(aclrtMemcpy(data, bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE), "copy input");
    }

    std::vector<uint8_t> CopyOut() const
    {
        std::vector<uint8_t> host(bytes);
        Check(aclrtMemcpy(host.data(), bytes, data, bytes, ACL_MEMCPY_DEVICE_TO_HOST), "copy output");
        return host;
    }

    void CheckGuards() const
    {
        if (allocation == nullptr) return;
        std::vector<uint8_t> before(kGuard), after(kGuard);
        Check(aclrtMemcpy(before.data(), kGuard, allocation, kGuard, ACL_MEMCPY_DEVICE_TO_HOST), "read prefix");
        Check(aclrtMemcpy(after.data(), kGuard, static_cast<uint8_t *>(data) + bytes,
                          kGuard, ACL_MEMCPY_DEVICE_TO_HOST), "read suffix");
        const auto intact = [](uint8_t value) { return value == kCanary; };
        if (!std::all_of(before.begin(), before.end(), intact) ||
            !std::all_of(after.begin(), after.end(), intact)) {
            throw std::runtime_error("device buffer guard overwritten");
        }
    }
};

struct DeviceTensor {
    GuardedBuffer buffer;
    aclTensor *desc = nullptr;
    void Release()
    {
        if (desc != nullptr) aclDestroyTensor(desc);
        desc = nullptr;
        buffer.Release();
    }
};

struct Resources {
    bool aclReady = false;
    bool deviceReady = false;
    int deviceId = 0;
    aclrtStream stream = nullptr;
    GuardedBuffer workspace;
    DeviceTensor x, w, s, y;
    ~Resources()
    {
        if (stream != nullptr) aclrtSynchronizeStream(stream);
        workspace.Release();
        y.Release(); s.Release(); w.Release(); x.Release();
        if (stream != nullptr) aclrtDestroyStream(stream);
        if (deviceReady) aclrtResetDevice(deviceId);
        if (aclReady) aclFinalize();
    }
};

void MakeTensor(DeviceTensor &tensor, const std::vector<int64_t> &origin,
                const std::vector<int64_t> &storage, aclDataType dtype, aclFormat format,
                size_t bytes, const std::vector<uint8_t> *host = nullptr)
{
    tensor.buffer.Allocate(bytes);
    if (host != nullptr) tensor.buffer.CopyIn(*host);
    // Strides describe the logical ND tensor, while storage describes actual NZ
    // bytes. Never reinterpret [8,4,16,32] as an ND logical weight argument.
    std::vector<int64_t> strides(origin.size(), 1);
    for (size_t i = origin.size(); i > 1; --i) strides[i - 2] = strides[i - 1] * origin[i - 1];
    tensor.desc = aclCreateTensor(origin.data(), origin.size(), dtype, strides.data(), 0,
                                  format, storage.data(), storage.size(), tensor.buffer.data);
    if (tensor.desc == nullptr) throw std::runtime_error("aclCreateTensor failed");
}

void CheckScales(const std::vector<uint8_t> &raw)
{
    for (size_t i = 0; i < raw.size(); i += 2) {
        const uint16_t bits = static_cast<uint16_t>(raw[i]) | (static_cast<uint16_t>(raw[i + 1]) << 8);
        if ((bits & 0x8000) || (bits & 0x7C00) == 0x7C00 || (bits & 0x7FFF) == 0) {
            throw std::runtime_error("scale must be positive finite FP16");
        }
    }
}
}  // namespace

int main(int argc, char **argv)
{
    std::string dataDir;
    try {
        if (argc != 3) throw std::runtime_error("usage: dflash_group_quant_linear_test DEVICE_ID CASE_DIR");
        dataDir = argv[2];
        // Invalidate previous successful evidence before touching the device.
        std::ofstream(dataDir + "/execution.json") << "{\"status\":\"RUNNING\"}\n";
        auto x = ReadBytes(dataDir + "/x.bin", kM * kK * 2);
        auto w = ReadBytes(dataDir + "/w_nz.bin", kN * kK);
        auto s = ReadBytes(dataDir + "/s_gn.bin", (kK / 128) * kN * 2);
        CheckScales(s);
        Resources r;
        r.deviceId = std::stoi(argv[1]);
        Check(aclInit(nullptr), "aclInit"); r.aclReady = true;
        Check(aclrtSetDevice(r.deviceId), "aclrtSetDevice"); r.deviceReady = true;
        Check(aclrtCreateStream(&r.stream), "aclrtCreateStream");
        MakeTensor(r.x, {kM, kK}, {kM, kK}, ACL_FLOAT16, ACL_FORMAT_ND, x.size(), &x);
        MakeTensor(r.w, {kN, kK}, {kK / 32, kN / 16, 16, 32}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, w.size(), &w);
        MakeTensor(r.s, {kK / 128, kN}, {kK / 128, kN}, ACL_FLOAT16, ACL_FORMAT_ND, s.size(), &s);
        MakeTensor(r.y, {kM, kN}, {kM, kN}, ACL_FLOAT16, ACL_FORMAT_ND, kM * kN * 2);

        uint64_t peakWorkspace = 0;
        for (int repeat = 0; repeat < kRepetitions; ++repeat) {
            // Different FP16 NaN poison on each call exposes unwritten output.
            Check(aclrtMemset(r.y.buffer.data, r.y.buffer.bytes, 0x7F + repeat * 128,
                              r.y.buffer.bytes), "poison output");
            uint64_t workspaceBytes = 0;
            aclOpExecutor *executor = nullptr;
            Check(aclnnDFlashGroupQuantLinearGetWorkspaceSize(r.x.desc, r.w.desc, r.s.desc, r.y.desc,
                                                              &workspaceBytes, &executor),
                  "aclnnDFlashGroupQuantLinearGetWorkspaceSize");
            if (workspaceBytes > 0 && r.workspace.bytes != workspaceBytes) r.workspace.Allocate(workspaceBytes);
            peakWorkspace = std::max(peakWorkspace, workspaceBytes);
            Check(aclnnDFlashGroupQuantLinear(r.workspace.data, workspaceBytes, executor, r.stream),
                  "aclnnDFlashGroupQuantLinear");
            Check(aclrtSynchronizeStream(r.stream), "aclrtSynchronizeStream");
            r.x.buffer.CheckGuards(); r.w.buffer.CheckGuards(); r.s.buffer.CheckGuards();
            r.y.buffer.CheckGuards(); r.workspace.CheckGuards();
            if (r.x.buffer.CopyOut() != x || r.w.buffer.CopyOut() != w || r.s.buffer.CopyOut() != s) {
                throw std::runtime_error("read-only input changed");
            }
            WriteBytes(dataDir + "/actual-" + std::to_string(repeat) + ".bin", r.y.buffer.CopyOut());
        }
        std::ofstream report(dataDir + "/execution.json");
        report << "{\"status\":\"PASS\",\"runtime\":\"AscendCL ACLNN\","
                  "\"op\":\"DFlashGroupQuantLinear\",\"cpu_fallback\":false,"
                  "\"input_readonly\":true,\"guards_intact\":true,\"repetitions\":" << kRepetitions <<
                  ",\"device_id\":" << r.deviceId << ",\"workspace_bytes\":" << peakWorkspace << "}\n";
        if (!report) throw std::runtime_error("cannot write execution report");
        std::cout << "ACLNN tiny execution completed; workspace=" << peakWorkspace
                  << " bytes, guards/inputs intact; numerical comparison follows\n";
        return EXIT_SUCCESS;
    } catch (const std::exception &e) {
        if (!dataDir.empty()) std::ofstream(dataDir + "/execution.json") << "{\"status\":\"FAIL\"}\n";
        std::cerr << "FAIL: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
