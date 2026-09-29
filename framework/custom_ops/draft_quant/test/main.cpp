#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "runner_common.h"
#include "aclnn_d_flash_group_quant_linear.h"
#include "../op_host/d_flash_group_quant_linear_contract.h"

namespace {
constexpr int kRepetitions = 2;
using namespace draft_quant_test;

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

void MakeTensor(DeviceTensor &tensor, const std::vector<int64_t> &view,
                const std::vector<int64_t> &storage, aclDataType dtype, aclFormat format,
                size_t bytes, const std::vector<uint8_t> *host = nullptr)
{
    tensor.buffer.Allocate(bytes);
    if (host != nullptr) tensor.buffer.CopyIn(*host);
    // aclCreateTensor's first shape is ViewShape, not GE origin_shape. CANN
    // 9.0 individual ACLNN uses that view for BOTH Host Tiling shapes. The NZ
    // weight must therefore use its physical view with physical NZ strides.
    std::vector<int64_t> strides(view.size(), 1);
    for (size_t i = view.size(); i > 1; --i) strides[i - 2] = strides[i - 1] * view[i - 1];
    tensor.desc = aclCreateTensor(view.data(), view.size(), dtype, strides.data(), 0,
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

int64_t ParseDimension(const char *text)
{
    size_t end = 0;
    const std::string value(text);
    const int64_t result = std::stoll(value, &end);
    if (end != value.size()) throw std::runtime_error("invalid dimension: " + value);
    return result;
}
}  // namespace

int main(int argc, char **argv)
{
    std::string dataDir;
    try {
        if (argc != 3 && argc != 6 && argc != 8) {
            throw std::runtime_error("usage: dflash_group_quant_linear_test DEVICE_ID CASE_DIR [M K N [WARMUP REPETITIONS]]");
        }
        dataDir = argv[2];
        // Invalidate previous evidence even when dimension validation fails.
        std::ofstream(dataDir + "/execution.json") << "{\"status\":\"RUNNING\"}\n";
        const int64_t kM = argc >= 6 ? ParseDimension(argv[3]) : 16;
        const int64_t kK = argc >= 6 ? ParseDimension(argv[4]) : 256;
        const int64_t kN = argc >= 6 ? ParseDimension(argv[5]) : 64;
        if (!draft_quant_contract::IsSupportedShape(kM, kK, kN)) {
            throw std::runtime_error("unsupported A1/A2 shape");
        }
        Timing timing;
        if (argc == 8) {
            const auto warmup = ParseDimension(argv[6]), repetitions = ParseDimension(argv[7]);
            if (warmup < 3 || warmup > 100 || repetitions < 10 || repetitions > 1000)
                throw std::runtime_error("timing requires 3..100 warmups and 10..1000 repetitions");
            timing.warmup = static_cast<int>(warmup);
            timing.repetitions = static_cast<int>(repetitions);
        }
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
        const std::vector<int64_t> nzShape = {kK / 32, kN / 16, 16, 32};
        MakeTensor(r.w, nzShape, nzShape, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, w.size(), &w);
        MakeTensor(r.s, {kK / 128, kN}, {kK / 128, kN}, ACL_FLOAT16, ACL_FORMAT_ND, s.size(), &s);
        MakeTensor(r.y, {kM, kN}, {kM, kN}, ACL_FLOAT16, ACL_FORMAT_ND, kM * kN * 2);

        uint64_t peakWorkspace = 0;
        std::vector<uint8_t> firstOutput;
        for (int repeat = 0; repeat < kRepetitions + timing.warmup + timing.repetitions; ++repeat) {
            // Different FP16 NaN poison on each call exposes unwritten output.
            Check(aclrtMemset(r.y.buffer.data, r.y.buffer.bytes, 0x7F + (repeat % 2) * 128,
                              r.y.buffer.bytes), "poison output");
            uint64_t workspaceBytes = 0;
            aclOpExecutor *executor = nullptr;
            const auto begin = Clock::now();
            Check(aclnnDFlashGroupQuantLinearGetWorkspaceSize(r.x.desc, r.w.desc, r.s.desc, r.y.desc,
                                                              &workspaceBytes, &executor),
                  "aclnnDFlashGroupQuantLinearGetWorkspaceSize");
            if (workspaceBytes > 0 && r.workspace.bytes != workspaceBytes) r.workspace.Allocate(workspaceBytes);
            peakWorkspace = std::max(peakWorkspace, workspaceBytes);
            const auto prepared = Clock::now();
            Check(aclnnDFlashGroupQuantLinear(r.workspace.data, workspaceBytes, executor, r.stream),
                  "aclnnDFlashGroupQuantLinear");
            Check(aclrtSynchronizeStream(r.stream), "aclrtSynchronizeStream");
            const auto completed = Clock::now();
            if (repeat >= kRepetitions + timing.warmup) {
                timing.prepare.push_back(Milliseconds(begin, prepared));
                timing.execute.push_back(Milliseconds(prepared, completed));
            }
            r.x.buffer.CheckGuards(); r.w.buffer.CheckGuards(); r.s.buffer.CheckGuards();
            r.y.buffer.CheckGuards(); r.workspace.CheckGuards();
            if (r.x.buffer.CopyOut() != x || r.w.buffer.CopyOut() != w || r.s.buffer.CopyOut() != s) {
                throw std::runtime_error("read-only input changed");
            }
            const auto actual = r.y.buffer.CopyOut();
            if (repeat == 0) firstOutput = actual;
            if (actual != firstOutput) throw std::runtime_error("custom output repeat drift");
            if (repeat < kRepetitions)
                WriteBytes(dataDir + "/actual-" + std::to_string(repeat) + ".bin", actual);
        }
        std::ofstream report(dataDir + "/execution.json");
        report << "{\"status\":\"PASS\",\"runtime\":\"AscendCL ACLNN\","
                  "\"op\":\"DFlashGroupQuantLinear\",\"cpu_fallback\":false,"
                  "\"input_readonly\":true,\"guards_intact\":true,\"repetitions\":" << kRepetitions <<
                  ",\"m\":" << kM << ",\"k\":" << kK << ",\"n\":" << kN <<
                  ",\"tile_n\":64,\"tile_k\":" << draft_quant_contract::KTile(kK) <<
                  ",\"device_id\":" << r.deviceId << ",\"workspace_bytes\":" << peakWorkspace
                  << ",\"tracked_device_allocation_bytes\":"
                  << (x.size() + w.size() + s.size() + kM * kN * 2 + peakWorkspace +
                      (4 + (peakWorkspace ? 1 : 0)) * 2 * kGuard);
        timing.Write(report);
        report << "}\n";
        if (!report) throw std::runtime_error("cannot write execution report");
        std::cout << "ACLNN execution completed; M=" << kM << " K=" << kK << " N=" << kN
                  << "; workspace=" << peakWorkspace
                  << " bytes, guards/inputs intact; numerical comparison follows\n";
        return EXIT_SUCCESS;
    } catch (const std::exception &e) {
        if (!dataDir.empty()) std::ofstream(dataDir + "/execution.json") << "{\"status\":\"FAIL\"}\n";
        std::cerr << "FAIL: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
