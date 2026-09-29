// Shared guarded AscendCL buffers for isolated custom-op/OM diagnostics.
#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>
#include "acl/acl.h"

namespace draft_quant_test {
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


using Clock = std::chrono::steady_clock;
inline double Milliseconds(Clock::time_point begin, Clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

struct Timing {
    int warmup = 0, repetitions = 0;
    std::vector<double> execute, prepare;
    void Write(std::ostream &out) const
    {
        out << std::setprecision(9) << ",\"timing\":{\"status\":\""
            << (repetitions ? "MEASURED" : "NOT_RUN")
            << "\",\"scope\":\"isolated host launch plus stream synchronization; transfers/checks excluded\","
               "\"warmup\":" << warmup << ",\"repetitions\":" << repetitions;
        auto series = [&](const char *name, const std::vector<double> &values) {
            out << ",\"" << name << "\":{\"samples_ms\":[";
            for (size_t i = 0; i < values.size(); ++i) out << (i ? "," : "") << values[i];
            out << "]";
            if (!values.empty()) {
                auto sorted = values;
                std::sort(sorted.begin(), sorted.end());
                const auto count = sorted.size();
                out << ",\"median_ms\":" << (sorted[(count - 1) / 2] + sorted[count / 2]) / 2
                    << ",\"p95_ms\":" << sorted[static_cast<size_t>(std::ceil(0.95 * count)) - 1];
            }
            out << "}";
        };
        series("execute_sync", execute);
        series("prepare", prepare);
        out << "},\"process_peak_device_memory\":\"NOT_MEASURED\"";
    }
};
}  // namespace draft_quant_test
