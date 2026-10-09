// Shared guarded AscendCL buffers for isolated custom-op/OM diagnostics.
#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <initializer_list>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
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

    // Inspect both complete guards, even when the prefix is corrupt. Offsets
    // identify the observed changed bytes, not an inferred DMA write extent.
    bool InspectGuards(const std::string &name, std::ostream &out) const
    {
        if (allocation == nullptr) {
            out << "guard buffer=" << name << " status=UNALLOCATED\n";
            return true;
        }
        std::vector<uint8_t> before(kGuard), after(kGuard);
        Check(aclrtMemcpy(before.data(), kGuard, allocation, kGuard, ACL_MEMCPY_DEVICE_TO_HOST), "read prefix");
        Check(aclrtMemcpy(after.data(), kGuard, static_cast<uint8_t *>(data) + bytes,
                          kGuard, ACL_MEMCPY_DEVICE_TO_HOST), "read suffix");
        bool intact = true;
        auto inspect = [&](const char *side, const std::vector<uint8_t> &guard, bool prefix) {
            size_t first = kGuard, last = 0, changed = 0;
            for (size_t i = 0; i < guard.size(); ++i) {
                if (guard[i] != kCanary) {
                    if (!changed) first = i;
                    last = i; ++changed;
                }
            }
            out << "guard buffer=" << name << " side=" << side
                << " payload_bytes=" << bytes << " allocation_request_bytes=" << bytes + 2 * kGuard
                << " guard_bytes=" << kGuard << " expected=0xa5 status=" << (changed ? "CORRUPT" : "INTACT")
                << " changed_bytes=" << changed;
            if (changed) {
                intact = false;
                out << " first_changed_guard_offset=" << first << " last_changed_guard_offset=" << last
                    << " first_changed_data_offset=";
                if (prefix) out << '-' << kGuard - first;
                else out << bytes + first;
                out << " last_changed_data_offset=";
                if (prefix) out << '-' << kGuard - last;
                else out << bytes + last;
                // Use a separate stream so callers' numeric formatting survives.
                std::ostringstream value;
                value << std::hex << std::setfill('0') << std::setw(2) << unsigned(guard[first]);
                out << " first_actual=0x" << value.str();
            }
            out << '\n';
        };
        inspect("prefix", before, true);
        inspect("suffix", after, false);
        return intact;
    }

    void CheckGuards(const std::string &name = "unnamed") const
    {
        std::ostringstream detail;
        if (!InspectGuards(name, detail))
            throw std::runtime_error("device buffer guard overwritten\n" + detail.str());
    }
};

// Report every buffer before failing, so one damaged allocation cannot hide a
// second one. No caller may use a false result to continue numerical acceptance.
inline void CheckNamedGuards(
    std::initializer_list<std::pair<const char *, const GuardedBuffer *>> buffers, std::ostream &out)
{
    std::string corrupt;
    for (const auto &entry : buffers) {
        if (!entry.second->InspectGuards(entry.first, out)) {
            if (!corrupt.empty()) corrupt += ',';
            corrupt += entry.first;
        }
    }
    out.flush();
    if (!corrupt.empty()) throw std::runtime_error("device buffer guard overwritten: " + corrupt);
}


using Clock = std::chrono::steady_clock;
inline double Milliseconds(Clock::time_point begin, Clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

struct Timing {
    int warmup = 0, repetitions = 0;
    bool continuous = false, collectPrepare = true;
    std::vector<double> execute, prepare;
    void Write(std::ostream &out) const
    {
        out << std::setprecision(9) << ",\"timing\":{\"status\":\""
            << (repetitions ? "MEASURED" : "NOT_RUN")
            << "\",\"protocol\":\"" << (continuous ? "continuous-v1" : "checked-v1")
            << "\",\"scope\":\"isolated host launch plus stream synchronization; transfers/checks excluded\","
               "\"per_timed_call_readback\":" << (continuous ? "false" : "true")
            << ",\"per_timed_call_poison\":" << (continuous ? "false" : "true")
            << ",\"correctness_before_calls\":2,\"correctness_after_calls\":" << (continuous ? 1 : 0)
            << ",\"timed_tail_checked\":" << (continuous ? "true" : "false") << ","
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

// Bracket uninterrupted warmup/measurement with the same guard/input/output
// checks as correctness runs. No readback, poison or file writes occur between
// continuous samples. Both runners use this scheduler to keep boundaries equal.
template <typename Prepare, typename Execute, typename Poison, typename Verify>
void RunMeasuredCalls(Timing &timing, Prepare prepare, Execute execute, Poison poison, Verify verify)
{
    timing.prepare.reserve(timing.repetitions);
    timing.execute.reserve(timing.repetitions);
    auto call = [&](bool measured) {
        const auto begin = Clock::now();
        prepare();
        const auto prepared = Clock::now();
        execute();
        const auto completed = Clock::now();
        if (measured) {
            if (timing.collectPrepare) timing.prepare.push_back(Milliseconds(begin, prepared));
            timing.execute.push_back(Milliseconds(prepared, completed));
        }
    };
    for (int repeat = 0; repeat < 2; ++repeat) {
        poison(repeat);
        call(false);
        verify("actual-" + std::to_string(repeat) + ".bin");
    }
    for (int repeat = 0; repeat < timing.warmup + timing.repetitions; ++repeat) {
        if (!timing.continuous) poison(repeat + 2);
        call(repeat >= timing.warmup);
        if (!timing.continuous) verify("");
    }
    if (timing.continuous) {
        verify("benchmark-last.bin");
        // A fresh poisoned call after measurement also detects unwritten
        // output that might have been masked by a previous valid timed result.
        poison(1);
        call(false);
        verify("postcheck.bin");
    }
}
}  // namespace draft_quant_test
