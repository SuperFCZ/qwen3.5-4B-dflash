#pragma once
#include <acl/acl.h>
#include <array>
#include <chrono>
#include <filesystem>
#include <string>
#include <vector>
#include <map>

namespace latency {
using Clock = std::chrono::steady_clock;
void check(bool ok, const std::string& message);
void acl_check(aclError status, const char* operation);
void configure(const std::filesystem::path& roles, const std::string& policy);
std::string preceding_model();

struct Transfer {
    void* dst;
    size_t capacity;
    const void* src;
    size_t bytes;
    aclrtMemcpyKind kind;
};
struct Sample {
    explicit Sample(bool mark);
    ~Sample();
    Sample(const Sample&) = delete;
    Sample& operator=(const Sample&) = delete;
    void begin();
    void finish(bool copies, const std::string& expected_role);
    void repeat(const Sample& first, bool copies);
    std::map<std::string, std::string> snapshot(bool inputs, const std::filesystem::path& folder = {}) const;
    void mark(unsigned index);
    double interval(unsigned a, unsigned b) const;
    bool marked;
    uint32_t model{};
    aclrtStream stream{};
    const aclmdlDataset* input{};
    aclmdlDataset* output{};
    std::array<Transfer, 5> transfers{};
    unsigned ncopy{}, executes{}, syncs{};
    Clock::time_point start{}, end{};
    std::array<double, 4> host_ms{}; // synchronous host occupation of each API family
    double wall{}, native_call{};
#ifndef PROBE_FAKE_ACL
    std::array<aclrtEvent, 4> events{};
#endif
};
}  // namespace latency

extern "C" {
decltype(aclrtMemcpyAsync) ObserveCopy;
decltype(aclmdlExecuteAsync) ObserveExecute;
decltype(aclrtSynchronizeStream) ObserveSync;
decltype(aclrtMalloc) ObserveMalloc;
decltype(aclmdlLoadFromFileWithMem) ObserveLoad;
decltype(aclmdlUnload) ObserveUnload;
}
