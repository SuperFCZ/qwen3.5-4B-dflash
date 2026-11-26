#include "capture.hpp"
#include "qwen35_dflash/sha256.hpp"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace latency {
namespace {
Sample* active = nullptr;
std::string policy, previous;
std::map<std::string, std::string> roles;
std::map<uint32_t, std::string> loaded;
std::map<uint32_t, void*> owned;
double ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b-a).count();
}
}
void check(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}
void acl_check(aclError status, const char* name) {
    if (status != ACL_SUCCESS) throw std::runtime_error(std::string(name) + " status=" + std::to_string(status));
}
void configure(const std::filesystem::path& path, const std::string& mode) {
    std::ifstream in(path);
    std::string role, file;
    while (in >> role >> std::quoted(file)) roles[std::filesystem::canonical(file).string()] = role;
    check(in.eof() && roles.size() == 5, "expected the five measured models");
    policy = mode;
}
std::string preceding_model() { return previous; }
Sample::Sample(bool enable) : marked(enable) {
#ifndef PROBE_FAKE_ACL
    if (marked) {
        for (auto& e : events) {
            const auto rc = aclrtCreateEventWithFlag(&e, ACL_EVENT_TIME_LINE);
            if (rc != ACL_SUCCESS) {
                for (auto created : events) if (created) aclrtDestroyEvent(created);
                acl_check(rc, "EVENT_UNAVAILABLE create");
            }
        }
    }
#else
    check(!marked, "EVENT_UNAVAILABLE fake ACL");
#endif
}
Sample::~Sample() {
    if (active == this) active = nullptr;
#ifndef PROBE_FAKE_ACL
    for (auto e : events) if (e) aclrtDestroyEvent(e);
#endif
}
void Sample::mark(unsigned index) {
#ifndef PROBE_FAKE_ACL
    if (marked) acl_check(aclrtRecordEvent(events.at(index), stream), "EVENT_UNAVAILABLE record");
#else
    (void)index;
#endif
}
double Sample::interval(unsigned a, unsigned b) const {
#ifndef PROBE_FAKE_ACL
    if (marked) {
        float value = 0;
        acl_check(aclrtEventElapsedTime(&value, events.at(a), events.at(b)), "EVENT_UNAVAILABLE elapsed");
        check(std::isfinite(value) && value >= 0, "invalid event interval");
        return value;
    }
#else
    (void)a;
    (void)b;
#endif
    return -1;
}
void Sample::begin() {
    check(!active && !executes, "unexpected nested/reused sample");
    active = this;
}
void Sample::finish(bool copies, const std::string& expected) {
    active = nullptr;
    check(executes == 1 && syncs == 1 && loaded.at(model) == expected, "wrong measured model or calls");
    check(ncopy == (copies ? 5U : 0U), "different copy count");
    const std::array<size_t, 5> bytes{8, 2, 8, 2, 120};
    for (unsigned i = 0; i < ncopy; ++i)
        check(transfers[i].bytes == bytes[i] &&
              transfers[i].kind == (i < 4 ? ACL_MEMCPY_HOST_TO_DEVICE : ACL_MEMCPY_DEVICE_TO_HOST),
              "different scalar/Top1 copy ABI");
    wall = ms(start, end);
}
void Sample::repeat(const Sample& first, bool copies) {
    auto copy = [&](unsigned i) {
        const auto& v = first.transfers[i];
        acl_check(ObserveCopy(v.dst, v.capacity, v.src, v.bytes, v.kind, first.stream), "repeat copy");
    };
    begin();
    if (copies) for (unsigned i = 0; i < 4; ++i) copy(i);
    acl_check(ObserveExecute(first.model, first.input, first.output, first.stream), "repeat execute");
    if (copies) copy(4);
    acl_check(ObserveSync(first.stream), "repeat sync");
    finish(copies, loaded.at(first.model));
}

std::map<std::string, std::string> Sample::snapshot(bool inputs, const std::filesystem::path& folder) const {
    std::vector<std::string> names = inputs
        ? std::vector<std::string>{"features","start_position","valid_rows","anchor","proposal_count"}
        : std::vector<std::string>{"draft_top1"};
    for (int i = 0; i < 5; ++i)
        for (auto kind : {"_key","_value"}) names.push_back("d" + std::to_string(i) + kind);
    const auto* ds = inputs ? input : output;
    check(aclmdlGetDatasetNumBuffers(ds) == names.size(), "unexpected static Draft ABI");
    if (!folder.empty()) std::filesystem::create_directories(folder);
    std::map<std::string, std::string> result;
    for (size_t i = 0; i < names.size(); ++i) {
        auto* data = aclmdlGetDatasetBuffer(ds, i);
        check(data && aclGetDataBufferAddr(data), "missing bound device buffer");
        std::string bytes(aclGetDataBufferSizeV2(data), '\0');
        check(!bytes.empty(), "empty device buffer");
        acl_check(aclrtMemcpy(bytes.data(), bytes.size(), aclGetDataBufferAddr(data), bytes.size(),
                              ACL_MEMCPY_DEVICE_TO_HOST), "snapshot after synchronization");
        result[names[i]] = qwen35::dflash::Sha256(bytes);
        if (!folder.empty()) {
            std::ofstream file(folder / (names[i] + ".bin"), std::ios::binary);
            file.exceptions(std::ios::badbit | std::ios::failbit);
            file.write(bytes.data(), bytes.size());
        }
    }
    return result;
}
}  // namespace latency

using namespace latency;
aclError ObserveCopy(void* dst, size_t cap, const void* src, size_t bytes, aclrtMemcpyKind kind, aclrtStream stream) {
    if (!active) return aclrtMemcpyAsync(dst, cap, src, bytes, kind, stream);
    auto& s = *active;
    if (!s.ncopy) {
        s.start = Clock::now();
        s.stream = stream;
        s.mark(0);
    }
    check(s.stream == stream && s.ncopy < s.transfers.size(), "unexpected copy stream/count");
    s.transfers[s.ncopy++] = {dst, cap, src, bytes, kind};
    auto before = Clock::now();
    auto rc = aclrtMemcpyAsync(dst, cap, src, bytes, kind, stream);
    s.host_ms[kind == ACL_MEMCPY_HOST_TO_DEVICE ? 0 : 2] += ms(before, Clock::now());
    return rc;
}
aclError ObserveExecute(uint32_t model, const aclmdlDataset* in, aclmdlDataset* out, aclrtStream stream) {
    previous = loaded.at(model);
    if (!active) return aclmdlExecuteAsync(model, in, out, stream);
    auto& s = *active;
    if (!s.ncopy) s.start = Clock::now();
    else check(s.stream == stream, "wrong execute stream");
    s.stream = stream;
    s.model = model;
    s.input = in;
    s.output = out;
    ++s.executes;
    s.mark(1);
    auto before = Clock::now();
    auto rc = aclmdlExecuteAsync(model, in, out, stream);
    s.host_ms[1] = ms(before, Clock::now());
    s.mark(2);
    return rc;
}
aclError ObserveSync(aclrtStream stream) {
    if (!active) return aclrtSynchronizeStream(stream);
    auto& s = *active;
    check(s.stream == stream, "wrong sync stream");
    if (s.ncopy) s.mark(3);
    auto before = Clock::now();
    auto rc = aclrtSynchronizeStream(stream);
    s.end = Clock::now();
    s.host_ms[3] = ms(before, s.end);
    ++s.syncs;
    return rc;
}
aclError ObserveMalloc(void** ptr, size_t bytes, aclrtMemMallocPolicy original) {
    return aclrtMalloc(ptr, bytes, policy == "huge-io-work" && original == ACL_MEM_MALLOC_NORMAL_ONLY
                       ? ACL_MEM_MALLOC_HUGE_FIRST : original);
}
aclError ObserveLoad(const char* path, uint32_t* id, void* work, size_t work_bytes, void* weight, size_t weight_bytes) {
    const auto& role = roles.at(std::filesystem::canonical(path).string());
    void* allocation = nullptr;
    if ((role == "c16" || role == "c64") && (policy == "draft-weights" || policy == "draft-work")) {
        size_t w = 0, k = 0;
        check(work && !weight && !weight_bytes, "expected shared workspace and GE-owned weights");
        auto rc = aclmdlQuerySize(path, &w, &k);
        if (rc != ACL_SUCCESS) return rc;
        if (policy == "draft-weights") {
            rc = aclrtMalloc(&allocation, k, ACL_MEM_MALLOC_HUGE_FIRST);
            weight = allocation;
            weight_bytes = k;
        } else {
            rc = aclrtMalloc(&allocation, w, ACL_MEM_MALLOC_NORMAL_ONLY);
            work = allocation;
            work_bytes = w;
        }
        if (rc != ACL_SUCCESS) return rc;
    }
    const auto rc = aclmdlLoadFromFileWithMem(path, id, work, work_bytes, weight, weight_bytes);
    if (rc == ACL_SUCCESS) {
        loaded[*id] = role;
        if (allocation) owned[*id] = allocation;
    } else if (allocation) aclrtFree(allocation);
    return rc;
}
aclError ObserveUnload(uint32_t id) {
    const auto rc = aclmdlUnload(id);
    if (rc != ACL_SUCCESS) return rc;
    loaded.erase(id);
    const auto it = owned.find(id);
    if (it == owned.end()) return rc;
    auto* allocation = it->second;
    owned.erase(it);
    return aclrtFree(allocation);
}
