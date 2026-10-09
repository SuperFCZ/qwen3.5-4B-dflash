// Minimal CPU model for testing the production kernel's indexing/control flow.
// NOT CANN headers, an Ascend compiler, or evidence of device execution.
#pragma once
#include <cassert>
#include <array>
#include <cstdint>
#include <cstring>
#include <deque>
#include <map>
#include <stdexcept>
#include <type_traits>
#include <vector>

#define __aicore__
#define __global__
#define __gm__
using half = _Float16;
using GM_ADDR = void *;
constexpr int PIPE_ALL = 0;
constexpr int PIPE_V = 1;

struct CpuCubeTiling {
    uint32_t orgN, orgKa, singleM, singleN, singleK;
};
struct CpuTiling {
    uint32_t globalM, globalK, globalN, tileK;
    uint64_t matmulUbBytes;
    CpuCubeTiling cubeTilingData;
};
struct CpuMetrics {
    uint64_t ubBytes = 0;
    uint32_t iterations = 0, partials = 0, outputs = 0, stores = 0, ends = 0;
    uint32_t castCalls = 0, mulsCalls = 0, vectorBarriers = 0;
    uint32_t brcbCalls = 0, mulCalls = 0;
    uint32_t copies = 0, castsWithPendingDma = 0, eventAllocations = 0, libraryWithPendingSignals = 0;
    uint32_t aStageCalls = 0;
    uint64_t aStageBytes = 0;
};
inline CpuMetrics cpuMetrics;
inline uint32_t cpuBlockIdx = 0, cpuBlockNum = 1;
inline half *cpuOutputBase = nullptr;
inline std::vector<uint32_t> cpuOutputWrites;
inline std::vector<int64_t> cpuOutputOwners;
#define GET_TILING_DATA(name, addr) const auto &name = *static_cast<const CpuTiling *>(addr)
inline void *GetSysWorkSpacePtr() { return nullptr; }

namespace AscendC {
template <bool C, typename T, typename F> using Conditional = std::conditional<C, T, F>;
struct DataCopyParams { uint16_t blockCount, blockLen, srcStride, dstStride; };
inline int64_t GetBlockIdx() { return cpuBlockIdx; }
inline int64_t GetBlockNum() { return cpuBlockNum; }
enum class TPosition { GM, VECIN, VECOUT, VECCALC };
enum class RoundMode { CAST_NONE };
enum class HardEvent { MTE2_S, MTE2_V, V_S };
using TEventID = int;
struct UnaryRepeatParams {
    uint16_t dstBlkStride, srcBlkStride;
    uint8_t dstRepStride, srcRepStride;
};
struct BrcbRepeatParams { uint16_t dstBlkStride, dstRepStride; };
struct BinaryRepeatParams { uint8_t dstBlkStride, src0BlkStride, src1BlkStride, dstRepStride, src0RepStride, src1RepStride; };

// Deliberately defer MTE2 writes. This is a dependency/lifetime checker, not
// a model of Ascend scheduling, cache coherence, bandwidth or execution time.
inline void Require(bool value, const char *message) {
    if (!value) throw std::logic_error(message);
}
struct Range {
    uintptr_t begin;
    size_t bytes;
};
inline Range Bytes(const void *p, size_t bytes) { return {reinterpret_cast<uintptr_t>(p), bytes}; }
inline bool Overlaps(Range a, Range b) { return a.begin < b.begin + b.bytes && b.begin < a.begin + a.bytes; }
struct Dma { Range target; const void *source; uint64_t sequence; };
struct Event { bool reserved = false, signalled = false; uint64_t fence = 0; };
struct SyncModel {
    uint64_t dmaIssued = 0, dmaDone = 0, scalarReady = 0, vectorReady = 0;
    uint64_t vectorIssued = 0, vectorScalarReady = 0;
    std::deque<Dma> pending;
    std::map<uintptr_t, Dma> writes;
    std::vector<Range> vectorAccesses, cubeReads;
    std::array<std::array<Event, 8>, 3> events{};
};
inline SyncModel cpuSync;

inline void DrainDma(uint64_t fence) {
    while (!cpuSync.pending.empty() && cpuSync.pending.front().sequence <= fence) {
        const auto task = cpuSync.pending.front();
        std::memcpy(reinterpret_cast<void *>(task.target.begin), task.source, task.target.bytes);
        cpuSync.dmaDone = task.sequence;
        cpuSync.pending.pop_front();
    }
}
inline void CheckReadable(Range range, bool scalar) {
    auto it = cpuSync.writes.lower_bound(range.begin);
    if (it != cpuSync.writes.begin()) --it;
    for (; it != cpuSync.writes.end() && it->first < range.begin + range.bytes; ++it) {
        if (!Overlaps(range, it->second.target)) continue;
        const auto seq = it->second.sequence;
        Require(seq <= cpuSync.dmaDone, "read before DMA completion");
        Require(seq <= (scalar ? cpuSync.scalarReady : cpuSync.vectorReady), "missing MTE2 consumer dependency");
    }
}
inline void VectorAccess(Range dst, Range src) {
    CheckReadable(src, false);
    for (auto range : cpuSync.cubeReads) Require(!Overlaps(dst, range), "W reused while Cube consumer is live");
    cpuSync.vectorAccesses.push_back(src);
    cpuSync.vectorAccesses.push_back(dst);
    ++cpuSync.vectorIssued;
}
inline void BeforeMatmul(Range weight) {
    Require(cpuSync.vectorScalarReady == cpuSync.vectorIssued, "Matmul issued before vector writes complete");
    cpuSync.cubeReads.push_back(weight);
}
template <HardEvent E> void SetFlag(int id) {
    auto &event = cpuSync.events.at(static_cast<size_t>(E)).at(id);
    Require(!event.signalled, "event ID collision / signal not consumed");
    event.signalled = true;
    event.fence = E == HardEvent::V_S ? cpuSync.vectorIssued : cpuSync.dmaIssued;
}
template <HardEvent E> void WaitFlag(int id) {
    auto &event = cpuSync.events.at(static_cast<size_t>(E)).at(id);
    Require(event.signalled, "wait without signal");
    event.signalled = false;
    if constexpr (E == HardEvent::V_S) {
        cpuSync.vectorScalarReady = event.fence;
        cpuSync.vectorAccesses.clear();
    } else {
        DrainDma(event.fence);
        if constexpr (E == HardEvent::MTE2_S) cpuSync.scalarReady = event.fence;
        else cpuSync.vectorReady = event.fence;
    }
}

template <typename T>
struct Tensor {
    T *data = nullptr;
    uint32_t count = 0;
    void SetGlobalBuffer(T *address, uint32_t size) { data = address; count = size; }
    Tensor operator[](uint32_t offset) const {
        assert(offset < count);
        return {data + offset, count - offset};
    }
    T GetValue(uint32_t offset) const {
        assert(offset < count);
        CheckReadable(Bytes(data + offset, sizeof(T)), true);
        return data[offset];
    }
};
template <typename T> using GlobalTensor = Tensor<T>;
template <typename T> using LocalTensor = Tensor<T>;

template <TPosition position = TPosition::VECCALC>
class TBuf {
public:
    void Allocate(uint64_t bytes) {
        bytes_ = bytes;
        memory_.resize((bytes + 7) / 8);
    }
    template <typename T> LocalTensor<T> Get(uint32_t count) {
        assert(static_cast<uint64_t>(count) * sizeof(T) <= bytes_);
        return {reinterpret_cast<T *>(memory_.data()), count};
    }
private:
    uint64_t bytes_ = 0;
    std::vector<uint64_t> memory_;
};
class TPipe {
public:
    TPipe() { cpuSync = {}; }
    ~TPipe() {
        assert(cpuSync.pending.empty() && cpuSync.vectorAccesses.empty() && cpuSync.cubeReads.empty());
        for (const auto &pool : cpuSync.events)
            for (const auto &event : pool) assert(!event.reserved && !event.signalled);
    }
    int FetchEventID(HardEvent type) {
        const auto &pool = cpuSync.events[static_cast<size_t>(type)];
        for (int id = 0; id < 8; ++id) if (!pool[id].reserved) return id;
        throw std::logic_error("event pool exhausted");
    }
    template <HardEvent E> int AllocEventID() {
        const int id = FetchEventID(E);
        auto &event = cpuSync.events[static_cast<size_t>(E)][id];
        Require(!event.signalled, "allocating an outstanding unreserved event");
        event.reserved = true;
        ++cpuMetrics.eventAllocations;
        return id;
    }
    template <HardEvent E> void ReleaseEventID(int id) {
        auto &event = cpuSync.events[static_cast<size_t>(E)].at(id);
        Require(event.reserved && !event.signalled, "release of live/unreserved event");
        event.reserved = false;
    }
    template <TPosition P> void InitBuffer(TBuf<P> &buffer, uint64_t bytes) {
        assert(bytes % 32 == 0);
        buffer.Allocate(bytes);
        cpuMetrics.ubBytes += bytes;
    }
};
inline TPipe *cpuPipe = nullptr;
template <int P> void PipeBarrier() {
    cpuSync.vectorAccesses.clear();
    if constexpr (P == PIPE_V) ++cpuMetrics.vectorBarriers;
    else {
        DrainDma(cpuSync.dmaIssued);
        cpuSync.scalarReady = cpuSync.vectorReady = cpuSync.dmaIssued;
        cpuSync.vectorScalarReady = cpuSync.vectorIssued;
        cpuSync.cubeReads.clear();
    }
}
template <typename T> void DataCopy(Tensor<T> dst, Tensor<T> src, uint32_t count) {
    assert(count <= dst.count && count <= src.count);
    assert(count * sizeof(T) % 32 == 0);
    const auto target = Bytes(dst.data, count * sizeof(T));
    for (auto range : cpuSync.cubeReads) Require(!Overlaps(target, range), "A reused while Cube consumer is live");
    for (auto range : cpuSync.vectorAccesses) Require(!Overlaps(target, range), "raw bank reused before vector completion");
    for (const auto &task : cpuSync.pending) Require(!Overlaps(target, task.target), "overlapping pending DMA writes");
    Dma task{target, src.data, ++cpuSync.dmaIssued};
    cpuSync.pending.push_back(task);
    cpuSync.writes[target.begin] = task;
    ++cpuMetrics.copies;
}
template <typename T> void DataCopy(Tensor<T> dst, Tensor<T> src, const DataCopyParams &params) {
    static_assert(sizeof(T) == 2);
    assert(params.blockCount == 80 && params.blockLen == 8 && params.srcStride == 152 && params.dstStride == 0);
    const auto count = params.blockLen * 32 / sizeof(T);
    for (uint32_t row = 0; row < params.blockCount; ++row)
        DataCopy(dst[row * (params.blockLen + params.dstStride) * 32 / sizeof(T)],
                 src[row * (params.blockLen + params.srcStride) * 32 / sizeof(T)], count);
    cpuMetrics.copies -= params.blockCount; // raw q/scale counter stays separate
    ++cpuMetrics.aStageCalls;
    cpuMetrics.aStageBytes += params.blockCount * params.blockLen * 32;
}
template <typename D, typename S> void Cast(Tensor<D> dst, Tensor<S> src, RoundMode, uint32_t count) {
    ++cpuMetrics.castCalls;
    assert(count <= dst.count && count <= src.count);
    VectorAccess(Bytes(dst.data, count * sizeof(D)), Bytes(src.data, count * sizeof(S)));
    for (uint32_t i = 0; i < count; ++i) dst.data[i] = static_cast<D>(src.data[i]);
}
template <typename D, typename S>
void Cast(Tensor<D> dst, Tensor<S> src, RoundMode, uint64_t mask, uint8_t repeats, const UnaryRepeatParams &params) {
    ++cpuMetrics.castCalls;
    if (!cpuSync.pending.empty()) ++cpuMetrics.castsWithPendingDma;
    static_assert(sizeof(D) == 2 && sizeof(S) == 1);
    assert(mask == 32 && repeats == 64 && params.dstBlkStride == 1 && params.srcBlkStride == 1);
    // Repeat strides use 32-byte units, independently of element widths.
    // This overload models only the contiguous-in-repeat pattern used here.
    for (uint32_t repeat = 0; repeat < repeats; ++repeat) {
        VectorAccess(Bytes(dst.data + repeat * params.dstRepStride * 32 / sizeof(D), mask * sizeof(D)),
                     Bytes(src.data + repeat * params.srcRepStride * 32 / sizeof(S), mask * sizeof(S)));
        for (uint32_t lane = 0; lane < mask; ++lane) {
            const auto d = repeat * params.dstRepStride * 32 / sizeof(D) + lane;
            const auto s = repeat * params.srcRepStride * 32 / sizeof(S) + lane;
            assert(d < dst.count && s < src.count);
            dst.data[d] = static_cast<D>(src.data[s]);
        }
    }
}
inline void Muls(Tensor<half> dst, Tensor<half> src, half scale, uint32_t count) {
    ++cpuMetrics.mulsCalls;
    assert(count <= dst.count && count <= src.count);
    VectorAccess(Bytes(dst.data, count * sizeof(half)), Bytes(src.data, count * sizeof(half)));
    for (uint32_t i = 0; i < count; ++i) {
        dst.data[i] = static_cast<half>(static_cast<float>(src.data[i]) * static_cast<float>(scale));
    }
}
inline void Duplicate(Tensor<half> dst, half value, uint32_t count) {
    assert(count <= dst.count);
    VectorAccess(Bytes(dst.data,count*2),Bytes(dst.data,0));
    for(uint32_t i=0;i<count;++i) dst.data[i]=value;
}
inline void Brcb(Tensor<half> dst, Tensor<half> src, uint8_t repeats, BrcbRepeatParams params) {
    ++cpuMetrics.brcbCalls;
    assert(repeats==8 && params.dstBlkStride==1 && params.dstRepStride==8);
    // Include dav_m200's read/restore tail in the lifetime/bounds model.
    assert(dst.count>=1152 && src.count>=64);
    VectorAccess(Bytes(dst.data,2304),Bytes(src.data,128));
    for(uint32_t row=0;row<64;++row)
        for(uint32_t lane=0;lane<16;++lane) dst.data[row*16+lane]=src.data[row];
    // The SDK's internal 24 PIPE_V barriers are not simulated/counted here.
}
inline void Mul(Tensor<half> dst, Tensor<half> x, Tensor<half> scale,
                uint64_t mask, uint8_t repeats, BinaryRepeatParams p) {
    ++cpuMetrics.mulCalls;
    assert(mask==128 && repeats==64 && p.dstBlkStride==1 && p.src0BlkStride==1 && p.src1BlkStride==0);
    for(uint32_t row=0;row<repeats;++row) {
        const auto d=row*p.dstRepStride*16, a=row*p.src0RepStride*16, b=row*p.src1RepStride*16;
        assert(d+mask<=dst.count && a+mask<=x.count && b+16<=scale.count);
        VectorAccess(Bytes(dst.data+d,mask*2),Bytes(x.data+a,mask*2));
        CheckReadable(Bytes(scale.data+b,32),false);
        for(uint32_t lane=0;lane<mask;++lane)
            dst.data[d+lane]=static_cast<half>(static_cast<float>(x.data[a+lane])*static_cast<float>(scale.data[b+lane%16]));
    }
}
}  // namespace AscendC
