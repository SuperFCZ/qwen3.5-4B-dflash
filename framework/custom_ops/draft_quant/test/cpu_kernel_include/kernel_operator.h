// Minimal CPU model for testing the production kernel's indexing/control flow.
// NOT CANN headers, an Ascend compiler, or evidence of device execution.
#pragma once
#include <cassert>
#include <cstdint>
#include <cstring>
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
    uint32_t globalK, globalN, tileK;
    uint64_t matmulUbBytes;
    CpuCubeTiling cubeTilingData;
};
struct CpuMetrics {
    uint64_t ubBytes = 0;
    uint32_t iterations = 0, partials = 0, outputs = 0, stores = 0, ends = 0;
    uint32_t castCalls = 0, mulsCalls = 0, vectorBarriers = 0;
};
inline CpuMetrics cpuMetrics;
inline uint32_t cpuBlockIdx = 0, cpuBlockNum = 1;
inline half *cpuOutputBase = nullptr;
inline std::vector<uint32_t> cpuOutputWrites;
inline std::vector<int64_t> cpuOutputOwners;
#define GET_TILING_DATA(name, addr) const auto &name = *static_cast<const CpuTiling *>(addr)
inline void *GetSysWorkSpacePtr() { return nullptr; }

namespace AscendC {
inline int64_t GetBlockIdx() { return cpuBlockIdx; }
inline int64_t GetBlockNum() { return cpuBlockNum; }
enum class TPosition { GM, VECIN, VECOUT, VECCALC };
enum class RoundMode { CAST_NONE };
enum class HardEvent { MTE2_S, MTE2_V };
struct UnaryRepeatParams {
    uint16_t dstBlkStride, srcBlkStride;
    uint8_t dstRepStride, srcRepStride;
};
template <HardEvent E> void SetFlag(int) {}
template <HardEvent E> void WaitFlag(int) {}

template <typename T>
struct Tensor {
    T *data = nullptr;
    uint32_t count = 0;
    void SetGlobalBuffer(T *address, uint32_t size) { data = address; count = size; }
    Tensor operator[](uint32_t offset) const {
        assert(offset < count);
        return {data + offset, count - offset};
    }
    T GetValue(uint32_t offset) const { assert(offset < count); return data[offset]; }
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
    int FetchEventID(HardEvent) { return 0; }
    template <TPosition P> void InitBuffer(TBuf<P> &buffer, uint64_t bytes) {
        assert(bytes % 32 == 0);
        buffer.Allocate(bytes);
        cpuMetrics.ubBytes += bytes;
    }
};
template <int P> void PipeBarrier() { if (P == PIPE_V) ++cpuMetrics.vectorBarriers; }
template <typename T> void DataCopy(Tensor<T> dst, Tensor<T> src, uint32_t count) {
    assert(count <= dst.count && count <= src.count);
    assert(count * sizeof(T) % 32 == 0);
    std::memcpy(dst.data, src.data, count * sizeof(T));
}
template <typename D, typename S> void Cast(Tensor<D> dst, Tensor<S> src, RoundMode, uint32_t count) {
    ++cpuMetrics.castCalls;
    assert(count <= dst.count && count <= src.count);
    for (uint32_t i = 0; i < count; ++i) dst.data[i] = static_cast<D>(src.data[i]);
}
template <typename D, typename S>
void Cast(Tensor<D> dst, Tensor<S> src, RoundMode, uint64_t mask, uint8_t repeats, const UnaryRepeatParams &params) {
    ++cpuMetrics.castCalls;
    static_assert(sizeof(D) == 2 && sizeof(S) == 1);
    assert(mask == 32 && repeats == 64 && params.dstBlkStride == 1 && params.srcBlkStride == 1);
    // Repeat strides use 32-byte units, independently of element widths.
    // This overload models only the contiguous-in-repeat pattern used here.
    for (uint32_t repeat = 0; repeat < repeats; ++repeat) {
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
    for (uint32_t i = 0; i < count; ++i) {
        dst.data[i] = static_cast<half>(static_cast<float>(src.data[i]) * static_cast<float>(scale));
    }
}
}  // namespace AscendC
