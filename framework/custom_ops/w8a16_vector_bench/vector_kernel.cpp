#include "kernel_operator.h"
#include "bench_shape.h"
#include "scale_expand.inc"

namespace {
using namespace AscendC;
using namespace vb;

template<bool Fma>
__aicore__ inline void Arithmetic(GM_ADDR output, uint32_t iterations) {
    TPipe pipe;
    TBuf<TPosition::VECCALC> leftBuf, rightBuf, dstBuf;
    pipe.InitBuffer(leftBuf, Elements * sizeof(half));
    pipe.InitBuffer(rightBuf, Elements * sizeof(half));
    pipe.InitBuffer(dstBuf, Elements * sizeof(half));
    auto left = leftBuf.Get<half>();
    auto right = rightBuf.Get<half>();
    auto dst = dstBuf.Get<half>();
    Duplicate(left, half(0.5f), Elements);
    Duplicate(right, half(Fma ? 0.25f : -1.0f), Elements);
    Duplicate(dst, half(Fma ? 0.0f : 0.5f), Elements);
    PipeBarrier<PIPE_V>();
    const BinaryRepeatParams dense{1, 1, 1, 8, 8, 8};
    for (uint32_t i = 0; i < iterations; ++i) {
        // 16384 independent accumulators. Reuse each only after the preceding vector op.
        if constexpr (Fma) MulAddDst(dst, left, right, uint64_t(128), uint8_t(Elements / 128), dense);
        else Mul(dst, dst, right, uint64_t(128), uint8_t(Elements / 128), dense);
        PipeBarrier<PIPE_V>();
    }
    PipeBarrier<PIPE_ALL>();
    GlobalTensor<half> out;
    out.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(output));
    DataCopy(out[GetBlockIdx() * Elements], dst, Elements);
    PipeBarrier<PIPE_ALL>();
}

template<bool Flat>
__aicore__ inline void LoadTile(LocalTensor<int8_t> q, LocalTensor<half> s,
                                GlobalTensor<int8_t> qGm, GlobalTensor<half> sGm,
                                uint32_t nt, uint32_t kt, uint32_t n, uint32_t kTiles) {
    if constexpr (Flat) {
        DataCopy(q, qGm[(nt * kTiles + kt) * Elements], Elements);
    } else {
        // Same 8 GM NZ plane slices and compact UB representation as CopyRawTile.
        for (uint32_t p = 0; p < KTile / Plane; ++p) {
            const uint32_t offset = ((kt * (KTile / Plane) + p) * n + nt * NTile) * Plane;
            DataCopy(q[p * NTile * Plane], qGm[offset], NTile * Plane);
        }
    }
    for (uint32_t g = 0; g < KTile / Group; ++g)
        DataCopy(s[g * NTile], sGm[(kt * (KTile / Group) + g) * n + nt * NTile], NTile);
}

template<bool Flat, bool Scale, bool B16>
__aicore__ inline void Convert(LocalTensor<half> w, LocalTensor<int8_t> q,
                               LocalTensor<half> scales, LocalTensor<half> expanded) {
    if constexpr (Flat) {
        // Packed [N64,K256]; one full-width Cast into the same consumer layout.
        const UnaryRepeatParams dense{1, 1, 8, 4};
        Cast(w, q, RoundMode::CAST_NONE, uint64_t(128), uint8_t(Elements / 128), dense);
    } else {
        const UnaryRepeatParams scatter{1, 1, KTile * sizeof(half) / 32, 1};
        for (uint32_t p = 0; p < KTile / Plane; ++p)
            Cast(w[p * Plane], q[p * NTile * Plane], RoundMode::CAST_NONE,
                 uint64_t(Plane), uint8_t(NTile), scatter);
    }
    PipeBarrier<PIPE_V>();
    if constexpr (Scale) {
        const BinaryRepeatParams groupMul{1, 1, 0, KTile / 16, KTile / 16, 1};
        for (uint32_t g = 0; g < KTile / Group; ++g) {
            if constexpr (B16) ExpandGateScale(expanded, scales[g * NTile]);
            else Brcb(expanded, scales[g * NTile], 8, {1, 8});
            PipeBarrier<PIPE_V>();
            Mul(w[g * Group], w[g * Group], expanded, uint64_t(Group), uint8_t(NTile), groupMul);
            PipeBarrier<PIPE_V>();
        }
        PipeBarrier<PIPE_V>();
    }
}

template<bool Flat, bool Scale, bool B16, bool Stream, bool CopyOnly = false>
__aicore__ inline void WeightPath(GM_ADDR codes, GM_ADDR scales, GM_ADDR output,
                                  uint32_t n, uint32_t kTiles, uint32_t iterations) {
    TPipe pipe;
    TBuf<TPosition::VECIN> qBuf, sBuf;
    TBuf<TPosition::VECOUT> wBuf;
    TBuf<TPosition::VECCALC> expandBuf;
    constexpr uint32_t banks = Stream ? 1 : 2;
    pipe.InitBuffer(qBuf, banks * Elements);
    pipe.InitBuffer(sBuf, banks * ScaleWords * sizeof(half));
    pipe.InitBuffer(wBuf, Elements * sizeof(half));
    pipe.InitBuffer(expandBuf, ExpandedBytes);
    auto q = qBuf.Get<int8_t>();
    auto s = sBuf.Get<half>();
    auto w = wBuf.Get<half>();
    auto expanded = expandBuf.Get<half>();
    // m200 Brcb backs up/restores the rear of the broadcast buffer.
    if constexpr (Scale && !B16) Duplicate(expanded, half(0.0f), ExpandedBytes / sizeof(half));
    GlobalTensor<int8_t> qGm, byteOut;
    GlobalTensor<half> sGm, out;
    qGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(codes));
    sGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(scales));
    out.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(output));
    byteOut.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(output));
    const uint32_t block = GetBlockIdx();
    if constexpr (!Stream) {
        for (uint32_t b = 0; b < banks; ++b)
            LoadTile<Flat>(q[b * Elements], s[b * ScaleWords], qGm, sGm, block, b, n, kTiles);
        PipeBarrier<PIPE_ALL>();
        for (uint32_t i = 0; i < iterations; ++i) {
            const uint32_t b = i & 1;
            Convert<Flat, Scale, B16>(w, q[b * Elements], s[b * ScaleWords], expanded);
            // Compiler fence only: every hot-loop result is observable to the compiler.
            asm volatile("" ::: "memory");
        }
        PipeBarrier<PIPE_ALL>();
        DataCopy(out[block * Elements], w, Elements);
    } else {
        // Serial DMA control. The completed input and compact output taps are fenced
        // before this block reuses UB. No Matmul or model-prefetch overlap in this probe.
        for (uint32_t nt = block; nt < n / NTile; nt += GetBlockNum()) {
            for (uint32_t kt = 0; kt < kTiles; ++kt) {
                const uint32_t tap = (nt * kTiles + kt) * TapBytes;
                LoadTile<Flat>(q, s, qGm, sGm, nt, kt, n, kTiles);
                PipeBarrier<PIPE_ALL>();
                if constexpr (CopyOnly) {
                    // Observe all eight 2 KiB slices and both scale transfers.
                    DataCopy(byteOut[tap], q, DataCopyParams{8, 1, 63, 0});
                    DataCopy(out[(tap + 256) / 2], s, DataCopyParams{2, 1, 3, 0});
                } else {
                    Convert<Flat, Scale, B16>(w, q, s, expanded);
                    PipeBarrier<PIPE_ALL>();
                    // First 16 FP16 words from every K32 plane in channel zero.
                    DataCopy(out[tap / 2], w, DataCopyParams{8, 1, 1, 0});
                }
                PipeBarrier<PIPE_ALL>();
            }
        }
    }
    PipeBarrier<PIPE_ALL>();
}
}  // namespace

extern "C" __global__ __aicore__ void dflash_vector_probe(
    GM_ADDR codes, GM_ADDR scales, GM_ADDR output, uint32_t mode, uint32_t n, uint32_t kTiles, uint32_t iterations) {
    using namespace vb;
    switch (mode) {
        case Fp16Fma: Arithmetic<true>(output, iterations); break;
        case Fp16Mul: Arithmetic<false>(output, iterations); break;
        case CastFlat: WeightPath<true, false, false, false>(codes, scales, output, n, kTiles, iterations); break;
        case CastNz: WeightPath<false, false, false, false>(codes, scales, output, n, kTiles, iterations); break;
        case NzBrcb: WeightPath<false, true, false, false>(codes, scales, output, n, kTiles, iterations); break;
        case NzB16: WeightPath<false, true, true, false>(codes, scales, output, n, kTiles, iterations); break;
        case FlatB16: WeightPath<true, true, true, false>(codes, scales, output, n, kTiles, iterations); break;
        case CopyNz: WeightPath<false, false, false, true, true>(codes, scales, output, n, kTiles, iterations); break;
        case CopyFlat: WeightPath<true, false, false, true, true>(codes, scales, output, n, kTiles, iterations); break;
        case StreamNzBrcb: WeightPath<false, true, false, true>(codes, scales, output, n, kTiles, iterations); break;
        case StreamNzB16: WeightPath<false, true, true, true>(codes, scales, output, n, kTiles, iterations); break;
        case StreamFlatB16: WeightPath<true, true, true, true>(codes, scales, output, n, kTiles, iterations); break;
    }
}
