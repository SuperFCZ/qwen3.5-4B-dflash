#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "d_flash_group_quant_linear_build_config.h"

static_assert(DFLASH_GROUP_QUANT_DEQUANT_MODE <= 1U, "unsupported dequantization mode");

using namespace matmul;

namespace {
constexpr uint32_t kM = 16;
constexpr uint32_t kTileN = 64;
constexpr uint32_t kGroup = 128;
constexpr uint32_t kNzK0 = 32;
constexpr uint32_t kNzN0 = 16;
}  // namespace

using AType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>;
using BType = MatmulType<AscendC::TPosition::VECOUT, CubeFormat::ND, half, true>;
using CType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>;
using BiasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;

extern "C" __global__ __aicore__ void d_flash_group_quant_linear(
    GM_ADDR x, GM_ADDR w_nz, GM_ADDR s, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    const uint32_t globalK = tilingData.globalK;
    const uint32_t globalN = tilingData.globalN;
    const uint32_t tileK = tilingData.tileK;
    const uint32_t nTiles = globalN / kTileN;
    const uint32_t blockIdx = static_cast<uint32_t>(AscendC::GetBlockIdx());
    const uint32_t blockNum = static_cast<uint32_t>(AscendC::GetBlockNum());
    if (blockNum == 0 || blockIdx >= blockNum || blockIdx >= nTiles) return;
    AscendC::TPipe pipe;
    Matmul<AType, BType, CType, BiasType> mm;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tilingData.cubeTilingData);

    AscendC::TBuf<AscendC::TPosition::VECIN> codesBuf, scaleBuf;
    AscendC::TBuf<AscendC::TPosition::VECOUT> weightBuf;
    AscendC::TBuf<> matmulUb;
    pipe.InitBuffer(codesBuf, kTileN * tileK * sizeof(int8_t));
    pipe.InitBuffer(scaleBuf, (tileK / kGroup) * kTileN * sizeof(half));
    pipe.InitBuffer(weightBuf, kTileN * tileK * sizeof(half));
    pipe.InitBuffer(matmulUb, tilingData.matmulUbBytes);
    mm.SetLocalWorkspace(matmulUb.Get<uint8_t>(tilingData.matmulUbBytes));

    AscendC::GlobalTensor<half> xGm, sGm, yGm;
    AscendC::GlobalTensor<int8_t> qGm;
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(x), kM * globalK);
    qGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(w_nz), globalN * globalK);
    sGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(s), (globalK / kGroup) * globalN);
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(y), kM * globalN);
    auto qLocal = codesBuf.Get<int8_t>(kTileN * tileK);
    auto sLocal = scaleBuf.Get<half>((tileK / kGroup) * kTileN);
    auto wLocal = weightBuf.Get<half>(kTileN * tileK);

    // Core b owns tiles b, b + blockNum, ... . Every output element has one
    // owner, and that owner completes the unchanged full-K reduction. UB/L0C
    // and the Matmul instance are private to the core; there is no split-K.
    for (uint32_t nTile = blockIdx; nTile < nTiles; nTile += blockNum) {
        const uint32_t nBegin = nTile * kTileN;
        for (uint32_t kBegin = 0; kBegin < globalK; kBegin += tileK) {
            // NZ's K32 planes have globalN/16 channel blocks. A 64-column
            // slice is contiguous within a plane, but successive planes are not.
            for (uint32_t k1 = 0; k1 < tileK / kNzK0; ++k1) {
                const uint32_t src = ((kBegin / kNzK0 + k1) * (globalN / kNzN0) +
                                       nBegin / kNzN0) * kNzN0 * kNzK0;
                AscendC::DataCopy(qLocal[k1 * kTileN * kNzK0], qGm[src], kTileN * kNzK0);
            }
            for (uint32_t group = 0; group < tileK / kGroup; ++group) {
                AscendC::DataCopy(sLocal[group * kTileN],
                    sGm[(kBegin / kGroup + group) * globalN + nBegin], kTileN);
            }
            // Scalar GetValue(scale) and vector Cast(q) are separate consumers
            // of MTE2. Make both cross-pipeline dependencies explicit on reuse.
            const auto scalarReady = pipe.FetchEventID(AscendC::HardEvent::MTE2_S);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(scalarReady);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(scalarReady);
            const auto vectorReady = pipe.FetchEventID(AscendC::HardEvent::MTE2_V);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(vectorReady);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(vectorReady);
            AscendC::PipeBarrier<PIPE_ALL>();
#if DFLASH_GROUP_QUANT_DEQUANT_MODE == 1
            // INT8 NZ tile: [tileK/32, 64, 32]. Each repeat casts one N row's
            // K32 slice into its final ND [64,tileK] location. Repeat strides
            // are in 32-byte units: source=32B, destination=tileK*sizeof(half).
            const AscendC::UnaryRepeatParams castParams{
                1, 1, static_cast<uint8_t>(tileK * sizeof(half) / 32), 1};
            for (uint32_t k1 = 0; k1 < tileK / kNzK0; ++k1) {
                AscendC::Cast(wLocal[k1 * kNzK0], qLocal[k1 * kTileN * kNzK0],
                    AscendC::RoundMode::CAST_NONE, static_cast<uint64_t>(kNzK0),
                    static_cast<uint8_t>(kTileN), castParams);
            }
            // The Casts write disjoint slices; all must finish before the
            // in-place Muls reads the assembled rows. Muls writes disjoint
            // N/group slices, preserving the original FP16 multiplication.
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t group = 0; group < tileK / kGroup; ++group) {
                for (uint32_t n = 0; n < kTileN; ++n) {
                    const half scale = sLocal.GetValue(group * kTileN + n);
                    const uint32_t dst = n * tileK + group * kGroup;
                    AscendC::Muls(wLocal[dst], wLocal[dst], scale, kGroup);
                }
            }
            AscendC::PipeBarrier<PIPE_V>();
#else
            // A3 control: keep the original per-K32 instruction sequence.
            for (uint32_t group = 0; group < tileK / kGroup; ++group) {
                for (uint32_t n = 0; n < kTileN; ++n) {
                    const half scale = sLocal.GetValue(group * kTileN + n);
                    for (uint32_t part = 0; part < kGroup / kNzK0; ++part) {
                        const uint32_t k1 = group * (kGroup / kNzK0) + part;
                        const uint32_t src = ((k1 * (kTileN / kNzN0) + n / kNzN0) * kNzN0 +
                                               n % kNzN0) * kNzK0;
                        const uint32_t dst = n * tileK + k1 * kNzK0;
                        AscendC::Cast(wLocal[dst], qLocal[src], AscendC::RoundMode::CAST_NONE, kNzK0);
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Muls(wLocal[dst], wLocal[dst], scale, kNzK0);
                        AscendC::PipeBarrier<PIPE_V>();
                    }
                }
            }
#endif
            AscendC::PipeBarrier<PIPE_ALL>();
            mm.SetTensorA(xGm[kBegin]);
            mm.SetTensorB(wLocal, true);
            if (tileK == globalK) {
                // K=256 retains the verified whole-K MatMul path.
                mm.IterateAll(yGm[nBegin]);
            } else {
                // SetTensorA/B restarts the single-output-block iterator.
                // false allocates/initializes C once; true reuses that L0C
                // accumulator. No GetTensorC or End between K tiles!
                if (!mm.Iterate(kBegin != 0)) {
                    mm.End();
                    return;  // Remaining poisoned outputs make the test fail.
                }
            }
            // All consumers of q/scale/W and MatMul's workspace must finish
            // before the next tile reuses those buffers. L0C stays allocated.
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        if (tileK != globalK) {
            mm.GetTensorC(yGm[nBegin]);  // The only FP16 output rounding for this N tile.
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }
    mm.End();
}
