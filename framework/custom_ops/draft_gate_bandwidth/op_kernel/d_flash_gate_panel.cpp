#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "panel_spec.h"
#include "panel_vector.h"

using namespace AscendC;
using namespace matmul;
namespace {
using namespace gate_panel;
using A = MatmulType<TPosition::GM, CubeFormat::ND, half>;
using B = MatmulType<TPosition::VECOUT, CubeFormat::NZ, half, false>;
using C = MatmulType<TPosition::GM, CubeFormat::ND, half>;
using Bias = MatmulType<TPosition::GM, CubeFormat::ND, float>;

template<uint32_t Slabs>
__aicore__ inline void ScaleSlabs(LocalTensor<half> w, LocalTensor<half> s, uint32_t firstSlab)
{
    // One 32-byte scale block is reused in every block/repeat of this Group128.
    // Consecutive instructions write disjoint [128,16] slices; s is read-only.
    const BinaryRepeatParams broadcast16{1, 1, 0, 8, 8, 0};
    for (uint32_t slab = firstSlab; slab < firstSlab + Slabs; ++slab) {
        for (uint32_t group = 0; group < Groups; ++group) {
            auto part = w[(slab * PanelK + group * Group) * 16];
            Mul(part, part, s[group * Columns + slab * 16], uint64_t(128), uint8_t(16), broadcast16);
        }
    }
}

__aicore__ inline void LoadPanel(LocalTensor<int8_t> q, LocalTensor<half> s,
    GlobalTensor<int8_t> codes, GlobalTensor<half> scales, uint32_t index)
{
    DataCopy(q, codes[index * Columns * PanelK], Columns * PanelK);
    DataCopy(s, scales[index * Groups * Columns], Groups * Columns);
}

// K1280: keep a complete W16 panel but only 32 channels of raw codes. After the
// first Cast, Vector no longer reads q. MTE2 may refill q while ScaleSlabs uses w.
__aicore__ inline void PrepareSplit(TPipe& pipe, LocalTensor<int8_t> q, LocalTensor<half> s,
    LocalTensor<half> w, GlobalTensor<int8_t> codes, GlobalTensor<half> scales, uint32_t index)
{
    constexpr uint32_t halfPanel = Columns / 2 * PanelK;
    const uint32_t offset = index * Columns * PanelK;
    DataCopy(q, codes[offset], halfPanel);
    DataCopy(s, scales[index * Groups * Columns], Groups * Columns);
    Handoff<HardEvent::MTE2_V>(pipe);
    CastCodes<halfPanel>(w, q);
    Handoff<HardEvent::V_MTE2>(pipe); // q source read complete before overwriting it
    DataCopy(q, codes[offset + halfPanel], halfPanel);
    PipeBarrier<PIPE_V>(); // first-half Mul consumes the Cast result
    ScaleSlabs<2>(w, s, 0);
    Handoff<HardEvent::MTE2_V>(pipe); // second-half Cast consumes the refill
    CastCodes<halfPanel>(w[halfPanel], q);
    PipeBarrier<PIPE_V>();
    ScaleSlabs<2>(w, s, 2);
}
}  // namespace

extern "C" __global__ __aicore__ void d_flash_gate_panel(GM_ADDR x, GM_ADDR w_tiles,
    GM_ADDR s, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    using namespace gate_panel;
    GET_TILING_DATA(td, tiling);
    const uint32_t block = GetBlockIdx(), blocks = GetBlockNum();
    if (!blocks || block >= Tiles || td.globalM != M || td.globalK != K ||
        td.globalN != N || td.tileK != PanelK) return;

    TPipe pipe;
    Matmul<A, B, C, Bias> mm;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &td.cubeTilingData);
    TBuf<TPosition::VECIN> qbuf, sbuf;
    TBuf<TPosition::VECOUT> wbuf;
    TBuf<> scratch;
    pipe.InitBuffer(qbuf, CodeBytes);
    pipe.InitBuffer(sbuf, ScaleBytes);
    pipe.InitBuffer(wbuf, WeightBytes);
    pipe.InitBuffer(scratch, td.matmulUbBytes);
    mm.SetLocalWorkspace(scratch.Get<uint8_t>(td.matmulUbBytes));
    auto q = qbuf.Get<int8_t>(CodeBytes);
    auto scales = sbuf.Get<half>(ScaleBytes / 2);
    auto w = wbuf.Get<half>(WeightBytes / 2);
    GlobalTensor<int8_t> qgm;
    GlobalTensor<half> sgm, xgm, ygm;
    qgm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(w_tiles), N * K);
    sgm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(s), N * K / Group);
    xgm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(x), M * K);
    ygm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(y), M * N);

    uint32_t bank = 0;
    if constexpr (!SplitRaw) {
        LoadPanel(q, scales, qgm, sgm, block * Panels);
        PipeBarrier<PIPE_ALL>(); // only the first panel lacks a preceding retirement fence
    }
    for (uint32_t nt = block; nt < Tiles; nt += blocks) {
        for (uint32_t kp = 0; kp < Panels; ++kp) {
            const uint32_t index = nt * Panels + kp;
            if constexpr (SplitRaw) {
                PrepareSplit(pipe, q, scales, w, qgm, sgm, index);
            } else {
                const bool sameN = kp + 1 < Panels;
                if (sameN || nt + blocks < Tiles) {
                    const uint32_t next = sameN ? index + 1 : (nt + blocks) * Panels;
                    LoadPanel(q[(1 - bank) * RawElements], scales[(1 - bank) * Groups * Columns],
                              qgm, sgm, next);
                }
                CastCodes<Columns * PanelK>(w, q[bank * RawElements]);
                PipeBarrier<PIPE_V>();
                ScaleSlabs<4>(w, scales[bank * Groups * Columns], 0);
                bank ^= 1;
            }
            Handoff<HardEvent::V_S>(pipe); // publish all W16 writes before issuing its SDK consumer
            mm.SetTensorA(xgm[kp * PanelK]);
            mm.SetTensorB(w, false); // physical FP16 NZ [N/16,K,16], logical B[K,N]
            if (!mm.Iterate(kp != 0)) {
                PipeBarrier<PIPE_ALL>();
                mm.End();
                return;
            }
            // Protect W16, SDK scratch, L0C and raw-bank reuse; also completes any lookahead DMA.
            PipeBarrier<PIPE_ALL>();
        }
        mm.GetTensorC(ygm[nt * Columns]);
        PipeBarrier<PIPE_ALL>();
    }
    mm.End();
}
