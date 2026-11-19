// Included in the identity-checked copy of the accepted Raw kernel.
#if defined(__NPU_ARCH__) && __NPU_ARCH__ != 2002
#error W16 requires the coupled dav_2002 architecture
#endif
#if defined(SPLIT_CORE_CUBE) || defined(SPLIT_CORE_VEC) || defined(ASCENDC_MATMUL_AICORE)
#error W16 requires the direct same-core MatmulImpl, not a client or service
#endif
namespace w16 {
using namespace AscendC;
using GateMatmul = Matmul<AType, BType, CType, BiasType>;
static_assert(CFG_NORM.doNorm && !CFG_NORM.enVecND2NZ && !CFG_NORM.enableMixDualMaster,
              "unexpected Matmul configuration");

struct Operations {
    TPipe &pipe;
    GateMatmul &mm;
    GlobalTensor<half> x, scale, y;
    GlobalTensor<int8_t> q;
    LocalTensor<int8_t> codes;
    LocalTensor<half> scales, weights, expanded;
    TEventID scalarID, vectorID;

    __aicore__ inline LocalTensor<half> Weight(uint32_t k) {
        auto w = weights[(k % 2) * 16384];
        w.SetSize(16384);
        return w;
    }
    __aicore__ inline void Copy(uint32_t n, uint32_t k) {
        CopyRawTile(codes[(k % 2) * 16384], scales[(k % 2) * 128], q, scale,
                    19456, 256, n, k * 256);
        SetFlag<HardEvent::MTE2_S>(scalarID);
        SetFlag<HardEvent::MTE2_V>(vectorID);
    }
    __aicore__ inline void WaitRaw() {
        WaitFlag<HardEvent::MTE2_S>(scalarID);
        WaitFlag<HardEvent::MTE2_V>(vectorID);
    }
    __aicore__ inline void Dequant(uint32_t k) {
        auto w = Weight(k);
        auto qBank = codes[(k % 2) * 16384];
        auto sBank = scales[(k % 2) * 128];
        const UnaryRepeatParams cast{1, 1, 16, 1};
        for (uint32_t plane = 0; plane < 8; ++plane) {
            Cast(w[plane * 32], qBank[plane * 2048], RoundMode::CAST_NONE,
                 static_cast<uint64_t>(32), static_cast<uint8_t>(64), cast);
        }
        PipeBarrier<PIPE_V>();
        for (uint32_t group = 0; group < 2; ++group) {
            Brcb(expanded, sBank[group * 64], 8, {1, 8});
            PipeBarrier<PIPE_V>();
            const BinaryRepeatParams mul{1, 1, 0, 16, 16, 1};
            Mul(w[group * 128], w[group * 128], expanded,
                static_cast<uint64_t>(128), static_cast<uint8_t>(64), mul);
            PipeBarrier<PIPE_V>();
        }
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline bool Iterate(uint32_t k) {
        const auto ready = pipe.FetchEventID(HardEvent::V_S);
        if (static_cast<uint32_t>(ready) >= QUE_MAX_EVENT) return false;
        SetFlag<HardEvent::V_S>(ready);
        WaitFlag<HardEvent::V_S>(ready);
        mm.SetTensorA(x[k * 256]);
        mm.SetTensorB(Weight(k), true);
        return mm.Iterate(k != 0);
    }
    __aicore__ inline void Fence() { PipeBarrier<PIPE_ALL>(); }
    __aicore__ inline void Output(uint32_t n) { mm.GetTensorC(y[n]); }
};

template<bool Overlap, typename Tiling>
__aicore__ inline void Run(GM_ADDR x, GM_ADDR w, GM_ADDR s, GM_ADDR y, Tiling &td) {
    const uint32_t blocks = GetBlockNum(), block = GetBlockIdx();
    if (blocks != 8 || block >= blocks || td.tileK != 256 || td.matmulUbBytes != 152832) return;
    TPipe pipe;
    GateMatmul mm;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &td.cubeTilingData);
    TBuf<TPosition::VECIN> codes, scales;
    TBuf<TPosition::VECOUT> weights;
    TBuf<TPosition::VECCALC> expanded;
    TBuf<> scratch;
    if (!(pipe.InitBuffer(codes, 32768) && pipe.InitBuffer(scales, 512) &&
          pipe.InitBuffer(weights, 65536) && pipe.InitBuffer(expanded, 2304) &&
          pipe.InitBuffer(scratch, 152832))) {
        mm.End();
        return;
    }
    Operations op{pipe, mm};
    op.codes = codes.Get<int8_t>(32768);
    op.scales = scales.Get<half>(256);
    op.weights = weights.Get<half>(32768);
    op.expanded = expanded.Get<half>(1152);
    auto work = scratch.Get<uint8_t>(152832);
    // Inspect actual addresses returned by TPipe before ANY payload access.
    // The final 8192 bytes include the m200 Brcb fixed temporary at 254976.
    if (op.codes.GetPhyAddr() != 0 || op.codes[16384].GetPhyAddr() != 16384 ||
        op.scales.GetPhyAddr() != 32768 || op.scales[128].GetPhyAddr() != 33024 ||
        op.Weight(0).GetPhyAddr() != 33280 || op.Weight(1).GetPhyAddr() != 66048 ||
        op.expanded.GetPhyAddr() != 98816 || work.GetPhyAddr() != 101120) {
        mm.End();
        return;
    }
    mm.SetLocalWorkspace(work);
    // Fetch peeks without reserving. Check capacity before Alloc mutates the
    // same-core pool. Neither call enters Matmul or issues a DMA.
    if (static_cast<uint32_t>(pipe.FetchEventID(HardEvent::MTE2_S)) >= QUE_MAX_EVENT ||
        static_cast<uint32_t>(pipe.FetchEventID(HardEvent::MTE2_V)) >= QUE_MAX_EVENT) {
        mm.End();
        return;
    }
    op.scalarID = pipe.AllocEventID<HardEvent::MTE2_S>();
    if (static_cast<uint32_t>(op.scalarID) >= QUE_MAX_EVENT) {
        mm.End();
        return;
    }
    op.vectorID = pipe.AllocEventID<HardEvent::MTE2_V>();
    if (static_cast<uint32_t>(op.vectorID) >= QUE_MAX_EVENT) {
        pipe.ReleaseEventID<HardEvent::MTE2_S>(op.scalarID);
        mm.End();
        return;
    }
    op.x.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(x), 16 * 2560);
    op.q.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(w), 19456 * 2560);
    op.scale.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(s), 20 * 19456);
    op.y.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(y), 16 * 19456);
    Duplicate(op.expanded, static_cast<half>(0), 1152);
    PipeBarrier<PIPE_V>();
    for (uint32_t tile = block; tile < 304; tile += blocks) {
        if (!RunW16NTile<Overlap>(op, tile * 64)) break;
    }
    pipe.ReleaseEventID<HardEvent::MTE2_S>(op.scalarID);
    pipe.ReleaseEventID<HardEvent::MTE2_V>(op.vectorID);
    mm.End();
}
} // namespace w16
