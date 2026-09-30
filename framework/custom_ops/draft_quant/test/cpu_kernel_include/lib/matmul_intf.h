// CPU model of the public calls used by A1. Synchronization/hardware behavior
// and CANN compilation remain separate server acceptance requirements.
#pragma once
#include "kernel_operator.h"

namespace matmul {
inline int cpuFailIteration = -1;
enum class CubeFormat { ND };
template <AscendC::TPosition P, CubeFormat F, typename D, bool TRANS = false>
struct MatmulType {};

template <typename A, typename B, typename C, typename Bias>
class Matmul {
public:
    void Init(const CpuCubeTiling *tiling) { tiling_ = *tiling; }
    void SetLocalWorkspace(AscendC::LocalTensor<uint8_t>) {}
    void SetTensorA(AscendC::GlobalTensor<half> x) { x_ = x; ready_ = true; }
    void SetTensorB(AscendC::LocalTensor<half> w, bool transpose) {
        assert(transpose);
        AscendC::BeforeMatmul(AscendC::Bytes(w.data, tiling_.singleN * tiling_.singleK * sizeof(half)));
        // Library helpers fetch temporary IDs too. A prefetched raw tile's
        // pending signals must be reserved even if PIPE_ALL completes its DMA.
        if (AscendC::cpuSync.events[0][0].signalled) ++cpuMetrics.libraryWithPendingSignals;
        const auto scalar = AscendC::cpuPipe->FetchEventID(AscendC::HardEvent::MTE2_S);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(scalar);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(scalar);
        const auto vector = AscendC::cpuPipe->FetchEventID(AscendC::HardEvent::MTE2_V);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(vector);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(vector);
        w_ = w;
        ready_ = true;
    }
    bool Iterate(bool partial = false) {
        assert(ready_);
        ready_ = false;
        if (cpuFailIteration == static_cast<int>(cpuMetrics.iterations)) {
            // Fault injection only: let End release an aborted accumulation.
            accumulatorLive_ = false;
            return false;
        }
        assert((tiling_.singleM == 16 || tiling_.singleM == 32 || tiling_.singleM == 64 || tiling_.singleM == 80) && tiling_.singleN == 64);
        assert(w_.count == tiling_.singleN * tiling_.singleK);
        if (partial) {
            assert(accumulatorLive_);
            ++cpuMetrics.partials;
        } else {
            assert(!accumulatorLive_);
            accumulator_.assign(tiling_.singleM * tiling_.singleN, 0.0f);
            accumulatorLive_ = true;
        }
        for (uint32_t m = 0; m < tiling_.singleM; ++m) {
            // Sparse CPU fixtures can cover the full A2 address space without
            // spending billions of host operations multiplying finite B by 0.
            // This is only an oracle optimization, never production code.
            std::vector<std::pair<uint32_t, float>> terms;
            for (uint32_t k = 0; k < tiling_.singleK; ++k) {
                const float value = static_cast<float>(x_.GetValue(m * tiling_.orgKa + k));
                if (value != 0) terms.emplace_back(k, value);
            }
            for (uint32_t n = 0; n < tiling_.singleN; ++n) {
                float &sum = accumulator_[m * tiling_.singleN + n];
                for (const auto &term : terms) {
                    sum += term.second * static_cast<float>(w_.GetValue(n * tiling_.singleK + term.first));
                }
            }
        }
        ++cpuMetrics.iterations;
        return true;
    }
    void GetTensorC(AscendC::GlobalTensor<half> y) {
        assert(accumulatorLive_);
        for (uint32_t m = 0; m < tiling_.singleM; ++m) {
            for (uint32_t n = 0; n < tiling_.singleN; ++n) {
                const auto index = m * tiling_.orgN + n;
                assert(index < y.count);
                const auto globalIndex = static_cast<size_t>(y.data + index - cpuOutputBase);
                assert(globalIndex < cpuOutputWrites.size());
                // Independently derive the sole owner from the global column.
                assert((globalIndex % tiling_.orgN / 64) % cpuBlockNum == cpuBlockIdx);
                assert(cpuOutputWrites[globalIndex]++ == 0);
                cpuOutputOwners[globalIndex] = cpuBlockIdx;
                y.data[index] = static_cast<half>(accumulator_[m * tiling_.singleN + n]);
                ++cpuMetrics.stores;
            }
        }
        accumulatorLive_ = false;
        ++cpuMetrics.outputs;
    }
    void IterateAll(AscendC::GlobalTensor<half> y) { Iterate(false); GetTensorC(y); }
    void End() { assert(!accumulatorLive_); ++cpuMetrics.ends; }
private:
    CpuCubeTiling tiling_{};
    AscendC::Tensor<half> x_, w_;
    std::vector<float> accumulator_;
    bool ready_ = false, accumulatorLive_ = false;
};
}  // namespace matmul
#define REGIST_MATMUL_OBJ(pipe, workspace, mm, tiling) AscendC::cpuPipe = pipe; mm.Init(tiling)
