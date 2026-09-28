// CPU model of the public calls used by A1. Synchronization/hardware behavior
// and CANN compilation remain separate server acceptance requirements.
#pragma once
#include "kernel_operator.h"

namespace matmul {
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
        w_ = w;
        ready_ = true;
    }
    bool Iterate(bool partial = false) {
        assert(ready_);
        ready_ = false;
        assert(tiling_.singleM == 16 && tiling_.singleN == 64);
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
            for (uint32_t n = 0; n < tiling_.singleN; ++n) {
                float &sum = accumulator_[m * tiling_.singleN + n];
                for (uint32_t k = 0; k < tiling_.singleK; ++k) {
                    sum += static_cast<float>(x_.GetValue(m * tiling_.orgKa + k)) *
                           static_cast<float>(w_.GetValue(n * tiling_.singleK + k));
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
#define REGIST_MATMUL_OBJ(pipe, workspace, mm, tiling) mm.Init(tiling)
