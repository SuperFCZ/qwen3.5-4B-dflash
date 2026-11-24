#pragma once
#include "kernel_operator.h"
#include "kv_spec.h"

namespace kv_panel {
// All paths deliver the same A1 NZ address ((k/16)*M + row)*16 + k%16.
template<uint32_t M>
__aicore__ inline void LoadActivation(AscendC::LocalTensor<half> a, AscendC::GlobalTensor<half> x,
                                      AscendC::LocalTensor<half> wbuf)
{
    using namespace AscendC;
    const DataCopyParams scatter{K / 16, 1, 0, M - 1};
#if DFLASH_KV_INPUT == 0
    for (uint32_t row = 0; row < M; ++row) DataCopy(a[row*16], x[row*K], scatter);
#elif DFLASH_KV_INPUT == 1
    // No weights have been issued. Both W16 banks form one temporary contiguous ND buffer.
    for (uint32_t row = 0; row < M; row += StagingRows) {
        const uint32_t count = M-row < StagingRows ? M-row : StagingRows;
        DataCopy(wbuf, x[row*K], count*K);
        PipeBarrier<PIPE_ALL>();
        for (uint32_t j = 0; j < count; ++j) {
            DataCopy(a[(row+j)*16], wbuf[j*K], scatter);
        }
        // Complete every UB->A1 source read before the next band or the first weight Cast.
        PipeBarrier<PIPE_ALL>();
    }
#elif DFLASH_KV_INPUT == 2
    // The timed model contains the ND -> [K/16,M,16] transpose.
    DataCopy(a, x, M*K);
#else
#error Unknown KV activation layout
#endif
}
} // namespace kv_panel
