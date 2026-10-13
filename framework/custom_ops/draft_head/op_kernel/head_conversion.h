#pragma once
// Experimental Mins -> Maxs. Special-value semantics MUST be measured on 310P3.
// In-place FP32 Vector operations; no scalar patching or second FP16 boundary.
__aicore__ inline void HeadFiniteClamp(const AscendC::LocalTensor<float>& values, uint32_t count) {
    AscendC::Mins(values,values,65504.0f,count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Maxs(values,values,-65504.0f,count);
    AscendC::PipeBarrier<PIPE_V>();
}
