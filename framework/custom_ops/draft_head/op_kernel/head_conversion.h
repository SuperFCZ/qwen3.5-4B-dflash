#pragma once
// Experimental Mins -> Maxs. Special-value semantics MUST be measured on 310P3.
// In-place FP32 Vector operations; no scalar patching or second FP16 boundary.
__aicore__ inline void HeadFiniteClamp(const AscendC::LocalTensor<float>& values, uint32_t count) {
    AscendC::Mins(values,values,65504.0f,count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Maxs(values,values,-65504.0f,count);
    AscendC::PipeBarrier<PIPE_V>();
}

// dav_m200 mode 0 uses ONE 64-bit mask for float; never repeat that mask over
// different groups. Every call below handles exactly one 64-float Vector block.
// scratch: 64 floats for the selected bound, then 32 aligned bytes for the mask.
// The normal Head reuses its not-yet-written half output tile (2048 bytes).
__aicore__ inline void HeadSelectLimit(const AscendC::LocalTensor<float>& values,
    const AscendC::LocalTensor<float>& bound, const AscendC::LocalTensor<uint8_t>& mask,
    float limit, AscendC::CMPMODE comparison, uint32_t count) {
    using namespace AscendC;
    Duplicate(bound,limit,64);
    // dav_m200 Duplicate does not support uint8_t: initialize the same 32 B as u32.
    Duplicate(mask.ReinterpretCast<uint32_t>(),uint32_t(0),8);
    PipeBarrier<PIPE_V>();
    // count is 64 (probe) or 1024 (Head); no scalar element loads/stores.
    for(uint32_t offset=0;offset<count;offset+=64) {
        CompareScalar(mask,values[offset],limit,comparison,64);
        PipeBarrier<PIPE_V>();
        // True chooses bound; false copies the original raw FP32 bits.
        Select(values[offset],mask,bound,values[offset],SELMODE::VSEL_CMPMASK_SPR,64);
        PipeBarrier<PIPE_V>(); // finish reading mask before the next comparison
    }
}
__aicore__ inline void HeadCompareSelectClamp(const AscendC::LocalTensor<float>& values,
    const AscendC::LocalTensor<uint8_t>& scratch, uint32_t count) {
    auto bound=scratch.ReinterpretCast<float>();
    auto mask=scratch[256]; // both addresses 32-byte aligned, disjoint from values
    HeadSelectLimit(values,bound,mask,65504.0f,AscendC::CMPMODE::GT,count);
    HeadSelectLimit(values,bound,mask,-65504.0f,AscendC::CMPMODE::LT,count);
}
