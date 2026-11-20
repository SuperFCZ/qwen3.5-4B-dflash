// Input is one contiguous Group128 slice of 64 FP16 channel scales in UB.
__aicore__ inline void ExpandGateScale(const AscendC::LocalTensor<half>& expanded,
                                      const AscendC::LocalTensor<half>& scales)
{
    uint64_t srcRows[16];
    uint64_t dstRows[16];
    const uint64_t srcBase = (uint64_t)scales.GetPhyAddr();
    const uint64_t dstBase = (uint64_t)expanded.GetPhyAddr();
    for (uint32_t row = 0; row < 16; ++row) {
        srcRows[row] = srcBase;
        dstRows[row] = dstBase + row * 32;
    }
    AscendC::SetMaskNorm();
    // Each repeat reads one 32 B block (16 equal, read-only row addresses)
    // and transposes it into sixteen distinct 32 B rows of replicated bits.
    const AscendC::TransDataTo5HDParams params{false, false, 4, 16, 1};
    AscendC::TransDataTo5HD<half>(dstRows, srcRows, params);
    AscendC::ResetMask();  // match the normal, full B16 mask at Brcb exit
}
