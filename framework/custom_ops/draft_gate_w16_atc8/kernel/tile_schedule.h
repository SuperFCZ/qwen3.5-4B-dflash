// Both device builds and the CPU event-order test instantiate this exact loop.
template<bool Overlap, typename Operations>
__aicore__ inline bool RunW16NTile(Operations &op, uint32_t nBegin)
{
    constexpr uint32_t steps = 10;
    op.Copy(nBegin, 0);
    op.WaitRaw();
    op.Dequant(0);
    for (uint32_t k = 0; k < steps; ++k) {
        const bool next = k + 1 < steps;
        if (next) op.Copy(nBegin, k + 1);
        if (!op.Iterate(k)) {
            if (next) op.WaitRaw();
            op.Fence();
            return false;
        }
        if (next) op.WaitRaw();
        if constexpr (!Overlap) op.Fence();
        if (next) op.Dequant(k + 1);
        if constexpr (Overlap) op.Fence();
    }
    op.Output(nBegin);
    op.Fence();
    return true;
}
