    if (down2d) {
        // Basic (not padded) copies. Length and both gaps are in units of 32 B.
        const AscendC::DataCopyParams qCopy{8, 64, 2496, 0};
        const AscendC::DataCopyParams sCopy{2, 4, 156, 0};
        const uint32_t src = ((kBegin / 32) * (2560 / 16) + nBegin / 16) * 16 * 32;
        AscendC::DataCopy(qLocal, qGm[src], qCopy);
        AscendC::DataCopy(sLocal, sGm[(kBegin / 128) * 2560 + nBegin], sCopy);
        return;
    }
