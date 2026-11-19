    const bool doubleW = DFLASH_GROUP_QUANT_GATE_PREFETCH(m, k, n, tileK);
    const uint64_t oneWeightBytes = kTileN * tileK * sizeof(uint16_t);
    const uint64_t weightBytes = oneWeightBytes * (doubleW ? 2U : 1U);
    const uint64_t extraWeightBytes = weightBytes - oneWeightBytes;
