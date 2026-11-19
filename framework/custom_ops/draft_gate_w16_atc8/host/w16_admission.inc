    // Observe scheduler A's selected blockDim; never force a platform result.
    if (doubleW && (blockDim != 8 || availableCores != 8 || coreLimit != 0 ||
        rawBanks != 2 || weightBytes != 65536 || matmulUbBytes != 152832 || userUbBytes != 109312 ||
        codesBytes != 32768 || scaleBytes != 512 || broadcastBytes != 2304 || broadcastReservedBytes != 8192 ||
        l1Used != 40960 || l0aUsed != 16384 || l0bUsed != 65536 || l0bCapacity != 65536 || l0cUsed != 4096 ||
        cube.get_baseM() != 16 || cube.get_baseN() != 64 || cube.get_baseK() != 256 ||
        cube.get_depthA1() != 1 || cube.get_depthB1() != 1 || cube.get_stepM() != 1 || cube.get_stepN() != 1 ||
        cube.get_stepKa() != 1 || cube.get_stepKb() != 1 || cube.get_dbL0A() != 2 || cube.get_dbL0B() != 2 ||
        cube.get_dbL0C() != 1 || cube.get_transLength() != 34816)) {
        std::fprintf(stderr, "W16 Formal Host: SDK launch, buffers or Cube tiling outside fixed Gate contract\n");
        return ge::GRAPH_FAILED;
    }
