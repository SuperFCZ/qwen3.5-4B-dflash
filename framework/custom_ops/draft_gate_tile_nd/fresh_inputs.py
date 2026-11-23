"""Seeded W8A16 operands and an independent exact-integer reference for this shape."""

import numpy as np

from layout import K, M, N, require


def synthetic_operands(m=M, k=K, n=N, seed=310):
    require(k % 128 == 0 and n % 16 == 0, "unaligned input shape")
    rng = np.random.Generator(np.random.PCG64(seed))
    x = rng.integers(1, 5, size=(m, k), dtype=np.int8)
    # Positive and negative output channels, dense weights, group/channel-specific scales.
    # These dyadic FP16 values and all intermediate sums are exactly representable in FP32.
    weights = rng.integers(1, 10, size=(n, k), dtype=np.int8)
    weights[1::2] *= -1
    scale_units = rng.integers(1, 5, size=(k // 128, n), dtype=np.uint8)
    total = np.zeros((m, n), dtype=np.int32)
    for group, units in enumerate(scale_units):
        ks = slice(128 * group, 128 * (group + 1))
        subtotal = x[:, ks].astype(np.int32) @ weights[:, ks].astype(np.int32).T
        total += subtotal * units
    require(4 * 9 * 4 * k < 2**24, "FP32 exact reference requires bounded dot products")
    nz = weights.reshape(n // 16, 16, k // 32, 32).transpose(2, 0, 1, 3)
    return {
        "x.bin": x.astype("<f2").tobytes(),
        "w_nz.bin": nz.tobytes(),
        "s_gn.bin": (scale_units.astype("<f2") / np.float16(64)).tobytes(),
        "golden.bin": (total.astype(np.float32) / 64).astype("<f2").tobytes(),
    }


def materialize(folder):
    folder.mkdir()
    for name, data in synthetic_operands().items():
        (folder / name).write_bytes(data)
