"""Deterministic CPU/PyTorch oracle for the fixed FP16 MatMul smoke case."""

import sys
import struct
from pathlib import Path

import torch

M, K, N = 16, 256, 64


def inputs():
    x_index = torch.arange(M * K, dtype=torch.int32).reshape(M, K)
    w_index = torch.arange(N * K, dtype=torch.int32).reshape(N, K)
    x = (((x_index * 13 + 7) % 17) - 8).to(torch.float16) / 8
    w = (((w_index * 7 + 3) % 19) - 9).to(torch.float16) / 8
    return x.contiguous(), w.contiguous()


def write_half(path, tensor):
    bits = tensor.view(torch.int16).flatten().tolist()
    path.write_bytes(struct.pack(f"<{len(bits)}h", *bits))


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in {"prepare", "check"}:
        raise SystemExit("usage: reference.py {prepare|check} DATA_DIR")
    mode, data_dir = sys.argv[1], Path(sys.argv[2])
    x, w = inputs()
    if mode == "prepare":
        data_dir.mkdir(parents=True, exist_ok=True)
        write_half(data_dir / "x.bin", x)
        write_half(data_dir / "w.bin", w)
        (data_dir / "actual.bin").unlink(missing_ok=True)
        print("Prepared FP16 inputs for X @ W.T")
        return

    raw = (data_dir / "actual.bin").read_bytes()
    if len(raw) != M * N * 2:
        raise SystemExit(f"FAIL: actual.bin has {len(raw)} bytes, expected {M * N * 2}")
    bits = struct.unpack(f"<{M * N}h", raw)
    actual = torch.tensor(bits, dtype=torch.int16).view(torch.float16).reshape(M, N)
    expected = (x.float() @ w.float().T).half()
    if not torch.equal(actual, expected):
        delta = (actual.float() - expected.float()).abs()
        index = int(delta.argmax())
        row, col = divmod(index, N)
        raise SystemExit(
            f"FAIL: {int((actual != expected).sum())} mismatches; max abs error "
            f"{delta.max().item():.6g}; first max at [{row},{col}]: "
            f"got {actual[row, col].item()}, expected {expected[row, col].item()}"
        )
    print(f"PASS: SmokeMatmul matches CPU/PyTorch FP32-accumulate -> FP16 reference ({M}x{N})")


if __name__ == "__main__":
    main()
