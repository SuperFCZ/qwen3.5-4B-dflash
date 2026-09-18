"""CANN built-in A16W8 grouped matmul; no activation quantization or fallback."""
from __future__ import annotations

import torch
from torch import Tensor


TORCH_OP = "npu::npu_weight_quant_batchmatmul"
GE_OP = "WeightQuantBatchMatmulV2"


def require_weight_quant_matmul():
    try:
        return torch.ops.npu.npu_weight_quant_batchmatmul.default
    except AttributeError as error:
        raise RuntimeError(
            "Quantized Draft requires torch_npu.npu_weight_quant_batchmatmul "
            "with CANN group-128 A16W8 support. Load torch_npu before export. "
            "Use draft_quant_matmul=dequant only for an explicit baseline; "
            "no automatic fallback is used."
        ) from error


def weight_quant_linear(value: Tensor, weight: Tensor, scales: Tensor) -> Tensor:
    """Linear [N,K] -> CANN [K,N]; grouped scales -> [K/128,N], single group -> [N].

    Keep transposes as views. TorchAir's built-in converter emits a fused
    WeightQuantBatchMatmulV2 with group_size=128, inner_precise=0. Casting the
    activation to INT8 or folding K-group scales outside MatMul is forbidden.
    The 310P AIR lowering folds the weight transpose and inserts built-in
    TransData to NZ, matching ACLNN's conversion before its WeightNz kernel.
    Eager success alone does not test this graph-mode conversion.
    """
    if value.dtype != torch.float16 or weight.dtype != torch.int8 or scales.dtype != torch.float16:
        raise ValueError("weight_quant requires FP16 activations/scales and INT8 weight codes")
    if weight.ndim != 2 or value.ndim < 1:
        raise ValueError("weight_quant requires weight[N,K] and a non-scalar activation")
    k, n = weight.shape[1], weight.shape[0]
    if k <= 0 or n <= 0 or scales.shape != (n, k // 128) or k % 128 or value.shape[-1] != k:
        raise ValueError("weight_quant requires weight[N,K] and scales[N,K/128]")
    # CANN disallows group_size == K. A single group is exactly per-channel.
    group_size = 128 if k > 128 else 0
    scale = scales.transpose(0, 1) if group_size else scales.reshape(n)
    result = require_weight_quant_matmul()(
        value.reshape(-1, k), weight.transpose(0, 1), scale,
        antiquant_group_size=group_size, inner_precise=0,
    )
    return result.reshape(*value.shape[:-1], n)


@torch.inference_mode()
def preflight_weight_quant_matmul(device: str) -> dict:
    """Execute an exact-valued group-128 check before loading model weights.

    Dispatcher/Meta presence alone does not establish 310P kernel support.
    These small dyadic inputs expose swapped groups/channels while avoiding
    reduction-rounding ambiguity. Full-checkpoint parity remains separate.
    """
    if not str(device).startswith("npu"):
        raise ValueError("weight_quant device preflight requires an NPU; CPU is not target evidence")
    require_weight_quant_matmul()
    x = ((torch.arange(16 * 256).reshape(16, 256) % 7) - 3).half() / 16
    q = ((torch.arange(64 * 256).reshape(64, 256) % 11) - 5).to(torch.int8)
    scales = (1 + torch.arange(64 * 2).reshape(64, 2) % 4).half() / 32
    golden = (x.float() @ (q.float() * scales.float().repeat_interleave(128, 1)).t()).half()
    try:
        actual = weight_quant_linear(x.to(device), q.to(device), scales.to(device)).cpu()
    except RuntimeError as error:
        raise RuntimeError(
            "CANN group-128 A16W8 preflight failed before weight loading; "
            "check 310P CANN/torch_npu kernel support. No FP16 fallback was selected. "
            + str(error)
        ) from error
    if not torch.equal(actual, golden):
        raise RuntimeError("CANN group-128 A16W8 exact-valued preflight differs from the reference")
    return {"status": "PASS", "scope": "small exact-valued NPU group-128 probe",
            "device": str(device), "torch_op": TORCH_OP, "ge_op_type": GE_OP,
            "activation_dtype": "float16", "weight_dtype": "int8",
            "group_size": 128, "inner_precise": 0, "cpu_fallback": False}
