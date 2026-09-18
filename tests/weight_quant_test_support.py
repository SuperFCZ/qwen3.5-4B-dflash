"""Explicit CPU-only oracle for native-call wiring; never device evidence."""
import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "framework/python"))

from qwen35_dflash.ascend310p.custom_op_export import _fake_npu_weight_quant_matmul

_LIBRARIES = []


@pytest.fixture
def weight_quant_cpu():
    if not _LIBRARIES:
        definition = torch.library.Library("npu", "FRAGMENT")
        definition.define(
            "npu_weight_quant_batchmatmul(Tensor x, Tensor weight, Tensor antiquant_scale, "
            "Tensor? antiquant_offset=None, Tensor? quant_scale=None, Tensor? quant_offset=None, "
            "Tensor? bias=None, int antiquant_group_size=0, int inner_precise=0) -> Tensor")
        implementation = torch.library.Library("npu", "IMPL", "CPU")

        def oracle(x, weight, antiquant_scale, antiquant_offset=None, quant_scale=None,
                   quant_offset=None, bias=None, antiquant_group_size=0, inner_precise=0):
            assert x.dtype == antiquant_scale.dtype == torch.float16
            assert weight.dtype == torch.int8 and inner_precise == 0
            assert all(t is None for t in (antiquant_offset, quant_scale, quant_offset, bias))
            group = antiquant_group_size or weight.shape[0]
            dense = (weight.float() * antiquant_scale.float().repeat_interleave(group, 0)).half()
            return torch.mm(x, dense)

        implementation.impl("npu_weight_quant_batchmatmul", oracle)
        torch.library.register_fake("npu::npu_weight_quant_batchmatmul", lib=definition)(
            _fake_npu_weight_quant_matmul)
        _LIBRARIES.extend([definition, implementation])
    return torch.ops.npu.npu_weight_quant_batchmatmul.default
