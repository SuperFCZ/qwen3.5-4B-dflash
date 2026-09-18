"""Check probe data flow and dynamic capture; CPU oracle is not CANN evidence."""
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from probe_draft_matmul_atc import make_spec, parser
from weight_quant_test_support import weight_quant_cpu
from test_incremental_air_om import small_threads

pytestmark = pytest.mark.usefixtures("weight_quant_cpu", "small_threads")


@pytest.mark.parametrize("bits", [4, 8])
def test_probe_keeps_flat_compressed_inputs_and_dynamic_gears(bits):
    spec = make_spec(bits, "tiny", "cpu")
    x, packed, scales = spec.example_args
    assert packed.shape == (64 * 256 * bits // 8,) and scales.shape == (64 * 2,)
    assert not any(buffer.numel() for buffer in spec.model.buffers())
    exported = torch.export.export(spec.model, spec.example_args, dynamic_shapes=(
        {0: torch.export.Dim("rows", min=16, max=64)}, None, None)).module()
    for rows in (16, 64):
        value = torch.ones(rows, 256).half() / 16
        # Runtime inputs must remain live and cannot be frozen into the AIR.
        output = exported(value, packed, scales)
        torch.testing.assert_close(output, spec.model(value, packed, scales), rtol=0, atol=0)
        zeros = torch.full_like(packed, 0x88 if bits == 4 else 0)
        assert torch.count_nonzero(exported(value, zeros, scales)) == 0
        assert torch.count_nonzero(output) > 0


def test_probe_cli_defaults_to_small_two_bitwidth_test():
    args = parser().parse_args(["--output-dir", "probe", "--atc", "/bin/true", "--soc-version", "Ascend310P3"])
    assert args.bits == [4, 8] and args.projection == ["tiny"]
