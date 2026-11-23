import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from layout import (
    K,
    N,
    TILE,
    OLD_STEM,
    STEM,
    HEADER,
    HEADER_SHA,
    OLD_OP,
    OP,
    host_source,
    kernel_source,
    once,
    pack,
)
from bench import NZ_MODE, TILE_MODE, atc_args, paired, selected_image
from air import protobuf_classes, transform
from tile_gate_graph import model_proto
import scale_contracts as scale
from a3_launch import configure_build


def weights(n, k):
    plane = np.arange(k // 32)[:, None, None].astype(np.int32)
    channels = np.arange(n)[None, :, None].astype(np.int32)
    lane = np.arange(32)[None, None, :].astype(np.int32)
    return ((plane + 7 * channels + lane) % 255 - 127).astype(np.int8)


class TileND(unittest.TestCase):
    def test_each_logical_coordinate(self):
        for n, k in ((64, 256), (192, 768)):
            nz = weights(n, k)
            packed, _ = pack(nz.tobytes(), n, k)
            q = np.frombuffer(packed, np.int8).reshape(n // 64, k // 256, 64, 256)
            for ch in range(n):
                np.testing.assert_array_equal(q[ch // 64, :, ch % 64, :].reshape(k), nz[:, ch, :].reshape(k))
        for n, k, data in ((63, 256, b""), (64, 255, b"x" * 16320), (64, 256, b"x" * 16383)):
            with self.assertRaises(ValueError):
                pack(data, n, k)

    def test_only_weight_path_changes(self):
        raw = (scale.c.QUANT / "op_kernel" / (OLD_STEM + ".cpp")).read_bytes()
        old = scale.render(raw)
        new = kernel_source(old)
        for token in (
            b"AscendC::Mul(",
            b"ExpandGateScale(",
            b"CopyRawTile(",
            b"SetFlag<",
            b"WaitFlag<",
            b"ReleaseEventID<",
            b"PipeBarrier<PIPE_V>",
            b"PipeBarrier<PIPE_ALL>",
            b"mm.SetTensorB(",
            b"mm.Iterate(",
            b"mm.GetTensorC(",
            b"REGIST_MATMUL_OBJ(",
            b"pipe.InitBuffer(",
        ):
            self.assertEqual(old.count(token), new.count(token), token)
        self.assertNotIn(b"qLocal[k1 * kTileN * kNzK0]", new)
        with self.assertRaises(ValueError):
            kernel_source(raw)
        host = host_source((scale.c.QUANT / "op_host" / (OLD_STEM + ".cpp")).read_text())
        self.assertIn(
            'Input("w_tiles").ParamType(REQUIRED).DataType({ge::DT_INT8}).Format({ge::FORMAT_ND})', host
        )
        self.assertIn("m == 16 && k == 2560 && n == 19456", host)
        self.assertIn("{304,10,64,256}", host)

    def test_full_gate_both_implementations(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            inc = p / "inc"
            shutil.copytree(scale.c.QUANT / "test/cpu_kernel_include", inc)
            header = inc / "kernel_operator.h"
            h = header.read_text()
            if sys.platform.startswith("linux") and platform.machine() == "aarch64":
                h = once(h, "using half = _Float16;", "using half = __fp16;")
            h = once(
                h,
                "    Tensor operator[]",
                "    uintptr_t GetPhyAddr() const { return reinterpret_cast<uintptr_t>(data); }\n    Tensor operator[]",
            )
            h = once(
                h,
                "assert(mask == 32 && repeats == 64 && params.dstBlkStride == 1 && params.srcBlkStride == 1);",
                "assert(params.dstBlkStride == 1 && params.srcBlkStride == 1 && ((mask == 32 && repeats == 64 && params.dstRepStride == 16 && params.srcRepStride == 1) || (mask == 128 && repeats == 128 && params.dstRepStride == 8 && params.srcRepStride == 4)));",
            )
            header.write_text(h + (scale.HERE / "test/portable_model.inc").read_text())
            configure_build(
                p / HEADER,
                p / "config.json",
                0,
                "batched",
                None,
                "prefetch",
                "a-ub",
                "broadcast",
                256,
                "loop",
                "raw",
            )
            self.assertEqual(scale.c.sha(p / HEADER), HEADER_SHA)
            old = scale.render((scale.c.QUANT / "op_kernel" / (OLD_STEM + ".cpp")).read_bytes())
            # Fill each real NZ plane independently to limit peak host memory.
            with (p / "nz.bin").open("wb") as f:
                ch = np.arange(N, dtype=np.int32)[:, None]
                lane = np.arange(32, dtype=np.int32)
                for plane in range(K // 32):
                    f.write(((plane + 7 * ch + lane) % 255 - 127).astype(np.int8).tobytes())
            (p / "tile.bin").write_bytes(pack((p / "nz.bin").read_bytes())[0])
            for name, code, entry in (("nz", old, OLD_STEM), ("tile", kernel_source(old), STEM)):
                (p / "variant.cpp").write_bytes(code)
                exe = p / (name + "-test")
                subprocess.run(
                    [
                        "c++",
                        "-std=c++17",
                        "-O2",
                        "-ffp-contract=off",
                        "-DENTRY=" + entry,
                        "-I" + str(inc),
                        "-I" + str(p),
                        str(HERE / "test/gate.cpp"),
                        "-o",
                        str(exe),
                    ],
                    check=True,
                )
                self.assertEqual(
                    subprocess.check_output([str(exe), str(p / (name + ".bin")), name], text=True).strip(),
                    "311296 exact, all tiles covered",
                )

    def test_paired_samples_and_original_atc_flags(self):
        rows = [
            dict(
                round=i,
                mode=m,
                folder=f"/timing/{i}/{m}",
                bit_mismatches=0,
                launch_authorized=True,
                profiled=False,
                validation={"timing": {"samples_ms": [2 if m == NZ_MODE else 1.5] * 30}},
            )
            for i in range(3)
            for m in ((NZ_MODE, TILE_MODE) if i % 2 == 0 else (TILE_MODE, NZ_MODE))
        ]
        r = paired(rows)
        self.assertEqual(r["reduction_pct"], 25)
        self.assertEqual(r["variants"][NZ_MODE]["count"], 90)
        with self.assertRaises(ValueError):
            paired(rows[::-1])
        rows[2]["bit_mismatches"] = 1
        with self.assertRaises(ValueError):
            paired(rows)
        a = [
            "atc",
            "--mode=0",
            "--model=/old",
            "--output=/a",
            "--soc_version=Ascend310P3",
            "--precision_mode=must_keep_origin_dtype",
        ]
        self.assertEqual(atc_args(a, "/new", "/b"), a[:2] + ["--model=/new", "--output=/b"] + a[4:])

    def test_selected_kernel_must_match_actual_launch(self):
        key = "te_gate_" + ("b" * 64)
        info = {
            "_kernel_bin_id": {"s": key + "a" * 64},
            "_kernelname": {"s": key + "__kernel0"},
            "_tbeKernelSize": {"i": 1300},
            "tvm_blockdim": {"i": 8},
            "_ge_attr_op_kernel_lib_name": {"s": "AIcoreEngine"},
        }
        node = {"type": OP, "name": "single", "attributes": [{"key": k, "value": v} for k, v in info.items()]}
        self.assertEqual(selected_image(node, 8)["sha256"], "a" * 64)
        with self.assertRaisesRegex(ValueError, "blockDim"):
            selected_image(node, 7)

    @unittest.skipUnless(
        os.environ.get("GATE_TEST_GE_DESCRIPTOR"), "requires a GE descriptor compiled from the installed SDK"
    )
    def test_real_schema_roundtrip(self):
        types = protobuf_classes(Path(os.environ["GATE_TEST_GE_DESCRIPTOR"]).read_bytes())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nz = bytes(K * N)
            scale_bits = b"\x00\x3c" * (K // 128 * N)
            original = model_proto(types, nz, scale_bits)
            src = root / "source.air"
            dest = root / "tile.air"
            src.write_bytes(original.SerializeToString())
            report = transform(src, dest, nz, scale_bits, types)
            self.assertEqual(report["bytes"], K * N)
            after = types[0].FromString(dest.read_bytes())
            nodes = {n.name: n for n in after.graph[0].op}
            self.assertEqual(nodes[OLD_OP].type, OP)
            for desc in (
                nodes["weight"].output_desc[0],
                nodes["weight"].attr["value"].t.desc,
                nodes[OLD_OP].input_desc[1],
            ):
                self.assertEqual(
                    (desc.layout, list(desc.shape.dim), list(desc.attr["origin_shape"].list.i)),
                    ("ND", list(TILE), list(TILE)),
                )
            self.assertEqual(nodes["scale"], original.graph[0].op[2])
            self.assertEqual(nodes["x"], original.graph[0].op[0])
            (root / "bad").mkdir()
            with self.assertRaisesRegex(ValueError, "scale"):
                transform(src, root / "bad/tile.air", nz, bytes(len(scale_bits)), types)
            self.assertEqual(src.read_bytes(), original.SerializeToString())


if __name__ == "__main__":
    unittest.main()
