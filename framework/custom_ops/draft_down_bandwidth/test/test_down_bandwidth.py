from pathlib import Path
import os
import struct
import sys
import subprocess
import tempfile
import unittest
import copy

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import down_panel as d
from gate_panel import pack_cube_panels
from layout import once
from run_down import BOUNDARIES, boundary_case, check_w_pair
from run_gate import pair_statistics

sys.path.insert(0, str(d.GATE / "test"))
from test_panels import cpu_headers


def down_cpu_headers(tmp):
    inc = cpu_headers(tmp)
    p = inc / "kernel_operator.h"
    p.write_text(
        once(
            p.read_text(),
            "if(sizeof(T)==2 && params.blockCount==16)\n        assert(params.blockLen==1 && params.srcStride==159 && params.dstStride==0);",
            "if(sizeof(T)==2 && params.blockCount==608)\n        assert(params.blockLen==1 && params.srcStride==0 && params.dstStride==15);",
        )
    )
    p = inc / "lib/matmul_intf.h"
    p.write_text(
        once(
            p.read_text(),
            "tiling_.singleM==16 && tiling_.orgKa==2560",
            "tiling_.singleM==16 && tiling_.orgKa==4864",
        )
    )
    return inc


class Down(unittest.TestCase):
    def test_real_kernels_and_buffer_lifetimes(self):
        m, k, n = d.SHAPE
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inc = down_cpu_headers(root)
            rng = np.random.default_rng(649728)
            q = rng.integers(-127, 128, (n, k), dtype=np.int8)
            gn = rng.uniform(0.004, 0.25, (k // 128, n)).astype("<f2")
            nz = q.reshape(n // 16, 16, k // 32, 32).transpose(2, 0, 1, 3).tobytes()
            codes, scales, _ = pack_cube_panels(nz, gn.tobytes(), d.PANEL, k=k, n=n)
            (root / "codes").write_bytes(codes)
            (root / "scales").write_bytes(scales)
            x = np.zeros((m, k), dtype="<f2")
            acc = np.zeros((m, n), dtype=np.float32)
            for row in range(m):
                for group in range(k // 128):
                    kk = group * 128 + (group * 31 + row * 11) % 128
                    x[row, kk] = np.float16((-1 if (group + row) % 2 else 1) * (1 + row / 31))
                    w16 = q[:, kk].astype(np.float16) * gn[group]
                    acc[row] += np.float32(x[row, kk]) * w16.astype(np.float32)
            x.tofile(root / "x")
            acc.astype("<f2").tofile(root / "golden")
            for banks in (1, 2):
                (root / "down_spec.h").write_bytes(d.header(banks))
                exe = root / "portable"
                cc = [
                    "c++",
                    "-std=c++17",
                    "-O2",
                    "-ffp-contract=off",
                    "-I" + str(inc),
                    "-I" + str(root),
                    "-I" + str(HERE / "op_kernel"),
                    "-I" + str(d.GATE),
                    str(HERE / "test/portable_down.cpp"),
                    "-o",
                    exe,
                ]
                subprocess.run(cc, check=True)
                subprocess.run([exe, root], cwd=root, check=True)
                subprocess.run([exe, root, "fail-iterate"], cwd=root, check=True)
                source = HERE / "op_kernel" / (d.STEM + ".cpp")
                anchor = (
                    "if (piece != 0) Pair<HardEvent::MTE3_V>(weightStored);"
                    if banks == 1
                    else "if (piece > 0 && piece + 1 < Pieces) Pair<HardEvent::MTE3_V>(weightStored);"
                )
                fault = root / source.name
                fault.write_text(once(source.read_text(), anchor, "// CPU fault: omitted W16 consumer fence"))
                subprocess.run(cc, check=True)
                result = subprocess.run(
                    [exe, root], cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("W16 overwritten during UB-to-L1 copy", result.stdout)
                fault.unlink()

    @unittest.skipUnless(os.getenv("GATE_TEST_GE_DESCRIPTOR"), "requires SDK GE schema")
    def test_real_ge_schema_and_full_down_reference(self):
        from air import protobuf_classes

        types = protobuf_classes(Path(os.environ["GATE_TEST_GE_DESCRIPTOR"]).read_bytes())
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            files, label, _ = d.inputs(out)
            self.assertEqual(label, "SYNTHETIC_SEEDED_C16_DOWN")
            self.assertEqual({n: i["bytes"] for n, i in files.items()}, d.EXTENTS)
            self.assertEqual(d.LOGICAL_BYTES, 25686016)
            self.assertEqual(d.FLOPS, 796917760)
            sparse = boundary_case(out / "boundary", files, shape=d.SHAPE, pairs=BOUNDARIES)
            self.assertEqual(sparse["golden.bin"]["bytes"], 81920)
            self.assertNotEqual(sparse["golden.bin"]["sha256"], files["golden.bin"]["sha256"])
            for control in (True, False):
                folder = out / str(control)
                result = d.prepare_air(folder, types, files, control)
                msg = types[0].FromString((folder / "down.air").read_bytes())
                nodes = {x.name: x for x in msg.graph[0].op}
                self.assertEqual(list(nodes["x"].output_desc[0].shape.dim), [16, 9728])
                op = nodes[d.OLD_OP]
                self.assertEqual(op.type, d.OLD_OP if control else d.OP)
                self.assertEqual(list(op.output_desc[0].shape.dim), [16, 2560])
                for i, name in enumerate(("weight", "scale"), 1):
                    node = nodes[name]
                    desc = node.output_desc[0]
                    self.assertEqual(desc, node.attr["value"].t.desc)
                    self.assertEqual(desc, op.input_desc[i])
                    self.assertEqual(tuple(desc.shape.dim), result["physical_shapes"][i - 1])
                    self.assertEqual(
                        node.attr["value"].t.data, d.c.verify(result["constants"][i - 1]).read_bytes()
                    )
                    if name == "weight" or not control:
                        self.assertTrue(node.attr["_out_shape_locked"].b)
                    if control and name == "weight":
                        self.assertEqual(list(desc.attr["origin_shape"].list.i), [2560, 9728])

    def test_candidate_resource_capacity_and_cube_equality(self):
        offsets = {f["name"]: 24 + f["offset"] for f in d.schema()}
        platform = dict(
            blockDim=8,
            availableCores=8,
            coreLimit=0,
            workspace=2097152,
            capacity=dict(UB=262144, L1=1048576, L0A=65536, L0B=65536, L0C=262144),
        )
        models = {}
        for mode, w, scratch in ((d.SINGLE, 77824, 132352), (d.DOUBLE, 155648, 54528)):
            data = bytearray(224)
            struct.pack_into("<4IQ", data, 0, *d.SHAPE, d.PANEL, scratch)
            fields = dict(
                singleCoreM=16,
                singleCoreN=64,
                singleCoreK=4864,
                baseM=16,
                baseN=64,
                baseK=256,
                stepKa=1,
                stepKb=1,
                depthA1=2,
                depthB1=2,
                dbL0A=2,
                dbL0B=2,
                dbL0C=1,
                transLength=2048,
            )
            for key, val in fields.items():
                struct.pack_into("<I", data, offsets[key], val)
            sched = dict(
                baseline_assignments=[list(range(i, 40, 8)) for i in range(8)],
                tiling_hex=data.hex(),
                resources=dict(
                    codes_bytes=38912,
                    scale_bytes=4864,
                    weight_bytes=w,
                    raw_banks=1,
                    tile_k=4864,
                    user_ub=262144 - scratch,
                    scratch=scratch,
                    workspace=2097152,
                    l1_used=1015808,
                    l0a_used=16384,
                    l0b_used=65536,
                    l0c_used=4096,
                ),
            )
            q = dict(tiling_sha256=d.sha(data), platform=platform, schedule=sched)
            proof = d.plan(q, mode)
            self.assertEqual(proof["outer_iterates_per_n64"], 2)
            self.assertEqual(proof["resident_l1_bytes"], 933888)
            models[mode] = dict(air={"sha256": "same"}, plans={d.TAG: dict(query=q, contract=proof)})
            bad = copy.deepcopy(q)
            bad["schedule"]["resources"]["l1_used"] -= 1024
            with self.assertRaisesRegex(ValueError, "capacity"):
                d.plan(bad, mode)
            bad = copy.deepcopy(q)
            bad["platform"]["blockDim"] = 7
            with self.assertRaisesRegex(ValueError, "8-block"):
                d.plan(bad, mode)
        check_w_pair(models)
        models[d.DOUBLE]["plans"][d.TAG]["contract"]["cube_sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "Cube"):
            check_w_pair(models)

    def test_down_pair_stats_use_down_dimensions(self):
        rows = []
        for i in range(3):
            for mode in ((d.SINGLE, d.DOUBLE) if i % 2 == 0 else (d.DOUBLE, d.SINGLE)):
                rows.append(
                    dict(
                        mode=mode,
                        round=i,
                        folder=str((mode, i)),
                        bit_mismatches=0,
                        launch_authorized=True,
                        profiled=False,
                        validation=dict(timing=dict(samples_ms=[1 if mode == d.SINGLE else 0.8] * 30)),
                    )
                )
        t = pair_statistics(rows, d.DOUBLE, d.SINGLE, flops=d.FLOPS, logical_bytes=d.LOGICAL_BYTES)
        self.assertAlmostEqual(t["equivalent_matrix_tflops"], 0.9961472)
        self.assertAlmostEqual(t["logical_gbps"], 32.10752)
        self.assertEqual(t["decision"], "GO")
