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
import gate_panel as panels
from layout import once
from run_gate import TILE_MODE, pair_statistics, boundary_case


def cpu_headers(tmp):
    inc = tmp / "cpu"
    shutil.copytree(panels.c.QUANT / "test/cpu_kernel_include", inc)
    h = inc / "kernel_operator.h"
    text = once(
        h.read_text(), "using TEventID = int;", "using TEventID = int; constexpr uint32_t QUE_MAX_EVENT=8;"
    )
    if sys.platform.startswith("linux") and platform.machine() == "aarch64":
        text = once(text, "using half = _Float16;", "using half = __fp16;")
    text = once(text, "{ MTE2_S, MTE2_V, V_S }", "{ MTE2_S, MTE2_V, V_S, V_MTE2, V_MTE3, MTE3_V }")
    text = once(text, "std::array<std::array<Event, 8>, 3>", "std::array<std::array<Event, 8>, 6>")
    text = once(
        text,
        "event.fence = E == HardEvent::V_S ?",
        "event.fence = (E == HardEvent::V_S || E == HardEvent::V_MTE2 || E == HardEvent::V_MTE3) ?",
    )
    text = once(
        text,
        "    if constexpr (E == HardEvent::V_S) {",
        """    if constexpr (E == HardEvent::V_MTE2) {
        Require(cpuSync.vectorIssued == event.fence, "wrong Vector to MTE2 fence generation");
        cpuSync.vectorAccesses.clear();
    } else if constexpr (E == HardEvent::V_MTE3) {
        cpuSync.storeReady = event.fence;
    } else if constexpr (E == HardEvent::MTE3_V) {
        ++storeWaits; FinishStores();
    } else if constexpr (E == HardEvent::V_S) {""",
    )
    text = once(
        text,
        "assert(mask == 32 && repeats == 64 && params.dstBlkStride == 1 && params.srcBlkStride == 1);",
        "assert(mask==128 && repeats && params.dstBlkStride==1 && params.srcBlkStride==1 && params.dstRepStride==8 && params.srcRepStride==4);",
    )
    text = once(
        text,
        "mask==128 && repeats==64 && p.dstBlkStride",
        "mask==128 && repeats==16 && p.src1RepStride==0 && p.dstBlkStride",
    )
    text = once(text, "GM, VECIN, VECOUT, VECCALC", "GM, VECIN, VECOUT, VECCALC, A1, B1")
    text = once(
        text,
        "struct SyncModel {",
        """inline uint64_t storeWaits=0, vectorWhileStorePending=0;
    struct SyncModel {
        Range l1B{0,0};
        std::deque<Dma> stores;
        uint64_t storeReady=0;""",
    )
    text = once(
        text,
        "inline void DrainDma(",
        """inline void FinishStores() {
        for(auto task:cpuSync.stores) std::memcpy(reinterpret_cast<void*>(task.target.begin),task.source,task.target.bytes);
        cpuSync.stores.clear();
    }
    inline void DrainDma(""",
    )
    text = once(
        text,
        "    CheckReadable(src, false);",
        """    CheckReadable(src, false);
        if(!cpuSync.stores.empty()) ++vectorWhileStorePending;
        for(auto task:cpuSync.stores)
            Require(!Overlaps(dst,Bytes(task.source,task.target.bytes)), "W16 overwritten during UB-to-L1 copy");""",
    )
    text = once(
        text, "        DrainDma(cpuSync.dmaIssued);", "        FinishStores(); DrainDma(cpuSync.dmaIssued);"
    )
    text = once(
        text,
        "        buffer.Allocate(bytes);",
        """        buffer.Allocate(bytes);
        if constexpr(P==TPosition::B1) cpuSync.l1B=Bytes(buffer.template Get<uint8_t>(bytes).data,bytes);""",
    )
    text = once(
        text,
        "    for (auto range : cpuSync.vectorAccesses) Require(!Overlaps(target, range)",
        """    if (cpuSync.l1B.bytes && Overlaps(target,cpuSync.l1B)) {
        Require(cpuSync.storeReady==cpuSync.vectorIssued,"UB-to-L1 copy before final Mul");
        cpuSync.stores.push_back({target,src.data,0}); ++cpuMetrics.copies;return;
    }
    for (auto range : cpuSync.vectorAccesses) Require(!Overlaps(target, range)""",
    )
    text = once(
        text,
        "    else assert((sizeof(T)==1 && params.blockCount==8",
        """    else if(sizeof(T)==2 && params.blockCount==16)
        assert(params.blockLen==1 && params.srcStride==159 && params.dstStride==0);
    else assert((sizeof(T)==1 && params.blockCount==8""",
    )
    h.write_text(text)
    h = inc / "lib/matmul_intf.h"
    text = once(h.read_text(), "enum class CubeFormat { ND };", "enum class CubeFormat { ND, NZ };")
    text = once(
        text,
        "static constexpr auto position = P;",
        "static constexpr auto position = P; static constexpr auto format=F; static constexpr bool trans=TRANS;",
    )
    text = once(
        text,
        "assert(transpose);",
        "static_assert(B::format==CubeFormat::NZ && !B::trans); assert(!transpose);",
    )
    # Independent SDK local NZ contract for logical [K,N] (FP16 C0=16).
    text = once(
        text,
        "w_.GetValue(n * tiling_.singleK + term.first)",
        "w_.GetValue(((n/16)*tiling_.singleK+term.first)*16+n%16)",
    )
    text = once(
        text,
        "assert(tiling_.singleM == 80 && tiling_.orgKa == 128);",
        "static_assert(A::position==AscendC::TPosition::A1 && A::format==CubeFormat::NZ); assert(tiling_.singleM==16 && tiling_.orgKa==2560);",
    )
    text = once(
        text,
        "x_.GetValue(m * tiling_.orgKa + k)",
        "x_.GetValue(A::position==AscendC::TPosition::A1 ? ((k/16)*tiling_.singleM+m)*16+k%16 : m*tiling_.orgKa+k)",
    )
    h.write_text(text)
    return inc


class CubePanels(unittest.TestCase):
    def test_coordinates_and_scale_bits(self):
        rng = np.random.default_rng(23)
        k, n = 2560, 128
        q = rng.integers(-128, 128, size=(n, k), dtype=np.int8)
        gn = rng.integers(0, 65536, size=(k // 128, n), dtype=np.uint16)
        nz = q.reshape(n // 16, 16, k // 32, 32).transpose(2, 0, 1, 3).tobytes()
        for p in panels.PANELS:
            codes, s, _ = panels.pack_inputs(nz, gn.tobytes(), p, k=k, n=n)
            w = np.frombuffer(codes, dtype=np.int8).reshape(n // 64, k // p, 4, p, 16)
            sc = np.frombuffer(s, dtype=np.uint16).reshape(n // 64, k // 128, 64)
            for channel in range(n):
                np.testing.assert_array_equal(
                    q[channel], w[channel // 64, :, (channel % 64) // 16, :, channel % 16].reshape(k)
                )
                np.testing.assert_array_equal(gn[:, channel], sc[channel // 64, :, channel % 64])
        with self.assertRaises(ValueError):
            panels.pack_inputs(nz, gn.tobytes(), 384, k=k, n=n)

    def test_actual_kernels_rounding_and_bank_lifetimes(self):
        n, k, m = panels.N, panels.K, panels.M
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inc = cpu_headers(root)
            rng = np.random.default_rng(367)
            q = rng.integers(-127, 128, (n, k), dtype=np.int8)
            gn = rng.uniform(0.004, 0.25, (k // 128, n)).astype("<f2")
            nz = q.reshape(n // 16, 16, k // 32, 32).transpose(2, 0, 1, 3).tobytes()
            x = np.zeros((m, k), dtype="<f2")
            acc = np.zeros((m, n), dtype=np.float32)
            for row in range(m):
                for g in range(k // 128):
                    kk = g * 128 + (row * 13 + g * 31) % 128
                    x[row, kk] = np.float16((-1 if (g + row) % 2 else 1) * (1 + row / 31))
                    w = (q[:, kk].astype(np.float16) * gn[g]).astype(np.float32)
                    acc[row] += np.float32(x[row, kk]) * w
            x.tofile(root / "x")
            acc.astype("<f2").tofile(root / "reference")
            for p, banks in [(p, 1) for p in panels.PANELS] + [(panels.K, 2)]:
                with self.subTest(panel=p, w16_banks=banks):
                    codes, sc, _ = panels.pack_inputs(nz, gn.tobytes(), p)
                    (root / "codes").write_bytes(codes)
                    (root / "scales").write_bytes(sc)
                    (root / "panel_spec.h").write_bytes(panels.panel_header(p, banks))
                    exe = root / "portable"
                    compiler = [
                        "c++",
                        "-std=c++17",
                        "-O2",
                        "-ffp-contract=off",
                        "-I" + str(inc),
                        "-I" + str(root),
                        "-I" + str(HERE / "op_kernel"),
                        "-I" + str(HERE),
                        str(HERE / "test/portable_gate.cpp"),
                        "-o",
                        exe,
                    ]
                    subprocess.run(compiler, check=True)
                    subprocess.run([exe, root], check=True)
                    if p == 2560:
                        source = (HERE / "op_kernel/full_l1.cpp").read_text()
                        for fence, hazard in (
                            (
                                (
                                    "if(slab!=0) Pair<HardEvent::MTE3_V>(weightStored);"
                                    if banks == 1
                                    else "if (slab == 1 || slab == 2) Pair<HardEvent::MTE3_V>(weightStored);"
                                ),
                                "W16 overwritten during UB-to-L1 copy",
                            ),
                            (
                                "Pair<HardEvent::V_MTE2>(rawConsumed);",
                                "raw bank reused before vector completion",
                            ),
                        ):
                            (root / "full_l1.cpp").write_text(once(source, fence, "// CPU fault injection"))
                            subprocess.run(compiler, check=True)
                            fault = subprocess.run(
                                [exe, root],
                                cwd=root,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                text=True,
                            )
                            self.assertNotEqual(fault.returncode, 0)
                            self.assertIn(hazard, fault.stdout)
                        (root / "full_l1.cpp").unlink()

    def test_paired_statistics(self):
        mode = "panel-k512"
        rows = []
        for i in range(3):
            for m in ((mode, TILE_MODE) if i % 2 == 0 else (TILE_MODE, mode)):
                rows.append(
                    dict(
                        mode=m,
                        round=i,
                        folder=str((m, i)),
                        bit_mismatches=0,
                        launch_authorized=True,
                        profiled=False,
                        validation={"timing": {"samples_ms": [1 if m == mode else 2] * 30}},
                    )
                )
        self.assertEqual(pair_statistics(rows, mode)["decision"], "GO")
        rows[2]["validation"]["timing"]["samples_ms"] = [0.5] * 30
        self.assertEqual(pair_statistics(rows, mode)["decision"], "NO_GO")
        with self.assertRaises(ValueError):
            pair_statistics(rows[:-1], mode)

    def test_w16_pair_uses_new_single_w_control_samples(self):
        control, mode = "panel-k2560", "panel-k2560-w2"
        rows = []
        for i in range(3):
            for m in ((control, mode) if i % 2 == 0 else (mode, control)):
                rows.append(
                    dict(
                        mode=m,
                        round=i,
                        folder=str((m, i)),
                        bit_mismatches=0,
                        launch_authorized=True,
                        profiled=False,
                        validation={"timing": {"samples_ms": [0.50 if m == control else 0.46] * 30}},
                    )
                )
        self.assertEqual(pair_statistics(rows, mode, control)["decision"], "GO")
        with self.assertRaisesRegex(ValueError, "invalid paired schedule"):
            pair_statistics(rows, mode)
        with self.assertRaisesRegex(ValueError, "identical"):
            pair_statistics(rows, mode, mode)

    @unittest.skipUnless(os.getenv("GATE_TEST_GE_DESCRIPTOR"), "requires CANN GE descriptor")
    def test_compiled_input_contracts(self):
        from air import protobuf_classes
        from fresh_run import data_files

        types = protobuf_classes(Path(os.environ["GATE_TEST_GE_DESCRIPTOR"]).read_bytes())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files, _ = data_files(root, None)
            stress = boundary_case(root / "stress", files)
            self.assertEqual(stress["w_nz.bin"], files["w_nz.bin"])
            self.assertNotEqual(stress["golden.bin"], files["golden.bin"])
            for panel in panels.PANELS:
                d = root / str(panel)
                packed = panels.make_air(d, types, files, panel)
                msg = types[0].FromString((d / "gate.air").read_bytes())
                nodes = {node.name: node for node in msg.graph[0].op}
                op = nodes["DFlashGroupQuantLinear"]
                self.assertEqual(op.type, panels.OP)
                for i, name in enumerate(("weight", "scale"), 1):
                    node = nodes[name]
                    shape = packed["physical_shapes"][i - 1]
                    self.assertTrue(node.attr["_out_shape_locked"].b)
                    self.assertEqual(list(node.output_desc[0].shape.dim), list(shape))
                    self.assertEqual(list(node.output_desc[0].attr["origin_shape"].list.i), list(shape))
                    self.assertEqual(node.attr["value"].t.desc, op.input_desc[i])
                    self.assertEqual(
                        node.attr["value"].t.data, panels.c.verify(packed["constants"][i - 1]).read_bytes()
                    )
