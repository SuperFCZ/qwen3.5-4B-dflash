import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import gate_next as g
from gate_input import reference_graph, single_graph
from air import protobuf_classes
from layout import once

sys.path.insert(0, str(g.q.HERE / "test"))
from test_q import q_cpu_headers


class GateNext(unittest.TestCase):
    def test_full_weight_schedule_bits_and_dma_handoffs(self):
        m, k, n = g.SHAPE
        rng = np.random.default_rng(31019456)
        weight = rng.integers(-128, 128, (n, k), dtype=np.int8)
        gn = rng.uniform(0.005, 0.035, (k // 128, n)).astype("<f2")
        nz = weight.reshape(n // 16, 16, k // 32, 32).transpose(2, 0, 1, 3).tobytes()
        w, s, _ = g.gate.pack_cube_panels(nz, gn.tobytes(), k, k=k, n=n)
        x = np.zeros((m, k), "<f2")
        acc = np.zeros((m, n), np.float32)
        for row in range(m):
            for group in range(k // 128):
                j = group * 128 + (11 * group + 13 * row) % 128
                x[row, j] = (-1 if (row + group) % 2 else 1) * (1 + row / 37)
                acc[row] += np.float32(x[row, j]) * (weight[:, j].astype("<f2") * gn[group]).astype(
                    np.float32
                )
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            inc = q_cpu_headers(d)
            h = inc / "kernel_operator.h"
            text = once(
                h.read_text(),
                "params.blockCount==160)\n        assert(params.blockLen==1 && params.srcStride==0 && params.dstStride==15);",
                "params.blockCount==16)\n        assert(params.blockLen==1 && params.srcStride==159 && params.dstStride==0);",
            )
            text = once(
                text,
                "struct SyncModel {",
                "inline std::vector<size_t> pendingAtFullFence;\nstruct SyncModel {",
            )
            h.write_text(
                once(
                    text,
                    "++cpuMetrics.fullBarriers;",
                    "pendingAtFullFence.push_back(cpuSync.pending.size()); ++cpuMetrics.fullBarriers;",
                )
            )
            x.tofile(d / "x")
            acc.astype("<f2").tofile(d / "gold")
            (d / "w").write_bytes(w)
            (d / "s").write_bytes(s)
            (d / "panel_spec.h").write_bytes(g.gate.panel_header(k, 2))
            control = (g.gate.HERE / "op_kernel/full_l1.cpp").read_text()
            activation = control[
                control.index("__aicore__ inline void CacheActivation(") : control.index("} // namespace")
            ]
            for index, mode in enumerate((g.CONTROL, *g.CANDIDATES)):
                src = control if mode == g.CONTROL else g.device_source(mode)
                self.assertIn(activation, src)
                impl = d / "gate_impl.cpp"
                impl.write_text(src)
                exe = d / "run"
                cc = [
                    "c++",
                    "-std=c++17",
                    "-ffp-contract=off",
                    "-O2",
                    f"-DTEST_SCHEDULE={index}",
                    "-I" + str(inc),
                    "-I" + str(d),
                    "-I" + str(g.gate.HERE),
                    str(HERE / "test/portable.cpp"),
                    "-o",
                    str(exe),
                ]
                subprocess.run(cc, check=True)
                for fail in (-1, 0, 18, 37):
                    subprocess.run([exe, d, str(fail)], check=True)
                if mode == g.EARLY:
                    impl.write_text(
                        once(
                            src,
                            "Pair<HardEvent::V_MTE2>(rawConsumed); // retire the last current raw and scale reads",
                            "// omitted last-Vector -> next-MTE2 protection in CPU fault injection",
                        )
                    )
                    subprocess.run(cc, check=True)
                    bad = subprocess.run([exe, d, "-1"], cwd=d, capture_output=True, text=True)
                    self.assertNotEqual(bad.returncode, 0)
                    self.assertRegex(bad.stderr, "raw overwrite|raw bank reused")

    def test_formal_pairs_use_fresh_uninstrumented_samples(self):
        from run_next import benchmark

        for reverse in (True, False):

            def call(folder, mode, files, timed, index):
                self.assertTrue(timed)
                return dict(
                    folder=str(folder),
                    mode=mode,
                    round=index,
                    launch_authorized=True,
                    profiled=False,
                    bit_mismatches=0,
                    validation={"timing": {"samples_ms": [1.0 if mode == g.CONTROL else 0.95] * 30}},
                )

            with tempfile.TemporaryDirectory() as d:
                result = benchmark(Path(d), g.CONTROL, g.NEXT, call, {}, reverse)
            expected = (g.NEXT, g.CONTROL) if reverse else (g.CONTROL, g.NEXT)
            self.assertEqual([r["mode"] for r in result["runs"]], list(expected + expected[::-1] + expected))
            self.assertEqual(result["statistics"]["decision"], "GO")
            self.assertEqual(result["statistics"]["variants"][g.NEXT]["count"], 90)

    @unittest.skipUnless(
        os.getenv("DFLASH_GE_DESCRIPTOR")
        and importlib.util.find_spec("torch")
        and importlib.util.find_spec("transformers"),
        "CPU Draft frontend and CANN GE descriptor",
    )
    def test_real_graph_cut_and_standalone_packing(self):
        import torch

        sys.path.insert(0, str(g.OPS / "draft_full_panels/test"))
        from test_full import fixture, execute, fe

        torch.manual_seed(31038)
        torch.set_num_threads(1)
        net, ops = fixture()
        net = fe.export_net(net, ops)
        types = protobuf_classes(Path(os.environ["DFLASH_GE_DESCRIPTOR"]).read_bytes())
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            tables = {}
            for name in ("embedding", "head"):
                f = d / name
                torch.randn(net.config.vocab_size, net.config.hidden_size).half().numpy().tofile(f)
                tables[name] = dict(fp16=g.c.identity(f))
            graph, abi = fe.trace_graph(types, net, ops, dict(target=tables), 16, 96)
            site = graph.projections[g.ROLE]
            shape = tuple(site[a] for a in ("m", "k", "n"))
            ref = reference_graph(
                d, types, graph.model, dict(projections=graph.projections, modules=graph.owners), shape=shape
            )
            snapshot = types[0].FromString(g.c.verify(ref["reference_air"]).read_bytes())
            inputs = []
            scalars = dict(anchor=42, start_position=4, valid_rows=16, proposal_count=15)
            for t in abi["inputs"]:
                inputs.append(
                    torch.tensor([scalars[t["name"]]], dtype=getattr(torch, t["dtype"]))
                    if t["name"] in scalars
                    else torch.randn(t["shape"]).half()
                )
            # Tap the uncut original for the two boundary tensors, Gate input and raw Gate output.
            end = graph.graph.op[-1]
            del end.input[:]
            names = list(ref["cut_nodes"])
            end.input.extend([x + ":0" for x in [*names, ref["x_node"], site["node"]]])
            refs = execute(graph, inputs)
            replay = dict(zip((ref["cut_nodes"][n] for n in names), refs[:2]))
            x, y = execute(
                SimpleNamespace(graph=snapshot.graph[0]),
                [replay[t["name"]] for t in ref["reference_abi"]["inputs"]],
            )
            self.assertTrue(torch.equal(x.view(torch.uint8), refs[2].view(torch.uint8)))
            self.assertTrue(torch.equal(y.view(torch.uint8), refs[3].view(torch.uint8)))
            gate, w, s = (graph.ops[site[n]] for n in ("node", "weight", "scale"))
            fe.pack_projection(gate, w, s, site, g.gate.OP, shape[1])
            air, _ = single_graph(d, types, graph.model, site, ref["weights"], shape=shape)
            mm = types[0].FromString(g.c.verify(air).read_bytes())
            self.assertEqual(
                [n.type for n in mm.graph[0].op], ["Data", "Const", "Const", g.gate.OP, "NetOutput"]
            )
            (output,) = execute(SimpleNamespace(graph=mm.graph[0]), [x])
            self.assertTrue(torch.equal(y.view(torch.uint8), output.view(torch.uint8)))
            s.attr["value"].t.data = bytes(len(s.attr["value"].t.data))
            with self.assertRaisesRegex(ValueError, "packing"):
                single_graph(d, types, graph.model, site, ref["weights"], shape=shape)


if __name__ == "__main__":
    unittest.main()
