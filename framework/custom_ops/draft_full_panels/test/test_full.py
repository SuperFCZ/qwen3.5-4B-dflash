"""CPU-only complete five-layer export and differential execution using a reduced-width checkpoint fixture."""

from pathlib import Path
import os
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import numpy as np
import torch
import model_frontend as fe
from models.dflash_v1.dflash_config import Qwen35DFlashConfig
from models.dflash_v1.draft_quantization import GroupQuantLinear
from models.dflash_v1.modeling_dflash import WeightOnlyLinear
from ge_frontend import GE, PROTO
from full_replay import capture_reference, compare_rounds
from module_profile import scopes, union, summarize


def fixture():
    cfg = Qwen35DFlashConfig(
        hidden_size=128,
        intermediate_size=256,
        vocab_size=256,
        num_hidden_layers=5,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        num_target_layers=10,
        target_layer_ids=(0, 2, 4, 6, 8),
        layer_types=(
            "full_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
        ),
        block_size=16,
        mask_token_id=255,
        rms_norm_eps=1e-6,
        rope_theta=1e7,
        max_position_embeddings=256,
        sliding_window=16,
        use_sliding_window=True,
        attention_bias=False,
        attention_dropout=0.0,
        hidden_act="silu",
        dtype="float16",
    )
    ops = fe.ExportOps()
    net = fe.DFlashDraftModel(cfg, ops, dtype=torch.float16)
    for name, m in list(net.named_modules()):
        if isinstance(m, WeightOnlyLinear):
            n, k = m.weight.shape
            mod = GroupQuantLinear(
                torch.randint(-4, 5, (n, k), dtype=torch.int8),
                torch.rand(n, k // 128).half() * 0.008,
                bits=8,
                in_features=k,
                ops=ops,
            )
            path, _, key = name.rpartition(".")
            setattr(net.get_submodule(path) if path else net, key, mod)
    return net, ops


def execute(graph, inputs):
    vals = {}
    dtype = {v: k for k, v in PROTO.items()}
    for n in graph.graph.op:
        t = n.type
        a = n.attr
        args = [vals[s.rsplit(":", 1)[0]] for s in n.input]
        x = args[0] if args else None
        if t == "NetOutput":
            return args
        if t == "Data":
            v = inputs[a["index"].i]
        elif t == "FileConstant":
            v = torch.from_numpy(
                np.fromfile(a["file_path"].s.decode(), np.float16).reshape(tuple(a["shape"].list.i))
            )
        elif t == "Const":
            pt = a["value"].t
            v = torch.frombuffer(bytearray(pt.data), dtype=dtype[pt.desc.dtype]).reshape(
                tuple(pt.desc.shape.dim)
            )
        elif t == "Cast":
            v = x.to({i: k for k, i in GE.items()}[a["dst_type"].i])
        elif t == "Reshape":
            v = x.reshape(tuple(args[1].tolist()))
        elif t == "BroadcastTo":
            v = x.expand(tuple(args[1].tolist()))
        elif t == "Slice":
            v = x[tuple(slice(i, i + l) for i, l in zip(args[1].tolist(), args[2].tolist()))]
        elif t == "Transpose":
            v = x.permute(tuple(args[1].tolist()))
        elif t == "ConcatD":
            v = torch.cat(args, a["concat_dim"].i)
        elif t == "ReduceMean":
            v = x.mean(tuple(args[1].tolist()), keepdim=a["keep_dims"].b)
        elif t == "SelectV2":
            v = torch.where(*args)
        elif t == "GatherV2":
            v = args[0][args[1]]
        elif t == "SoftmaxV2":
            v = x.softmax(a["axes"].list.i[0])
        elif t == "ArgMaxV2":
            v = x.argmax(int(args[1])).int()
        elif t == "TensorMove":
            v = x.clone()
        elif t == "ScatterNdUpdate":
            v = x.index_copy(0, args[1][:, 0].long(), args[2])
        elif t == "BatchMatMulV2":
            v = x @ args[1]
        elif t == "MatMul":
            v = x @ (args[1].T if a["transpose_x2"].b else args[1])
        elif t in (fe.RAW, fe.GATE, fe.DOWN):
            codes, s = args[1:]
            k = x.shape[-1]
            if t == fe.RAW:
                n_out = s.shape[1]
                q = codes.permute(1, 2, 0, 3).reshape(n_out, k)
            else:
                n_out = codes.shape[0] * 64
                panel_k = codes.shape[-1] // 64
                q = (
                    codes.reshape(n_out // 64, k // panel_k, 4, panel_k, 16)
                    .permute(0, 2, 4, 1, 3)
                    .reshape(n_out, k)
                )
                s = s.transpose(0, 1).reshape(k // 128, n_out)
            w = (q.reshape(n_out, k // 128, 128).half() * s.T.unsqueeze(-1)).reshape(n_out, k)
            v = torch.nn.functional.linear(x, w)
        else:
            unary = dict(
                Sin=torch.sin,
                Cos=torch.cos,
                Rsqrt=torch.rsqrt,
                Swish=torch.nn.functional.silu,
                Square=torch.square,
                Neg=torch.neg,
                LogicalNot=torch.logical_not,
            )
            binary = dict(
                Add=torch.add,
                Sub=torch.sub,
                Mul=torch.mul,
                LogicalAnd=torch.logical_and,
                Less=torch.lt,
                Greater=torch.gt,
                GreaterEqual=torch.ge,
                LessEqual=torch.le,
            )
            v = unary[t](x) if t in unary else binary[t](*args)
        if list(v.shape) != list(n.output_desc[0].shape.dim) or v.dtype != dtype[n.output_desc[0].dtype]:
            raise AssertionError((t, n.name, v.dtype, v.shape))
        vals[n.name] = v
    raise AssertionError("no output")


class TestFull(unittest.TestCase):
    def test_modules(self):
        graph = {
            "modules": {"g": "layer0.gate_up", "d": "layer0.down", "h": "head"},
            "node_types": {"h": "MatMul", "g": fe.GATE, "d": fe.DOWN},
        }
        _, m = scopes(
            graph,
            [
                dict(
                    name="f",
                    type="T",
                    attributes=[dict(key="_datadump_original_op_names", value={"list": {"s": ["g", "d"]}})],
                ),
                dict(name="h", type="MatMul", attributes=[]),
            ],
        )
        self.assertEqual(m, {"f": "Fused/layer0.down,layer0.gate_up", "h": "head.matmul"})
        self.assertEqual(union([(0, 6), (4, 9), (20, 21)]), 10)
        a, _ = summarize([dict(module="a", lo=0, hi=1000, call=i) for i in range(7, 37)], ["a", "b"])
        self.assertEqual((a[0]["p50_task_sum_ms"], a[1]["p50_task_sum_ms"]), (1, None))

    def test_capture_checks_whole_kv_and_untouched_rows(self):
        from joint import c

        with tempfile.TemporaryDirectory() as root:
            d = Path(root)
            e = dict(
                status="PASS",
                cpu_fallback=False,
                reference_capture=True,
                golden_match=False,
                verification_calls=2,
                guards=True,
                readonly=True,
                repeat=True,
                poison_verified=True,
                sync_completed=True,
            )
            c.write(d / "execution.json", e)
            original = np.zeros((1, 2, 6, 4), np.float16)
            updated = original.copy()
            updated[:, :, 1:3] = 1
            original.tofile(d / "input.bin")
            io = dict(
                files={"d0_key": c.identity(d / "input.bin")},
                source="REAL_CHECKPOINT_SEEDED_INPUTS",
                scalars={"start_position": 1, "proposal_count": 15},
            )
            abi = dict(
                outputs=[dict(name="draft_top1", shape=(1, 15)), dict(name="d0_key", shape=original.shape)]
            )
            for i in (0, 1):
                np.ones((1, 15), np.int64).tofile(d / f"draft_top1-actual-{i}.bin")
                updated.tofile(d / f"d0_key-actual-{i}.bin")
            self.assertEqual(set(capture_reference(d, io, abi, 2)), {"draft_top1", "d0_key"})
            updated[:, :, 5] = 2
            for i in (0, 1):
                updated.tofile(d / f"d0_key-actual-{i}.bin")
            with self.assertRaisesRegex(ValueError, "untouched KV"):
                capture_reference(d, io, abi, 2)

    def test_compiled_projection_provenance(self):
        from compile_model import projection_sites

        roles = {str(i): {"node": f"p{i}"} for i in range(26)}
        nodes = [
            dict(
                name=f"c{i}",
                type=fe.RAW,
                attributes=[dict(key="_datadump_original_op_names", value={"list": {"s": [f"p{i}"]}})],
            )
            for i in range(26)
        ]
        self.assertEqual(projection_sites(nodes, roles)["p9"]["name"], "c9")
        nodes[-1]["attributes"][0]["value"]["list"]["s"].append("p0")
        with self.assertRaisesRegex(ValueError, "fused projection"):
            projection_sites(nodes, roles)

    def test_timing_keeps_profile_samples_out(self):
        def run(t):
            return dict(profiled=False, validation={"timing": dict(samples_ms=[t] * 30, median_ms=t)})

        rows = [dict(baseline=run(2), optimized=run(1)) for _ in range(3)]
        self.assertEqual(compare_rounds(rows)["ratio"], 2)
        rows[-1]["optimized"]["profiled"] = True
        with self.assertRaisesRegex(ValueError, "profiling samples"):
            compare_rounds(rows)

    @unittest.skipUnless(
        os.environ.get("DFLASH_GE_DESCRIPTOR"), "set DFLASH_GE_DESCRIPTOR to a CANN ge_ir FileDescriptorSet"
    )
    def test_actual_draft_forward(self):
        torch.set_num_threads(1)
        torch.manual_seed(772)
        from models.dflash_v1.dflash_ops import TorchDFlashOps

        types = fe.protobuf_classes(Path(os.environ["DFLASH_GE_DESCRIPTOR"]).read_bytes())
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            net, ops = fixture()
            # Same checkpoint, two independent evaluations: native PyTorch and the serialized GE model.
            import copy

            eager = copy.deepcopy(net)
            from qwen35_dflash.ascend310p.quant_factory import AirDFlashOps

            class ReferenceOps(AirDFlashOps):
                rms_norm = TorchDFlashOps.rms_norm

            eagerops = ReferenceOps(attention_matmul_dtype="float16", quant_matmul_backend="dequant")
            for m in eager.modules():
                if hasattr(m, "ops"):
                    m.ops = eagerops
            emb = torch.nn.Embedding(256, 128, dtype=torch.float16)
            head = torch.nn.Linear(128, 256, bias=False, dtype=torch.float16)
            reference = fe.DraftGraph(eager, emb, head)
            source = {"target": {}}
            for name, w in (("head", head.weight), ("embedding", emb.weight)):
                path = folder / name
                w.detach().numpy().tofile(path)
                source["target"][name] = {"fp16": {"path": str(path)}}
            exported = fe.export_net(net, ops)
            for gear in (16, 64):
                g, abi = fe.trace_graph(types, exported, ops, source, gear, 96)
                args = [
                    torch.rand(1, gear, 640).half(),
                    torch.tensor([7]),
                    torch.tensor([gear - 1], dtype=torch.int16),
                    torch.tensor([11]),
                    torch.tensor([12], dtype=torch.int16),
                ] + [torch.rand(1, 2, 96, 32).half() for _ in range(10)]
                with torch.inference_mode():
                    expected = reference(*args)
                    actual = execute(g, args)
                before = fe.node_hashes(g)
                expected_changes = set()
                for role, r in g.projections.items():
                    if role.endswith((".gate_up", ".down")):
                        parts = [g.ops[r[k]] for k in ("node", "weight", "scale")]
                        typ = fe.GATE if role.endswith(".gate_up") else fe.DOWN
                        fe.pack_projection(*parts, r, typ, r["k"] if typ == fe.GATE else r["k"] // 2)
                        expected_changes.update(p.name for p in parts)
                after = fe.node_hashes(g)
                self.assertEqual({n for n in before if before[n] != after[n]}, expected_changes)
                self.assertEqual(len(expected_changes), 30)
                optimized = execute(g, args)
                self.assertEqual(len(g.projections), 26)
                self.assertEqual(len(actual), 11)
                for spec, a, b, opt in zip(abi["outputs"], expected, actual, optimized):
                    self.assertTrue(torch.equal(a.view(torch.uint8), opt.view(torch.uint8)), spec["name"])
                    self.assertTrue(torch.equal(a.view(torch.uint8), b.view(torch.uint8)), spec["name"])


if __name__ == "__main__":
    unittest.main()
