"""Run the actual five-layer DraftGraph.forward with a static GE export backend and real CPU-loaded weights."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
OPS = HERE.parent
REPO = OPS.parents[1]
sys.path[:0] = [str(REPO), str(REPO / "framework/python"), str(OPS / "draft_gate_bandwidth")]
from gate_panel import pack_cube_panels
from air import protobuf_classes
from ge_frontend import Graph
from models.dflash_v1.draft_quantization import load_quantized_draft
from models.dflash_v1.modeling_dflash import DFlashDraftModel
from models.dflash_v1.dflash_ops import TorchDFlashOps
from qwen35_dflash.ascend310p.incremental import DraftGraph
from qwen35_dflash.ascend310p.quant_factory import AirDFlashOps
from qwen35_dflash.ascend310p.draft_ac_export import projection_modules
from qwen35_dflash.ascend310p.utils import sha256_file

RAW = "DFlashGroupQuantLinear"
GATE = "DFlashGatePanel"
DOWN = "DFlashDownPanel"


def identity(p):
    p = Path(p).resolve(strict=True)
    return dict(path=str(p), bytes=p.stat().st_size, sha256=sha256_file(p))


def target_table(target, key, shard, folder, shape):
    from safetensors import safe_open

    path = (target / shard).resolve(strict=True)
    if target.resolve() not in path.parents:
        raise ValueError("Target shard escapes checkpoint")
    out = folder / (key + ".fp16.bin")
    with safe_open(path, framework="pt", device="cpu") as f, out.open("xb") as dst:
        tensor = f.get_slice(key)
        if list(tensor.get_shape()) != list(shape) or tensor.get_dtype() not in ("BF16", "F16", "F32"):
            raise ValueError("Target embedding/head shape or dtype: " + key)
        for i in range(0, shape[0], 2048):
            data = tensor[i : i + 2048].to(torch.float16).contiguous().numpy()
            if not np.isfinite(data).all():
                raise ValueError("nonfinite FP16 Target table: " + key)
            dst.write(memoryview(data).cast("B"))
    return dict(key=key, shard=str(path), fp16=identity(out))


def tables(folder, target, config):
    folder.mkdir()
    cfg = json.loads((target / "config.json").read_text())
    txt = cfg.get("text_config", cfg)
    index = target / "model.safetensors.index.json"
    mapping = json.loads(index.read_text())["weight_map"]
    shape = (config.vocab_size, config.hidden_size)
    if (txt["vocab_size"], txt["hidden_size"]) != shape:
        raise ValueError("Target and Draft dimensions differ")
    embeddings = [
        k for k in mapping if k in ("model.language_model.embed_tokens.weight", "model.embed_tokens.weight")
    ]
    if len(embeddings) != 1:
        raise ValueError("expected one Target language embedding; found " + str(embeddings))
    embedding = embeddings[0]
    heads = [k for k in mapping if k == "lm_head.weight" or k.endswith(".lm_head.weight")]
    if len(heads) == 1:
        head = heads[0]
    elif not heads and txt.get("tie_word_embeddings", cfg.get("tie_word_embeddings")) is True:
        head = embedding
    else:
        raise ValueError("ambiguous Target head; an explicit head or tied embeddings are required")
    saved = {k: target_table(target, k, mapping[k], folder, shape) for k in dict.fromkeys((embedding, head))}
    return dict(
        config=identity(target / "config.json"),
        index=identity(index),
        shards=[identity(p) for p in sorted({r["shard"] for r in saved.values()})],
        embedding=saved[embedding],
        head=saved[head],
    )


class ExportOps(AirDFlashOps):
    def __init__(self):
        super().__init__(attention_matmul_dtype="float16", quant_matmul_backend="dequant")
        self.g = None

    def rms_norm(self, value, weight, eps):
        # Same FP32 normalization, cast-before-gamma boundary as the receiver ADN lowering.
        return TorchDFlashOps.rms_norm(self, value, weight, eps)

    def attention(self, *args):
        with self.g.scope(f"layer{self.g.layer}.attention"):
            return super().attention(*args)

    def rotary(self, *args):
        with self.g.scope(f"layer{self.g.layer}.rope"):
            return super().rotary(*args)

    def swiglu(self, *args):
        with self.g.scope(f"layer{self.g.layer}.swiglu"):
            return super().swiglu(*args)

    def top1(self, *args):
        with self.g.scope("head"):
            ids = super().top1(*args)
        self.g.module = "head.output"
        return ids


class ExternalEmbedding(nn.Module):
    def __init__(self, ops):
        super().__init__()
        self.ops = ops

    def forward(self, ids):
        g = self.ops.g
        with g.scope("embedding"):
            table = self.weight
            return g.emit(
                "GatherV2",
                [("x", table), ("indices", ids), ("axis", torch.tensor(0, dtype=torch.int32))],
                ids.shape + (table.shape[1],),
                torch.float16,
                {"batch_dims": ("i", 0)},
            )


class ExportLinear(nn.Module):
    def __init__(self, ops, name, q, scales):
        super().__init__()
        self.ops = ops
        self.role = name
        self.n, self.k = q.shape
        self.nz = (
            q.reshape(self.n // 16, 16, self.k // 32, 32).permute(2, 0, 1, 3).contiguous().numpy().tobytes()
        )
        self.gn = scales.t().contiguous().numpy().tobytes()
        if len(self.gn) != 2 * self.n * self.k // 128:
            raise ValueError("non-Group128 scales")

    def forward(self, x):
        g = self.ops.g
        role = self.role
        with g.scope(role):
            flat = x.reshape(-1, self.k)
            stem = role.replace(".", "_")
            w = g.blob(
                self.nz,
                torch.int8,
                (self.k // 32, self.n // 16, 16, 32),
                name=stem + "_weight",
                layout="FRACTAL_NZ",
                origin=(self.n, self.k),
            )
            s = g.blob(self.gn, torch.float16, (self.k // 128, self.n), name=stem + "_scale")
            y = g.emit(
                RAW,
                [("x", flat), ("w_nz", w), ("s", s)],
                (flat.shape[0], self.n),
                torch.float16,
                name="proj_" + stem,
            )
            g.projections[role] = dict(
                node=y.name,
                weight=w.name,
                scale=s.name,
                m=flat.shape[0],
                k=self.k,
                n=self.n,
                nz_sha256=hashlib.sha256(self.nz).hexdigest(),
                gn_sha256=hashlib.sha256(self.gn).hexdigest(),
            )
            return y.reshape(*x.shape[:-1], self.n)


class ScopedNorm(nn.Module):
    def __init__(self, original, ops, scope, layer=None):
        super().__init__()
        self.original = original
        self.ops = ops
        self.scope = scope
        self.layer = layer

    def forward(self, x):
        g = self.ops.g
        if self.layer is not None:
            g.layer = self.layer
            g.module = f"layer{self.layer}.glue"
        with g.scope(self.scope):
            return self.original(x)


def model(draft_path, target_path, folder):
    ops = ExportOps()
    draft = load_quantized_draft(
        DFlashDraftModel, str(draft_path), variant="w8a16", ops=ops, device="cpu", dtype=torch.float16
    )
    cfg = draft.config
    if (
        cfg.num_hidden_layers,
        cfg.block_size,
        cfg.hidden_size,
        cfg.intermediate_size,
        cfg.vocab_size,
        cfg.feature_size,
        cfg.num_key_value_heads,
        cfg.head_dim,
    ) != (5, 16, 2560, 9728, 248320, 12800, 8, 128):
        raise ValueError("requires the pinned five-layer W8 DFlash checkpoint")
    source = dict(
        draft=draft.draft_quantization_audit, target=tables(folder / "target-tables", target_path, cfg)
    )
    return export_net(draft, ops), ops, source


def export_net(draft, ops):
    def row_update(x, idx, v):
        g = ops.g
        with g.scope(f"layer{g.layer}.kv_update"):
            # TorchAir's functional npu_scatter_nd_update copies the caller's cache before the ref update.
            private = x.unary("TensorMove")
            return g.emit(
                "ScatterNdUpdate", [("var", private), ("indices", idx), ("updates", v)], x.shape, x.dtype
            )

    graph = DraftGraph(draft, ExternalEmbedding(ops), nn.Module(), row_update=row_update, consume_source=True)
    del draft
    for name, parent, attr in projection_modules(graph):
        original = getattr(parent, attr)
        role = name.replace("layer-", "layer").replace("-", ".")
        setattr(parent, attr, ExportLinear(ops, role, original.integer_weight(), original.scales))
    graph.hidden_norm = ScopedNorm(graph.hidden_norm, ops, "fc_norm")
    graph.norm = ScopedNorm(graph.norm, ops, "final_norm")
    for i, layer in enumerate(graph.layers):
        for attr in ("input_norm", "q_norm", "k_norm", "post_norm"):
            setattr(
                layer,
                attr,
                ScopedNorm(
                    getattr(layer, attr), ops, f"layer{i}.{attr}", i if attr == "input_norm" else None
                ),
            )
    return graph


def pack_projection(op, w, s, r, typ, panel):
    from ge_frontend import desc

    packed, gn, _ = pack_cube_panels(
        w.attr["value"].t.data, s.attr["value"].t.data, panel, k=r["k"], n=r["n"]
    )
    shapes = ((r["n"] // 64, r["k"] // panel, 64 * panel), (r["n"] // 64, r["k"] // 128, 64))
    for j, (node, port, dtype, shape, data) in enumerate(
        zip((w, s), ("w_tiles", "s"), (torch.int8, torch.float16), shapes, (packed, gn)), 1
    ):
        out = node.output_desc[0]
        out.Clear()
        desc(out, port, dtype, shape)
        op.input_desc[j].CopyFrom(out)
        node.attr["value"].t.desc.CopyFrom(out)
        node.attr["value"].t.data = data
    op.type = typ
    return hashlib.sha256(packed).hexdigest(), hashlib.sha256(gn).hexdigest()


def rewrite(g):
    allowed = set()
    rows = []
    for role, r in g.projections.items():
        typ, panel = (
            (GATE, 2560)
            if role.endswith(".gate_up")
            else (DOWN, 4864) if role.endswith(".down") else (None, None)
        )
        if not typ:
            continue
        op, w, s = (g.ops[r[k]] for k in ("node", "weight", "scale"))
        if op.type != RAW or list(op.input[1:]) != [w.name + ":0", s.name + ":0"] or r["m"] != 16:
            raise ValueError("incorrect projection binding: " + role)
        nk = (r["k"], r["n"])
        if nk != ((2560, 19456) if typ == GATE else (9728, 2560)):
            raise ValueError("incorrect projection shape")
        wsha, ssha = pack_projection(op, w, s, r, typ, panel)
        allowed.update((op.name, w.name, s.name))
        rows.append(
            dict(
                role=role,
                node=op.name,
                type=typ,
                panel_k=panel,
                w16_banks=2 if typ == GATE else 1,
                packed_sha256=wsha,
                scale_sha256=ssha,
            )
        )
    if {r["role"] for r in rows} != {f"layer{i}.{t}" for i in range(5) for t in ("gate_up", "down")}:
        raise ValueError("requires all ten distinct MLP sites")
    return allowed, rows


def node_hashes(g):
    return {x.name: hashlib.sha256(x.SerializeToString(deterministic=True)).hexdigest() for x in g.graph.op}


def trace_graph(types, net, ops, source, gear, capacity):
    from ge_frontend import Value

    g = Graph(types, f"reconstructed_draft_c{gear}")
    ops.g = g
    names = ["features", "start_position", "valid_rows", "anchor", "proposal_count"] + [
        f"d{i}_{t}" for i in range(5) for t in ("key", "value")
    ]
    dt = [torch.float16, torch.int64, torch.int16, torch.int64, torch.int16] + [torch.float16] * 10
    shapes = (
        [(1, gear, net.config.feature_size)]
        + [(1,)] * 4
        + [(1, net.config.num_key_value_heads, capacity, net.config.head_dim)] * 10
    )
    args = [g.data(n, t, s, i) for i, (n, t, s) in enumerate(zip(names, dt, shapes))]
    with g.scope("embedding"):
        net.embedding.weight = g.external(
            source["target"]["embedding"]["fp16"]["path"],
            (net.config.vocab_size, net.config.hidden_size),
            "target_embedding",
        )
    with g.scope("head"):
        net.head.weight = g.external(
            source["target"]["head"]["fp16"]["path"],
            (net.config.vocab_size, net.config.hidden_size),
            "target_head",
        )
    with torch.inference_mode():
        result = net(*args)
    if not all(isinstance(v, Value) for v in result):
        raise ValueError("Draft forward returned a host output")
    outs = g.outputs(result, ["draft_top1", *names[5:]])
    abi = dict(
        inputs=[
            dict(name=n, dtype=str(t).split(".")[-1], shape=list(s)) for n, t, s in zip(names, dt, shapes)
        ],
        outputs=outs,
    )
    if (
        len(outs) != 11
        or outs[0] != dict(name="draft_top1", dtype="int64", shape=[1, 15])
        or any(o != a for o, a in zip(outs[1:], abi["inputs"][5:]))
    ):
        raise ValueError("Draft outputs differ from 15 IDs / 10 full KV ABI")
    if len(g.projections) != 26:
        raise ValueError("wrong projection inventory")
    return g, abi


def gear_graph(types, net, ops, source, gear, capacity, out):
    out.mkdir()
    g, abi = trace_graph(types, net, ops, source, gear, capacity)
    base = node_hashes(g)

    def save(name):
        path = out / (name + ".air")
        if g.model.ByteSize() > 2**31 - (64 << 20):
            raise ValueError("AIR exceeds safe protobuf size")
        path.write_bytes(g.model.SerializeToString(deterministic=True))
        return identity(path)

    raw = save("baseline")
    changed, rows = rewrite(g)
    after = node_hashes(g)
    if set(base) != set(after) or {n for n in base if base[n] != after[n]} != changed or len(changed) != 30:
        raise ValueError("non-target node, edge, descriptor or constant changed")
    opt = save("optimized")
    inventory = dict(
        abi=abi,
        gear=gear,
        baseline=raw,
        optimized=opt,
        modules=g.owners,
        projections=g.projections,
        replacements=rows,
        unchanged_nodes=len(base) - len(changed),
        node_types={n: op.type for n, op in g.ops.items()},
    )
    net.head.weight = None
    net.embedding.weight = None
    ops.g = None
    return inventory


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("descriptor", type=Path)
    p.add_argument("draft", type=Path)
    p.add_argument("target", type=Path)
    p.add_argument("--capacity", type=int, default=2112)
    p.add_argument("--gears", nargs="+", type=int, choices=(16, 64), default=[16, 64])
    args = p.parse_args()
    torch.set_num_threads(2)
    graph, ops, source = model(args.draft, args.target, args.out)
    types = protobuf_classes(args.descriptor.read_bytes())
    cases = {}
    for gear in args.gears:
        print("Exporting checkpoint DraftGraph.forward C" + str(gear), flush=True)
        cases[str(gear)] = gear_graph(types, graph, ops, source, gear, args.capacity, args.out / f"c{gear}")
    source.update(
        gears=cases,
        frontend="DraftGraph.forward; AirDFlashOps FP16 grouped attention; FP32 RMS reduction and cast-before-scale",
        source_files=[
            identity(p)
            for p in (
                Path(sys.modules[DraftGraph.__module__].__file__),
                Path(sys.modules[AirDFlashOps.__module__].__file__),
                Path(__file__),
                HERE / "ge_frontend.py",
            )
        ],
    )
    (args.out / "graphs.json").write_text(json.dumps(source, indent=2) + "\n")


if __name__ == "__main__":
    main()
