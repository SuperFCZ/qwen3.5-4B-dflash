"""Build the v4 Target directly from the intact W8A8 receiver GE graphs.

Only the feature selection, cache capacity and recurrent state boundary differ
between these pinned receiver artifacts and the documented Chunk-v4 contract.
Inputs and outputs are bound by port. Network weights and the GDR/CacheUpdate
algorithms stay on their original native GE operators.
"""

from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import numpy as np

from winners import c, require, UNION
from targets import RECEIVER
from air import protobuf_classes

SOURCE_SHA = {
    "target_prefill": "71e10234801b0ea05a1817a1f88a90b19040f20e51f14a83d1a77fbfa4803807",
    "target_verify": "00a7296549565338e32a15a5b55faf07210cb1eef3c4e82cb138f525146f41a7",
}
OLD_LAYERS = (1, 5, 9, 13, 17, 21, 25, 29)
DTYPE = {"float32": 1, "float16": 2, "int16": 5, "int32": 7, "int64": 8}
INT_DTYPE = {7: "<i4", 8: "<i8"}
OLD_LENGTH = 576
LENGTH = 2112
PAGE = 64


def digest(data):
    return hashlib.sha256(data).hexdigest()


def node_name(edge):
    return edge.rsplit(":", 1)[0]


def tensor_update(d, old, new, *, public=False):
    """Retain unknown intermediate shapes, update known physical/origin shapes."""
    before = DTYPE[old["dtype"]]
    after = DTYPE[new["dtype"]]
    require(d.dtype in (0, before, after), f"dtype conflicts with {old['name']}: {d.dtype}")
    d.dtype = after
    dims = list(d.shape.dim)
    require(not dims or dims in (old["shape"], new["shape"]), f"wrong shape on {old['name']}: {dims}")
    if dims or public:
        d.shape.dim[:] = new["shape"]
    if "origin_shape" in d.attr and d.attr["origin_shape"].list.i:
        shape = list(d.attr["origin_shape"].list.i)
        require(shape in (old["shape"], new["shape"]), "conflicting origin shape")
        d.attr["origin_shape"].list.i[:] = new["shape"]
    if "origin_data_type" in d.attr and before != after:
        orig = d.attr["origin_data_type"]
        if orig.HasField("s") and orig.s in (b"DT_FLOAT16", b"DT_FLOAT", b"FLOAT16", b"FLOAT"):
            orig.s = b"DT_FLOAT" if orig.s.startswith(b"DT_") else b"FLOAT"


def const_int(op):
    if op.type != "Const" or "value" not in op.attr:
        return None
    tensor = op.attr["value"].t
    dt = tensor.desc.dtype
    if dt not in INT_DTYPE:
        return None
    # TorchAir encodes even scalar constants as tensor byte strings.
    values = np.frombuffer(tensor.data, dtype=INT_DTYPE[dt])
    require(
        values.size == int(np.prod(tensor.desc.shape.dim, dtype=np.int64)), f"bad integral Const: {op.name}"
    )
    return values.reshape(tuple(tensor.desc.shape.dim))


def set_literal(op, arr):
    t = op.attr["value"].t
    prev = list(t.desc.shape.dim)
    if prev != list(arr.shape):
        for d in (t.desc, *op.output_desc):
            if list(d.shape.dim) == prev:
                d.shape.dim[:] = list(arr.shape)
            if "origin_shape" in d.attr and list(d.attr["origin_shape"].list.i) == prev:
                d.attr["origin_shape"].list.i[:] = list(arr.shape)
    t.data = arr.astype(INT_DTYPE[t.desc.dtype], copy=False).tobytes()


class TargetIR:
    def __init__(self, graph, old, new):
        self.g = graph
        self.old = old
        self.new = new
        self.nodes = {p.name: p for p in graph.op}
        require(len(self.nodes) == len(graph.op), "duplicate AIR node")
        self.before = {n: digest(p.SerializeToString(deterministic=True)) for n, p in self.nodes.items()}
        self.users = defaultdict(list)
        for p in graph.op:
            for i, v in enumerate(p.input):
                if v:
                    self.users[node_name(v)].append((p.name, i))
        (self.net,) = (p for p in graph.op if p.type == "NetOutput")
        data = sorted((p for p in graph.op if p.type in ("Data", "RefData")), key=lambda p: p.attr["index"].i)
        ports = len(old["outputs"])
        require(
            [p.attr["index"].i for p in data] == list(range(len(old["inputs"])))
            and len(self.net.input) >= ports
            and len(self.net.input_desc) in (ports, len(self.net.input))
            and all(e and not e.endswith(":-1") for e in self.net.input[:ports])
            and all(e.endswith(":-1") for e in self.net.input[ports:]),
            "source public port inventory",
        )
        self.data = {s["name"]: p for s, p in zip(old["inputs"], data)}
        self.out_index = {s["name"]: i for i, s in enumerate(old["outputs"])}
        self.known_changes = set()
        self.drop = set()
        self.insert = defaultdict(list)
        self.old_to_new = {}
        self.port_log = []

    def parent(self, p, slot=0):
        return self.nodes[node_name(p.input[slot])]

    def ancestors(self, names, stop=()):
        seen = set()
        stack = list(names)
        stop = set(stop)
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            if n not in stop:
                stack.extend(node_name(v) for v in self.nodes[n].input if v and not v.endswith(":-1"))
        return seen

    def rewire(self, op, slot, edge):
        before = op.input[slot]
        self.known_changes.add(op.name)
        self.port_log.append((op.name, slot, before, edge))
        op.input[slot] = edge

    def add_before(self, original, copy, name):
        require(name not in self.nodes, "Target graph name collision: " + name)
        copy.name = name
        self.nodes[name] = copy
        self.insert[original.name].append(copy)
        return copy

    def features(self, rows):
        idx = self.out_index["features"]
        last = self.parent(self.net, idx)
        cat = self.parent(last) if last.type == "PadV3" else last
        require(
            cat.type == "ConcatV2" and len(cat.input) == 9 and len(cat.input_desc) == 9,
            "old feature ConcatV2 layout",
        )
        axis = const_int(self.parent(cat, 8))
        require(axis is not None and int(axis.item()) in (-1, 2), "feature concat axis")
        old_ids = [self.parent(cat, i) for i in range(8)]
        require(
            all(p.type == "Identity" and len(p.input) == 1 for p in old_ids),
            "exported hidden-state clone missing",
        )
        # Each layer ends in where(valid, residual+MLP, zeros_like(...)). The one other
        # SelectV2 builds the attention mask; it never supplies the true residual Add.
        h = [
            p
            for p in self.g.op
            if p.type == "SelectV2" and len(p.input) == 3 and self.parent(p, 1).type == "Add"
        ]
        require(
            len(h) == 32 and [node_name(p.input[0]) for p in old_ids] == [h[i].name for i in OLD_LAYERS],
            "32-layer residual order or original eight feature taps differ",
        )
        for i in range(1, len(h)):
            outer = self.parent(h[i], 1)
            require(
                any(
                    node_name(v) == h[i - 1].name
                    for z in outer.input
                    if z and self.nodes[node_name(z)].type == "Add"
                    for v in self.nodes[node_name(z)].input
                ),
                "broken residual chain at layer " + str(i),
            )
        by_layer = dict(zip(OLD_LAYERS, old_ids))
        for i in UNION:
            if i in by_layer:
                continue
            n = type(cat)()
            n.CopyFrom(old_ids[0])
            n.input[0] = h[i].name + ":0"
            by_layer[i] = self.add_before(cat, n, "e2e_feature_layer_" + str(i))
        old_desc = type(cat.input_desc[0])()
        old_desc.CopyFrom(cat.input_desc[0])
        axis_desc = type(old_desc)()
        axis_desc.CopyFrom(cat.input_desc[-1])
        old_axis = cat.input[-1]
        del cat.input[:]
        del cat.input_desc[:]
        for j, i in enumerate(UNION):
            cat.input.append(by_layer[i].name + ":0")
            desc = cat.input_desc.add()
            desc.CopyFrom(old_desc)
            if old_desc.name.endswith("0"):
                desc.name = old_desc.name[:-1] + str(j)
        cat.input.append(old_axis)
        cat.input_desc.add().CopyFrom(axis_desc)
        if "N" in cat.attr:
            require(cat.attr["N"].i == 8, "wrong original feature arity")
            cat.attr["N"].i = len(UNION)
        resize_feature_ports(cat)
        self.known_changes.add(cat.name)

        def feature_spec(n, rows):
            return dict(name="features", dtype="float16", shape=[1, rows, n])

        old = feature_spec(len(OLD_LAYERS) * 2560, rows)
        new = feature_spec(len(UNION) * 2560, rows)
        tensor_update(cat.output_desc[0], old, new)
        if last is not cat:
            tensor_update(last.input_desc[0], old, new)
            tensor_update(last.output_desc[0], self.old["outputs"][idx], self.new["outputs"][idx])
            self.known_changes.add(last.name)
        tensor_update(self.net.input_desc[idx], self.old["outputs"][idx], self.new["outputs"][idx])
        self.known_changes.add(self.net.name)
        return {str(i): h[i].name for i in UNION}

    def recurrent(self, verify):
        oldins = {a["name"]: a for a in self.old["inputs"]}
        newins = {a["name"]: a for a in self.new["inputs"]}
        valid = self.data["valid_rows"].name + ":0"
        seen = set()
        rows = {}
        accepted = self.net.input[self.out_index["accepted_count"]] if verify else None
        for name, d in self.data.items():
            if not name.endswith("_recurrent"):
                continue
            old, new = oldins[name], newins[name]
            require(old["dtype"] == "float16" and new == dict(old, dtype="float32"), "recurrent state ABI")
            idx = self.out_index[name]
            outcast = self.parent(self.net, idx)
            committed = self.parent(outcast)
            require(
                outcast.type == "Cast"
                and outcast.input[0] == committed.name + ":1"
                and committed.type == "ChunkGatedDeltaRule"
                and committed.output_desc[1].dtype == 1
                and self.users[outcast.name] == [(self.net.name, idx)],
                "wrong published recurrent producer: " + name,
            )
            users = self.users[d.name]
            require(
                users and all(self.nodes[n].type == "Cast" and j == 0 for n, j in users),
                "unexpected old recurrent reader",
            )
            pair = []
            for cname, _ in users:
                cast = self.nodes[cname]
                require(
                    cast.output_desc[0].dtype == 1 and len(cast.input) == 1, "wrong GDR initial state cast"
                )
                for gname, j in self.users[cname]:
                    g = self.nodes[gname]
                    require(
                        g.type == "ChunkGatedDeltaRule"
                        and len(g.input) == 7
                        and j == 5
                        and g.output_desc[0].dtype == 2
                        and g.output_desc[1].dtype == 1
                        and gname not in seen,
                        "GDR input or output contract",
                    )
                    self.rewire(g, j, d.name + ":0")
                    tensor_update(g.input_desc[j], new, new)
                    pair.append(g)
                    seen.add(gname)
                self.drop.add(cname)
                self.old_to_new[cname + ":0"] = d.name + ":0"
            (first,) = (g for g in pair if g.input[6] == valid)
            if verify:
                require(
                    len(pair) == 2
                    and first is not committed
                    and committed in pair
                    and node_name(accepted) in self.ancestors([node_name(committed.input[6])])
                    and list(first.input)[:6] == list(committed.input)[:6],
                    "incorrect second-pass commit: " + name,
                )
                discard = self.out_index["verify_discard_" + name]
                require(self.net.input[discard] == first.name + ":1", "wrong verify discard state")
            else:
                require(len(pair) == 1 and first is committed, "incorrect prefill GDR")
            for desc in (*d.output_desc, *d.input_desc):
                tensor_update(desc, old, new, public=True)
            if "dtype" in d.attr:
                d.attr["dtype"].dt = 0
            self.known_changes.add(d.name)
            self.rewire(self.net, idx, committed.name + ":1")
            tensor_update(self.net.input_desc[idx], old, new)
            self.old_to_new[outcast.name + ":0"] = committed.name + ":1"
            self.drop.add(outcast.name)
            rows[name] = [first.name, committed.name]
        require(len(rows) == 24 and len(seen) == (48 if verify else 24), "incomplete native GDR inventory")
        return rows

    def attention(self, rows, table_files, out):
        """Extend exactly the named attention lengths, the common causal range and eight physical page maps."""
        attns = [n for n in self.g.op if n.type == "AdnFusedInferAttention"]
        require(len(attns) == 8, "wrong native attention count")
        caps = defaultdict(set)
        tables = {}
        mask_limits = set()
        witness = {}
        ins = {n["name"]: n for n in self.old["inputs"]}
        newins = {n["name"]: n for n in self.new["inputs"]}
        keys = {n.name: k for k, n in self.data.items()}
        for a in attns:
            require(
                len(a.input) >= 17 and not a.input[3] and self.parent(a, 4).type == "Cast",
                "unexpected native attention ABI",
            )
            for j, length in ((5, rows), (6, OLD_LENGTH), (16, OLD_LENGTH)):
                v = self.parent(a, j)
                ints = const_int(v)
                require(
                    ints is not None and ints.size == 1 and int(ints.item()) == length,
                    f"wrong {a.name} sequence-length port {j}",
                )
                if j != 5:
                    caps[v.name].add((a.name, j))
            block = self.parent(a, 14)
            require(
                block.type == "FileConstant"
                and block.attr["dtype"].dt == 3
                and list(block.attr["shape"].list.i) == [1, OLD_LENGTH // PAGE],
                "wrong input page map: " + block.name,
            )
            tables[block.name] = block
            mask = self.ancestors([self.parent(a, 4).name])
            public = {n for n in mask if self.nodes[n].type in ("Data", "RefData")}
            require(
                public == {self.data[k].name for k in ("start_position", "valid_rows")}
                and {"Less", "LessEqual", "LogicalAnd", "SelectV2"} <= {self.nodes[k].type for k in mask},
                "wrong causal mask inputs",
            )
            ranges = [self.nodes[n] for n in mask if self.nodes[n].type == "Range"]
            require(len(ranges) == 2, "unexpected causal columns/positions")
            stops = set()
            for window in ranges:
                v0, v1, vs = (const_int(self.parent(window, i)) for i in range(3))
                require(
                    all(x is not None and x.size == 1 for x in (v0, v1, vs))
                    and v0.item() == 0
                    and vs.item() == 1,
                    "non-unit mask range",
                )
                end = v1.item()
                stops.add(end)
                if end == OLD_LENGTH:
                    caps[node_name(window.input[1])].add((window.name, 1))
                    mask_limits.add(window.name)
            require(stops == {rows, OLD_LENGTH}, "causal columns do not use the physical cache capacity")
            # Source mask expressions have no other shape-bearing literals for the old capacity.
            for n in mask:
                node = self.nodes[n]
                val = const_int(node)
                if val is not None and np.any(val == OLD_LENGTH):
                    require(n in caps, "additional unclassified mask capacity literal: " + n)
                for desc in (*node.input_desc, *node.output_desc):
                    self.length_desc(node, desc)
            branches = []
            for slot, suffix in ((1, "_key"), (2, "_value")):
                ref = a.input[slot]
                chain = []
                while self.nodes[node_name(ref)].type == "CacheUpdate":
                    q = self.nodes[node_name(ref)]
                    require(ref == q.name + ":0" and len(q.input) == 4, "invalid CacheUpdate chain")
                    chain.append(q)
                    ref = q.input[0]
                k = keys.get(node_name(ref), "")
                require(
                    k.endswith(suffix)
                    and ref == self.data[k].name + ":0"
                    and len(chain) == (1 if rows == 64 else 16)
                    and self.net.input[self.out_index[k]] == a.input[slot],
                    "incorrect paged state: " + a.name + suffix,
                )
                require(k not in witness, "paged KV shared by multiple attention layers")
                old, new = ins[k], newins[k]
                require(
                    old == dict(new, shape=[9, 64, 64, 16]) and new["shape"] == [33, 64, 64, 16],
                    "unexpected paged cache ABI",
                )
                for desc in (*self.data[k].input_desc, *self.data[k].output_desc):
                    tensor_update(desc, old, new, public=True)
                self.known_changes.add(self.data[k].name)
                for op in chain:
                    tensor_update(op.input_desc[0], old, new)
                    tensor_update(op.output_desc[0], old, new)
                    self.known_changes.add(op.name)
                # In a TorchAir save some operator input descriptors are untyped placeholders.
                self.shape_only(a.input_desc[slot], old["shape"], new["shape"])
                self.known_changes.add(a.name)
                tensor_update(self.net.input_desc[self.out_index[k]], old, new, public=True)
                branches.append(k)
                witness[k] = [v.name for v in reversed(chain)]
            require(branches[0].split("_")[0] == branches[1].split("_")[0], "crossed attention layer KV")
            self.length_desc(a, a.input_desc[4])
        require(
            len(tables) == 8 and len(witness) == 16 and len(mask_limits) == 1,
            "unexpected page/cache/range inventory",
        )
        for key, users in caps.items():
            require(
                set(self.users[key]) == users,
                "attention capacity constant shared with another operation: " + key,
            )
            v = self.nodes[key]
            arr = const_int(v)
            set_literal(v, np.full_like(arr, LENGTH))
            self.known_changes.add(key)
        new_pages = out / "page-table-2112.i32"
        np.arange(LENGTH // PAGE, dtype="<i4").tofile(new_pages)
        previous = np.arange(OLD_LENGTH // PAGE, dtype="<i4").tobytes()
        for key, op in tables.items():
            file = table_files[key]
            require(
                file.stat().st_size == len(previous)
                and file.read_bytes() == previous
                and all(
                    self.nodes[n].type == "AdnFusedInferAttention" and j == 14 for n, j in self.users[key]
                ),
                "non-contiguous or shared source page map: " + key,
            )
            op.attr["file_path"].s = str(new_pages).encode()
            op.attr["shape"].list.i[:] = [1, LENGTH // PAGE]
            for desc in op.output_desc:
                self.shape_only(desc, [1, 9], [1, 33])
            for n, j in self.users[key]:
                self.shape_only(self.nodes[n].input_desc[j], [1, 9], [1, 33])
            self.known_changes.add(key)
        return dict(
            capacity_constants=sorted(caps),
            mask_column_range=sorted(mask_limits),
            page_maps=sorted(tables),
            page_payload=c.identity(new_pages),
            cache_update=witness,
        )

    @staticmethod
    def shape_only(d, old, new):
        if list(d.shape.dim) == old:
            d.shape.dim[:] = new
        if "origin_shape" in d.attr and list(d.attr["origin_shape"].list.i) == old:
            d.attr["origin_shape"].list.i[:] = new

    def length_desc(self, node, d):
        dims = list(d.shape.dim)
        if OLD_LENGTH in dims:
            self.shape_only(d, dims, [LENGTH if i == OLD_LENGTH else i for i in dims])
            self.known_changes.add(node.name)
        if "origin_shape" in d.attr:
            shp = list(d.attr["origin_shape"].list.i)
            if OLD_LENGTH in shp:
                d.attr["origin_shape"].list.i[:] = [LENGTH if i == OLD_LENGTH else i for i in shp]
                self.known_changes.add(node.name)

    def finish(self):
        g = self.g
        original = list(g.op)
        changed = {
            p.name for p in original if digest(p.SerializeToString(deterministic=True)) != self.before[p.name]
        }
        require(
            changed <= self.known_changes,
            "unexpected Target changes: " + str(sorted(changed - self.known_changes)),
        )
        all_new = {p.name for nodes in self.insert.values() for p in nodes}
        kept = (set(self.before) - self.drop) | all_new
        # Emit unmodified serialized nodes in their original order. Only new nodes need fresh IDs.
        nextid = max(p.id for p in original) + 1
        out = []
        for p in original:
            for q in self.insert[p.name]:
                q.id = nextid
                nextid += 1
                out.append(q)
            if p.name not in self.drop:
                out.append(p)
        require(
            all(node_name(e) in kept for p in out for e in p.input if e),
            "dangling public/state edge in reconstructed Target",
        )
        require(not self.drop.intersection(map(node_name, g.input)), "public input removed")
        for i, e in enumerate(g.output):
            if e in self.old_to_new:
                g.output[i] = self.old_to_new[e]
        require(all(node_name(e) in kept for e in g.output), "dangling graph output")
        del g.op[:]
        g.op.extend(out)
        return dict(
            changed=sorted(changed),
            removed=sorted(self.drop),
            added=sorted(all_new),
            byte_identical_original_nodes=len(self.before) - len(changed | self.drop),
            rewired=self.port_log,
        )


def port_map(op):
    keys = op.attr["_input_name_key"].list.s
    values = op.attr["_input_name_value"].list.i
    require(len(keys) == len(values) and len(set(keys)) == len(keys), "unusable GE port map: " + op.name)
    return dict(zip(keys, values))


def resize_feature_ports(op):
    if "_input_name_key" in op.attr:
        old = port_map(op)
        require(
            old == {b"concat_dim": 8, **{f"x{i}".encode(): i for i in range(8)}},
            "unknown ConcatV2 input map: " + str(old),
        )
        new = {b"concat_dim": len(UNION), **{f"x{i}".encode(): i for i in range(len(UNION))}}
        names = sorted(new)
        op.attr["_input_name_key"].list.s[:] = names
        op.attr["_input_name_value"].list.i[:] = [new[n] for n in names]


def receiver_specs(history):
    root = RECEIVER / "artifacts"
    manifest = c.read(root / "deployment-manifest.json")
    ref = manifest["air_manifest"]
    p = c.linked(root, ref)
    air = c.read(p)
    result = {}
    for rec in air["graphs"]:
        role = rec["name"]
        if role not in SOURCE_SHA:
            continue
        require(rec["air"]["sha256"] == SOURCE_SHA[role], "unrecognized Target AIR: " + role)
        af = c.linked(root, rec["air"])
        md = rec["metadata"]
        old = md["incremental_contract"]
        require(
            old["abi"] == "qwen35-dflash-chunk-v3"
            and old["capacity"] == 512
            and old["cache_capacity"] == OLD_LENGTH
            and old["feature_width"] == 20480
            and md.get("target_quant_mode") == "w8a8_dynamic",
            "unexpected 512 receiver contract",
        )
        source = md["tensor_abi"]
        dest = history["graphs"][role]["abi"]
        for side in ("inputs", "outputs"):
            require(
                [v["name"] for v in source[side]] == [v["name"] for v in dest[side]],
                "changed ordered Target IO",
            )
            for v, z in zip(source[side], dest[side]):
                n = v["name"]
                if n.endswith("_recurrent") and not n.startswith("verify_discard_"):
                    expected = dict(v, dtype="float32")
                elif n.endswith(("_key", "_value")):
                    expected = dict(v, shape=[33, 64, 64, 16])
                elif n == "features":
                    expected = dict(v, shape=[1, 64, len(UNION) * 2560])
                else:
                    expected = v
                require(expected == z, "unexpected receiver/2048 plan ABI difference: " + n)
        result[role] = dict(
            root=str(root),
            air=c.identity(af),
            source_abi=source,
            abi=dest,
            record=rec,
            manifest=c.identity(p),
        )
    require(set(result) == set(SOURCE_SHA), "missing canonical native source graphs")
    return result


def reconstruct(out, types, role, spec):
    source = c.verify(spec["air"])
    rec = spec["record"]
    root = Path(spec["root"])
    payloads = {p.resolve(): v for v in rec["payload_files"] for p in (root / v["path"],)}
    require(
        len(payloads) == len(rec["payload_files"]) and source in payloads, "invalid archived Target payloads"
    )
    verified = {p: dict(v, path=str(c.linked(root, v))) for p, v in payloads.items()}
    msg = types[0].FromString(source.read_bytes())
    (graph,) = msg.graph
    ir = TargetIR(graph, spec["source_abi"], spec["abi"])
    counts = Counter(n.type for n in graph.op)
    rows = 64 if role == "target_prefill" else 16
    require(
        all(
            counts[n] == v
            for n, v in {
                "FileConstant": 701,
                "ChunkGatedDeltaRule": 24 if rows == 64 else 48,
                "CacheUpdate": 16 if rows == 64 else 256,
                "QuantBatchMatmulV4444": 249,
                "DynamicQuant": 249,
                "AdnRmsNorm": 105,
                "AdnFusedInferAttention": 8,
            }.items()
        ),
        "wrong real Target operator inventory",
    )
    files = {}
    for n in graph.op:
        if n.type == "FileConstant":
            raw = Path(n.attr["file_path"].s.decode())
            p = (source.parent / raw).resolve(strict=True)
            require(p in verified, "unmanifested FileConstant " + n.name + ": " + str(p))
            files[n.name] = p
            if not raw.is_absolute():
                n.attr["file_path"].s = str(p).encode()
                ir.known_changes.add(n.name)
    h = ir.features(rows)
    gdr = ir.recurrent(rows == 16)
    geometry = ir.attention(rows, files, out)
    diff = ir.finish()
    expected = {k: v for k, v in files.items() if k not in geometry["page_maps"]}
    after = {
        n.name: Path(n.attr["file_path"].s.decode()).resolve() for n in graph.op if n.type == "FileConstant"
    }
    require(
        all(after[k] == v for k, v in expected.items()) and len(expected) == 693 and len(after) == 701,
        "changed model weight binding",
    )
    path = out / (role + ".air")
    path.write_bytes(msg.SerializeToString(deterministic=True))
    proof = dict(
        source=spec["air"],
        source_manifest=spec["manifest"],
        source_abi=spec["source_abi"],
        output_abi=spec["abi"],
        source_counts=dict(counts),
        learned_and_fixed_payloads={k: verified[v] for k, v in expected.items()},
        old_eight_verified=list(OLD_LAYERS),
        eleven_feature_sources=h,
        recurrence=gdr,
        geometry=geometry,
        node_identity=diff,
    )
    c.write(out / "audit.json", proof)
    print(
        role,
        "701 file constants verified, 693 original bindings plus 8 resized page tables;",
        len(gdr),
        "FP32 state slots,",
        counts["CacheUpdate"],
        "original CacheUpdates,",
        len(h),
        "feature taps",
        flush=True,
    )
    return dict(
        air=c.identity(path),
        abi=spec["abi"],
        target_layers=list(UNION),
        payloads=[
            c.identity(path),
            geometry["page_payload"],
            *[verified[v] for v in sorted(set(expected.values()))],
        ],
        origin="CANN_GE_RECEIVER_RECONSTRUCTION",
        source=spec["air"],
        audit=c.identity(out / "audit.json"),
    )
