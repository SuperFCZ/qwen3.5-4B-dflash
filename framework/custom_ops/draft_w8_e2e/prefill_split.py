"""Split the verified Target Prefill at the last valid FP16 row feeding its W8A8 vocabulary head."""

from collections import Counter
from pathlib import Path

from winners import c, require
from air import protobuf_classes
from packed_panel import nd_desc
from receiver_air import node_name


ROW = "prefill_head_input"


def split_model(source, abi, vocab=248320):
    (g,) = source.graph
    table = {n.name: n for n in g.op}
    (net,) = (n for n in g.op if n.type == "NetOutput")
    count = len(abi["outputs"])
    require(
        len(table) == len(g.op)
        and ROW not in table
        and abi["outputs"][0] == dict(name="target_top1", dtype="int64", shape=[1, 1])
        and abi["outputs"][1]["name"] == "features"
        and len(net.input_desc) >= count
        and len(net.input) >= count
        and all(e.endswith(":-1") for e in net.input[count:]),
        "unexpected Target output ABI",
    )
    top = net.input[0]
    argmax = table[node_name(top)]
    mm = table[node_name(argmax.input[0])]
    dq = table[node_name(mm.input[0])]
    cut = dq.input[0]
    row = table[node_name(cut)]
    weight = table[node_name(mm.input[1])]
    shape = list(weight.attr["shape"].list.i)
    require(
        argmax.type == "ArgMaxV2"
        and mm.type == "QuantBatchMatmulV4444"
        and dq.type == "DynamicQuant"
        and weight.type == "FileConstant"
        and len(shape) == 2
        and shape[0] > 0
        and shape[1] == vocab
        and row.type == "GatherV2"
        and cut == row.name + ":0"
        and len(dq.input) == 1
        and row.output_desc[0].dtype == dq.input_desc[0].dtype == 2,
        "cannot identify the last-row Gather -> DynamicQuant -> vocabulary matmul -> ArgMax",
    )
    handoff = dict(name=ROW, dtype="float16", shape=[1, 1, shape[0]])

    def ancestry(edges, boundary=None):
        seen = set()
        pending = [node_name(e) for e in edges if e and e != boundary]
        while pending:
            name = pending.pop()
            if name in seen:
                continue
            seen.add(name)
            pending.extend(node_name(e) for e in table[name].input if e and e != boundary)
        return seen

    head = ancestry([top], cut)
    # Preserve every non-Head receiver operation, including ordering edges. Only
    # the exclusive suffix is removed from the body; shared literals may be copied.
    other = set(table) - head - {net.name}
    keep = ancestry([cut, *net.input[1:], *(n + ":0" for n in other)])
    require(
        not ({argmax.name, mm.name, dq.name, weight.name} & keep)
        and not any(table[n].type in ("Data", "RefData", "ChunkGatedDeltaRule", "CacheUpdate") for n in head)
        and all(node_name(e) in keep for e in g.input)
        and row.name in keep,
        "Head is required by a retained state/feature or another public input",
    )
    require(
        sum(table[n].type == "QuantBatchMatmulV4444" for n in head) == 1
        and sum(table[n].type == "DynamicQuant" for n in head) == 1,
        "ambiguous vocabulary branch",
    )
    if g.output:
        require(list(g.output) == list(net.input[:count]), "unexpected secondary graph-output mapping")

    body = type(source)()
    body.CopyFrom(source)
    bg = body.graph[0]
    del bg.op[:]
    for n in g.op:
        if n.name in keep or n.name == net.name:
            v = bg.op.add()
            v.CopyFrom(n)
            if n.name == net.name:
                v.input[0] = cut
                d = v.input_desc[0]
                name = d.name
                d.Clear()
                nd_desc(d, name, 2, handoff["shape"])
    if bg.output:
        bg.output[0] = cut

    model = type(source)(
        name="target_prefill_head", version=source.version, custom_version=source.custom_version
    )
    hg = model.graph.add(name=model.name)
    inp = hg.op.add(name=ROW, type="Data")
    inp.attr["index"].i = 0
    nd_desc(inp.output_desc.add(), "y", 2, handoff["shape"])
    inp.input_desc.add().CopyFrom(inp.output_desc[0])
    hg.input.append(ROW + ":0")
    for n in g.op:
        if n.name not in head:
            continue
        v = hg.op.add()
        v.CopyFrom(n)
        for i, edge in enumerate(v.input):
            if edge == cut:
                v.input[i] = ROW + ":0"
                name = v.input_desc[i].name
                v.input_desc[i].CopyFrom(inp.output_desc[0])
                v.input_desc[i].name = name
    end = hg.op.add(name=net.name, type="NetOutput")
    end.input.append(top)
    end.input_desc.add().CopyFrom(net.input_desc[0])
    if "_input_name_key" in net.attr:
        for key in ("_input_name_key", "_input_name_value"):
            end.attr[key].CopyFrom(net.attr[key])
        end.attr["_input_name_key"].list.s[:] = [b"input0"]
        end.attr["_input_name_value"].list.i[:] = [0]
    if g.output:
        hg.output.append(top)
    for i, v in enumerate(hg.op):
        v.id = i
    abis = {
        "body": dict(inputs=abi["inputs"], outputs=[handoff, *abi["outputs"][1:]]),
        "head": dict(inputs=[handoff], outputs=abi["outputs"][:1]),
    }
    return (
        {"body": body, "head": model},
        abis,
        dict(
            cut=cut,
            projection=mm.name,
            head_nodes=sorted(head),
            removed_from_body=sorted(head - keep),
            shared_nodes=sorted(head & keep),
        ),
    )


def export(folder, desc, source):
    source_path = c.verify(source["air"])
    original = protobuf_classes(c.verify(desc).read_bytes())[0].FromString(source_path.read_bytes())
    models, abis, proof = split_model(original, source["abi"])
    records = {str(Path(r["path"]).resolve()): r for r in source["payloads"]}
    result = {}
    for side, model in models.items():
        nodes = model.graph[0].op
        files = set()
        for n in nodes:
            if n.type == "FileConstant":
                raw = Path(n.attr["file_path"].s.decode())
                file = (source_path.parent / raw).resolve(strict=True)
                if not raw.is_absolute():
                    n.attr["file_path"].s = str(file).encode()
                files.add(str(file))
        require(all(f in records for f in files), "unbound receiver payload in split Target")
        path = folder / (side + ".air")
        path.write_bytes(model.SerializeToString(deterministic=True))
        result[side] = dict(
            air=c.identity(path),
            abi=abis[side],
            payloads=[records[f] for f in sorted(files)],
            nodes=dict(Counter(n.type for n in nodes)),
        )
    c.write(folder / "split.json", proof)
    print(
        f"Target split at {proof['cut']}: body {result['body']['nodes'].get('QuantBatchMatmulV4444', 0)} "
        f"quantized projections, final Head one vocabulary projection, {abis['head']['inputs'][0]['shape']} FP16",
        flush=True,
    )
    return dict(parts=result, proof=c.identity(folder / "split.json"), source=source["air"])
