"""Compare unredacted GE node records and their constants in the actual full OMs."""

from collections import defaultdict

from archive import c, require
from graph_data import const_mapping, differences, digest, encode, mapped_string, nodes, short
from evidence import GATE, NODES
from w16_graph import _checked_vendor

# mode-1 prints GE map fields as arrays. Only these descriptor metadata entries may
# have differently ordered, unequal duplicate values. Both typed multisets stay exact.
ORIGIN_KEYS = {
    "origin_format",
    "origin_format_for_int",
    "format_for_int",
    "origin_shape_initialized",
    "origin_shape",
}


def raw_inventory(model):
    return nodes(c.verify(model["dump"]))


def graph_interfaces(model, mapping):
    result = []

    def visit(value):
        if isinstance(value, list):
            for v in value:
                visit(v)
        elif isinstance(value, dict):
            if isinstance(value.get("op"), list):
                result.append(
                    dict(
                        name=value.get("name"),
                        nodes=sorted(mapping.get(op["name"], op["name"]) for op in value["op"]),
                        **{
                            port: [mapped_string(edge, mapping) for edge in value.get(port, [])]
                            for port in ("input", "output")
                        },
                    )
                )
            for k, v in value.items():
                if k not in ("op", "data", "tensor_content"):
                    visit(v)

    visit(c.read(c.verify(model["dump"])))
    return result


def bound_code_field(node, key, value, image):
    """Only derived references of an independently verified image acquire a shared comparison value."""
    expect = {"_kernel_bin_id": "s", "_tbeKernelSize": "i", "_kernelname": "s"}[key]
    require(isinstance(value, dict) and set(value) == {expect}, f"{node}: malformed {key}")
    actual = value[expect]
    if key == "_kernel_bin_id":
        require(
            isinstance(actual, str)
            and actual.endswith(image["sha256"])
            and ("id" not in image or actual == image["id"]),
            f"{node}: digest does not bind compiled ELF",
        )
        return {expect: actual[:-64]}  # compiler key prefix still compared
    if key == "_tbeKernelSize":
        require(type(actual) is int and actual == image["bytes"], f"{node}: queried ELF length differs")
        return {expect: "VERIFIED_IMAGE_LENGTH"}
    require(actual == image["symbol"], f"{node}: entry symbol differs")
    return value


def canonical_node(row, mapping, model, images, fusion):
    name = row["name"]
    kind = row["type"]
    references = {"name", "input", "src_name", "dst_name"}
    provenance = {
        "_datadump_original_op_names",
        "_datadump_original_op_input_name",
        "_datadump_original_op_output_name",
        "_original_op_names",
    }

    def visit(v, path=()):
        if isinstance(v, dict):
            return {k: visit(w, path + (k,)) for k, w in v.items()}
        if isinstance(v, list):
            if path and path[-1] in ("attr", "attributes"):
                groups = defaultdict(list)
                for entry in v:
                    require(
                        isinstance(entry, dict) and isinstance(entry.get("key"), str) and entry["key"],
                        f"{name}: malformed GE attribute collection",
                    )
                    k = entry["key"]
                    root = path == ("attr",)
                    item = {field: visit(w, path + (k, field)) for field, w in entry.items()}
                    if root and k in ("_kernel_bin_id", "_tbeKernelSize", "_kernelname") and name in images:
                        require(k not in groups, f"{name}: duplicate {k}")
                        item["value"] = bound_code_field(name, k, item["value"], images[name])
                    elif root and kind == GATE and k == "custom_op_file_path":
                        require(k not in groups, f"{name}: duplicate private source")
                        item["value"] = _checked_vendor(item["value"], model["opp_root"], name)
                    elif root and k == "fusion_failed":
                        require(
                            k not in groups
                            and set(item["value"]) == {"i"}
                            and type(item["value"]["i"]) is int,
                            f"{name}: malformed fusion scope ID",
                        )
                        label = item["value"]["i"]
                        if label >= 0:
                            fusion[mapping.get(name, name)] = label
                            item["value"] = {"i": "FE_FAILED_SCOPE_ID"}
                    groups[k].append(item)
                for k, items in groups.items():
                    if k in ORIGIN_KEYS and any(
                        s in ("input_desc", "output_desc", "desc", "td") for s in path
                    ):
                        items.sort(key=encode)
                return dict(groups)
            return [visit(w, path + (i,)) for i, w in enumerate(v)]
        if isinstance(v, str):
            # Data/weights and arbitrary string attributes are never rewritten as graph identifiers.
            if not any(k in ("data", "tensor_content", "weight_data") for k in path) and (
                (path and path[0] in references)
                or (
                    path
                    and path[-1] == "name"
                    and any(k in ("input_desc", "output_desc", "desc", "td") for k in path)
                )
                or any(k in provenance for k in path)
            ):
                return mapped_string(v, mapping)
        return v

    return visit(row)


def compare_graphs(raw, candidate, old_images, new_images):
    left, right = map(raw_inventory, (raw, candidate))
    aliases, missing = const_mapping(left, right)
    require(
        not any(missing),
        f"unbound generated Const; Raw/candidate examples: {missing[0][:4]} / {missing[1][:4]}",
    )
    mapped = {aliases.get(n, n): n for n in right}
    require(
        len(mapped) == len(right) and mapped.keys() == left.keys(), "missing/unpaired compiled graph node"
    )
    fusion = [{}, {}]
    constants = {}
    interfaces = graph_interfaces(raw, {})
    require(
        interfaces == graph_interfaces(candidate, aliases),
        "subgraph membership or ordered graph ports changed",
    )
    count = 0
    changed_labels = {}
    residuals = []
    for name in sorted(left):
        other = mapped[name]
        a = canonical_node(left[name], {}, raw, old_images, fusion[0])
        b = canonical_node(right[other], aliases, candidate, new_images, fusion[1])
        delta = next(differences(a, b), None)
        if delta is not None:
            p, v, w = delta
            if len(residuals) < 8:
                residuals.append(f"{name}{p}: {short(v)} -> {short(w)}")
        # Store the digest only after comparing full, typed Const payloads and descriptor/offset fields.
        if a["type"] in ("Const", "Constant"):
            constants[name] = dict(candidate_node=other, sha256=digest(a))
        count += 1
    require(not residuals, "protected graph field differs: " + " | ".join(residuals))
    require(bool(constants) and fusion[0].keys() == fusion[1].keys(), "Const/fusion label inventory differs")
    renumber, reverse = {}, {}
    for name, l in fusion[0].items():
        r = fusion[1][name]
        require(
            renumber.setdefault(l, r) == r and reverse.setdefault(r, l) == l,
            "changed fusion scope membership",
        )
        if l != r:
            changed_labels[name] = [l, r]
    return dict(
        node_count=count,
        interfaces=interfaces,
        generated_constants=aliases,
        const_records=constants,
        protected_nodes=count - len(NODES),
        fusion_scope_id_changes=changed_labels,
    )
