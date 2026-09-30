"""Exact full-projection inventories for real C16 and C64 replay captures."""
ABI = "dflash-group-quant-linear-a4-real-v1"
KINDS = ("kv", "q", "o", "gate_up", "down")
CASE_NAMES = ("fc",) + tuple(f"layer-{layer}-{kind}" for layer in range(5) for kind in KINDS)


def cases(context_rows):
    if type(context_rows) is not int or context_rows not in (16, 64):
        raise ValueError("A4 needs an actual frozen C16 or C64 replay; changing input rows cannot fabricate a gear")
    shapes = {"fc": (context_rows, 12800, 2560), "kv": (context_rows + 16, 2560, 2048),
              "q": (16, 2560, 4096), "o": (16, 4096, 2560),
              "gate_up": (16, 2560, 19456), "down": (16, 9728, 2560)}
    result = []
    for name in CASE_NAMES:
        layer, kind = (None, "fc") if name == "fc" else (int(name.split("-")[1]), name.split("-", 2)[2])
        m, k, n = shapes[kind]
        result.append(dict(name=name, layer=layer, projection=kind, context_rows=context_rows,
                           m=m, k=k, n=n, group_size=128, layout="nz_int8_v1"))
    return result


def case_shape(case):
    expected = next((row for row in cases(case.get("context_rows")) if row["name"] == case.get("name")), None)
    if (expected is None or any(case.get(key) != value for key, value in expected.items()) or
            any(type(case.get(d)) is not int for d in ("m", "k", "n")) or
            (expected["layer"] is not None and type(case.get("layer")) is not int)):
        raise ValueError("invalid A4 projection identity/shape/context gear")
    return expected["m"], expected["k"], expected["n"]


def capture_targets(graph, context_rows):
    """Hook exactly the production packed projections, in execution order."""
    if len(graph.layers) != 5:
        raise ValueError("A4 requires all five production Draft layers")
    modules = [graph.fc]
    for layer in graph.layers:
        modules.extend((layer.kv_linear, layer.q_proj, layer.o_proj, layer.gate_up_linear, layer.down_proj))
    return tuple(zip(modules, cases(context_rows)))


def paired_gears(c16, c64):
    """Both captures must bind the same real checkpoint, weights and feature order."""
    for bundle, rows in ((c16, 16), (c64, 64)):
        if (bundle.get("abi") != ABI or bundle.get("context_rows") != rows or
                bundle.get("source", {}).get("context_rows") != rows or
                tuple(c.get("name") for c in bundle.get("cases", [])) != CASE_NAMES):
            raise ValueError("A4 requires separate complete C16 and C64 captures")
    for key in ("config_sha256", "model_sha256"):
        if not c16["checkpoint"].get(key) or c16["checkpoint"][key] != c64["checkpoint"].get(key):
            raise ValueError("C16/C64 checkpoint identity differs")
    if (not c16["source"].get("feature_layers") or
            c16["source"]["feature_layers"] != c64["source"].get("feature_layers")):
        raise ValueError("C16/C64 feature layer order differs")
    for left, right in zip(c16["cases"], c64["cases"]):
        case_shape(left); case_shape(right)
        for name in ("q_nk.bin", "w_nz.bin", "s_gn.bin"):
            if left["files"][name]["sha256"] != right["files"][name]["sha256"]:
                raise ValueError(f"C16/C64 weight/scale identity differs: {left['name']}/{name}")
