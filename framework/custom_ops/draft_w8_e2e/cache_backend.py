"""Identify the four-tensor Ascend310P CacheUpdate provider and the actual compiled kernels."""

import re
from pathlib import Path

from winners import c, require
from tools.audit_internal_custom_ops import _extract_header_declarations
from package import command
from bench import read_om_json, embedded

SHARED = Path("/home/l00886453/custom_ops_llm/vendors/customize_scatter")


def provider(path):
    root = Path(path).resolve(strict=True)
    h = root / "op_api/include/aclnn_cache_update.h"
    require(h.is_file(), f"missing CacheUpdate ACLNN header at {h}")
    decl = [x for x in _extract_header_declarations(h) if x["symbol"] == "aclnnCacheUpdateGetWorkspaceSize"]
    require(len(decl) == 1, f"cannot resolve CacheUpdate ABI from {h}")
    params = {i["name"].replace("_", "").lower(): i["c_type"] for i in decl[0]["parameters"]}
    require(
        all(params.get(n) == "const aclTensor*" for n in ("targetblock", "offsetinblock")),
        "need the tensor-index CacheUpdate provider, got " + decl[0]["canonical_signature"],
    )
    sdk = root / "op_impl/ai_core/tbe"
    code = list((sdk / "kernel/ascend310p/cache_update").glob("*.o"))
    require(code, "missing Ascend310P CacheUpdate binary in " + str(root))
    artifacts = []
    for p in sorted(code):
        f = c.identity(p)
        meta = p.with_suffix(".json")
        with p.open("rb") as stream:
            require(
                stream.read(6) == b"\x7fELF\x02\x01", "CacheUpdate device binary is not a little-endian ELF64"
            )
        require(meta.is_file(), "CacheUpdate kernel JSON is missing: " + str(meta))
        row = c.read(meta)
        if row.get("sha256"):
            require(row["sha256"] == f["sha256"], "CacheUpdate .o and kernel JSON disagree")
        artifacts.append(dict(binary=f, metadata=c.identity(meta)))
    wrappers = [
        c.identity(p) for p in sorted(sdk.glob("*/dynamic/cache_update.*")) if p.suffix in (".cpp", ".py")
    ]
    libs = [c.identity(p) for p in sorted((root / "op_api/lib").glob("*.so")) if p.is_file()]
    require(libs, "missing receiver CacheUpdate ACLNN library")
    return dict(
        path=str(root),
        header=c.identity(h),
        signature=decl[0]["canonical_signature"],
        artifacts=artifacts,
        dynamic=wrappers,
        libraries=libs,
    )


def verify_nodes(nodes, vendor, expected):
    expected = {f"CacheUpdate_{i}" if i else "CacheUpdate" for i in range(expected)}
    rows = {}
    refs = []
    for node in nodes:
        if node["type"] != "CacheUpdate":
            continue
        attrs = {v["key"]: v.get("value") for v in node["attributes"]}

        def get(k, typ):
            return (attrs.get(k) or {}).get(typ)

        names = set((attrs.get("_original_op_names") or {}).get("list", {}).get("s", []))
        names.update((attrs.get("_datadump_original_op_names") or {}).get("list", {}).get("s", []))
        names.add(node["name"])
        keys = names & expected
        require(len(keys) == 1, "ambiguous CacheUpdate source: " + str(names))
        key = keys.pop()
        engine = get("_ge_attr_op_kernel_lib_name", "s")
        sym = get("_kernelname", "s")
        bd = get("tvm_blockdim", "i")
        path = get("custom_op_file_path", "s")
        require(
            engine == "AIcoreEngine" and sym and bd is not None and int(bd) > 0,
            f"{node['name']} compiled to engine={engine}, symbol={sym}, blockDim={bd}; required the receiver AI Core CacheUpdate",
        )
        if path:
            actual = Path(path).resolve()
            require(
                actual == vendor or vendor in actual.parents,
                "wrong CacheUpdate private registration: " + path,
            )
        require(
            key not in rows and len([e for e in node["inputs"] if e and not e.endswith(":-1")]) == 4,
            "CacheUpdate arity/duplicate: " + key,
        )
        bid = get("_kernel_bin_id", "s")
        size = get("_tbeKernelSize", "i")
        rec = dict(
            node=node["name"], engine=engine, symbol=sym, block_dim=int(bd), vendor=path, binary_id=bid
        )
        if bid and size and re.search("[a-f0-9]{64}$", bid):
            refs.append(
                dict(
                    node=node["name"],
                    id=bid,
                    symbol=sym,
                    bytes=int(size),
                    sha256=bid[-64:],
                    block_dim=int(bd),
                    engine=engine,
                )
            )
        rows[key] = rec
    require(
        set(rows) == expected,
        "compiled CacheUpdate inventory: missing " + str(sorted(expected - rows.keys())[:12]),
    )
    return rows, refs


def audit_compiled(d, name, om, atc, env, vendor):
    path = d / "compiled-target.json"
    command([atc, "--mode=1", "--om=" + str(om), "--json=" + str(path)], d / "inspect-target.log", env, d)
    nodes = read_om_json(path)
    rows, refs = verify_nodes(nodes, Path(vendor).resolve(), 16 if name == "target_prefill" else 256)
    images = embedded(om, refs) if refs else {}
    report = dict(nodes=rows, embedded=images, dump=c.identity(path))
    p = d / "cache-update-backend.json"
    c.write(p, report)
    print(
        f"{name}: {len(rows)}/{len(rows)} CacheUpdate on AI Core; {len(images)} embedded ELF bindings",
        flush=True,
    )
    return c.identity(p)
