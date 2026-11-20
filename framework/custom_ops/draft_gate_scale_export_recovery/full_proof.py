"""Full static-OM export admission for the guarded, unprofiled Draft experiment."""

from collections import Counter
from pathlib import Path
import struct

from archive import c, require
from evidence import CANDIDATE, CONTROL, NODES, raw_air, verify_compiled_graph
from expansion_models import selected_plans, verify_worker
from graph_contract import compare_graphs
from inspect_export import blob, part_data
from elf_identity import equivalent
from native_dfx import compare as dfx_compare
from om_constants import inline_constants, weight_partition
from routing import check_package, verify_dispatch
from scale_contracts import historical
from scale_review import allocated_sections
from w16_export import native_images

SDK = {
    "aarch64-linux/include/proto/ge_ir.proto": "20cde9c443d712d802c46de87a6c1214ad96858ac3bee9b2056eb2e29dcebb7e",
    "python/site-packages/tbe/tikcpp/compile_op.py": "fe4ad8ac0a724f7270ba2ccdbe392f41ee8caf2d33e4867a9087767f3e52d7f2",
    "python/site-packages/tbe/tikcpp/ascendc_common_utility.py": "8faadf4b2a17356a28d99a9c089c27379e2252b96cf1d0b1d0a198d5e691d2cf",
}


def sdk_identity(cann):
    paths = {k: c.identity(Path(cann) / k) for k in SDK}
    require({k: v["sha256"] for k, v in paths.items()} == SDK, "different SDK source filename contract")
    return paths


def compare_parts(left, right):
    a, b = part_data(left), part_data(right)
    require(set(a) == set(b) == {8, 0, 1, 3, 2}, "unreviewed IMOD partitions")
    for kind in (1, 2, 8):
        require(a[kind] == b[kind], f"IMOD partition {kind} changed (size/SHA256)")
    return {
        str(k): dict(
            raw=dict(bytes=x[0], sha256=x[1]),
            candidate=dict(bytes=b[k][0], sha256=b[k][1]),
            exact_equal=x == b[k],
        )
        for k, x in a.items()
    }


def without_image_id(item):
    return {k: v for k, v in item.items() if k not in ("id", "offset", "bytes", "sha256")}


def compare_images(raw, new, targets, native):
    """Independently hash every graph-selected slice; proof caching is per identical pair AND entry symbol."""
    old, cand = native
    require(old.keys() == cand.keys() and "MatMul" in old, "native/Head image inventory changed")
    lq, rq = raw["atc_kernels"], new["atc_kernels"]
    require(
        lq.keys() == rq.keys()
        and len(lq) == 26
        and len(targets) == 5
        and targets <= lq.keys()
        and not (lq.keys() & old.keys()),
        "wrong compiled Gate/other projection inventory",
    )
    left = dict(old, **lq)
    right = dict(cand, **rq)
    changed, protected, proofs = {}, {}, {}
    with open(raw["om"]["path"], "rb") as a, open(new["om"]["path"], "rb") as b:
        for n in sorted(left):
            x, y = left[n], right[n]
            if n in lq:
                require(
                    x["engine"] == y["engine"] == "AIcoreEngine"
                    and x["block_dim"] == y["block_dim"] == 8
                    and x["id"][:-64] == y["id"][:-64]
                    and x["id"].endswith(x["sha256"])
                    and y["id"].endswith(y["sha256"]),
                    "scheduler or compiler key changed: " + n,
                )
            if n in targets:
                require(
                    x["sha256"] != y["sha256"]
                    and x["executable_sections"] != y["executable_sections"]
                    and x["symbol"] == y["symbol"],
                    "target did not change code under the bound entry: " + n,
                )
                changed[n] = dict(raw=x, candidate=y)
                continue
            require(
                without_image_id(x) == without_image_id(y),
                "protected entry or executable section changed: " + n,
            )
            key = x["sha256"] + "/" + y["sha256"] + "/" + x["symbol"]
            if key not in proofs:
                l, r = blob(a, x), blob(b, y)
                if l == r or struct.unpack_from("<H", l, 16)[0] == 2:
                    proof = equivalent(l, r, x["symbol"])
                else:
                    require(n in old, "DFX exception outside native code: " + n)
                    proof = dfx_compare(l, r, x["symbol"])
                proofs[key] = proof
            protected[n] = dict(
                proof=key,
                raw_sha256=x["sha256"],
                candidate_sha256=y["sha256"],
                status=proofs[key]["status"],
                native=n in old,
            )
    counts = {
        k: dict(Counter(v["status"] for v in protected.values() if v["native"] == is_native))
        for k, is_native in (("non_gate_projections", False), ("native", True))
    }
    return left, right, dict(targets=changed, protected=protected, unique_proofs=proofs, counts=counts)


def check_full(ctx, chain, gear, package, model):
    src = ctx["src"]
    control = ctx["full"][gear]
    bound = chain[f"c{gear}-source"]["binding"]
    require(
        control["mode"] == CONTROL
        and model["mode"] == CANDIDATE
        and model["scope"] == control["scope"] == "full"
        and model["key"] == control["key"] == gear
        and model["gate_nodes"] == control["gate_nodes"] == NODES
        and model["om"] != control["om"]
        and model["abi"] == control["abi"]
        and model["air"] == control["air"]
        and model["opp_root"] == package["opp"]["install"] != control["opp_root"]
        and model["private_kernel"] == package["kernel"],
        "Full OM/control/OPP or ABI mismatch",
    )
    check_package(ctx, gear, package)
    for m in (control, model):
        verify_compiled_graph(m, src, True)
    worker = c.read(c.verify(model["worker_report"]))
    historical.verify_tree(worker, {})
    folder = c.verify(model["dump"]).parent
    compile_receipt = verify_worker(worker, model, src, package, raw_air(ctx, gear))
    require(
        model["plans"] == selected_plans(folder, worker, package, src, control, write=False),
        "actual ATC resource plan differs",
    )
    targets = set(NODES.values())
    dispatch = verify_dispatch(
        c.identity(folder / "scale-dispatch.jsonl"), package, model["atc_kernels"], targets
    )
    partitions = compare_parts(control, model)
    # Reconcile the immutable Raw per-node digest inventory with the archived Source evidence.
    require(
        weight_partition(c.verify(control["om"])) == bound["raw_weight_partition"]
        and inline_constants(c.verify(control["dump"])) == bound["raw_const_nodes"],
        "frozen Raw constants lost their original Source binding",
    )
    left, right, images = compare_images(
        control, model, targets, [native_images(m) for m in (control, model)]
    )
    graph = compare_graphs(control, model, left, right)
    require(set(graph["const_records"]) == set(bound["raw_const_nodes"]), "per-Const inventory drift")
    resources = {}
    for n in sorted(targets):
        before, after = (allocated_sections(m, n) for m in (control, model))
        require(before == after, "target writable storage or SDK metadata changed: " + n)
        resources[n] = before
    return dict(
        scope="FULL_OM_UNPROFILED_EXECUTION",
        raw_om=control["om"],
        candidate_om=model["om"],
        compile=compile_receipt,
        dispatch=dispatch,
        partitions=partitions,
        graph=graph,
        images=images,
        target_resources=resources,
        sdk=sdk_identity(src["cann"]),
        device_execution="NOT_RUN",
    )
