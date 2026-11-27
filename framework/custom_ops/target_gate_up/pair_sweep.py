from pathlib import Path
import hashlib
import inspect
import shutil
import numpy as np

from pair_graph import HERE, OP, MM, DQ, N, c, require, load_model, single_graph, save_graph
from pair_kernel import VARIANTS, build_operator, use_operator, pack, custom_nodes
from pair_measure import compiled_inventory, validate_two_rounds, smoke, paired, profile, WEIGHT_BYTES, PEAK
from pair_reference import compile_om, matrix_view, read_case
from panel_profiler import bind_msprof


def package_key(stages):
    files = [
        HERE / "TargetGateUpPanel.json",
        *sorted((HERE / "op_kernel").glob("*.cpp")),
        *sorted((HERE / "op_host").glob("*")),
    ]
    builder = hashlib.sha256(inspect.getsource(build_operator).encode()).hexdigest()
    sources = {str(p.relative_to(HERE)): c.sha(p) for p in files}
    # The first board sweep keyed both C++ translation units by basename. The
    # installed, hash-bound dynamic source disambiguates the missing device entry.
    if "new-operator" in stages.previous:
        receipt = stages.completed("new-operator")
        legacy = {f.name: sources[str(f.relative_to(HERE))] for f in files}
        legacy["pair_kernel.py"] = "37a422eb1d9badc2cf02fc142f21a18f8a4f134ff667dcec376a2c7aa6d17a75"
        if (
            receipt["contract"]["inputs"] == legacy
            and builder == "5cdf9756a60b4f807a636715aa608afbefcac6b0fe4fd396dc65256ae8420a98"
            and receipt["result"]["implementation"]["sha256"] == sources["op_kernel/target_gate_up_panel.cpp"]
        ):
            return legacy
    return dict(sources=sources, builder=builder)


def rounding_examples(proof, ref, layouts):
    """Record the exact inputs to a rare discrepancy; never adjust a result by token or fixture."""
    examples = {}
    for key, failure in proof["mismatches"].items():
        if failure["count"] > 32:
            continue
        case, branch = key.split("/")
        q, scales, _ = read_case(ref["cases"][case], ref["rows"])
        source = ref["weights"][branch]
        w = matrix_view(np.fromfile(c.verify(source["weight"]), np.int8), layouts[branch])
        sw = np.fromfile(c.verify(source["scale"]), "<f4")[0]
        r, n = divmod(int(failure["first"]), N)
        acc = int(q[r].astype(np.int32) @ w[:, n].astype(np.int32))
        pre = np.float16(np.float32(acc) * sw)
        sr = scales[r, 0]
        examples[key] = dict(
            row=r,
            column=n,
            acc=acc,
            weight_bits=int(sw.view("<u4")),
            row_bits=int(sr.view("<u4")),
            mid_bits=int(pre.view("<u2")),
            half_row_bits=int(np.float16(sr).view("<u2")),
            expected=failure["expected"],
            actual=failure["actual"],
        )
    return examples


def run_sweep(out, stage, assets, refs, env, cann, desc, state, stages):
    package = stage("new-operator", package_key(stages), lambda d: build_operator(d, cann, env))
    private = use_operator(env, package)
    state["package"] = package
    cli = bind_msprof(out, cann, env)
    atc = Path(shutil.which("atc", path=env["PATH"])).resolve(strict=True)
    runner = c.verify(assets["tools"]["ac_om_runner"])
    cap = assets["workspace"]
    results = {}
    for role, ref in refs.items():
        rows = ref["rows"]
        original = assets["models"][role]
        src = load_model(desc, original["air"])
        math = stage(
            role + "-rounding",
            [ref["cases"], ref["weights"], c.sha(HERE / "pair_measure.py")],
            lambda d: validate_two_rounds(d, ref),
        )
        layouts = {b: v["layout"] for b, v in math.items()}
        packed_by_width = {}
        for width in sorted({v[0] for v in VARIANTS.values()}):
            packed_by_width[str(width)] = stage(
                f"{role}-b1-n{width}",
                [ref["weights"], layouts, width, c.sha(HERE / "pair_kernel.py")],
                lambda d: pack(d, ref["weights"], layouts, width),
            )
        candidates = {}
        envs = {}
        inv = {}
        failures = {}
        for name in ("native", "shared", *VARIANTS):
            try:
                if name in ("native", "shared"):
                    model = ref["models"][name]
                    e = env
                else:
                    packed = packed_by_width[str(VARIANTS[name][0])]

                    def make(d):
                        def builder(cls, b, m):
                            return custom_nodes(cls, b, m, c.verify(packed), name)

                        graph, abi = single_graph(src, original["abi"], rows, "custom", builder)
                        return save_graph(d, graph, abi)

                    air = stage(
                        role + "-air-" + name,
                        [original["air"], packed, name, c.sha(HERE / "pair_kernel.py")],
                        make,
                    )
                    model = stage(
                        role + "-atc-" + name,
                        [air["air"], package, c.identity(atc), assets["binding"]],
                        lambda d: compile_om(d, air, atc, private, dump=True),
                    )
                    e = private
                inventory = compiled_inventory(model)
                nmm = sum(v["type"] == MM for v in inventory.values())
                nop = [v for v in inventory.values() if v["type"] == OP]
                require(
                    (len(nop) == 0 and nmm == 2)
                    if name in ("native", "shared")
                    else (
                        len(nop) == 1
                        and nmm == 0
                        and (not nop[0]["blocks"] or int(nop[0]["blocks"]) == 8)
                        and (
                            not nop[0]["vendor"]
                            or Path(nop[0]["vendor"]).resolve() == Path(package["vendor"]).resolve()
                        )
                    ),
                    "wrong compiled matrix implementation or block count",
                )
                print(
                    f"M{rows} {name} compiled DQ / native matmul / new kernel "
                    f"{sum(v['type'] == DQ for v in inventory.values())}/{nmm}/{len(nop)}",
                    flush=True,
                )
                proof = stage(
                    role + "-check-" + name,
                    [model["om"], ref["x"], ref["cases"], c.sha(HERE / "pair_measure.py")],
                    lambda d: smoke(d, name, model, ref, runner, e),
                )
                if not proof["ok"]:
                    failures[name] = proof["mismatches"]
                    examples = rounding_examples(proof, ref, layouts)
                    if examples:
                        print(f"M{rows} {name} rounding witnesses {examples}", flush=True)
                    continue
                candidates[name] = model
                envs[name] = e
                inv[name] = inventory
            except (ValueError, RuntimeError, OSError, TypeError) as exc:
                if name == "native":
                    raise
                failures[name] = str(exc)
                print(f"M{rows} {name} unavailable: {exc}", flush=True)
        identities = {k: m["om"] for k, m in candidates.items()}
        timing = stage(
            role + "-paired",
            [identities, ref["x"]["real"], cap, c.sha(HERE / "pair_measure.py")],
            lambda d: paired(d, candidates, ref, runner, cap, envs),
        )
        profiles = {}
        for name, model in candidates.items():
            try:
                profiles[name] = stage(
                    role + "-profile-" + name,
                    [model["om"], ref["x"]["real"], cli["file"], cap, c.sha(HERE / "pair_measure.py")],
                    lambda d: profile(d, name, model, inv[name], ref, runner, cap, envs[name], cli),
                )
            except (ValueError, RuntimeError, OSError, KeyError) as exc:
                profiles[name] = {"unresolved": str(exc)}
                print(f"M{rows} {name} profile unresolved: {exc}", flush=True)
        results[role] = dict(
            rows=rows,
            modes=identities,
            inventory=inv,
            smoke_failures=failures,
            timing=timing,
            profiles=profiles,
            layouts=layouts,
            rounding=math,
            packed_by_width=packed_by_width,
        )
    state["groups"] = results
    print("Target Gate/Up: real Layer 0, host Execute+Sync and msprof device times in us", flush=True)
    for role, g in results.items():
        for name, b in g["timing"]["modes"].items():
            s = b["stats"]
            pf = g["profiles"][name]
            detail = (
                f"kernel {pf['core_us']['p50']:.2f}, DQ/prep→last task {pf['span_us']['p50']:.2f}, "
                f"{pf['logical_kernel_GB_s']:.2f} logical GB/s"
                if "core_us" in pf
                else "profile unresolved"
            )
            print(f"{role:6} {name:18} {s['p50']:.2f}/{s['p95']:.2f} host  {detail}", flush=True)
        print(
            f"Selected {role} {g['timing']['selected']}; excluded {list(g['smoke_failures'])}; "
            f"weight-only reference {WEIGHT_BYTES / (1000 * PEAK):.2f} us",
            flush=True,
        )
    return results
