"""Use the unmodified receiver as the numerical oracle; inspect tensor bytes and public GE shapes."""

from pathlib import Path
import shutil

import numpy as np

from pair_graph import K, N, c, require, file_path, load_model, pairs, save_graph, single_graph
from run_prefill_head import capture
from package import command
from experiment import Stages
from diagnose_calls import receiver_runtime


ORDERS = ("weight-first", "row-first", "combined")
LAYOUTS = ("nd-kn", "nd-nk", "nz-kn", "nz-nk", "blocked-nk")


def deployment(out, root, cann):
    z = c.read(root / "result.json")
    require(
        z["status"] == "MEASURED_AND_PROFILED"
        and z["large_transposes_removed_in_both"]
        and z["same_workload"]
        and z["round_matches"] == z["token_matches"] == 60,
        "select the measured direct-convolution-window Target",
    )
    old = Stages(out, root)
    split = c.verify(z["baseline"]).parent
    tools = Stages(out, split).completed("runners")["result"]
    ws = c.read(c.verify(c.read(split / "result.json")["baseline"]))
    context = c.read(c.verify(ws["baseline"]))
    full = c.verify(context["baseline"]).parent
    env, binding = receiver_runtime(full, c.read(full / "result.json"), cann)
    models = {}
    for role in ("body", "verify"):
        s = old.completed("air-" + role)["result"]
        m = old.completed("atc-" + role)["result"]
        require(z["models"][role] == m["om"] and m["abi"] == s["abi"], "wrong source Target")
        csv = c.verify(z["profiles"][role]["candidate"]["csv"])
        folder = next(d / "case" for d in csv.parents if (d / "case/execution.json").is_file())
        models[role] = dict(
            air=s["air"], abi=s["abi"], om=m["om"], dump=m["dump"], input_dir=str(folder / "inputs")
        )
    return dict(
        models=models,
        plan=z["plan"],
        baseline=c.identity(root / "result.json"),
        workspace=z["workspace"]["shared_bytes"],
        tools=tools,
        binding=binding,
    ), env


def compile_om(d, graph, atc, env, *, dump=False):
    src = c.verify(graph["air"])
    prefix = d / "model"
    e = dict(env, ASCEND_CACHE_PATH=str(d / "cache"), ASCEND_WORK_PATH=str(d))
    (d / "cache").mkdir()
    args = [
        atc,
        "--mode=0",
        "--framework=1",
        "--model=" + str(src),
        "--output=" + str(prefix),
        "--soc_version=Ascend310P3",
        "--precision_mode=must_keep_origin_dtype",
    ]
    c.write(d / "atc-command.json", list(map(str, args)))
    command(args, d / "atc.log", e, src.parent)
    om = c.identity(prefix.with_suffix(".om"))
    result = dict(om=om, abi=graph["abi"])
    if dump:
        f = d / "compiled.json"
        command([atc, "--mode=1", "--om=" + om["path"], "--json=" + str(f)], d / "inspect.log", e, d)
        result["dump"] = c.identity(f)
    return result


def matrix_view(raw, layout, k=K, n=N):
    if layout == "nd-kn":
        return raw.reshape(k, n)
    if layout == "nd-nk":
        return raw.reshape(n, k).T
    if layout == "nz-kn":
        return raw.reshape(n // 32, k // 16, 16, 32).transpose(1, 2, 0, 3).reshape(k, n)
    if layout == "nz-nk":
        return raw.reshape(k // 32, n // 16, 16, 32).transpose(1, 2, 0, 3).reshape(n, k).T
    require(layout == "blocked-nk", "unknown weight encoding")
    return raw.reshape(n // 16, k // 32, 16, 32).transpose(0, 2, 1, 3).reshape(n, k).T


def converted(acc, sw, sr, order):
    acc = acc.astype(np.float32)
    if order == "weight-first":
        f = (acc * sw) * sr
    elif order == "row-first":
        f = (acc * sr) * sw
    else:
        require(order == "combined", "invalid dequantization order")
        f = acc * (sr * sw)
    return f.astype("<f2").view("<u2")


def ints(q, w):
    # Each 256-term partial is below 2^24 at the worst INT8 magnitudes, hence its
    # FP32 BLAS result is exact. The full K reduction is accumulated in INT32.
    out = np.zeros((len(q), w.shape[1]), dtype=np.int32)
    for k in range(0, q.shape[1], 256):
        partial = q[:, k : k + 256].astype(np.float32) @ w[k : k + 256].astype(np.float32)
        out += partial.astype(np.int32)
    return out


def read_case(outputs, rows):
    q = np.fromfile(c.verify(outputs["q_probe"]), "<f4").reshape(rows, K)
    require(
        np.isfinite(q).all() and np.array_equal(q, q.astype(np.int8).astype(np.float32)),
        "invalid quantized codes",
    )
    return (
        q.astype(np.int8),
        np.fromfile(c.verify(outputs["row_scale"]), "<f4").reshape(rows, 1),
        {name: np.fromfile(c.verify(outputs[name]), "<u2").reshape(rows, N) for name in ("gate", "up")},
    )


def audit_math(d, weights, cases, rows):
    data = {k: read_case(outputs, rows) for k, outputs in cases.items()}
    q, sr, gold = data["basis"]
    require(np.all((q != 0).sum(axis=1) == 1), "basis quantization changed zero entries")
    cols = np.argmax(q != 0, axis=1)
    totals = {order: 0 for order in ORDERS}
    report = {}
    for branch, pair in weights.items():
        raw = np.fromfile(c.verify(pair["weight"]), dtype=np.int8)
        sw = np.fromfile(c.verify(pair["scale"]), "<f4").item()
        scores = {}
        for layout in LAYOUTS:
            w = matrix_view(raw, layout)
            acc = q[np.arange(rows), cols].astype(np.int32)[:, None] * w[cols].astype(np.int32)
            scores[layout] = min(
                int(np.count_nonzero(converted(acc, np.float32(sw), sr, o) != gold[branch])) for o in ORDERS
            )
        layout = min(scores, key=scores.get)
        require(list(scores.values()).count(scores[layout]) == 1, "ambiguous weight order: " + str(scores))
        w = matrix_view(raw, layout)
        all_errors = {}
        for case, (q2, s2, y2) in data.items():
            mism = {o: 0 for o in ORDERS}
            for j in range(0, N, 512):
                acc = ints(q2, w[:, j : j + 512])
                for order in ORDERS:
                    mism[order] += int(
                        np.count_nonzero(
                            converted(acc, np.float32(sw), s2, order) != y2[branch][:, j : j + 512]
                        )
                    )
            all_errors[case] = mism
            for o in ORDERS:
                totals[o] += mism[o]
        report[branch] = dict(layout=layout, basis_layout_mismatches=scores, errors=all_errors)
        print(
            f"M{rows} {branch}: weight layout {layout}, basis mismatches {scores[layout]}, FP16 bit mismatches {all_errors}",
            flush=True,
        )
    order = min(totals, key=totals.get)
    result = dict(branches=report, order=order, totals=totals, exact=totals[order] == 0)
    c.write(d / "oracle-math.json", result)
    print(
        f"M{rows} nearest tested scale order {order}; {totals[order]} mismatches across both projections",
        flush=True,
    )
    return result


def inputs(d, rows, how):
    rng = np.random.default_rng(31003 + rows)
    if how == "basis":
        x = np.zeros((rows, K), "<f2")
        cols = np.linspace(0, K - 1, rows, dtype=np.int64)
        x[np.arange(rows), cols] = np.where(np.arange(rows) % 2, 1.0, -1.0)
    else:
        x = (rng.normal(size=(rows, K)) * np.exp2(np.arange(rows) % 9 - 4)[:, None]).astype("<f2")
        x[0] = 0
        x[1, ::37] = -x[1, 0]
    f = d / "x.bin"
    x.tofile(f)
    return c.identity(f)


def references(out, stage, assets, env, desc):
    runner = c.verify(assets["tools"]["ac_om_runner"])
    atc = Path(shutil.which("atc", path=env["PATH"])).resolve(strict=True)
    result = {}
    for role, rows in (("body", 64), ("verify", 16)):
        original = assets["models"][role]
        msg = load_model(desc, original["air"])
        branch = pairs(msg.graph[0])[0]
        table = {n.name: n for n in msg.graph[0].op}
        weights = {
            k: {t: c.identity(file_path(table[v[t]])) for t in ("weight", "scale")} for k, v in branch.items()
        }
        models = {}
        cases = {}
        xs = {}
        for mode in ("prefix", "oracle", "native", "shared"):

            def make(d):
                new, abi = single_graph(msg, original["abi"], rows, mode)
                return save_graph(d, new, abi)

            graph = stage(f"{role}-air-{mode}", [original["air"], weights, mode], make)
            models[mode] = stage(
                f"{role}-atc-{mode}",
                [graph["air"], assets["binding"], c.identity(atc)],
                lambda d: compile_om(d, graph, atc, env, dump=mode in ("native", "shared")),
            )
        prefix = models["prefix"]
        pin = {
            s["name"]: c.identity(Path(original["input_dir"]) / (s["name"] + ".bin"))
            for s in prefix["abi"]["inputs"]
        }

        def snap(d, model, ins):
            outputs, changed = capture(d / "device", runner, model, ins, env)
            require(not changed, "Gate/Up input was modified")
            return outputs

        full = stage(
            role + "-real",
            [prefix["om"], pin, assets["tools"]["ac_om_runner"]],
            lambda d: snap(d, prefix, pin),
        )
        xs["real"] = full["x"]
        cases["real"] = {k: v for k, v in full.items() if k != "x"}
        for test in ("basis", "stress"):
            x = stage(role + "-x-" + test, [rows, test], lambda d: inputs(d, rows, test))
            xs[test] = x
            cases[test] = stage(
                role + "-ref-" + test,
                [models["oracle"]["om"], x],
                lambda d: snap(d, models["oracle"], {"x": x}),
            )
        math = stage(role + "-math", [weights, cases], lambda d: audit_math(d, weights, cases, rows))
        result[role] = dict(rows=rows, models=models, cases=cases, x=xs, math=math, weights=weights)
    return result
