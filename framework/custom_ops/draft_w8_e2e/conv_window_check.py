"""Check complete Target outputs and count materialized convolution-window transposes in msprof."""

from collections import Counter
import csv
import math
import re

import numpy as np

from winners import c, require
from run_prefill_head import capture
from diagnose_calls import export_csv
from prefill_split import ROW


def input_files(folder, abi):
    out = {}
    for port in abi["inputs"]:
        name = port["name"]
        f = c.identity(folder / (name + ".bin"))
        require(
            f["bytes"] == np.dtype(port["dtype"]).itemsize * math.prod(port["shape"]), "input size: " + name
        )
        out[name] = f
    return out


def archived(folder, abi):
    case = folder / "case"
    row = c.read(case / "execution.json")
    require(
        row["status"] == "PASS"
        and row["reference_capture"]
        and not row["cpu_fallback"]
        and row["repeat"]
        and row["guards"]
        and row["timing"]["repetitions"] == 0
        and row["allocation"]["io"] == "normal"
        and row["allocation"]["workspace"] == "huge-first",
        "expected the checked two-call receiver sample in " + str(folder),
    )
    return dict(
        inputs=input_files(case / "inputs", abi),
        golden={v["name"]: c.identity(case / (v["name"] + "-actual-0.bin")) for v in abi["outputs"]},
        mutations={k: c.identity(case / (k + "-input-after.bin")) for k in row["mutated_inputs"]},
        workspace=row["allocation"]["workspace_bytes"],
    )


def compare_bytes(values, frozen, label):
    for kind, expected, actual in zip(("outputs", "ref inputs"), frozen, values, strict=True):
        require(set(actual) == set(expected), f"{label}: different {kind} names")
        bad = [n for n in expected if actual[n]["sha256"] != expected[n]["sha256"]]
        require(not bad, f"{label}: bitwise difference in {kind}: {bad}")


def integers(record, dtype):
    return np.fromfile(c.verify(record), dtype=dtype).tolist()


def validate_round(samples, case):
    prompt = case["prompt_token_ids"]
    m = case["dflash"]["measurements"][0]
    first = next(r for r in m["rounds"] if r["stage"] == "target_verify")
    ids = [m["generated_token_ids"][0], *first["proposed_token_ids"]]
    for role, values, pos in (("body", prompt, 0), ("verify", ids, len(prompt))):
        ins = samples[role]["inputs"]
        observed = integers(ins["input_ids"], "<i8")
        require(
            integers(ins["valid_rows"], "<i2") == [len(values)]
            and integers(ins["start_position"], "<i8") == [pos]
            and observed == values + [0] * (len(observed) - len(values)),
            f"{role} snapshot differs from the production prompt/round",
        )
    a, b = samples["body"]["golden"], samples["verify"]
    states = set(b["inputs"]) - {"input_ids", "start_position", "valid_rows"}
    require(
        all(a[k]["sha256"] == b["inputs"][k]["sha256"] for k in states), "Verify state differs from Prefill"
    )
    require(
        integers(b["golden"]["target_top1"], "<i8")[: len(ids)] == first["target_token_ids"]
        and integers(b["golden"]["accepted_count"], "<i8") == [len(ids) - 1]
        and len(first["accepted_draft_token_ids"]) == len(ids) - 1 == 15,
        "expected archived full-accept round",
    )


def boundaries(folder, runner, models, old_verify, proofs, sample, env):
    checks = []
    for proof in proofs:
        if proof["start"] == 0:
            continue
        name = f"prefill-{proof['start']}-{proof['valid']}"
        ins = input_files(c.verify(proof["hidden"]).parent / "inputs", models["body"]["abi"])
        golden = dict(proof["candidate"])
        del golden["target_top1"]
        golden[ROW] = proof["hidden"]
        outputs = capture(folder / name, runner, models["body"], ins, env)
        compare_bytes(outputs, (golden, proof["candidate_mutations"]), name)
        checks.append(name)
        print(
            f"Prefill pos={proof['start']} valid={proof['valid']}: {len(golden)} complete outputs and reference writes bitwise PASS",
            flush=True,
        )
    require(len(checks) == 2, "missing archived full and partial blocks")
    # The frozen first round accepted every proposal. Change proposal r only:
    # predictions for rows before r are causal and the first mismatch commits exactly r rows.
    base = sample["inputs"]
    original = np.fromfile(c.verify(base["input_ids"]), dtype="<i8")
    for commit in (1, 3):
        d = folder / f"verify-{commit}"
        d.mkdir()
        ids = original.copy()
        ids[commit] = (int(ids[commit]) + 1) % 248320
        path = d / "input_ids.bin"
        ids.tofile(path)
        inputs = dict(base, input_ids=c.identity(path))
        reference = capture(d / "reference", runner, old_verify, inputs, env)
        require(
            integers(reference[0]["accepted_count"], "<i8") == [commit - 1], "wrong forced first mismatch"
        )
        actual = capture(d / "candidate", runner, models["verify"], inputs, env)
        compare_bytes(actual, reference, d.name)
        checks.append(d.name)
        print(
            f"Verify accepted={commit - 1}, committed={commit}: {len(actual[0])} complete outputs and reference writes bitwise PASS",
            flush=True,
        )
    return checks


def op_summary(folder, rows):
    (path,) = (folder / "profiler").rglob("op_summary*.csv")
    visits, large, gathers = Counter(), Counter(), Counter()
    sums = Counter()
    unshaped = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for raw in csv.DictReader(stream):
            r = {re.sub("[^a-z0-9]", "", k.lower()): v for k, v in raw.items() if k}
            typ = r.get("optype", "")
            name = r.get("opname", "").rsplit("/", 1)[-1]
            if typ == "QuantBatchMatmulV4444":
                visits[name] += 1
            is_transpose = typ.lower().startswith("transpose") or typ == "Permute"
            is_gather = "target_conv_window_t" in name or typ.lower().startswith("gather")
            if not (is_transpose or is_gather):
                continue

            def first_shape(field):
                dims = tuple(map(int, re.findall(r"-?\d+", (r.get(field) or "").split(";")[0])))
                return dims[1:] if dims[:1] == (1,) else dims

            inp, out = first_shape("inputshapes"), first_shape("outputshapes")
            if is_transpose and not (inp and out):
                unshaped.append(name)
            us = float(r["taskdurationus"])
            require(math.isfinite(us) and us >= 0, "invalid device task duration at " + name)
            if is_transpose:
                sums["all_transpose"] += us
                if inp == (8192, rows, 4) and out == (rows, 8192, 4):
                    large[name] += 1
                    sums["window_transpose"] += us
            elif (inp == (8192, rows + 4) and out == (8192, 4)) or "target_conv_window_t" in name:
                gathers[name] += 1
                sums["direct_gather"] += us
    require(
        len(visits) == (248 if rows == 64 else 249) and set(visits.values()) == {2},
        "expected two complete executions of the single Target OM",
    )
    require(all(n == 2 for n in large.values()), "window transpose count differs between executions")
    return dict(
        csv=c.identity(path),
        large_nodes=len(large),
        large_names=sorted(large),
        unshaped_transposes=sorted(set(unshaped)),
        large_ms=sums["window_transpose"] / 2000,
        all_transpose_ms=sums["all_transpose"] / 2000,
        observed_gather_tasks=sum(gathers.values()) / 2,
        observed_gather_ms=sums["direct_gather"] / 2000,
    )


def profiled(folder, runner, model, sample, rows, old, cli, env):
    # Request the baseline receiver's shared-workspace capacity when the new OM fits.
    def replay(d, workspace, profiler):
        results = capture(d, runner, model, sample["inputs"], env, workspace=workspace, profiler=profiler)
        compare_bytes(results, (sample["golden"], sample["mutations"]), f"profile M{rows}")

    d = folder / "same-workspace"
    try:
        replay(d, sample["workspace"], cli)
    except (RuntimeError, ValueError):
        log = d / "capture.log"
        if not log.exists() or "incompatible receiver workspace capacity" not in log.read_text(
            errors="replace"
        ):
            raise
        d = folder / "model-workspace"
        replay(d, 0, cli)
    used = c.read(d / "case/execution.json")["allocation"]["workspace_bytes"]
    try:
        export_csv(d, cli, env)
        new = op_summary(d, rows)
    except (RuntimeError, ValueError, KeyError, OSError) as error:
        print(f"M{rows} msprof attribution incomplete: {error}; trace {d / 'profiler'}", flush=True)
        return dict(previous=old, incomplete=str(error), trace=str(d / "profiler"), workspace_bytes=used)
    observed = "observed (some shapes missing)" if new["unshaped_transposes"] else "per Execute"
    print(
        f"M{rows} msprof large window Transpose {old['large_nodes']} -> {new['large_nodes']} {observed}, "
        f"{old['large_ms']:.3f} -> {new['large_ms']:.3f} ms; all Transpose "
        f"{old['all_transpose_ms']:.3f} -> {new['all_transpose_ms']:.3f} ms; "
        f"observed direct gather {new['observed_gather_tasks']:g} tasks / {new['observed_gather_ms']:.3f} ms; "
        f"complete outputs and ref inputs bitwise PASS",
        flush=True,
    )
    return dict(previous=old, candidate=new, workspace_bytes=used)
