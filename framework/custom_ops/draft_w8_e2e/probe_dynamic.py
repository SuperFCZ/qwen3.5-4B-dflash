"""Isolate an installed DynamicQuant provider on Target Prefill, using archived short prompts."""

import argparse
from collections import Counter
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import traceback

from winners import c, require
from history import read_history, write_candidate_plan, SUITE
from experiment import Stages
from run_q import run_path
from targets import receiver_env, compile_target
from cache_backend import audit_compiled
from bench import read_om_json, embedded
from gate_plan import elf_sections
from tools.audit_internal_custom_ops import _read_ops_info

FAMILIES = (
    "DynamicQuant",
    "QuantBatchMatmulV4444",
    "AdnRmsNorm",
    "AdnFusedInferAttention",
    "ChunkGatedDeltaRule",
    "CacheUpdate",
)


def paths(s):
    return list(dict.fromkeys(str(Path(v).resolve()) for v in s.split(os.pathsep) if v))


def add_provider(base, vendor):
    """Keep the existing ADN/GDR/quant-matmul/cache registrations and add one sourced package."""
    root = Path(vendor).resolve(strict=True)
    script = root / "bin/set_env.bash"
    require(script.is_file(), "missing vendor environment script")
    source_info, errors = _read_ops_info(root, root.parent, "ascend310p", "DynamicQuant")
    require(not errors, "invalid 310P DynamicQuant registration: " + "; ".join(errors))
    keys = ("ASCEND_CUSTOM_OPP_PATH", "LD_LIBRARY_PATH", "PYTHONPATH", "PATH", "ASCEND_OPP_PATH")
    cmd = 'source "$1" >&2 || exit $?; shift; for k; do printf "%s=%s\\0" "$k" "${!k-}"; done'
    proc = subprocess.run(
        ["bash", "--noprofile", "--norc", "-c", cmd, "vendor", str(script), *keys],
        env=base,
        capture_output=True,
        check=True,
    )
    vals = dict(s.split("=", 1) for s in os.fsdecode(proc.stdout).split("\0") if s)
    require(set(vals) == set(keys), "incomplete sourced vendor environment")
    previous = paths(base.get("ASCEND_CUSTOM_OPP_PATH", ""))
    require(str(root) not in previous, "DynamicQuant vendor already present in this comparison")
    require(
        paths(vals["ASCEND_CUSTOM_OPP_PATH"]) == [str(root), *previous],
        "sourcing must add only the new DynamicQuant root at highest priority",
    )
    require(paths(vals["ASCEND_OPP_PATH"]) == paths(base.get("ASCEND_OPP_PATH", "")), "base CANN OPP changed")
    for key in ("LD_LIBRARY_PATH", "PYTHONPATH", "PATH"):
        old = paths(base.get(key, ""))
        new = paths(vals[key])
        require(
            [x for x in new if x in old] == old
            and all(Path(x).is_relative_to(root) for x in new if x not in old),
            "source changed unrelated search paths: " + key,
        )
    return dict(base, **vals), dict(
        script=c.identity(script), ops_info=source_info, order=paths(vals["ASCEND_CUSTOM_OPP_PATH"])
    )


def inventory(model):
    selected = {}
    bindings = []
    counts = Counter()
    for n in read_om_json(c.verify(model["dump"])):
        counts[n["type"]] += 1
        if n["type"] not in FAMILIES:
            continue
        a = {v["key"]: v.get("value") or {} for v in n["attributes"]}
        get = lambda key, typ: a.get(key, {}).get(typ)
        engine = get("_ge_attr_op_kernel_lib_name", "s")
        bid = get("_kernel_bin_id", "s") or ""
        symbol = get("_kernelname", "s")
        size = int(get("_tbeKernelSize", "i") or 0)
        require(n["name"] not in selected, "duplicate compiled target node")
        selected[n["name"]] = dict(
            role=n["type"],
            engine=engine,
            vendor=get("custom_op_file_path", "s"),
            block=get("tvm_blockdim", "i"),
            text=None,
        )
        if engine == "AIcoreEngine" and symbol and size > 0 and re.search("[0-9a-f]{64}$", bid):
            bindings.append(dict(node=n["name"], bytes=size, symbol=symbol, sha256=bid[-64:]))
    # Only linked machine text is compared; temporary ELF filenames can change between builds.
    if bindings:
        om = c.verify(model["om"])
        try:
            images = embedded(om, bindings)
        except ValueError as e:
            print("Cannot bind every auxiliary ELF; checking DynamicQuant separately:", e, flush=True)
            quant = [x for x in bindings if selected[x["node"]]["role"] == "DynamicQuant"]
            images = embedded(om, quant) if quant else {}
        for k, v in images.items():
            selected[k]["text"] = v["executable_sections"]
    return selected, dict(counts)


def package_fingerprint(vendor, family):
    """Summarize installed device code; this is separate from code selected in an OM."""
    root = Path(vendor) / "op_impl/ai_core/tbe"
    key = family.lower()
    matches = lambda p: key in p.stem.replace("_", "").lower()
    inputs = sorted(p for p in root.glob("*/dynamic/*.cpp") if matches(p))
    objects = [p for d in (root / "kernel/ascend310p").glob("*") if matches(d) for p in d.glob("*.o")]
    sections = sorted({(s["bytes"], s["sha256"]) for f in objects for s in elf_sections(f.read_bytes())})
    return dict(sources=sorted({c.sha(p) for p in inputs}), objects=len(objects), text=sections)


def compare(before, after, root):
    require(before.keys() == after.keys(), "compiled projection/native node inventory differs")
    roles = {}
    for name, new in after.items():
        old = before[name]
        role = new["role"]
        require(old["role"] == role, "compiled op type changed: " + name)
        g = roles.setdefault(role, dict(nodes=0, source_changes=0, code_changes=0, unbound_text=0, chosen=0))
        g["nodes"] += 1
        g["source_changes"] += old["vendor"] != new["vendor"]
        if not old["text"] or not new["text"]:
            g["unbound_text"] += 1
        else:
            g["code_changes"] += old["text"] != new["text"] or old["block"] != new["block"]
        if role == "DynamicQuant":
            require(new["engine"] == "AIcoreEngine", "unexpected DynamicQuant engine: " + str(new["engine"]))
            v = Path(new["vendor"]).resolve() if new["vendor"] else None
            g["chosen"] += v == root or (v is not None and root in v.parents)
        else:
            require(
                (new["engine"], new["vendor"]) == (old["engine"], old["vendor"]),
                "unrelated backend/registration changed: " + name,
            )
    require(set(roles) == set(FAMILIES), "missing a measured Target operator family")
    return roles


def probe_args(base_command, plan, batch, index):
    args = list(base_command)

    def value(k):
        require(args.count(k) == 1, "unexpected archived runner option " + k)
        return args[args.index(k) + 1]

    require(
        value("--mode") == "dflash"
        and value("--model-kind") == "chunk"
        and "--low-memory" in args
        and "--trace-rounds" in args,
        "requires measured DFlash-only chunk runner",
    )
    for k, v in (
        ("--model", plan["path"]),
        ("--model-sha256", plan["sha256"]),
        ("--prompt-batch", batch["path"]),
        ("--prompt-batch-sha256", batch["sha256"]),
        ("--output", str(index)),
        ("--warmup", "0"),
        ("--repetitions", "1"),
        ("--max-new-tokens", "1"),
    ):
        value(k)
        args[args.index(k) + 1] = v
    return args


def run_anchors(d, history, prior, models, drafts, env):
    d.mkdir()
    plan = write_candidate_plan(d / "prefill-plan.txt", history, models, drafts)
    cmdref = prior.completed("generation")["result"]
    base_cmd = c.read(c.verify(cmdref["command"]))
    runner = prior.completed("generation-runner")["result"]["file"]
    require(Path(base_cmd[0]).resolve() == c.verify(runner), "runner changed since measurement")
    selected = [p for p in history["prompts"] if p["group"] == "short"]
    require(
        len(selected) == 8 and all(0 < len(p["tokens"]) <= 64 for p in selected),
        "expected eight single-chunk prompts",
    )
    lines = c.verify(history["batch"]).read_text().splitlines()
    names = {p["id"] for p in selected}
    batch = d / "prompts.txt"
    batch.write_text("\n".join([lines[0], *[l for l in lines[1:] if shlex.split(l)[0] in names], ""]))
    ref = c.identity(batch)
    index = d / "runner-batch.json"
    cmd = probe_args(base_cmd, plan, ref, index)
    c.write(d / "command.json", cmd)
    with (d / "runner.log").open("x") as log, subprocess.Popen(
        cmd, cwd=d, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace"
    ) as proc:
        for l in proc.stdout:
            log.write(l)
            if not l.startswith("{"):
                print(l, end="", flush=True)
        if proc.wait():
            raise RuntimeError("prefill probe failed; see " + str(d / "runner.log"))
    idx = c.read(index)
    require(
        idx["fake_acl"] is False
        and idx["model_sha256"] == plan["sha256"]
        and idx["prompt_batch_sha256"] == ref["sha256"]
        and idx["order"] == "DFlash only; ordinary baseline external"
        and [v["id"] for v in idx["cases"]] == [p["id"] for p in selected],
        "wrong prefill probe batch",
    )
    rows = []
    for prompt, rec in zip(selected, idx["cases"]):
        name = prompt["id"]
        f = Path(str(index) + ".cases") / (name + ".json")
        raw = c.read(f)
        require(
            Path(rec["report"]).resolve() == f
            and raw["status"] == rec["status"] == "PASS"
            and raw["cpu_fallback"] is False
            and raw["prompt_token_ids"] == prompt["tokens"]
            and raw["model"]["sha256"] == plan["sha256"]
            and raw["device_id"] == 0
            and raw["limits"]["max_new_tokens"] == 1
            and "ordinary" not in raw
            and raw["dflash"]["warmup"] == 0
            and raw["dflash"]["repetitions"] == 1,
            "unbound anchor " + name,
        )
        (m,) = raw["dflash"]["measurements"]
        (t,) = m["generated_token_ids"]
        require(
            len(m["stage_ms"]["target_prefill"]) == 1
            and all(not m["stage_ms"].get(k) for k in ("draft", "target_verify", "target_decode"))
            and m["counters"]["drafted_tokens"] == 0
            and m["counters"]["graph_calls"] == 1,
            "non-prefill work in anchor probe",
        )
        old = c.read(c.verify(history["cases"][name]))
        prev = c.read(c.verify(cmdref["reports"][name]))
        historical = old["dflash"]["measurements"][0]["generated_token_ids"][0]
        baseline = prev["dflash"]["measurements"][0]["generated_token_ids"][0]
        rows.append(
            dict(prompt=name, historical=historical, previous=baseline, candidate=t, report=c.identity(f))
        )
        print(name, "historical", historical, "previous", baseline, "candidate", t, flush=True)
    before = sum(r["previous"] == r["historical"] for r in rows)
    after = sum(r["candidate"] == r["historical"] for r in rows)
    changed = sum(r["candidate"] != r["previous"] for r in rows)
    print(f"Prefill anchors, previous → candidate: {before}/8 → {after}/8; changed {changed}/8", flush=True)
    return dict(
        rows=rows,
        previous_matches=before,
        matches=after,
        changed=changed,
        device_calls=8,
        runner_index=c.identity(index),
    )


def run(out, args, state):
    source = run_path(out.parent, args.from_run, "w8-e2e-")
    st = Stages(out, source)
    history = read_history(SUITE)
    result = c.read(source / "result.json")
    prior_target = c.read(source / "target.json")
    require(
        result["status"].startswith("E2E_MEASURED") and result["targets"] == prior_target["models"],
        "requires a completed generation",
    )
    active = [Path(v).resolve() for v in prior_target["vendors"]]
    cann = Path(os.environ["CANN_ROOT"]).resolve(strict=True)
    vendor = args.provider.resolve(strict=True)
    base = receiver_env(cann, os.environ, active)
    env, binding = add_provider(base, vendor)
    state["sourced"] = binding
    old = result["targets"]["target_prefill"]
    cache = c.read(c.verify(old["cache_backend"]))
    old_model = dict(om=old["om"], dump=cache["dump"])
    orig, count = inventory(old_model)
    print(
        "Previous Prefill providers",
        [(k, sorted({str(x["vendor"]) for x in orig.values() if x["role"] == k})) for k in FAMILIES],
        flush=True,
    )
    print(
        "GroupedMatmul nodes",
        sum(v for k, v in count.items() if k.lower().startswith("groupedmatmul")),
        "sourced OPP order",
        binding["order"],
        flush=True,
    )
    state["sibling_comparisons"] = {}
    for family, name in (
        ("QuantBatchMatmulV4444", "customize_quantMatmul"),
        ("ChunkGatedDeltaRule", "customize_linearAttention_new"),
    ):
        old = {x["vendor"] for x in orig.values() if x["role"] == family and x["vendor"]}
        other = vendor.parent / name
        if len(old) == 1 and other.is_dir():
            a = package_fingerprint(old.pop(), family)
            b = package_fingerprint(other, family)
            value = dict(
                same_source=a["sources"] == b["sources"] if a["sources"] and b["sources"] else None,
                same_prebuilt_text=a["text"] == b["text"] if a["text"] and b["text"] else None,
                previous=a,
                shared=b,
            )
            state["sibling_comparisons"][family] = value
            print(
                f"{family} shared candidate vs current: same C++ {value['same_source']}, same prebuilt executable sections {value['same_prebuilt_text']}",
                flush=True,
            )
    if args.inspect_only:
        return "ENVIRONMENT_INSPECTED"
    atc = Path(shutil.which("atc", path=env["PATH"])).resolve(strict=True)
    require(atc == Path(shutil.which("atc", path=base["PATH"])).resolve(strict=True), "compiler changed")
    source_graph = st.completed("reconstruct-target_prefill")["result"]
    require(source_graph["air"] == old["source_air"], "different Target Prefill AIR")
    d = out / "prefill"
    d.mkdir()
    model = compile_target(d, "target_prefill", source_graph, history, env, atc)
    backend = audit_compiled(
        d, "target_prefill", c.verify(model["om"]), atc, env, prior_target["cache_provider"]["path"]
    )
    new_model = dict(om=model["om"], dump=c.read(c.verify(backend))["dump"])
    new, new_count = inventory(new_model)
    comparisons = compare(orig, new, vendor)
    state.update(
        compiled=model["om"],
        cache_backend=backend,
        comparison=comparisons,
        other_operator_counts_changed=count != new_count,
    )
    print("Compiled families", comparisons, flush=True)
    quant = comparisons["DynamicQuant"]
    if quant["chosen"] != quant["nodes"]:
        return "PROVIDER_SELECTION_UNRESOLVED"
    # Even identical instruction bytes can receive a different host tiling. Measure anchors after a verified binding.
    drafts = {}
    for gear in (16, 64):
        a = st.completed(f"air-c{gear}")["result"]
        m = st.completed(f"audit-c{gear}")["result"]
        require(m["om"] == result["drafts"][str(gear)]["om"], "different Draft OM")
        drafts[gear] = dict(abi=a["abi"], om=m["om"])
    models = dict(prior_target["models"], target_prefill=model)
    state["anchors"] = run_anchors(out / "anchors", history, st, models, drafts, env)
    return "ALL_ANCHORS_MATCH" if state["anchors"]["matches"] == 8 else "ANCHOR_DIFFERENCES"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--from-run", required=True)
    p.add_argument("--provider", type=Path, required=True)
    p.add_argument("--inspect-only", action="store_true")
    args = p.parse_args()
    out = Path(os.environ["DFLASH_OUTPUT"])
    s = {"status": "RUNNING"}
    try:
        s["status"] = run(out, args, s)
    except Exception as e:
        s.update(status="FAILED", error=repr(e))
        traceback.print_exc()
        (out / "traceback.txt").write_text(traceback.format_exc())
    c.write(out / "result.json", s)
    print(s["status"], "REPORT", out / "result.json", flush=True)
    return int(s["status"] in ("FAILED", "PROVIDER_SELECTION_UNRESOLVED"))


if __name__ == "__main__":
    raise SystemExit(main())
