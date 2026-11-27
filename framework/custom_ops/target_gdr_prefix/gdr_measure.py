"""Matched native states, unprofiled wall time, and per-pass msprof device-task time."""

from collections import Counter, defaultdict
import csv
import math
from pathlib import Path
import re
from statistics import median
import subprocess

import numpy as np

from gdr_graph import GDR, COMMIT
from pair_graph import c, require
from conv_window_check import archived, compare_bytes, integers
from run_prefill_head import capture
from full_replay import prepare_case
from diagnose_calls import export_csv
from snapshot_profiles import check_application
from pair_measure import stats


def state_files(out, base, assets, previous):
    model = assets["models"]["verify"]
    profile = c.verify(previous["profiles"]["verify"]["candidate"]["csv"])
    live = next(d for d in profile.parents if (d / "case/execution.json").is_file())
    st = c.read(base / "stages.json")
    checked = c.verify(st["state-check"]).parent
    sources = {16: live, 1: checked / "verify-1/candidate", 3: checked / "verify-3/candidate"}
    frozen = {}
    for length, src in sources.items():
        record = archived(src, model["abi"])
        require(
            integers(record["golden"]["accepted_count"], "<i8") == [length - 1]
            and integers(record["inputs"]["valid_rows"], "<i2") == [16],
            "wrong saved Verify boundary",
        )
        frozen[length] = record
    return frozen, profile


def pass_times(path, sites, *, custom=False, calls=2):
    names = {v["first"]: "first" for v in sites.values()}
    names.update({(v.get("new") or v["commit"]): "commit" for v in sites.values()})
    visits = defaultdict(list)
    metrics = defaultdict(list)
    model_ids = set()
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        for raw in csv.DictReader(f):
            r = {re.sub("[^a-z0-9]", "", k.lower()): v for k, v in raw.items() if k}
            if r.get("optype") not in (GDR, COMMIT):
                continue
            name = r["opname"].rsplit("/", 1)[-1]
            require(
                name in names and r["optype"] == (COMMIT if custom and names[name] == "commit" else GDR),
                "unexpected GDR device task: " + name,
            )
            d = float(r["taskdurationus"])
            require(math.isfinite(d) and d > 0, "invalid task duration")
            visits[name].append(d)
            model_ids.add(r.get("modelid", ""))
            for k in ("mactimeus", "vectimeus", "mte2timeus", "mte3timeus", "aicoretimeus"):
                try:
                    value = float(r[k])
                    if math.isfinite(value):
                        metrics[names[name], k].append(value)
                except (KeyError, ValueError, TypeError):
                    pass
    require(
        set(visits) == set(names) and len(model_ids) == 1 and all(len(z) == calls for z in visits.values()),
        "incomplete first/commit profile on the sole loaded OM",
    )
    result = {}
    for side in ("first", "commit"):
        count = sum(v == side for v in names.values())
        result[side] = dict(
            nodes=count,
            task_sum_ms=sum(sum(vals) for n, vals in visits.items() if names[n] == side) / (1000 * calls),
            per_node_median_counters_us={
                k: median(v) for (s, k), v in metrics.items() if s == side and len(v) == count * calls
            },
        )
    return dict(**result, csv=c.identity(path))


def full_profile(d, length, model, sample, site_map, runner, env, cli, *, custom=False):
    vals = capture(d, runner, model, sample["inputs"], env, profiler=cli)
    compare_bytes(vals, (sample["golden"], sample["mutations"]), f"full Verify L{length}")
    export_csv(d, cli, env)
    (path,) = (d / "profiler").rglob("op_summary*.csv")
    return pass_times(path, site_map, custom=custom)


def length_files(d, operands, name, physical=16):
    inputs = dict(operands)
    if physical < 16:
        for key in ("query", "key", "value", "g", "beta"):
            scale = 32 * 128 if key in ("query", "key", "value") else 32
            dt = "<f4" if key == "g" else "<f2"
            a = np.fromfile(c.verify(inputs[key]), dt).reshape(16, scale)[:physical]
            f = d / (key + ".bin")
            a.tofile(f)
            inputs[key] = c.identity(f)
    f = d / "length.bin"
    np.array([name], "<i2").tofile(f)
    inputs["length"] = c.identity(f)
    return inputs


def freeze(d, model, inputs, runner, env):
    vals, changed = capture(
        d / "device", runner, model, {s["name"]: inputs[s["name"]] for s in model["abi"]["inputs"]}, env
    )
    require(not changed, "GDR modified a round-start operand")
    return vals["state"]


def equal_state(got, want):
    if got["sha256"] == want["sha256"]:
        return dict(exact=True, different=0)
    a = np.fromfile(c.verify(want), "<u4")
    b = np.fromfile(c.verify(got), "<u4")
    require(a.shape == b.shape, "state shape mismatch")
    ids = np.flatnonzero(a != b)
    i = int(ids[0])
    return dict(exact=False, different=len(ids), first=i, old_bits=hex(int(a[i])), new_bits=hex(int(b[i])))


def timed(d, model, inputs, golden, runner, workspace, env, cli=None):
    case = d / "case"
    prepare_case(
        case,
        model["abi"],
        {"files": {v["name"]: inputs[v["name"]] for v in model["abi"]["inputs"]}},
        {"state": golden},
    )
    cmd = list(
        map(str, [runner, 0, c.verify(model["om"]), case, case / "inputs", workspace, "huge-first", "normal"])
    )
    if cli:
        cmd = [cli["alias"], *cli["flags"], "--output=" + str(d / "profiler"), *cmd]
    c.write(d / "command.json", cmd)
    log = d / "runner.log"
    with log.open("xb") as f:
        rc = subprocess.run(cmd, cwd=d, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    receipt = case / "execution.json"
    check_application(rc, receipt.is_file(), log.read_text(errors="replace"))
    e = c.read(receipt)
    z = e["timing"]
    require(
        e["status"] == "PASS"
        and e["guards"]
        and e["readonly"]
        and e["repeat"]
        and not e["cpu_fallback"]
        and e["allocation"]["golden_outputs"] == 1
        and z["protocol"] == "continuous-v1"
        and z["warmup"] == 5
        and z["repetitions"] == 30
        and z["timed_tail_checked"],
        "incomplete single-layer timing",
    )
    return [1000 * x for x in z["execute_sync"]["samples_ms"]]


def pair_times(d, candidate, native, ins, gold, runner, workspace, env, newenv):
    all_samples = [[], []]
    ratios = []
    for ri, order in enumerate(((0, 1), (1, 0), (0, 1))):
        med = {}
        for side in order:
            m, e = (candidate, newenv) if side else (native, env)
            obs = timed(d / f"r{ri + 1}-{side}", m, ins, gold, runner, workspace, e)
            med[side] = median(obs)
            all_samples[side].extend(obs)
        ratios.append(med[0] / med[1])
    return dict(native_us=stats(all_samples[0]), candidate_us=stats(all_samples[1]), round_ratios=ratios)


def histogram(base, control):
    counts = Counter()
    for ref in control["reports"].values():
        m = c.read(c.verify(ref))["dflash"]["measurements"]
        for obs in m:
            counts.update(
                1 + len(r["accepted_draft_token_ids"]) for r in obs["rounds"] if r["stage"] == "target_verify"
            )
    golden = base["groups"]["all"]["conv_window"]
    require(
        sum(counts.values()) == golden["verify"]["count"]
        and sum(k * v for k, v in counts.items()) == golden["accepted"] + sum(counts.values()),
        "different full generation trace",
    )
    return dict(sorted(counts.items()))
