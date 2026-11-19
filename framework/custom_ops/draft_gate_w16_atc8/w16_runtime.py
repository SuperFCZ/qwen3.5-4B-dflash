"""Use the frozen Gate8 ACL OM binaries and their original pre-execute socket."""

import os
from pathlib import Path
import socket
import struct
import subprocess

from w16_common import LAYERS, OVERLAP, SERIAL, SINGLE, VARIANTS, c, flow, host, require, stats
from measurement import mismatch_evidence, use_specs, verify_maps, verify_outputs


def invocation(folder, runner, model, mode, env, selected, src, row):
    parent, child = socket.socketpair()
    with parent, child, (folder / "runner.log").open("xb") as log:
        parent.setsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED, 1)
        fd = child.fileno()
        endpoint = os.readlink("/proc/self/fd/" + str(fd))
        argv = [
            str(c.verify(runner)),
            "0",
            str(c.verify(model["om"])),
            str(folder),
            str(folder / "inputs"),
            str(folder / "golden"),
            mode,
            str(fd),
        ]
        c.write(folder / "runner.command.json", dict(argv=argv, env=env, fd=fd, endpoint=endpoint))
        with subprocess.Popen(
            argv, env=env, stdout=log, stderr=subprocess.STDOUT, pass_fds=(fd,)
        ) as child_proc:
            child.close()
            parent.settimeout(600)
            try:
                size = struct.calcsize("3i")
                token, control, flags, _ = parent.recvmsg(1, socket.CMSG_SPACE(size))
                creds = [
                    struct.unpack("3i", msg)
                    for level, kind, msg in control
                    if level == socket.SOL_SOCKET and kind == socket.SCM_CREDENTIALS
                ]
                require(
                    token == b"?" and not flags & socket.MSG_CTRUNC and len(creds) == 1,
                    "original Runner credentials missing",
                )
                pid, uid, gid = creds[0]
                proc = Path("/proc") / str(pid)
                require(
                    pid == child_proc.pid
                    and uid == os.getuid()
                    and gid == os.getgid()
                    and os.readlink(proc / "fd" / str(fd)) == endpoint
                    and c.identity(proc / "exe") == runner
                    and (proc / "cmdline").read_bytes().split(b"\0")[:-1] == [os.fsencode(s) for s in argv],
                    "wrong Runner PID, FD, executable or argv",
                )
                actual = dict(
                    os.fsdecode(x).split("=", 1) for x in (proc / "environ").read_bytes().split(b"\0") if x
                )
                require(actual == env, "child environment differs from exact admitted environment")
                live = host.dsos.maps_rows((proc / "maps").read_text())
                require(
                    live == host.dsos.maps_rows((folder / "pre-maps.txt").read_text()), "live pre-maps differ"
                )
                row["maps"]["pre"] = verify_maps(folder / "pre-maps.txt", selected, src)
                row["gate"] = dict(
                    pid=pid,
                    child_pid=child_proc.pid,
                    uid=uid,
                    gid=gid,
                    fd=fd,
                    endpoint=endpoint,
                    runner=runner,
                    control_token="?",
                    child_environment=actual,
                    live_pre_maps_equal=c.identity(folder / "pre-maps.txt"),
                )
                parent.sendall(b"G")
                row["launch_authorized"] = True
                require(child_proc.wait(timeout=600) == 0, "OM Runner failed: " + str(folder / "runner.log"))
            finally:
                parent.close()
                if child_proc.poll() is None:
                    try:
                        child_proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        child_proc.kill()
                        child_proc.wait()


def case_run(
    folder,
    src,
    built,
    model,
    gear,
    layer,
    mode,
    round_id,
    pair,
    timed,
    report,
    previous,
    *,
    spec=None,
    plan_key="16x2560x19456",
    check_outputs=verify_outputs,
    elements=311296,
    diagnose=mismatch_evidence,
    constants=None,
):
    if spec is None:
        spec = use_specs(src, gear, layer)
    binary = built["om" if layer == "full" else "runner"]
    if constants is None:
        constants = (
            src["full"][gear]["frozen"]["payloads"]
            if layer == "full"
            else {k: v for k, v in src["cases"][gear][layer]["inputs"].items() if k != "x.bin"}
        )
    require(
        model["mode"] == mode
        and model["key"] == (gear if layer == "full" else layer)
        and model["abi"] == spec["abi"]
        and model["opp_root"] == built["opp"]["install"]
        and model["private_kernel"] == built["kernel"],
        "cross-wired model/OPP/case",
    )
    folder.mkdir(parents=True)
    prior = c.read(c.verify(src["host_source"]))["binding"]["prior"]
    env = flow.build_env(prior, built)
    c.verify_opp(built["opp"])
    require(
        c.dependencies(binary["runner"], folder / "ldd.log", env) == binary["deps"],
        "runner dependencies changed",
    )
    for side in ("inputs", "outputs"):
        (folder / (side + ".tsv")).write_text(
            "".join(
                t["name"] + " " + t["dtype"] + " " + " ".join(map(str, t["shape"])) + "\n"
                for t in spec["abi"][side]
            )
        )
    for side in ("inputs", "golden"):
        directory = folder / side
        directory.mkdir()
        for name, item in spec[side].items():
            (directory / (name + ".bin")).symlink_to(c.verify(item))
    row = dict(
        gear=gear,
        layer=layer,
        mode=mode,
        round=round_id,
        pair=pair,
        folder=str(folder),
        device=0,
        profiled=False,
        launch_authorized=False,
        maps={},
        runner=binary["runner"],
        om=model["om"],
        opp_root=model["opp_root"],
        opp_kernel=built["kernel"]["elf"],
        manifest=built["manifest"],
        selected_kernels=model["atc_kernels"],
        atc_plan=model["plans"][plan_key]["contract"],
        inputs=spec["inputs"],
        golden=spec["golden"],
    )
    row.update(air=model["air"], deployment=model["deployment"], frozen_constants=constants)
    report["runs"].append(row)
    try:
        invocation(folder, binary["runner"], model, "benchmark" if timed else "smoke", env, built, src, row)
        row["maps"]["post"] = verify_maps(folder / "post-maps.txt", built, src)
        row["execution"] = c.identity(folder / "execution.json")
        suffixes = ["actual-0.bin", "actual-1.bin"] + (["benchmark-last.bin", "postcheck.bin"] if timed else [])
        row["outputs"] = {n: [c.identity(folder / (n + "-" + s)) for s in suffixes] for n in spec["golden"]}
        row["validation"] = check_outputs(row, spec, timed)
        row["bit_mismatches"] = 0
        if layer != "full":
            row.update(max_ulp=0, compared_elements=elements)
        if timed:
            require(
                previous
                and all(
                    row["validation"][k] == previous["validation"][k]
                    for k in ("workspace_bytes", "model_weight_bytes")
                ),
                "OM allocations drift since Smoke",
            )
        c.verify_opp(built["opp"])
        c.verify_tree(spec)
    except Exception:
        try:
            row["differences"] = diagnose(folder, spec)
        except (OSError, KeyError, ValueError) as exc:
            row["difference_read_error"] = str(exc)
        raise
    finally:
        row["artifacts"] = [c.identity(p) for p in sorted(folder.iterdir()) if p.is_file()]


PAIRS = {
    "schedule": (SERIAL, OVERLAP),
    "net": (SINGLE, OVERLAP),
    "resource-control": (SINGLE, SERIAL),
    "down-copy": ("down-loop", "down-2d"),
}


def verify_runtime_row(
    folder,
    row,
    src,
    selected,
    model,
    timed,
    previous,
    *,
    spec=None,
    plan_key="16x2560x19456",
    check_outputs=verify_outputs,
    elements=311296,
    constants=None,
):
    full = row["layer"] == "full"
    if spec is None:
        spec = use_specs(src, row["gear"], row["layer"])
    if constants is None:
        constants = (
            src["full"][row["gear"]]["frozen"]["payloads"]
            if full
            else {k: v for k, v in src["cases"][row["gear"]][row["layer"]]["inputs"].items() if k != "x.bin"}
        )
    require(
        row["frozen_constants"] == constants
        and row["air"] == model["air"]
        and row["deployment"] == model["deployment"],
        "model constant payload provenance mismatch",
    )
    runner = selected["om" if full else "runner"]["runner"]
    require(
        row["folder"] == str(folder)
        and row["om"] == model["om"]
        and row["selected_kernels"] == model["atc_kernels"]
        and row["manifest"] == selected["manifest"]
        and row["opp_root"] == selected["opp"]["install"]
        and row["opp_kernel"] == selected["kernel"]["elf"]
        and row["runner"] == runner
        and row["atc_plan"] == model["plans"][plan_key]["contract"]
        and row["device"] == 0
        and row["profiled"] is False
        and row["launch_authorized"] is True,
        "run file/model identity mismatch",
    )
    journal = c.read(folder / "runner.command.json")
    gate = row["gate"]
    require(
        gate["pid"] == gate["child_pid"] > 0
        and gate["fd"] == journal["fd"]
        and gate["endpoint"] == journal["endpoint"]
        and gate["runner"] == runner
        and gate["control_token"] == "?"
        and gate["child_environment"] == journal["env"] == selected["env"]["values"]
        and gate["live_pre_maps_equal"] == c.identity(folder / "pre-maps.txt")
        and journal["argv"]
        == [
            runner["path"],
            "0",
            model["om"]["path"],
            str(folder),
            str(folder / "inputs"),
            str(folder / "golden"),
            "benchmark" if timed else "smoke",
            str(gate["fd"]),
        ],
        "Runner gate evidence changed",
    )
    for side in ("pre", "post"):
        require(row["maps"][side] == verify_maps(folder / (side + "-maps.txt"), selected, src), "DSO map drift")
    for side in ("inputs", "golden"):
        require(row[side] == spec[side], "mixed frozen case")
        for name, item in spec[side].items():
            require(c.identity(folder / side / (name + ".bin")) == item, "case link changed")
    suffixes = ["actual-0.bin", "actual-1.bin"] + (["benchmark-last.bin", "postcheck.bin"] if timed else [])
    require(
        row["execution"] == c.identity(folder / "execution.json")
        and row["outputs"]
        == {n: [c.identity(folder / (n + "-" + s)) for s in suffixes] for n in spec["golden"]},
        "historic or swapped output files",
    )
    require(
        row["validation"] == check_outputs(row, spec, timed) and row["bit_mismatches"] == 0,
        "numeric/Runner evidence changed",
    )
    if full:
        require(
            len(spec["golden"]) == 11 and spec["golden"]["draft_top1"]["bytes"] == 15 * 8,
            "not all fifteen token IDs and ten full KV arrays",
        )
    else:
        require(row["max_ulp"] == 0 and row["compared_elements"] == elements, "wrong numerical coverage")
    if timed:
        require(
            previous
            and all(
                previous["validation"][k] == row["validation"][k]
                for k in ("workspace_bytes", "model_weight_bytes")
            ),
            "allocation drift",
        )


def order(gear, layers, pairs=(), *, full=False):
    if not pairs:
        return [
            (gear, layer, "smoke", 0, mode)
            for layer in layers
            for mode in ((SINGLE, OVERLAP) if full else VARIANTS)
        ]
    return [
        (gear, layer, pair, r, mode)
        for layer in layers
        for pair in pairs
        for r in range(3)
        for mode in (PAIRS[pair][::-1] if r % 2 else PAIRS[pair])
    ]


def row_key(row):
    return row["gear"], row["layer"], row["pair"], row["round"], row["mode"]


def paired(rows, gear, layer, pair):
    wanted = order(gear, [layer], [pair])
    require(
        [row_key(r) for r in rows] == wanted and len({r["folder"] for r in rows}) == 6,
        "wrong pairing or reused run",
    )
    control, candidate = PAIRS[pair]
    values = {}
    identities = {}
    per_round = {}
    for variant in PAIRS[pair]:
        picked = [r for r in rows if r["mode"] == variant]
        samples = []
        per_round[variant] = []
        for row in picked:
            require(
                row["profiled"] is False
                and row["launch_authorized"]
                and row["bit_mismatches"] == 0
                and row["atc_plan"]["block_dim"] == 8,
                "timing without safety/8-block admission",
            )
            each = stats(row["validation"]["timing"]["samples_ms"], 30)
            per_round[variant].append(each)
            samples.extend(each["samples_ms"])
        values[variant] = stats(samples, 90)
        first = picked[0]
        names = (
            "runner",
            "om",
            "opp_root",
            "opp_kernel",
            "manifest",
            "selected_kernels",
            "atc_plan",
            "inputs",
            "golden",
            "air",
            "deployment",
            "frozen_constants",
        )
        require(
            all(all(row[name] == first[name] for name in names) for row in picked),
            "variant identity drift across rounds",
        )
        identities[variant] = {name: first[name] for name in names}
        identities[variant].update(
            workspace_bytes=first["validation"]["workspace_bytes"],
            model_weight_bytes=first["validation"]["model_weight_bytes"],
        )
    ratios = [a["median_ms"] / b["median_ms"] for a, b in zip(per_round[control], per_round[candidate])]
    a, b = values[control], values[candidate]
    ratio = a["median_ms"] / b["median_ms"]
    regression = b["p95_ms"] > a["p95_ms"]
    return dict(
        variants=values,
        identities=identities,
        round_statistics=per_round,
        round_ratios=ratios,
        ratio=ratio,
        p95_regression=regression,
        decision="GO" if min(ratios) > 1 and ratio >= 1.03 and not regression else "NO_GO",
        median_saving_ms=a["median_ms"] - b["median_ms"],
        historical_samples_used=False,
    )


def decision(scheduling, net):
    # Two different samples of C; do not pretend B and A were directly paired in these rounds.
    return dict(
        schedule=scheduling,
        net=net,
        serial_minus_single_anchor_adjusted_ms=scheduling["median_saving_ms"] - net["median_saving_ms"],
        overlap_anchor_drift_ms=net["variants"][OVERLAP]["median_ms"]
        - scheduling["variants"][OVERLAP]["median_ms"],
        resource_only_cost="NOT_IDENTIFIABLE: additional W/scratch and common raw prefetch issue timing differ from A",
        decision="GO" if scheduling["decision"] == net["decision"] == "GO" else "NO_GO",
    )


def five_layer_decision(statistics, gear):
    outputs = {}
    for pair in ("schedule", "net"):
        a, b = PAIRS[pair]
        orders = {}
        for label, indices in (("control_first", (0, 2)), ("candidate_first", (1,))):
            layers = {}
            for layer in LAYERS:
                data = statistics["/".join((gear, layer, pair))]
                per = data["round_statistics"]
                sides = {
                    m: stats([x for i in indices for x in per[m][i]["samples_ms"]], 30 * len(indices))
                    for m in (a, b)
                }
                layers[layer] = dict(
                    variants=sides,
                    all_rounds_faster=all(x > 1 for x in data["round_ratios"])
                    and sides[a]["median_ms"] > sides[b]["median_ms"],
                    p95_regression=sides[b]["p95_ms"] > sides[a]["p95_ms"],
                )
            sums = {m: sum(v["variants"][m]["median_ms"] for v in layers.values()) for m in (a, b)}
            ratio = sums[a] / sums[b]
            orders[label] = dict(
                layers=layers,
                sum_isolated_medians_ms=sums,
                ratio_of_sums=ratio,
                decision="GO"
                if ratio >= 1.03
                and all(v["all_rounds_faster"] and not v["p95_regression"] for v in layers.values())
                else "NO_GO",
            )
        outputs[pair] = orders
    return dict(
        orders=outputs,
        scope="FIVE_INDEPENDENT_PROJECTIONS",
        decision="GO"
        if all(v["decision"] == "GO" for pair in outputs.values() for v in pair.values())
        else "NO_GO",
    )
