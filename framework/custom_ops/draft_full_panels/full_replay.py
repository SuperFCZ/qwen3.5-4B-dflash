"""Full-cache replay, reference capture, warmup and unprofiled paired timing."""

from pathlib import Path
import shutil

import numpy as np

from joint import c, require, OPS
from full_launch import launch
from package import command
from raw_protocol import verify_run, stats


def build_runner(out, cann, env):
    # Reuse the checked-in guarded runner. Only the first, separate control process may capture a reference.
    src = out / "runner-src"
    shutil.copytree(OPS / "draft_a_core_scheduler_flow/om", src)
    file = src / "runner.cpp"
    code = file.read_text()
    substitutions = [
        (
            'if(!benchmark && std::string(argv[6])!="smoke")',
            'const bool capture=std::string(argv[6])=="capture";\n        if(!capture && !benchmark && std::string(argv[6])!="smoke")',
        ),
        ("golden;for(const auto&t:outputs)", "golden;if(!capture)for(const auto&t:outputs)"),
        ("if(value!=golden[i])", "if(!capture && value!=golden[i])"),
        (
            r"\"scope\":\"full Draft frozen Golden checked in process\"",
            r'\"reference_capture\":"<<(capture?"true":"false")<<",\"scope\":\"checkpoint full Draft paired replay\"',
        ),
        (r"\"golden_match\":true", r'\"golden_match\":"<<(capture?"false":"true")<<"'),
    ]
    for old, new in substitutions:
        require(code.count(old) == 1, "unexpected full runner source: " + old)
        code = code.replace(old, new)
    file.write_text(code)
    dest = out / "runner-build"
    command(
        [
            "cmake",
            "-S",
            src,
            "-B",
            dest,
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCANN_ROOT=" + str(cann),
            "-DCOMMON_INCLUDE=" + str(c.QUANT / "test"),
        ],
        out / "configure.log",
        env,
        out,
    )
    command(["cmake", "--build", dest, "--parallel", "2"], out / "build.log", env, out)
    return c.identity(dest / "raw_om_runner")


def inputs(out, gear, abi, external):
    rng = np.random.default_rng(310300 + gear)
    d = out / "inputs"
    d.mkdir()
    scalar = {"start_position": 64, "valid_rows": gear, "anchor": 42, "proposal_count": 15}
    result = {}
    for item in abi["inputs"]:
        name = item["name"]
        typ = np.dtype(item["dtype"]).newbyteorder("<")
        shape = tuple(item["shape"])
        dest = d / (name + ".bin")
        if external:
            source = external / f"c{gear}" / dest.name
            require(
                source.is_file() and source.stat().st_size == np.prod(shape) * typ.itemsize,
                "invalid input: " + str(source),
            )
            shutil.copyfile(source, dest)
        else:
            a = (
                np.array([scalar[name]], dtype=typ)
                if name in scalar
                else (rng.standard_normal(shape, dtype=np.float32) * 0.125).astype(typ)
            )
            a.tofile(dest)
        if np.issubdtype(typ, np.floating):
            require(np.isfinite(np.fromfile(dest, dtype=typ)).all(), "nonfinite input: " + name)
        else:
            scalar[name] = int(np.fromfile(dest, dtype=typ)[0])
        result[name] = c.identity(dest)
    capacity = abi["outputs"][1]["shape"][2]
    require(
        0 <= scalar["start_position"] <= capacity - gear
        and 0 < scalar["valid_rows"] <= gear
        and 0 <= scalar["anchor"] < 248320
        and 0 < scalar["proposal_count"] <= 15,
        "invalid Draft replay positions/IDs",
    )
    return dict(
        files=result, scalars=scalar, source="USER_REPLAY" if external else "REAL_CHECKPOINT_SEEDED_INPUTS"
    )


def prepare_case(folder, abi, io, reference):
    folder.mkdir(parents=True)
    for side in ("inputs", "outputs"):
        (folder / (side + ".tsv")).write_text(
            "".join(
                x["name"] + " " + x["dtype"] + " " + " ".join(map(str, x["shape"])) + "\n" for x in abi[side]
            )
        )
    for sub, files in (("inputs", io["files"]), ("golden", reference or {})):
        d = folder / sub
        d.mkdir()
        for name, item in files.items():
            (d / (name + ".bin")).symlink_to(c.verify(item))


def capture_reference(folder, io, abi, gear):
    e = c.read(folder / "execution.json")
    require(
        e["status"] == "PASS"
        and e["reference_capture"] is True
        and e["golden_match"] is False
        and e["cpu_fallback"] is False
        and e["verification_calls"] == 2
        and all(e[k] is True for k in ("guards", "readonly", "repeat", "poison_verified", "sync_completed")),
        "control capture failed safety checks",
    )
    result = {}
    pos = io["scalars"]["start_position"]
    for item in abi["outputs"]:
        name = item["name"]
        a = folder / (name + "-actual-0.bin")
        other = folder / (name + "-actual-1.bin")
        require(c.sha(a) == c.sha(other), "control repeat mismatch: " + name)
        v = np.fromfile(a, dtype="<i8" if name == "draft_top1" else "<f2").reshape(item["shape"])
        if name == "draft_top1":
            require(
                np.all((v >= 0) & (v < 248320)) and not v[:, io["scalars"]["proposal_count"] :].any(),
                "invalid captured tokens",
            )
        else:
            old = np.fromfile(c.verify(io["files"][name]), dtype="<u2").reshape(v.shape)
            bits = v.view("<u2")
            require(
                np.isfinite(v).all()
                and np.array_equal(old[:, :, :pos], bits[:, :, :pos])
                and np.array_equal(old[:, :, pos + gear :], bits[:, :, pos + gear :]),
                "nonfinite or changed untouched KV rows: " + name,
            )
            if io["source"] == "REAL_CHECKPOINT_SEEDED_INPUTS":
                require(
                    np.any(bits[:, :, pos : pos + gear] != old[:, :, pos : pos + gear]),
                    "seeded replay did not update KV: " + name,
                )
        result[name] = c.identity(a)
    return result


def record_mismatches(folder, abi, reference):
    mismatches = {}
    for item in abi["outputs"]:
        name = item["name"]
        t = "<i8" if name == "draft_top1" else "<u2"
        expected = np.fromfile(c.verify(reference[name]), dtype=t)
        for p in sorted(folder.glob(name + "-*.bin")):
            actual = np.fromfile(p, dtype=t)
            if actual.shape != expected.shape:
                mismatches[p.name] = {"actual_elements": actual.size, "expected_elements": expected.size}
            else:
                unequal = np.flatnonzero(actual != expected)
                if unequal.size:
                    i = int(unequal[0])
                    mismatches[p.name] = dict(
                        count=int(unequal.size),
                        first_index=i,
                        expected_bits=int(expected[i]),
                        actual_bits=int(actual[i]),
                    )
    if mismatches:
        c.write(folder / "mismatches.json", mismatches)
        print("Output mismatches:", mismatches, flush=True)


def run_om(folder, abi, io, model, runner, env, dsos, phase, reference=None, profiler=None):
    require((reference is None) == (phase == "capture"), "invalid reference/phase pair")
    app = folder / "application"
    prepare_case(app, abi, io, reference)
    try:
        info = launch(folder, runner, model["om"], phase, env, dsos, profiler)
    except Exception:
        if reference:
            record_mismatches(app, abi, reference)
        raise
    if phase == "capture":
        return dict(
            reference=capture_reference(app, io, abi, abi["inputs"][0]["shape"][1]),
            execution=c.identity(app / "execution.json"),
        )
    timed = phase == "benchmark"
    outputs = {
        o["name"]: [
            c.identity(app / (o["name"] + "-" + suffix + ".bin"))
            for suffix in (
                ("actual-0", "actual-1", "benchmark-last", "postcheck") if timed else ("actual-0", "actual-1")
            )
        ]
        for o in abi["outputs"]
    }
    row = dict(execution=c.identity(app / "execution.json"), outputs=outputs)
    require(
        c.read(app / "execution.json")["reference_capture"] is False, "unexpected capture during comparison"
    )
    row.update(validation=verify_run(row, {"golden": reference}, 0, timed), profiled=bool(profiler), **info)
    return row


def compare_rounds(rows):
    require(len(rows) == 3, "three rounds required")
    out = {}
    for mode in ("baseline", "optimized"):
        runs = [r[mode] for r in rows]
        require(not any(r["profiled"] for r in runs), "profiling samples in official timing")
        out[mode] = stats([t for r in runs for t in r["validation"]["timing"]["samples_ms"]], 90)
    return dict(
        variants=out,
        ratio=out["baseline"]["median_ms"] / out["optimized"]["median_ms"],
        round_ratios=[
            r["baseline"]["validation"]["timing"]["median_ms"]
            / r["optimized"]["validation"]["timing"]["median_ms"]
            for r in rows
        ],
    )
