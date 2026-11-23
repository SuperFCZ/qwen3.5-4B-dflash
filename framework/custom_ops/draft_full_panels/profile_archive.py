"""Read a completed full-Draft measurement. This admission path has no build or execution callback."""

from pathlib import Path

from joint import c, require
from full_replay import compare_rounds
from raw_protocol import verify_run

MODES = ("baseline", "optimized")


def read_gear(stages, graph, gear, runner):
    models = {}
    for mode in MODES:
        atc = stages.completed(f"atc-c{gear}-{mode}")
        require(
            atc["contract"]["pkg"] == stages.refs["private-opp"]
            and atc["contract"]["air"] == atc["result"]["air"] == graph[mode],
            "different compiled Draft: " + mode,
        )
        models[mode] = atc["result"]
        c.verify(models[mode]["om"])
    c.verify(models["optimized"]["dump"])
    io = stages.completed(f"inputs-c{gear}")
    require(io["contract"]["abi"] == graph["abi"], "archived input ABI drift")
    for item in io["result"]["files"].values():
        c.verify(item)
    reference = stages.completed(f"reference-c{gear}")
    key = reference["contract"]
    require(
        key["runner"] == runner
        and key["io"] == stages.refs[f"inputs-c{gear}"]
        and key["models"] == [stages.refs[f"atc-c{gear}-{mode}"] for mode in MODES],
        "different archive reference/OM pair",
    )
    golden = reference["result"]["reference"]
    smoke = stages.completed(f"smoke-c{gear}")
    bench = stages.completed(f"benchmark-c{gear}")
    require(
        smoke["contract"] == dict(key, reference=golden)
        and bench["contract"] == dict(smoke["contract"], smoke=stages.refs[f"smoke-c{gear}"]),
        "archived smoke/benchmark predecessors changed",
    )
    rounds = bench["result"]["rounds"]
    seen = set()
    for i, pair in enumerate([smoke["result"], *rounds]):
        timed = i > 0
        require(set(pair) == set(MODES), "incomplete archived pair")
        for mode, row in pair.items():
            app = Path(row["execution"]["path"]).parent
            cmd = c.read(app.parent / "command.json")
            require(
                app not in seen
                and cmd["om"] == models[mode]["om"]
                and cmd["runner"] == runner
                and cmd["argv"] == cmd["application"]
                and cmd["argv"][:-1]
                == [
                    runner["path"],
                    "0",
                    models[mode]["om"]["path"],
                    str(app),
                    str(app / "inputs"),
                    str(app / "golden"),
                    "benchmark" if timed else "smoke",
                ],
                "archived launch/OM or round identity differs",
            )
            seen.add(app)
            require(
                row["profiled"] is False
                and row["launch_authorized"] is True
                and row["validation"] == verify_run(row, {"golden": golden}, 0, timed),
                "archived execution differs",
            )
    statistics = compare_rounds(rounds)
    require(statistics == bench["result"]["statistics"], "archived samples/statistics differ")
    return dict(graph=graph, io=io["result"], reference=golden, model=models["optimized"]), statistics


def measured_cases(stages, prior, cann, gears):
    graph_index = stages.completed("checkpoint-graphs")["result"]
    graphs = c.read(c.verify(graph_index))["gears"]
    private = stages.completed("private-opp")
    require(private["contract"]["cann"] == str(cann), "archive belongs to another CANN installation")
    pkg = private["result"]
    c.verify_opp(pkg["opp"])
    c.verify_tree(pkg["sources"])
    runner = stages.completed("runner")["result"]
    c.verify(runner)
    cases = {}
    timings = {}
    for gear in gears:
        require(str(gear) in graphs, f"no completed Draft C{gear} in {prior}")
        print(f"Checking sealed C{gear} smoke, 3 paired rounds and all 90 samples per variant", flush=True)
        cases[gear], timings[str(gear)] = read_gear(stages, graphs[str(gear)], gear, runner)
    summary = c.identity(prior / "latency.json")
    archived = c.read(c.verify(summary))
    require(all(archived[g] == t for g, t in timings.items()), "latency.json differs from sealed benchmark")
    c.write(stages.folder / "stages.json", stages.refs)
    c.write(stages.folder / "latency.json", timings)
    c.write(
        stages.folder / "profile-source.json",
        dict(
            source=str(prior),
            latency=summary,
            receipts=stages.refs.copy(),
            gears=gears,
            fresh_formal_samples=0,
        ),
    )
    return pkg, runner, cases
