"""Read the surviving 20-prompt report and the exact plan used by its C++ runner."""

import json
from pathlib import Path
import shlex

from winners import c, require, REPO, SELECT, UNION
from tools import benchmark_prompts as suite

ROOT = Path("/home/w00949577/z50058744/qwen35-w8nz-repro")
SUITE = ROOT / "gdr-lengths-w55diia2"
TARGET_SHA = {
    "target_prefill": "134888bcc577ca3f42bad994f2caaeb210214d7ab5676fc964a9d29855b459a7",
    "target_verify": "268e0b4cf11f39e3aa3eb53a4e78834127eaaf77065f91d7baf2a30a2ef30ec8",
}
RUNNER_SHA = "2c033fb0f80914d915d196a00adb8fabcf665cce0a3fb7584ab899770ce5a254"


def parse_plan(path):
    words = [shlex.split(l) for l in path.read_text().splitlines()]
    require(
        words[:3]
        == [
            ["qwen35-dflash-chunk-v4"],
            ["capacity", "2048", "248320"],
            ["draft_prefill_policy", "static_draft16_64_oms"],
        ]
        and words[-1] == ["done"],
        "expected documented 2048 Chunk v4 and two static Draft gears",
    )
    graphs = {}
    row = None
    for fields in words[3:-1]:
        kind = fields[0]
        if kind == "graph":
            require(row is None and len(fields) == 4 and fields[1] not in graphs, "invalid plan graph")
            row = graphs[fields[1]] = dict(
                path=fields[2], sha256=fields[3], abi={"inputs": [], "outputs": []}
            )
        elif kind in ("I", "O"):
            shape = list(map(int, fields[4:]))
            require(row is not None and len(shape) == int(fields[3]) and min(shape) > 0, "bad plan tensor")
            row["abi"]["inputs" if kind == "I" else "outputs"].append(
                dict(name=fields[1], dtype=fields[2], shape=shape)
            )
        elif kind == "static_gear64":
            require(
                row is graphs.get("draft") and len(fields) == 3 and "static64" not in row,
                "unexpected secondary OM",
            )
            row["static64"] = dict(path=fields[1], sha256=fields[2])
        else:
            require(fields == ["end"] and row is not None, "unexpected plan row " + str(fields))
            row = None
    require(
        row is None
        and set(graphs) == {"target_prefill", "target_verify", "target_decode", "draft"}
        and all(graphs[k]["sha256"] == v for k, v in TARGET_SHA.items())
        and "static64" in graphs["draft"],
        "different archived Target OMs",
    )
    feat = dict(name="features", dtype="float16", shape=[1, 64, len(UNION) * 2560])
    require(
        graphs["draft"]["abi"]["inputs"][0] == feat
        and len(graphs["draft"]["abi"]["inputs"]) == 15
        and len(graphs["draft"]["abi"]["outputs"]) == 11
        and all(feat in graphs[k]["abi"]["outputs"] for k in TARGET_SHA),
        "unexpected feature/state contract",
    )
    return graphs


def read_target_contract(root):
    (index,) = root.rglob("runner-batch.json")
    base = index.parent
    idx = c.read(index)
    require(idx["fake_acl"] is False, "historical batch has no real device identity")
    plans = [p for p in root.rglob("chunk-plan.txt") if c.sha(p) == idx["model_sha256"]]
    require(plans, "missing authentic historical plan")
    graphs = parse_plan(plans[0])
    return dict(index=c.identity(index), plan=c.identity(plans[0]), graphs=graphs)


def read_history(root):
    hist = read_target_contract(root)
    index = c.verify(hist["index"])
    base = index.parent
    batch = base / "prompts.txt"
    idx = c.read(index)
    req = c.read(base / "request.json")
    require(
        idx["models_reused_across_prompts"]
        and idx["low_memory"]
        and idx["prompt_batch_sha256"] == c.sha(batch),
        "different historical batch",
    )
    runner = Path(req["runner"]["path"])
    require(c.sha(runner) == req["runner"]["sha256"] == RUNNER_SHA, "different historical generation runner")
    mat = c.read(root / "summary.json")
    proto = mat["protocol"]
    require(
        (proto["warmup"], proto["repetitions"]) == (1, 3)
        and (req.get("enable_thinking") is True or proto.get("enable_thinking") is True),
        "different warmup/repeat/thinking",
    )
    lines = batch.read_text().splitlines()
    require(lines[0] == "QWEN35_PROMPT_BATCH_V1", "invalid prompt batch")
    prompts = suite.load_prompts(None)
    names = [v["id"] for v in prompts]
    require(
        len(prompts) == 20
        and sum(p["group"] == "short" for p in prompts) == 8
        and [r["id"] for r in idx["cases"]] == names,
        "different prompt suite/order",
    )
    files = {}
    ids = {}
    for line, p, row in zip(lines[1:], prompts, idx["cases"], strict=True):
        name, toks = shlex.split(line)
        ids[name] = list(map(int, toks.split(",")))
        f = Path(str(index) + ".cases") / (name + ".json")
        z = c.read(f)
        require(
            name == p["id"]
            and z["prompt_token_ids"] == ids[name]
            and z["model"]["sha256"] == idx["model_sha256"]
            and z["device_id"] == 0
            and z["cpu_fallback"] is False
            and z["eos_token_ids"] == [248044]
            and z["limits"] == {"max_new_tokens": 128, "max_draft_tokens": 15}
            and z["status"] == row["status"]
            and z["protocol"]["low_memory"]
            and z["protocol"]["warmup"] == 1
            and z["protocol"]["repetitions"] == 3
            and len(z["dflash"]["measurements"]) == len(z["ordinary"]["measurements"]) == 3,
            "invalid historical generation: " + name,
        )
        files[name] = c.identity(f)
    frozen = mat.get("prompts", [])
    # Matrix wrappers can store the prepared prompts in the member summary instead.
    if not frozen or "prompt_token_ids" not in frozen[0]:
        frozen = c.read(root / "w8a16/summary.json")["prompts"]
    require(
        [(r["id"], r["prompt_token_ids"]) for r in frozen] == list(ids.items()), "prompt token IDs changed"
    )
    return dict(
        hist,
        batch=c.identity(batch),
        runner=c.identity(runner),
        request=c.identity(base / "request.json"),
        historical=c.identity(root / "summary.json"),
        cases=files,
        prompts=[dict(p, tokens=ids[p["id"]]) for p in prompts],
        device=req.get("runtime_identity", mat.get("runtime_identity", {})),
    )


def write_candidate_plan(path, history, targets, drafts):
    """The original public tensors and native plan grammar, with exactly the four measured-role OMs."""
    graphs = parse_plan(c.verify(history["plan"]))
    lines = ["qwen35-dflash-chunk-v4", "capacity 2048 248320", "draft_prefill_policy static_draft16_64_oms"]
    for name in ("target_prefill", "target_verify", "draft"):
        original = graphs[name]["abi"]
        if name == "draft":
            for gear in (16, 64):
                got = drafts[gear]["abi"]
                need = dict(
                    original,
                    inputs=[dict(original["inputs"][0], shape=[1, gear, 28160]), *original["inputs"][1:]],
                )
                require(got == need, "candidate Draft C" + str(gear) + " is incompatible with saved plan")
            om = drafts[16]["om"]
            alternate = drafts[64]["om"]
        else:
            require(
                targets[name]["abi"] == original and targets[name]["target_layers"] == list(UNION),
                "Target ABI/layer order differs",
            )
            om = targets[name]["om"]

        def token(record):
            f = c.verify(record)
            require(not any(v in str(f) for v in "\r\n\t"), "invalid model path")
            return json.dumps(str(f)) + " " + record["sha256"]

        lines.append(f"graph {name} " + token(om))
        if name == "draft":
            lines.append("static_gear64 " + token(alternate))
        for direction, mark in (("inputs", "I"), ("outputs", "O")):
            for t in original[direction]:
                shape = t["shape"]
                lines.append(" ".join(map(str, (mark, t["name"], t["dtype"], len(shape), *shape))))
        lines.append("end")
    path.write_text("\n".join([*lines, "done", ""]))
    return c.identity(path)
