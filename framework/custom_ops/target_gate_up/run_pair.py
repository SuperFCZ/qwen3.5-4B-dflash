"""Measure the public Target Gate/Up pair with identical inputs on Ascend310P3."""

import argparse
import os
from pathlib import Path
import traceback

from pair_graph import HERE, c
from pair_reference import deployment, references
from experiment import Stages
from run_q import run_path


def run(out, args, state):
    base = run_path(out.parent, args.from_run, "target-conv-window-")
    cann = Path(os.environ["CANN_ROOT"]).resolve(strict=True)
    assets, env = deployment(out, base, cann)
    desc = c.identity(out / "cann-ge.desc")
    st = Stages(out, run_path(out.parent, args.resume, "target-gate-up-") if args.resume else None)
    sources = {n: c.sha(HERE / n) for n in ("pair_graph.py", "pair_reference.py")}

    def stage(name, key, action):
        state["phase"] = name
        return st.run(name, dict(base=assets["baseline"], sources=sources, inputs=key), action)

    refs = references(out, stage, assets, env, desc)
    state.update(baseline=assets["baseline"], math={k: v["math"] for k, v in refs.items()})
    return "REFERENCES_MEASURED"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--from-run", required=True)
    p.add_argument("--references-only", action="store_true")
    p.add_argument("--resume")
    args = p.parse_args()
    out = Path(os.environ["DFLASH_OUTPUT"])
    state = {"status": "RUNNING"}
    try:
        state["status"] = run(out, args, state)
    except Exception as exc:
        state.update(status="FAILED", error=str(exc))
        traceback.print_exc()
    c.write(out / "result.json", state)
    print(state["status"], "REPORT", out / "result.json", flush=True)
    raise SystemExit(state["status"] == "FAILED")
