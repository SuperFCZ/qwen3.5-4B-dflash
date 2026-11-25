"""Tie every projection to the measured single operator and the checkpoint bundle."""

from dataclasses import replace
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
OPS = HERE.parent
REPO = OPS.parents[1]
sys.path[:0] = [str(OPS / "draft_w8_transfer"), str(REPO / "framework/python"), str(REPO)]
from recipes import CONTROL, candidates, c, require, selections
from incumbents import GATE_OM, gate_archive, GATE_RUN, GATE_RESULT, check_elf
from gate_input import DEPLOYMENT
from experiment import Stages
from run_gate import pair_statistics

TRANSFER = "w8-transfer-mDK933"
TRANSFER_SHA = "45d1b39dbed04072901fb64e926f2f5678abab226239fd32f56316e751a60760"
WIN = {
    "down": ("down-next-w2", "33622f3b94c8284c499b6b0d2e14c5482b4ec46332f37bc5915d17596dbfae3b"),
    "q": ("q-n128-full", "887737d27c41953657411104e81cfc9e6548e1249a2183de37a69643ebfa762c"),
    "fc64": ("fc64-raw2", "a93f83e6c58751ca673bdbfe5dc6b67c0579141579dbe0066ad987b674a4e2ef"),
}
UNION = (1, 5, 8, 9, 13, 15, 17, 21, 22, 25, 29)
SELECT = (1, 8, 15, 22, 29)
FULL_OMS = {
    16: "16e822f10a339b984dc8a2d4c6bba2bf34b30dca5f3f750959e7533b6e73e2f1",
    64: "0ebfea2dbed4cf7eb6fa6fe4fd36f5890289f58447f9862aa22650db20a2b545",
}


def selected(gear):
    specs = selections(gear)
    specs["gate_up"] = replace(specs["gate_up"], mode=CONTROL["gate"].name)
    for key, (name, _) in WIN.items():
        if key == "fc64" and gear != 64:
            continue
        (r,) = (r for r in candidates(key) if r.name == name)
        specs[r.role] = replace(specs[r.role], mode=r.name, columns=r.width, panel=r.panel)
    return specs


def measured_asset(model, pkg, hashes):
    check_elf(model, model["selected"])
    c.verify_opp(pkg["opp"])
    c.verify(pkg["source"])
    c.verify(pkg["host_source"])
    return dict(
        device=pkg["source"],
        host=pkg["host_source"],
        query=model["query"],
        image=model["selected"],
        layer0_constants=hashes,
        mode=model["mode"],
        source_om=model["om"],
    )


def admit(out, runs):
    old = Stages(out, runs / DEPLOYMENT)
    arch = old.completed("admit")["result"]
    st = Stages(out, runs / TRANSFER)
    final_path = runs / TRANSFER / "result.json"
    require(c.sha(final_path) == TRANSFER_SHA, "different W8 transfer measurement")
    final = c.read(final_path)
    require(
        final["status"] == "MEASURED_AND_PROFILED" and set(final["groups"]) == set(CONTROL),
        "incomplete W8 sweep",
    )
    gate = gate_archive(out, runs)
    require(
        gate["model"]["om"]["sha256"] == GATE_OM
        and final["groups"]["gate"]["selected"] == gate["model"]["mode"],
        "wrong Gate incumbent",
    )
    from gate_panel import pack_cube_panels

    cfg = gate["cases"]["measured"]
    from rewrite import digest

    weights = pack_cube_panels(
        *(c.verify(cfg[k]).read_bytes() for k in ("w_nz.bin", "s_gn.bin")), 2560, k=2560, n=19456
    )
    updates = {"gate_up": measured_asset(gate["model"], gate["package"], [digest(v) for v in weights[:2]])}
    for key, (name, sha) in WIN.items():
        g = final["groups"][key]
        b = CONTROL[key]
        m, k, n = b.shape
        pair = g["pairs"][b.name + "__" + name]
        atc = st.completed(f"{key}/atc-{name}")
        model = atc["result"]
        pkg = st.completed(f"{key}/build-{name}")["result"]
        evidence = st.completed(f"{key}/pair-{b.name}__{name}")
        sm = st.completed(f"{key}/smoke-{name}")
        require(
            g["incumbent"] == b.name
            and g["selected"] == name
            and model["mode"] == name
            and model["op"] == b.spec.op
            and model["om"] == g["candidates"][name]["om"]
            and model["om"]["sha256"] == sha
            and pair["report"] == st.refs[evidence["name"]]
            and sm["result"]["accepted"]
            and sm["contract"]["om"] == model["om"]
            and st.refs[sm["name"]] in evidence["contract"]["smoke"]
            and atc["contract"]["build"] == st.refs[f"{key}/build-{name}"],
            "incorrect measured transfer: " + key,
        )
        stats = pair_statistics(
            evidence["result"]["runs"],
            name,
            b.name,
            flops=2 * m * k * n,
            logical_bytes=k * n + 2 * (k // 128) * n + 2 * m * (k + n),
        )
        require(
            stats == evidence["result"]["statistics"]
            and stats["decision"] == "GO"
            and pair["round_ratios"] == stats["round_ratios"],
            "invalid winning timing",
        )
        updates[key] = measured_asset(model, pkg, [v["sha256"] for v in model["constants"]])
    for gear in (16, 64):
        key = str(gear)
        case = arch["cases"][key]
        prior = old.completed(f"audit-c{gear}")["result"]
        require(prior["om"]["sha256"] == FULL_OMS[gear], "different full Draft benchmark")
        for role, spec in selected(gear).items():
            path = "fc64" if role == "fc" and gear == 64 else role
            if path in updates:
                case["assets"][role] = updates[path]
            else:
                require(
                    case["assets"][role]["mode"]
                    == spec.mode
                    == prior["inventory"]["fc" if role == "fc" else "layer0." + role]["mode"],
                    "different frozen selection: " + role,
                )
        c.verify_tree(case["io"]["files"])
        c.verify_tree(case["reference"])
    return dict(archive=arch, transfer=c.identity(final_path), gate=GATE_RESULT, receipts=st.refs)
