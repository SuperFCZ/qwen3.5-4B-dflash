"""Admit read-only Gate8 and profiling chains before compiling or opening a device."""

import hashlib
import subprocess
from pathlib import Path
from w16_common import CUBE_SHA, FIRST, REVIEW, SINGLE, c, historical, require, sdk_files
from gate_common import GEARS, case_contract

FIRST_HASHES = {
    "x.bin": "44fed068f079c18e95ef303c0d02cf5401470db7277ec17d9d024fbf29079256",
    "w_nz.bin": "c898ef7b555f6e3bcf3b183ec7742b012f0a4e40ab4a46cd34184b77218b5a2c",
    "s_gn.bin": "1365af20b6f09734c6d02f2f3d3045fb20e4af1f53d414c9b4dd9011c7ced333",
    "golden": "353839a7a41bdd44930e91f72ae750adc2f7f15e14f2b679d5f0799e1b5da76c",
}
RAW_FULL = {
    "16": "eeb1510326908a7890fd9d62ce7b2ec8801241abadf749e041cb16ce3d9fd17e",
    "64": "a120bc6e6b47ae8de9d0a771229fec363477f93bf6b0ba1e706987dee223075e",
}


def admit(base, profile, tool=None, *, needs_machine_decoder=True):
    audit = "3971424e8946c13b3a757bce887f3db53cc788d2"
    git = ["git", "-C", str(c.REPO)]
    require(
        subprocess.run(git + ["merge-base", "--is-ancestor", audit, "HEAD"]).returncode == 0,
        "missing W16 SDK audit revision",
    )
    for name in (
        "w16-double-buffer-revised-design.md",
        "w16-event-lifecycle-revised.md",
        "next-npu-experiment.md",
        "group-quant-scalar-vector-review.md",
    ):
        path = REVIEW / name
        require(
            c.sha(path)
            == hashlib.sha256(
                subprocess.check_output(git + ["show", audit + ":" + path.relative_to(c.REPO).as_posix()])
            ).hexdigest(),
            "reviewed contract changed: " + name,
        )
    last = c.read(base / "full-benchmark/report.json")
    outer = dict(sources=historical.sources(), history_roots=last["history_roots"], predecessors={})
    chain = historical.load_chain(base, "full-benchmark", outer)
    historical.stage_status("full-benchmark", last)
    require(
        last["sources"] == outer["sources"]
        and last["predecessors"] == outer["predecessors"]
        and last["root"] == str(base)
        and last["phase"] == "full-benchmark"
        and last["device_execution"] == "COMPLETED"
        and last["benchmark"] == "MEASURED",
        "incomplete frozen Gate8 chain",
    )
    historical.verify_tree(last, {})
    historical.validate_stage("full-benchmark", last, chain, base)
    refs = dict(outer["predecessors"])
    refs["full-benchmark"] = c.identity(base / "full-benchmark/report.json")
    src = chain["source"]["binding"]
    case_contract(src["cases"])
    case = src["cases"]["16"][FIRST]
    files = dict(case["inputs"], golden=case["golden"])
    require(
        {key: c.identity(c.verify(item))["sha256"] for key, item in files.items()} == FIRST_HASHES,
        "C16 Layer0 differs from frozen reviewed inputs/Golden",
    )
    paired_file = profile / "profile-paired-analysis/profile-report.json"
    paired = c.read(paired_file)
    require(
        paired["phase"] == "profile-paired-analysis"
        and paired["status"] == "DEEP_PROFILE_PAIRED_ANALYSIS_PASS"
        and not paired.get("error")
        and not paired.get("integrity_error"),
        "unaccepted paired profiling",
    )
    admit_file = c.verify(paired["predecessors"]["admit"])
    prof = c.read(admit_file)
    require(
        admit_file == profile / "admit/profile-report.json"
        and prof["status"] == "DEEP_ADMIT_PASS"
        and prof["gate8_root"] == str(base)
        and prof["binding"]["reports"] == refs
        and not prof.get("integrity_error")
        and not prof.get("error")
        and prof["sources"] == paired["sources"],
        "profile and Gate8 are not the same baseline",
    )
    # Only bind historical profile data used here; no timeline samples enter timing.
    historical.verify_tree(prof["sources"], {})
    cases = {}
    for gear in GEARS:
        raw = chain["full-export"]["models"]["raw"][gear]
        profile_model = prof["binding"]["variants"]["raw"]["cases"][gear]
        require(
            raw["om"] == profile_model["om"]
            and raw["om"]["sha256"] == RAW_FULL[gear]
            and raw["plans"]["16x2560x19456"]["contract"]["cube_sha256"] == CUBE_SHA,
            "Raw full OM identity",
        )
        item = paired["cases"]["raw-c" + gear]
        analysis = c.read(c.verify(item))
        require(
            item["path"] == str(paired_file.parent / ("raw-c" + gear) / "case-analysis.json")
            and analysis["mode"] == "raw"
            and analysis["gear"] == gear,
            "wrong raw profile case",
        )
        cases[gear] = item
    from w16_static import disassembler

    return dict(
        audit_revision=audit,
        history=refs,
        profile=c.identity(paired_file),
        profile_admit=c.identity(admit_file),
        profile_cases=cases,
        sdk=sdk_files(src["cann"]),
        tool=disassembler(src["cann"], tool) if needs_machine_decoder else None,
        first_case=files,
        historical_samples_used=False,
    )


def recovered(binding):
    refs = binding["history"]
    source = c.read(c.verify(refs["source"]))["binding"]
    build = c.read(c.verify(refs["build"]))["variants"]["raw"]
    models = {}
    for stage in ("export", "full-export"):
        frozen = c.read(c.verify(refs[stage]))["models"]["raw"]
        models[stage] = {key: dict(value, mode=SINGLE) for key, value in frozen.items()}
    return source, build, models


def verify_source(binding, base, profile):
    refs = binding["history"]
    require(
        set(refs) == set(historical.STAGES)
        and binding["audit_revision"] == "3971424e8946c13b3a757bce887f3db53cc788d2",
        "incomplete source inventory",
    )
    before = {}
    for phase in historical.STAGES:
        item = refs[phase]
        require(
            Path(item["path"]) == base / phase / "report.json", "historical report outside admitted experiment"
        )
        old = c.read(c.verify(item))
        historical.stage_status(phase, old)
        require(
            old["root"] == str(base) and old["phase"] == phase and old["predecessors"] == before,
            "historical report chain changed",
        )
        before[phase] = item
    src, raw, models = recovered(binding)
    case = src["cases"]["16"][FIRST]
    items = dict(case["inputs"], golden=case["golden"])
    require(
        items == binding["first_case"]
        and {k: c.sha(c.verify(v)) for k, v in items.items()} == FIRST_HASHES
        and sdk_files(src["cann"]) == binding["sdk"],
        "SDK or first case drift",
    )
    paired = c.read(c.verify(binding["profile"]))
    require(
        binding["profile"]["path"] == str(profile / "profile-paired-analysis/profile-report.json")
        and binding["profile_admit"] == paired["predecessors"]["admit"]
        and c.read(c.verify(binding["profile_admit"]))["binding"]["reports"] == refs,
        "different profiling chain",
    )
    for gear in GEARS:
        require(
            models["full-export"][gear]["om"]["sha256"] == RAW_FULL[gear]
            and paired["cases"]["raw-c" + gear] == binding["profile_cases"][gear],
            "historical Raw variant drift",
        )
