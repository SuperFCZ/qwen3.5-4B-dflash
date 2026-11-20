"""Source/SDK-level evidence for the two archived C16 Gate variants."""

from pathlib import Path
import hashlib
import json
import struct
import sys

HERE = Path(__file__).resolve().parent
OPS = HERE.parent
sys.path.insert(0, str(OPS / "draft_gate_w16_decoder_retry"))
import retry_decoder as archive
from w16_common import (
    CUBE_SHA,
    DOUBLE,
    FIRST as FIRST,
    HEADER,
    OVERLAP,
    RAW_HEADER,
    REVIEW,
    SERIAL,
    SHAPE_KEY as SHAPE_KEY,
    SINGLE as SINGLE,
    STEM,
    c,
    one as one,
    render_kernel,
    require,
)
from w16_source import FIRST_HASHES as FIRST_HASHES
from w16_static import REGIONS
from gate_plan import schema

sys.path.insert(0, str(REVIEW))
from review_contracts import check_trace, schedule

LABELS = ["EXPERIMENTAL_ONLY", "MACHINE_PROOF_UNRESOLVED", "PRODUCTION_NOT_APPROVED"]
ISA_ROOT = archive.DEFAULT_ROOT.parent / "w16-isa-audit-20yqefdn"
GATE_SOURCE = {
    SERIAL: "144ab97b82cf91d7a3ce8cf3d20018f8394fc180e2bb41a06f727656fc17a13b",
    OVERLAP: "4fd99adb3fb57b8946ca1ecebec4d1bf3100946632e05bd6c97e66f6a62972d5",
}
GATE_ELF = {
    SERIAL: "7de3096e7697b9056366321c2585d3e77079651957a9f9f44611c4d4a7e0f323",
    OVERLAP: "195d076c225e54db57d50dda7a550df65a5c6fcf3842b4f256ecaa0b1cf90eb2",
}
GATE_TEXT = {
    SERIAL: "6ec3564e76c5e716ed2bae4c906dee958fac8dec5ac8b8d9fdc7a285e44e1aa4",
    OVERLAP: "458e9ebf0ec3b1fd9701aed0cc1c194f7b076a3c6bdfe7ecc40f41c5cf2cded9",
}
EXTRA_SDK = {
    "include/adv_api/matmul/matmul.h": "4de4ecd342f1735f92cdad0d887f3984de7943b1269cc7a5c70cc1ea8dae48cb",
    "impl/adv_api/detail/kfc/kfc_register_obj.h": "28ea04257428760816b5bbfaf2ccfd23029fc42c6bd20b7ca685fdafb278f876",
    "impl/basic_api/kernel_tensor_impl.h": "1e5d6669d91e5687db76fcace1cb1b2a9af0dfde92c4b5ee65dd2b23b99b9e0a",
}


def same(actual, expected, message):
    # JSON booleans and integers have different meanings in stored stage contracts.
    require(
        json.dumps(actual, sort_keys=True, allow_nan=False)
        == json.dumps(expected, sort_keys=True, allow_nan=False),
        message,
    )


def pinned(path, sha):
    rec = c.identity(path)
    require(rec["sha256"] == sha, "reviewed identity changed: " + str(path))
    return rec


def source_manifest():
    extras = [Path(archive.__file__), REVIEW / "review_contracts.py"]
    return [
        c.identity(p)
        for p in sorted([*HERE.glob("*.py"), *HERE.glob("*.md"), *HERE.glob("test/*.py"), *extras])
    ]


def checked_sources(mode, raw, build):
    name = "op_kernel/" + STEM + ".cpp"
    frozen = c.verify(raw["generated_sources"][name]["file"]).read_bytes()
    cpp = build["generated_sources"][name]["file"]
    actual = c.verify(cpp).read_bytes()
    require(
        cpp["sha256"] == GATE_SOURCE[mode] and actual == render_kernel(frozen, mode),
        "unreviewed Gate implementation",
    )
    require(build["variant"] == mode and build["dynamic"], "variant or installed source missing")
    vendor = Path(build["opp"]["install"]) / "vendors/customize"
    for ref in build["dynamic"]:
        p = c.verify(ref)
        require(
            vendor in p.parents and p.read_bytes() == actual, "dynamic source differs from its private build"
        )
        pinned(p.parent / HEADER, RAW_HEADER)
    for side in ("op_kernel", "op_host"):
        require(
            build["generated_sources"][side + "/" + HEADER]["file"]["sha256"] == RAW_HEADER,
            "unreviewed build header",
        )
    host = build["generated_sources"]["op_host/" + STEM + ".cpp"]["file"]
    return dict(
        kernel=cpp,
        dynamic=build["dynamic"],
        generated_header=build["generated_sources"]["op_kernel/" + HEADER]["file"],
        host=pinned(c.verify(host), "8c491ac323b456643a9a1d4a1c4f4091e419c93df3cee93bb58a32b6b64df3e9"),
    )


def memory_contract(plan):
    r = plan["resources"]
    same(
        r,
        dict(
            raw_banks=2,
            codes_bytes=32768,
            scale_bytes=512,
            weight_bytes=65536,
            tile_k=256,
            user_ub=109312,
            scratch=152832,
            l1_used=40960,
            l0a_used=16384,
            l0b_used=65536,
            l0c_used=4096,
            workspace=2097152,
        ),
        "dual W resources drift",
    )
    require(
        plan["block_dim"] == 8
        and len(plan["cube_fields"]) == 50
        and plan["cube_sha256"] == CUBE_SHA
        and plan["workspace"] == r["workspace"],
        "ATC/Cube/workspace contract",
    )
    values = plan["cube_fields"]
    names = [field["name"] for field in schema()]
    require(
        set(values) == set(names)
        and all(type(values[k]) is int and 0 <= values[k] <= 0xFFFFFFFF for k in names)
        and hashlib.sha256(struct.pack("<50I", *(values[k] for k in names))).hexdigest() == CUBE_SHA,
        "TCube fields do not reconstruct the frozen 200 bytes",
    )
    same(plan["assignments"], [list(range(b, 304, 8)) for b in range(8)], "not the eight cyclic owners")
    intervals = {}
    start = 0
    for name, size in (
        ("q0", r["codes_bytes"] // 2),
        ("q1", r["codes_bytes"] // 2),
        ("s0", r["scale_bytes"] // 2),
        ("s1", r["scale_bytes"] // 2),
        ("w0", r["weight_bytes"] // 2),
        ("w1", r["weight_bytes"] // 2),
        ("expanded", 2304),
        ("sdk", r["scratch"]),
        ("reserved", 8192),
    ):
        require(start % 32 == size % 32 == 0 and size > 0, "UB alignment")
        intervals[name] = [start, start + size]
        start += size
    intervals["brcb_fixed"] = [248 * 1024 + 1024, 248 * 1024 + 2048]
    same(intervals, REGIONS, "SDK/TPipe half-open byte intervals")
    require(
        start == 262144 and intervals["expanded"][1] + 8192 == r["user_ub"], "incorrect UB reserve accounting"
    )
    # Each of the eight cast planes and two multiply groups covers a complete [64,256] half bank.
    cast = {(row * 256 + plane * 32 + i) for row in range(64) for plane in range(8) for i in range(32)}
    mul = {(row * 256 + group * 128 + i) for row in range(64) for group in range(2) for i in range(128)}
    # Aligned ND(t) UB->L1 takes 16 K16 slices, one 32 B block on each of 64 rows.
    load = {(row * 256 + sl * 16 + i) for sl in range(16) for row in range(64) for i in range(16)}
    require(cast == mul == load == set(range(16384)), "incomplete W16 footprint")
    return dict(
        intervals=intervals,
        unit="bytes, half-open per block",
        alignment=32,
        physical_evidence="SOURCE_RUNTIME_GUARD_BEFORE_PAYLOAD; NOT_DEVICE_OBSERVED",
        cast_mul_sdk_read_half_elements_per_bank=len(load),
        brcb_payload=2048,
        brcb_rear=256,
        actual_atc=plan,
    )


def lifetime_contract(mode):
    require(mode in DOUBLE, "unknown schedule")
    outcomes = {}
    for fail in [None, *range(10)]:
        steps = list(schedule(10, overlap=(mode == OVERLAP), fail_at=fail))
        accum, publish = check_trace(steps, 10, w_banks=2)
        require(
            accum == list(range(10 if fail is None else fail)) and publish == (fail is None),
            "partial/failure drain",
        )
        slices = range(9 if fail is None else fail)
        for k in slices:
            i, d, f = [steps.index(x) for x in (("iterate", k), ("dequant", k + 1), ("all", k))]
            require(
                (i < d < f if mode == OVERLAP else i < f < d) and f < steps.index(("v_to_s", k + 1)),
                "wrong source order at rebind",
            )
        outcomes["success" if fail is None else "failure-k" + str(fail)] = dict(
            accumulated_k=accum,
            publishes=publish,
            raw_s_waits=sum(op == "wait_s" for op, k in steps),
            raw_v_waits=sum(op == "wait_v" for op, k in steps),
            fences=sum(op in ("all", "output_all") for op, k in steps),
            trace_sha256=hashlib.sha256(json.dumps(steps, separators=(",", ":")).encode()).hexdigest(),
        )
    return dict(
        evidence="REVIEWED_SOURCE_AND_SDK_ORDER",
        paths=outcomes,
        tiles_per_block=38,
        transition="Output; N fence; next N starts with k=0 and partial=false",
        ordinary_failure="drain issued raw pair if present; fence; stop N loop; Release both IDs; End",
        hardware_failure="no drain claim; abort experiment on device or sync error",
    )
