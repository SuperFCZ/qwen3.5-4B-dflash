import copy
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import run_w16 as driver
from w16_common import DOUBLE, FIRST, LAYERS, OVERLAP, SERIAL, SHAPE, c
from w16_export import resource_contract
from w16_runtime import PAIRS, decision, order, paired
from w16_static import NoOverlap, StaticBlocked, admit_static, collect, decode, sections, walk_instructions
from down_fallback import ABI as DOWN_ABI, down_node
from flow_query import assignments


def plan(scratch):
    old = scratch == 185600
    cube = bytes.fromhex(
        json.loads((HERE.parent / "draft_w8a16_deep_profile/w16-ub-budget.json").read_text())["queries"][0][
            "cube_hex"
        ]
    )
    parent = struct.pack("<IIIIQ", *SHAPE, 256, scratch) + cube
    res = dict(
        user_ub=262144 - scratch,
        scratch=scratch,
        raw_banks=2,
        tile_k=256,
        codes_bytes=32768,
        scale_bytes=512,
        weight_bytes=32768 if old else 65536,
        l1_used=40960,
        l0a_used=16384,
        l0b_used=65536,
        l0c_used=4096,
        workspace=2097152,
    )
    return dict(
        platform=dict(
            blockDim=8,
            arch=2002,
            GetCoreNum=8,
            availableCores=8,
            coreLimit=0,
            route="ATC",
            workspace=2097152,
            capacity=dict(UB=262144, L0B=65536),
        ),
        schedule=dict(
            variant="A",
            route="ATC",
            core_limit=0,
            selected_block_dim=8,
            n_tiles=304,
            baseline_assignments=assignments(304, 8),
            resources=res,
            tiling_hex=parent.hex(),
        ),
        tiling_sha256=hashlib.sha256(parent).hexdigest(),
    )


def rows(pair):
    samples = []
    for gear, layer, p, r, m in order("16", [FIRST], [pair]):
        v = 1.1 if m == PAIRS[pair][0] else 1.0
        samples.append(
            dict(
                gear=gear,
                layer=layer,
                pair=p,
                round=r,
                mode=m,
                folder=f"{p}/{r}/{m}",
                profiled=False,
                launch_authorized=True,
                bit_mismatches=0,
                atc_plan=dict(block_dim=8),
                runner=m,
                om=m,
                opp_root=m,
                opp_kernel=m,
                manifest=m,
                selected_kernels=m,
                inputs="same",
                golden="frozen",
                air="same-air",
                deployment=m,
                frozen_constants="same-weights",
                validation=dict(workspace_bytes=32, model_weight_bytes=64, timing=dict(samples_ms=[v] * 30)),
            )
        )
    return samples


class ResourceTests(unittest.TestCase):
    def test_parent_full_cube_and_precise_delta(self):
        raw, actual = plan(185600), plan(152832)
        result = resource_contract(actual, raw)
        self.assertEqual(len(result["cube_fields"]), 50)
        self.assertEqual(result["assignments"], [list(range(b, 304, 8)) for b in range(8)])
        for field in (
            "tiling",
            "base",
            "partition",
            "workspace",
            "capacity",
            "l0b",
            "scratch",
            "core",
            "policy",
            "weight",
        ):
            x = copy.deepcopy(actual)
            s = x["schedule"]
            if field in ("tiling", "base"):
                data = bytearray.fromhex(s["tiling_hex"])
                data[20 if field == "tiling" else 33] ^= 1
                s["tiling_hex"] = data.hex()
                x["tiling_sha256"] = hashlib.sha256(data).hexdigest()
            elif field == "partition":
                s["baseline_assignments"][0][-1] = 1
            elif field == "workspace":
                x["platform"]["workspace"] += 8
            elif field == "capacity":
                x["platform"]["capacity"]["UB"] += 32
            elif field == "l0b":
                s["resources"]["l0b_used"] -= 32
            elif field == "scratch":
                s["resources"]["scratch"] += 32
            elif field == "core":
                x["platform"]["blockDim"] = 7
            elif field == "policy":
                s["variant"] = "B"
            else:
                s["resources"]["weight_bytes"] = 32768
            with self.subTest(field=field), self.assertRaises(ValueError):
                resource_contract(x, raw)

    def test_down_case_not_gate_or_other_layer(self):
        item = dict(
            nz_sha256="w",
            scale_gn_sha256="s",
            ge_type="DFlashGroupQuantLinear",
            roundtrip="BIT_EXACT",
            storage_shape=[304, 160, 16, 32],
            scale_shape=[76, 2560],
            k=9728,
            n=2560,
            nz_bytes=24903680,
        )
        graph = dict(runtime_input_abi=dict(draft_ac=dict(nodes=[item])))
        case = dict(inputs={"w_nz.bin": {"sha256": "w"}, "s_gn.bin": {"sha256": "s"}})
        self.assertEqual(down_node(graph, case), item)
        self.assertEqual(DOWN_ABI["outputs"][0]["shape"], [16, 2560])
        item["nz_sha256"] = "other"
        with self.assertRaises(ValueError):
            down_node(graph, case)


class StaticTests(unittest.TestCase):
    def test_decoding_requires_exact_bytes_coverage(self):
        table = [dict(name=".text", va=0, offset=0, bytes=8, flags=6)]
        raw = b"abcdefgh"
        prefix = "Disassembly of section .text:\n"
        text = prefix + " 0: 61 62 63 64    testop a\n 4: 65 66 67 68    testop b\n"
        self.assertEqual(len(decode(raw, table, text)[".text"]), 2)
        for bad in (
            prefix + "0: <not available>",
            text.replace("61 62", "62 61"),
            text.split(" 4:")[0],
            text.replace(" 4:", " 0:"),
            text.replace("testop a", "<unknown>"),
        ):
            with self.subTest(bad=bad), self.assertRaises((StaticBlocked, ValueError)):
                decode(raw, table, bad)
        with self.assertRaises(ValueError):
            sections(b"invalid")

    def test_branch_walk_cannot_invent_an_edge(self):
        block = [
            {"pc": 4 * i, "asm": s, "encoding": "00000000"}
            for i, s in enumerate(("testop a", "b.eq 0xc <done>", "testop b", "testop c", "ret"))
        ]
        self.assertEqual([x["pc"] for x in walk_instructions([[0, 4], [12, 12]], block)], [0, 4, 12])
        with self.assertRaises(ValueError):
            walk_instructions([[0, 4], [0, 0]], block)
        with self.assertRaises(ValueError):
            walk_instructions([[0, 8], [12, 12]], block)
        with self.assertRaises(ValueError):
            walk_instructions([[0, 0], [12, 12]], block)

    def test_no_testable_code_or_decoder_never_admits(self):
        record = {"elf": {"sha256": "same", "path": "exact.elf"}}
        with self.assertRaises(NoOverlap):
            admit_static({FIRST: {m: {"node": record} for m in DOUBLE}}, {}, {}, {}, None)
        cap = dict(record, status="BLOCKED_STATIC_ADMISSION", reason="not available", om={}, node={})
        model = {"gate_nodes": {"layer": "node"}, "om": {}, "atc_kernels": {"node": {}}}
        with self.assertRaises(StaticBlocked):
            admit_static({FIRST: {OVERLAP: {"node": cap}}}, {OVERLAP: {FIRST: model}}, {}, {}, None)

    def test_private_o_is_not_the_selected_om_elf(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            om = p / "M.om"
            om.write_bytes(b"not the reviewed machine image")
            node = dict(
                offset=0, bytes=3, sha256=hashlib.sha256(b"ELF").hexdigest(), symbol="name", block_dim=8
            )
            with self.assertRaisesRegex(ValueError, "embedded"):
                collect(p / "capture", c.identity(om), node, {})


class ProtocolTests(unittest.TestCase):
    def test_manual_dag_and_no_stage_promotion(self):
        self.assertEqual(driver.requirements("smoke")[-1], "static")
        self.assertEqual(driver.requirements("compare")[-2:], ("schedule-benchmark", "net-benchmark"))
        self.assertIn("five-c16-benchmark", driver.requirements("five-c64-smoke"))
        self.assertIn("five-c64-benchmark", driver.requirements("full-atc"))
        self.assertEqual(driver.requirements("full-smoke")[-1], "full-static")
        self.assertEqual(len(driver.schedule("smoke")), 3)
        self.assertEqual(len(driver.schedule("schedule-benchmark")), 6)
        self.assertEqual(len(driver.schedule("five-c64-smoke")), 15)
        self.assertEqual(len(driver.schedule("five-c16-benchmark")), 60)
        self.assertEqual(len(driver.schedule("full-benchmark")), 12)
        self.assertEqual(driver.atc_keys("atc"), (FIRST,))
        self.assertEqual(driver.atc_keys("five-c16-atc"), LAYERS[1:])

    def test_pair_order_fresh_series_threshold_and_anchor_estimate(self):
        r = rows("schedule")
        s = paired(r, "16", FIRST, "schedule")
        n = paired(rows("net"), "16", FIRST, "net")
        self.assertEqual(decision(s, n)["decision"], "GO")
        self.assertEqual([x["mode"] for x in r], [SERIAL, OVERLAP, OVERLAP, SERIAL, SERIAL, OVERLAP])
        self.assertEqual(s["variants"][SERIAL]["count"], 90)
        for defect in ("reorder", "count", "stale", "profile", "round", "p95", "nan", "identity"):
            bad = copy.deepcopy(r)
            if defect == "reorder":
                bad[0], bad[1] = bad[1], bad[0]
            elif defect == "count":
                bad[1]["validation"]["timing"]["samples_ms"].pop()
            elif defect == "stale":
                bad[3]["folder"] = bad[0]["folder"]
            elif defect == "profile":
                bad[4]["profiled"] = True
            elif defect == "round":
                bad[1]["validation"]["timing"]["samples_ms"] = [1.12] * 30
            elif defect == "p95":
                bad[1]["validation"]["timing"]["samples_ms"][-5:] = [2.0] * 5
            elif defect == "nan":
                bad[2]["validation"]["timing"]["samples_ms"][0] = float("nan")
            else:
                bad[3]["om"] = "alien"
            with self.subTest(defect=defect):
                if defect in ("round", "p95"):
                    self.assertEqual(paired(bad, "16", FIRST, "schedule")["decision"], "NO_GO")
                else:
                    with self.assertRaises(ValueError):
                        paired(bad, "16", FIRST, "schedule")
        for row in r:
            if row["mode"] == SERIAL:
                row["validation"]["timing"]["samples_ms"] = [1.029] * 30
        self.assertEqual(paired(r, "16", FIRST, "schedule")["decision"], "NO_GO")

    def test_blocked_predecessor_and_fresh_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve() / "exp"
            (root / "source").mkdir(parents=True)
            c.write(root / "source/report.json", dict(root=str(root)))
            out = root / "static"
            out.mkdir()
            file = out / "report.json"
            c.write(file, {"status": "BLOCKED_STATIC_ADMISSION"})
            before = c.identity(file)
            self.assertNotEqual(driver.fresh_stage(root, "static", []), out)
            self.assertEqual(before, c.identity(file))
            with self.assertRaises(ValueError):
                driver.previous(root, "static")
