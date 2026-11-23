import copy
from pathlib import Path
import struct
import sys
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from gate_panel import validate_plan, validate_w16_pair, panel_header, c, M, N, K, SHAPE_KEY, schema
import tempfile


class Resources(unittest.TestCase):
    def test_distinct_sdk_plans_require_the_declared_ub_and_l1(self):
        offsets = {f["name"]: 24 + f["offset"] for f in schema()}
        for panel, codes, scales, w, scratch, steps, trans, l1 in (
            (256, 32768, 512, 32768, 187904, 1, 8704, 40960),
            (512, 65536, 1024, 65536, 121856, 2, 16896, 81920),
            (1280, 40960, 1280, 163840, 47872, 5, 41472, 204800),
            (2560, 40960, 2560, 81920, 128512, 10, 2048, 819200),
            (2560, 40960, 2560, 163840, 46592, 10, 2048, 819200),
        ):
            data = bytearray(224)
            struct.pack_into("<4IQ", data, 0, M, K, N, panel, scratch)
            fields = dict(
                singleCoreM=16,
                singleCoreN=64,
                singleCoreK=panel,
                baseM=16,
                baseN=64,
                baseK=256,
                stepKa=steps,
                stepKb=steps,
                depthA1=steps,
                depthB1=steps,
                dbL0A=2,
                dbL0B=2,
                dbL0C=1,
                transLength=trans,
            )
            for name, v in fields.items():
                struct.pack_into("<I", data, offsets[name], v)
            assign = [list(range(i, 304, 8)) for i in range(8)]
            resources = dict(
                raw_banks=2 if panel <= 512 else 1,
                tile_k=panel,
                codes_bytes=codes,
                scale_bytes=scales,
                weight_bytes=w,
                user_ub=262144 - scratch,
                scratch=scratch,
                workspace=2097152,
                l1_used=l1,
                l0a_used=16384,
                l0b_used=65536,
                l0c_used=4096,
            )
            query = dict(
                tiling_sha256="fixture",
                platform=dict(blockDim=8, capacity={"UB": 262144}, workspace=2097152),
                schedule=dict(baseline_assignments=assign, tiling_hex=data.hex(), resources=resources),
            )
            banks = 2 if panel == 2560 and w == 163840 else 1
            base = {"plans": {SHAPE_KEY: {"query": copy.deepcopy(query)}}}
            self.assertEqual(
                validate_plan(query, base, panel, w16_banks=banks)["outer_iterates_per_n64"], K // panel
            )
            bad = copy.deepcopy(query)
            bad["schedule"]["resources"]["l1_used"] -= 1024
            with self.assertRaisesRegex(ValueError, "memory"):
                validate_plan(bad, base, panel, w16_banks=banks)
            struct.pack_into("<I", data, offsets["baseK"], 512)
            bad = copy.deepcopy(query)
            bad["schedule"]["tiling_hex"] = data.hex()
            with self.assertRaisesRegex(ValueError, "Cube plan"):
                validate_plan(bad, base, panel, w16_banks=banks)

    def test_direct_pair_requires_identical_air_cube_and_resource_delta(self):
        with tempfile.TemporaryDirectory() as tmp:
            air = Path(tmp) / "gate.air"
            air.write_bytes(b"shared input graph")
            before = dict(
                tiling_hex=(bytes(24) + bytes(range(200))).hex(),
                resources=dict(user_ub=133632, weight_bytes=81920, scratch=128512, raw_banks=1),
            )
            after = copy.deepcopy(before)
            after["resources"].update(user_ub=215552, weight_bytes=163840, scratch=46592)
            base = dict(
                mode="panel-k2560", air=c.identity(air), plans={SHAPE_KEY: {"query": {"schedule": before}}}
            )
            q = {"schedule": after}
            validate_w16_pair(q, base, air)
            with self.assertRaisesRegex(ValueError, "invalid W16 buffer"):
                panel_header(1280, 2)
            for key in ("raw_banks", "scratch", "weight_bytes"):
                bad = copy.deepcopy(q)
                bad["schedule"]["resources"][key] += 1
                with self.assertRaisesRegex(ValueError, "delta"):
                    validate_w16_pair(bad, base, air)
            bad = copy.deepcopy(q)
            bad["schedule"]["tiling_hex"] = "00" * 224
            with self.assertRaisesRegex(ValueError, "Cube tiling"):
                validate_w16_pair(bad, base, air)
            air.write_bytes(b"another graph")
            with self.assertRaisesRegex(ValueError, "identical AIR"):
                validate_w16_pair(q, base, air)
