import copy
from pathlib import Path
import struct
import sys
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from gate_panel import validate_plan, M, N, K, SHAPE_KEY, schema


class Resources(unittest.TestCase):
    def test_distinct_sdk_plans_require_the_declared_ub_and_l1(self):
        offsets = {f["name"]: 24 + f["offset"] for f in schema()}
        for panel, codes, scales, w, scratch, steps, trans, l1 in (
            (256, 32768, 512, 32768, 187904, 1, 8704, 40960),
            (512, 65536, 1024, 65536, 121856, 2, 16896, 81920),
            (1280, 40960, 1280, 163840, 47872, 5, 41472, 204800),
            (2560, 40960, 2560, 81920, 128512, 10, 2048, 819200),
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
            base = {"plans": {SHAPE_KEY: {"query": copy.deepcopy(query)}}}
            self.assertEqual(validate_plan(query, base, panel)["outer_iterates_per_n64"], K // panel)
            bad = copy.deepcopy(query)
            bad["schedule"]["resources"]["l1_used"] -= 1024
            with self.assertRaisesRegex(ValueError, "memory"):
                validate_plan(bad, base, panel)
            struct.pack_into("<I", data, offsets["baseK"], 512)
            bad = copy.deepcopy(query)
            bad["schedule"]["tiling_hex"] = data.hex()
            with self.assertRaisesRegex(ValueError, "Cube plan"):
                validate_plan(bad, base, panel)
