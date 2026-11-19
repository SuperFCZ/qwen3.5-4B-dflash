"""Run with Python unittest. Optional inputs are read-only, never start a profiler or runtime."""
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import unittest

from review_contracts import check_trace, projection_calls, schedule


HERE = Path(__file__).resolve().parent
OPS = HERE.parent
CUBE_SHA = "745b846a7e3b31de10827caa5e493026d3984e282dd698177d2da380b826359a"


def digest(data):
    return hashlib.sha256(data).hexdigest()


class LifetimeTests(unittest.TestCase):
    def test_all_lifetimes_and_placement_control(self):
        for extent in (1, 2, 3, 10, 38):
            serial = list(schedule(extent, overlap=False))
            ahead = list(schedule(extent, overlap=True))
            self.assertCountEqual(serial, ahead)
            for trace in (serial, ahead):
                self.assertEqual(check_trace(trace, extent), (list(range(extent)), True))
                self.assertEqual(trace.count(("output_all", -1)), 1)
            for k in range(extent - 1):
                self.assertLess(ahead.index(("iterate", k)), ahead.index(("dequant", k + 1)))
                self.assertLess(ahead.index(("dequant", k + 1)), ahead.index(("all", k)))
                self.assertLess(serial.index(("all", k)), serial.index(("dequant", k + 1)))

    def test_single_bank_cannot_support_overlap(self):
        check_trace(schedule(overlap=False), 10, w_banks=1)
        with self.assertRaisesRegex(ValueError, "W overwrite"):
            check_trace(schedule(overlap=True), 10, w_banks=1)

    def test_all_ordinary_failure_points_including_tail(self):
        for k in range(10):
            for overlap in (True, False):
                trace = list(schedule(overlap=overlap, fail_at=k))
                self.assertEqual(check_trace(trace, 10), (list(range(k)), False))
                self.assertNotIn(("dequant", k + 1), trace)
                self.assertNotIn(("get_c", -1), trace)
                self.assertLess(trace.index(("all", k)), trace.index(("release", -1)))

    def test_missing_sync_edges_and_false_success(self):
        for omit in (("wait_s", 1), ("wait_v", 7), ("v_to_s", 2), ("all", 5),
                     ("output_all", -1), ("release", -1)):
            with self.subTest(omit=omit):
                trace = list(schedule(overlap=True))
                trace.remove(omit)
                with self.assertRaises(ValueError):
                    check_trace(trace, 10)
        trace = list(schedule(overlap=True, fail_at=5))
        trace.insert(trace.index(("release", -1)), ("get_c", -1))
        with self.assertRaisesRegex(ValueError, "incomplete"):
            check_trace(trace, 10)

    def test_raw_dma_drain_and_event_reuse(self):
        trace = list(schedule(overlap=True, fail_at=6))
        trace.remove(("wait_s", 7))
        with self.assertRaisesRegex(ValueError, "drain"):
            check_trace(trace, 10)
        trace = list(schedule(overlap=True))
        trace.insert(trace.index(("wait_v", 1)), ("copy", 2))
        with self.assertRaisesRegex(ValueError, "signal"):
            check_trace(trace, 10)

    def test_rebind_and_tail_overrun(self):
        trace = list(schedule(overlap=True))
        trace.remove(("all", 0))
        with self.assertRaisesRegex(ValueError, "rebind"):
            check_trace(trace, 10)
        trace = list(schedule(overlap=True))
        trace.insert(trace.index(("all", 9)), ("copy", 10))
        with self.assertRaisesRegex(ValueError, "outside"):
            check_trace(trace, 10)

    def test_speculative_dequant_before_iterate_result(self):
        trace = list(schedule(overlap=True))
        for event in (("wait_s", 1), ("wait_v", 1), ("dequant", 1)):
            trace.remove(event)
            trace.insert(trace.index(("iterate", 0)), event)
        with self.assertRaisesRegex(ValueError, "successful previous Iterate"):
            check_trace(trace, 10)


class GeometryTests(unittest.TestCase):
    def test_real_sdk_query_bytes_and_ub_layout(self):
        saved = json.loads((OPS / "draft_w8a16_deep_profile/w16-ub-budget.json").read_text())
        self.assertEqual([row["scratch"] for row in saved["queries"]], [185600, 152832])
        for row in saved["queries"]:
            cube = bytes.fromhex(row["cube_hex"])
            self.assertEqual((row["get_tiling_rc"], len(cube), digest(cube)), (0, 200, CUBE_SHA))
        sizes = [2*64*256, 2*2*64*2, 64*256*2, 64*256*2, 2304, 152832, 8192]
        offset = 0
        for name, size in zip(("raw_codes", "raw_scales", "W0", "W1", "broadcast", "matmul", "brcb_reserve"), sizes):
            self.assertEqual(offset % 32, 0)
            self.assertEqual(saved["accounting"][name], [offset, offset + size])
            offset += size
        self.assertEqual(offset, 262144)
        self.assertEqual(offset - 152832, 109312)
        self.assertEqual(64 * 256 * 2 * 2, saved["L0B"])

    def test_brcb_rear_and_fixed_work_area(self):
        # One B16 repeat writes a 16x16 transpose and restores its rear 128 halfs.
        for repeats, allocated in ((8, 2304), (16, 4352)):
            high_water = max(i * 128 + 16 * 16 for i in range(repeats)) * 2
            self.assertEqual(high_water, allocated)
            self.assertGreater(high_water, repeats * 8 * 16 * 2)
        temp = 248 * 1024 + 1024
        self.assertTrue(253952 <= temp < temp + 512 * 2 <= 262144)

    def test_sdk_aligned_b_reads_are_confined_to_one_bank(self):
        # dav_m200 on-the-fly CopyND2NZ: a C0 slice per row, then the next C0.
        reads = []
        for bank in (33280, 66048):
            src_gap = 256 * 2 // 32 - 1
            starts = [bank + c0*32 + row*(1 + src_gap)*32
                      for c0 in range(256//16) for row in range(64)]
            self.assertEqual(sorted(starts), list(range(bank, bank + 32768, 32)))
            reads.append(set(starts))
        self.assertTrue(reads[0].isdisjoint(reads[1]))
        self.assertEqual(max(reads[1]) + 32, 98816)  # first broadcast byte

    def test_down_copy_spans_cover_both_frozen_layouts(self):
        # Compare the existing NZ/GN loops with the proposed 32-byte DMA units.
        n, k = 2560, 9728
        for repeats, blocks, gap, plane_k, elt_bytes in ((8, 64, 2496, 32, 1),
                                                         (2, 4, 156, 128, 2)):
            source_starts = []
            length = blocks * 32
            for nt in range(0, n, 64):
                for kt in range(0, k, 256):
                    if plane_k == 32:
                        legacy = [((kt//32 + j)*(n//16) + nt//16)*16*32
                                  for j in range(repeats)]
                    else:
                        legacy = [((kt//128 + j)*n + nt)*elt_bytes for j in range(repeats)]
                    moved = [legacy[0] + j*(blocks + gap)*32 for j in range(repeats)]
                    self.assertEqual(moved, legacy)
                    source_starts.extend(moved)
                    self.assertNotEqual(legacy[1], legacy[0] + (blocks + gap + 1)*32)
            total = n*k if plane_k == 32 else n*(k//128)*elt_bytes
            self.assertEqual(sorted(source_starts), list(range(0, total, length)))

    def test_projection_counts_and_tile_ownership(self):
        cases = ((16, 2560, 19456, 3040), (16, 9728, 2560, 1520),
                 (16, 2560, 4096, 640), (16, 4096, 2560, 640),
                 (32, 2560, 2048, 320), (80, 2560, 2048, 640),
                 (16, 12800, 2560, 2000), (64, 12800, 2560, 2000))
        for m, k, n, steps in cases:
            calls = projection_calls(m, k, n)
            self.assertEqual(calls["iterate"], steps)
            self.assertEqual(calls["get_c"], n // 64)
            owners = [list(range(i, n // 64, 8)) for i in range(8)]
            self.assertEqual(sorted(t for lane in owners for t in lane), list(range(n // 64)))
            if n == 19456:
                self.assertEqual([len(x) for x in owners], [38] * 8)
                self.assertEqual(calls["brcb_inner_pipe_v"], 145920)
                self.assertEqual(calls["explicit_pipe_v_loop"], 18240)
                self.assertEqual(calls["pipe_all"], 3344)
                self.assertEqual(calls["get_value"], 0)
            if m == 80:
                self.assertEqual((calls["get_value"], calls["muls"], calls["brcb"]), (40960, 40960, 0))
            if k == 9728:
                self.assertEqual((calls["explicit_pipe_v_loop"], calls["v_to_s"]), (9120, 1520))
            if k == 4096:
                self.assertEqual((calls["pipe_all"], calls["v_to_s"]), (1960, 0))

    def test_outside_frozen_shape_inventory(self):
        with self.assertRaisesRegex(ValueError, "inventory"):
            projection_calls(16, 512, 64)

    def test_group128_half_bits_during_bank_rotation(self):
        # CPU fixture: independent logical NK oracle versus eight K32-plane casts.
        # Signed codes, per-row/group scale, and a different generation every K.
        banks = [None, None]
        def code(n, k):
            return (n * 17 + (k // 256) * 11 + k * 3) % 256 - 128
        def scale(n, k):
            return 1 + ((n + 7 * (k // 128)) % 29) / 1024  # exactly FP16
        for op, k in schedule(overlap=True):
            if op == "dequant":
                q_nz = bytes(code(n, 256*k + j*32 + u) & 255
                             for j in range(8) for n in range(64) for u in range(32))
                out = bytearray(32768)
                for j in range(8):
                    for n in range(64):
                        factor = scale(n, 256*k + (j // 4)*128)
                        for u in range(32):
                            raw = q_nz[(j*64+n)*32+u]
                            signed = raw if raw < 128 else raw - 256
                            struct.pack_into("<e", out, 2*(n*256 + j*32 + u), signed * factor)
                banks[k % 2] = out
            elif op == "iterate":
                expected = b"".join(struct.pack("<e", code(n, i)*scale(n, i))
                                    for n in range(64) for i in range(k*256, (k+1)*256))
                self.assertEqual(banks[k % 2], expected)

    def test_fp32_order_and_final_half_boundary_are_observable(self):
        def fold(parts):
            acc = 0.0
            for part in parts:
                acc, = struct.unpack("<f", struct.pack("<f", acc + part))
            return struct.pack("<e", acc)
        self.assertNotEqual(fold([2**24, 1, -2**24]), fold([2**24, -2**24, 1]))
        midway, = struct.unpack("<e", struct.pack("<e", 1 + 2**-11))
        self.assertNotEqual(fold([1 + 2**-11, 2**-12]), fold([midway, 2**-12]))

    def test_head_capacity_and_source_identity(self):
        self.assertEqual(15520 * 160 * 16 * 16 * 2, 1271398400)
        self.assertEqual(15 * 248320 * 2, 7449600)
        kernel = OPS / "draft_quant/op_kernel/d_flash_group_quant_linear.cpp"
        self.assertEqual(digest(kernel.read_bytes()),
                         "a4861bb5f7e38aceced9e4d89fc6c2519cb4a80e0914d7dd418c50389e8f1a4f")

    def test_head_descriptor_does_not_determine_weight_load_count(self):
        # Small 15-row example with identical weights and exact FP16 outputs.
        x = [[(m + k) % 3 for k in range(16)] for m in range(15)]
        weight = [[(n + k) % 5 - 2 for k in range(16)] for n in range(32)]
        retained = [[0] * 32 for _ in x]
        reloaded = [[0] * 32 for _ in x]
        loads_once = loads_by_row = 0
        for n in range(32):
            for k in range(16):
                w = weight[n][k]
                loads_once += 1
                for m, row in enumerate(x):
                    retained[m][n] += row[k] * w
        for m, row in enumerate(x):
            for n in range(32):
                for k in range(16):
                    reloaded[m][n] += row[k] * weight[n][k]
                    loads_by_row += 1
        self.assertEqual(retained, reloaded)
        self.assertEqual(loads_by_row, 15 * loads_once)

    @unittest.skipUnless(os.environ.get("CANN_ASC_ROOT"), "optional installed CANN header audit")
    def test_installed_sdk_line_reference_identities(self):
        root = Path(os.environ["CANN_ASC_ROOT"])
        rows = re.findall(r"^\| `((?:include|impl)/[^`]+)` \| [^|]+ \| `([0-9a-f]{64})` \|",
                          (HERE / "w16-event-lifecycle-revised.md").read_text(), re.M)
        self.assertEqual(len(rows), 23)
        self.assertEqual(len(set(path for path, _ in rows)), len(rows))
        for path, expected in rows:
            with self.subTest(path=path):
                self.assertEqual(digest((root / path).read_bytes()), expected)


if __name__ == "__main__":
    unittest.main()
