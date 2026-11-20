"""Host-only tests, including counterexamples for producer-only fences."""

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fence_contract import (
    ALL,
    DESIGN,
    MTE3,
    NARROW,
    W_EVENT,
    Queues,
    missing_edges,
    reserve_fence_ids,
    simulate,
)
from review_evidence import research_signal, sdk_check, source_check


class FenceTests(unittest.TestCase):
    def test_preregistered_thresholds_and_p95(self):
        def samples(v):
            return [[v] * 30 for _ in range(3)]

        self.assertTrue(research_signal("single-w-raw", samples(1), samples(0.97))["signal"])
        self.assertFalse(research_signal("single-w-raw", samples(1.03), samples(1))["signal"])
        self.assertTrue(research_signal("double-w-overlap", samples(1.03), samples(1))["signal"])
        tail = samples(0.95)
        for r in tail:
            r[-2:] = [1.2, 1.2]
        self.assertFalse(research_signal("single-w-raw", samples(1), tail)["signal"])
        rounds = samples(0.9)
        rounds[1] = [1] * 30
        self.assertFalse(research_signal("single-w-raw", samples(1), rounds)["signal"])
        with self.assertRaises(ValueError):
            research_signal("single-w-raw", samples(1), [[0.8] * 29] * 3)

    def test_source_fingerprints_and_rendered_variants(self):
        source_check()

    def test_source_schedule_and_partial_order(self):
        for layout in DESIGN:
            narrow = layout == NARROW
            trace = simulate(layout, NARROW if narrow else ALL, n_tiles=2)
            self.assertEqual(trace.outputs, [0, 1])
            self.assertEqual(trace.cubes, [(n, k, k > 0) for n in range(2) for k in range(10)])
            self.assertFalse(trace.queues.signals)
            self.assertFalse(missing_edges(trace, 1 if layout == "single-w-raw" else 2))
            pos = trace.actions.index
            for k in range(9):
                d = pos((0, k + 1, "dequant"))
                f = pos((0, k, "fence"))
                i = pos((0, k, "iterate"))
                self.assertLess(i, f)
                if layout in ("double-w-overlap", NARROW):
                    self.assertTrue(i < d < f)
                else:
                    self.assertLess(f, d)
                self.assertLess(f, pos((0, k + 1, "iterate")))
            if layout == "single-w-raw":
                self.assertLess(pos((0, 1, "copy")), pos((0, 0, "dequant")))
            else:
                self.assertLess(pos((0, 0, "dequant")), pos((0, 1, "copy")))

    def test_same_pipe_barrier_is_not_cross_pipe_release(self):
        trace = simulate(fence=MTE3)
        misses = missing_edges(trace, 2)
        self.assertIn(("W reuse", "n0.k0.B1_write_W_read.end", "n0.k2.D.start"), misses)
        self.assertIn(("A1 reuse", "n0.k0.A1_load_L0.end", "n0.k1.A1_write.start"), misses)
        # No cycle/negative numbers are needed: these endpoints are simply unordered.
        self.assertFalse(trace.queues.before("n0.k2.D.start", "n0.k0.B1_write_W_read.end"))

    def test_mte3_vector_releases_w_but_not_a1(self):
        trace = simulate(fence=W_EVENT)
        misses = missing_edges(trace, 2)
        self.assertTrue(all(x[0] == "A1 reuse" for x in misses))
        self.assertEqual(len(misses), 9)
        self.assertTrue(trace.queues.before("n0.k0.B1_write_W_read.end", "n0.k2.D.start"))
        self.assertTrue(trace.queues.before("n0.k0.B1_write_W_read.end", "n0.k1.bind.start"))

    def test_candidate_directions_and_generation_acknowledgements(self):
        trace = simulate(NARROW, NARROW, n_tiles=2)
        self.assertEqual(sum(n.endswith("F_load.set") for n in trace.queues.bit), 18)
        self.assertEqual(sum(n.endswith("F_raw.set") for n in trace.queues.bit), 18)
        self.assertTrue(
            all(ok for key, _, _, ok in trace.queues.rearms if str(key[2]).startswith("candidate"))
        )
        for n in range(2):
            for k in range(9):
                first = f"n{n}.k{k}."
                nxt = f"n{n}.k{k + 1}."
                self.assertTrue(trace.queues.before(first + "B1_write_W_read.end", nxt + "bind.start"))
                self.assertTrue(trace.queues.before(first + "A1_load_L0.end", nxt + "A1_write.start"))
            self.assertIn(f"n{n}.k9.F_ALL", trace.queues.bit)
            self.assertIn(f"n{n}.k9.N_ALL", trace.queues.bit)
        # Isolate the two edges. Lost A1 protection, or lost raw-token acknowledgement.
        self.assertTrue(any(x[0] == "A1 reuse" for x in missing_edges(simulate(fence="V_MTE2-only"), 2)))
        t = simulate(fence="MTE1_V-only")
        self.assertTrue(
            any(
                key == ("MTE2", "V", "raw") and "k2.raw_V.set" in b and not ok
                for key, _, b, ok in t.queues.rearms
            )
        )

    def test_common_prologue_is_not_silently_certified(self):
        for layout in ("double-w-serial", "double-w-overlap", NARROW):
            t = simulate(layout, NARROW if layout == NARROW else ALL, n_tiles=2)
            unknown = [(a, b) for key, a, b, ok in t.queues.rearms if key == ("MTE2", "V", "raw") and not ok]
            self.assertEqual(unknown, [(f"n{n}.k0.raw_V.wait", f"n{n}.k1.raw_V.set") for n in range(2)])

    def test_event_leases_collision_capacity_and_rollback(self):
        directions = ("MTE1_V", "V_MTE2")
        occupancy = {d: 0b11 for d in directions}
        reserved, leases = reserve_fence_ids(occupancy, {d: 1 for d in directions})
        self.assertEqual(list(leases.values()), [2, 2])  # same number, different pools
        self.assertEqual(list(reserved.values()), [7, 7])
        for left, right, needed in ((255, 0, 0), (0, 255, 0), (127, 0, 1)):
            initial = dict(zip(directions, (left, right)))
            current, ids = reserve_fence_ids(initial, {d: needed for d in directions})
            self.assertEqual(current, initial)
            self.assertFalse(ids)

    def test_cube_hazard_independent_of_w_source_release(self):
        t = simulate(NARROW, NARROW, matrix_barrier=False)
        misses = missing_edges(t, 2)
        self.assertEqual([m[0] for m in misses], ["L0C partial"] * 9)
        self.assertTrue(t.queues.before("n0.k0.B1_write_W_read.end", "n0.k2.D.start"))
        self.assertFalse(t.queues.before("n0.k0.mad.end", "n0.k1.mad.start"))
        # The proposed handoffs allow unrelated MMAD to remain outstanding.
        self.assertFalse(simulate(NARROW, NARROW).queues.before("n0.k0.mad.end", "n0.k2.raw.start"))

    def test_output_failure_release_and_no_speculation(self):
        for n in range(2):
            for k in range(10):
                for site in ("V_S-unavailable", "MoveNext"):
                    t = simulate(NARROW, NARROW, n_tiles=2, fail=(n, k), failure_site=site)
                    self.assertEqual(t.outputs, list(range(n)))
                    self.assertEqual([(a, b) for a, b, _ in t.cubes if a == n], [(n, i) for i in range(k)])
                    self.assertNotIn((n, k + 1, "dequant"), t.actions)
                    self.assertEqual(t.drains[-1], f"n{n}.k{k}.failure_ALL")
                    self.assertFalse(t.queues.signals)
                    for name in t.queues.bit:
                        if name.endswith((".raw.end", ".D.end", ".mad.end", ".F_load.wait", ".F_raw.wait")):
                            self.assertTrue(t.queues.before(name, "release_end.start"), name)
                    self.assertFalse(any(".k10." in name for name in t.queues.bit))

    def test_w_footprint_and_guarded_boundaries(self):
        for bank in (33280, 66048):
            reads = [bank + 2 * (256 * row + 16 * stripe) for stripe in range(16) for row in range(64)]
            self.assertEqual(sorted(reads), list(range(bank, bank + 32768, 32)))
            self.assertEqual(max(reads) + 32, bank + 32768)
        self.assertEqual(33280 + 32768, 66048)
        self.assertEqual(66048 + 32768, 98816)
        self.assertEqual(101120 + 152832, 253952)
        self.assertEqual(253952 + 8192, 262144)
        self.assertEqual((16 // 16) * (64 // 16), 4)
        self.assertLess(4, 10)

    def test_only_explicit_completion_and_full_scope(self):
        g = Queues()
        g.op("m", "M")
        g.op("v", "V")
        g.op("dma", "MTE3")
        g.barrier("dma_barrier", "MTE3")
        g.op("later_vector", "V")
        self.assertFalse(g.before("dma.end", "later_vector.start"))
        g.barrier("full", "ALL")
        g.op("after", "S")
        for op in ("m", "v", "dma", "later_vector"):
            self.assertTrue(g.before(op + ".end", "after.start"))

    def test_sdk_manifest_has_unique_bindings_and_fails_closed(self):
        data = json.loads((HERE / "sdk-evidence.json").read_text())
        self.assertEqual(len(data["files"]), len({r["path"] for r in data["files"]}))
        self.assertEqual(sum(r["binding"] == "YHsAKu-admit" for r in data["files"]), 26)
        self.assertEqual(
            next(r["sha256"] for r in data["files"] if r["path"].endswith("/mmad_compute.h")),
            "c5f5dd39c23f845d4a8e4088030c6e173bf0273370d81152819432a952c66c5b",
        )
        with tempfile.TemporaryDirectory() as d:
            base = Path(d) / data["root"]
            base.mkdir(parents=True)
            first = base / data["files"][0]["path"]
            first.parent.mkdir(parents=True, exist_ok=True)
            first.touch()
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                sdk_check(Path(d))

    @unittest.skipUnless(os.environ.get("FENCE_CANN_ROOT"), "requires selected CANN 9.0 installation")
    def test_all_sdk_hashes_and_source_locations(self):
        self.assertEqual(len(sdk_check(Path(os.environ["FENCE_CANN_ROOT"]))), 43)


if __name__ == "__main__":
    unittest.main()
