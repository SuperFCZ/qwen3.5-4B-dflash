import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from probe_dynamic import add_provider, compare, probe_args, FAMILIES


class Probe(unittest.TestCase):
    def test_child_source_order_and_scope(self):
        with tempfile.TemporaryDirectory() as d:
            r = Path(d)
            v = r / "provider"
            p = v / "bin/set_env.bash"
            p.parent.mkdir(parents=True)
            info = v / "op_impl/ai_core/tbe/config/ascend310p/aic-ops-info.json"
            info.parent.mkdir(parents=True)
            info.write_text(json.dumps({"DynamicQuant": {}}))
            p.write_text(
                'v=${BASH_SOURCE[0]%/bin/*}\nexport ASCEND_CUSTOM_OPP_PATH="$v:$ASCEND_CUSTOM_OPP_PATH"\nexport LD_LIBRARY_PATH="$v/op_api/lib:$LD_LIBRARY_PATH"\n'
            )
            base = dict(os.environ, ASCEND_CUSTOM_OPP_PATH=str(r / "previous"))
            e, proof = add_provider(base, v)
            self.assertEqual(proof["order"], [str(v.resolve()), str((r / "previous").resolve())])
            self.assertNotEqual(base["ASCEND_CUSTOM_OPP_PATH"], e["ASCEND_CUSTOM_OPP_PATH"])
            p.write_text(p.read_text() + "export ASCEND_OPP_PATH=/unexpected\n")
            with self.assertRaisesRegex(ValueError, "base CANN OPP changed"):
                add_provider(base, v)

    def test_one_changed_family(self):
        def family(role):
            return dict(role=role, engine="AIcoreEngine", vendor=None, block=8, text=[{"sha256": "A"}])

        old = {f: family(f) for f in FAMILIES}
        new = dict(old, DynamicQuant=dict(old["DynamicQuant"], vendor="/new", text=[{"sha256": "B"}]))
        info = compare(old, new, Path("/new"))
        self.assertEqual((info["DynamicQuant"]["code_changes"], info["DynamicQuant"]["chosen"]), (1, 1))
        new["ChunkGatedDeltaRule"] = dict(old["ChunkGatedDeltaRule"], vendor="/wrong")
        with self.assertRaisesRegex(ValueError, "unrelated backend"):
            compare(old, new, Path("/new"))

    def test_single_token_dflash_only(self):
        old = [
            "runner",
            "--model-kind",
            "chunk",
            "--mode",
            "dflash",
            "--model",
            "old",
            "--model-sha256",
            "oldhash",
            "--prompt-batch",
            "batch",
            "--prompt-batch-sha256",
            "oldbatch",
            "--warmup",
            "1",
            "--repetitions",
            "3",
            "--output",
            "prior",
            "--max-new-tokens",
            "128",
            "--max-draft-tokens",
            "15",
            "--trace-rounds",
            "--low-memory",
        ]
        q = probe_args(old, {"path": "new", "sha256": "H"}, {"path": "short", "sha256": "B"}, "output")
        self.assertEqual(q[q.index("--max-new-tokens") + 1], "1")
        self.assertEqual(q[q.index("--warmup") + 1], "0")
        self.assertEqual(q[q.index("--mode") + 1], "dflash")
        self.assertIn("prior", old)


if __name__ == "__main__":
    unittest.main()
