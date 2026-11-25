import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from probe_dynamic import add_provider, compare, probe_args, FAMILIES
import probe_dynamic as probe


class Probe(unittest.TestCase):
    def test_full_driver_keeps_prefill_after_fingerprinting_both_siblings(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as ctx:
            root = Path(tmp)
            out = root / "new"
            out.mkdir()
            source = root / "w8-e2e-fixture"
            source.mkdir()
            vendor = root / "dynamic"
            vendor.mkdir()
            for name in ("customize_quantMatmul", "customize_linearAttention_new"):
                (root / name).mkdir()
            tool = root / "atc"
            tool.write_text("")
            tool.chmod(0o755)

            def file(name, content):
                p = root / name
                probe.c.write(p, content)
                return probe.c.identity(p)

            prior = dict(
                om=file("old-om", {}),
                source_air=file("air", {}),
                cache_backend=file("backend", {"dump": file("dump", {})}),
            )
            models = {"target_prefill": prior}
            draft_ref = file("draft-om", {})
            drafts = {str(g): dict(om=draft_ref) for g in (16, 64)}
            probe.c.write(
                source / "target.json", dict(models=models, vendors=[], cache_provider=dict(path=str(root)))
            )
            probe.c.write(source / "result.json", dict(status="E2E_MEASURED", targets=models, drafts=drafts))
            stages = Mock()

            def receipt(name):
                if name == "reconstruct-target_prefill":
                    return {"result": {"air": prior["source_air"]}}
                return {"result": {"om": draft_ref, "abi": {}}}

            stages.completed.side_effect = receipt
            env = {"PATH": str(root)}
            ctx.enter_context(patch.dict(os.environ, CANN_ROOT=str(root)))

            def stub(name, *args, **kw):
                return ctx.enter_context(patch.object(probe, name, *args, **kw))

            stub("Stages", return_value=stages)
            stub("read_history", return_value={})
            stub("receiver_env", return_value=env)
            stub("add_provider", return_value=(env, {"order": [str(vendor)]}))
            old = {
                f: dict(role=f, engine="AIcoreEngine", vendor=str(root / f), text=["code"], block=1)
                for f in FAMILIES
            }
            new = dict(old, DynamicQuant=dict(old["DynamicQuant"], vendor=str(vendor), text=["new"]))
            stub("inventory", side_effect=[(old, {}), (new, {})])
            stub("package_fingerprint", return_value=dict(sources=["same"], text=["same"]))
            compiled = stub("compile_target", return_value=dict(om=file("new-om", {})))
            stub("audit_compiled", return_value=prior["cache_backend"])
            anchors = stub("run_anchors", return_value=dict(matches=1))
            state = {}
            result = probe.run(
                out, SimpleNamespace(provider=vendor, from_run=source.name, inspect_only=False), state
            )
            self.assertEqual(result, "ANCHOR_DIFFERENCES")
            self.assertEqual(len(state["sibling_comparisons"]), 2)
            self.assertEqual(compiled.call_args.args[2]["air"], prior["source_air"])
            self.assertEqual(anchors.call_count, 1)

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
