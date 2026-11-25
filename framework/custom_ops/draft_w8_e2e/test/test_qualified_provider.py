import copy
import tempfile
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from qualified_provider import recheck_anchors
from winners import c


class QualifiedProvider(unittest.TestCase):
    def test_probe_is_bound_to_same_plan_input_and_all_eight_real_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            idx = d / "batch.json"
            folder = Path(str(idx) + ".cases")
            folder.mkdir()

            def write(p, x):
                c.write(p, x)
                return c.identity(p)

            plan = write(d / "plan", {})
            batch = write(d / "tokens", {"example": "test"})
            runner = write(d / "runner", {})
            write(
                d / "command.json",
                [
                    runner["path"],
                    "--model",
                    plan["path"],
                    "--model-sha256",
                    plan["sha256"],
                    "--prompt-batch",
                    batch["path"],
                ],
            )
            history = dict(prompts=[], cases={})
            prev = {"reports": {}}
            rows = []
            cases = []
            for i in range(8):
                n = f"q{i}"
                history["prompts"].append(dict(id=n, group="short", tokens=[i]))
                mode = {
                    "measurements": [
                        {
                            "generated_token_ids": [i + 1],
                            "counters": {"graph_calls": 1},
                            "stage_ms": {"target_prefill": [1.0]},
                        }
                    ]
                }
                raw = dict(
                    prompt_token_ids=[i],
                    status="PASS",
                    cpu_fallback=False,
                    device_id=0,
                    model={"sha256": plan["sha256"]},
                    limits={"max_new_tokens": 1},
                    protocol={"warmup": 0, "repetitions": 1},
                    dflash=mode,
                )
                ref = write(folder / (n + ".json"), raw)
                cases.append(dict(id=n, status="PASS", report=ref["path"]))
                history["cases"][n] = ref
                prev["reports"][n] = write(
                    d / ("previous" + n), {"dflash": {"measurements": [{"generated_token_ids": [i]}]}}
                )
                rows.append(dict(prompt=n, historical=i + 1, candidate=i + 1, previous=i, report=ref))
            index = write(
                idx,
                dict(
                    fake_acl=False,
                    status="PASS",
                    model_sha256=plan["sha256"],
                    order="DFlash only; ordinary baseline external",
                    prompt_batch_sha256=batch["sha256"],
                    cases=cases,
                ),
            )
            probe = {"anchors": {"runner_index": index, "rows": rows}}
            recheck_anchors(probe, history, plan, runner, prev)
            crossed = copy.deepcopy(probe)
            crossed["anchors"]["rows"][-1]["historical"] = -1
            with self.assertRaisesRegex(ValueError, "unmatched anchor"):
                recheck_anchors(crossed, history, plan, runner, prev)


if __name__ == "__main__":
    unittest.main()
