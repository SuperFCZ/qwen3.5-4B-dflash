import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gate_env_check as probe


class IndependentConda(unittest.TestCase):
    def test_sdk_and_packages_have_separate_prefixes(self):
        with tempfile.TemporaryDirectory() as tmp:
            top = Path(tmp).resolve()
            env = top / "env"
            cann = top / "cann"
            (env / "conda-meta").mkdir(parents=True)
            (env / "conda-meta/history").touch()
            tools = {
                k: str(root / "bin" / k)
                for k, root in (
                    ("python3", env),
                    ("atc", cann),
                    ("msopgen", cann),
                    ("cmake", top),
                    ("make", top),
                    ("c++", top),
                )
            }

            def mod(name):
                return SimpleNamespace(__file__=str(env / (name + ".py")))

            def sdk(name):
                p = cann / (name + ".py")
                p.touch()
                return SimpleNamespace(origin=str(p))

            cann.mkdir()
            for name in ("numpy", "scipy", "google.protobuf", "grpc_tools.protoc"):
                Path(mod(name).__file__).touch()
            with patch.object(sys, "version_info", (3, 11)), patch.object(
                sys, "prefix", str(env)
            ), patch.object(sys, "base_prefix", str(env)), patch.object(
                sys, "executable", tools["python3"]
            ), patch.object(
                probe.shutil, "which", side_effect=tools.get
            ), patch.object(
                probe.importlib, "import_module", side_effect=mod
            ), patch.object(
                probe.importlib.util, "find_spec", side_effect=sdk
            ), patch.object(
                probe.subprocess, "run"
            ) as command:
                probe.check(env, cann, top)
                self.assertEqual(command.call_args.args[0][0], tools["python3"])
                self.assertEqual(json.loads((top / "conda-environment.json").read_text())["prefix"], str(env))
                with patch.object(
                    probe.importlib, "import_module", return_value=SimpleNamespace(__file__=__file__)
                ):
                    with self.assertRaisesRegex(ValueError, "escaped"):
                        probe.check(env, cann, top)

    def test_venv_and_foreign_interpreter_are_rejected_before_imports(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "independent"):
                probe.check(Path(tmp), Path(tmp), Path(tmp))
