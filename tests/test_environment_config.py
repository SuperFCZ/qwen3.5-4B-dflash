from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "config/dflash_env.sh.example"
PROBE = """
import json, os, sys
from pathlib import Path
values = {k: os.environ.get(k) for k in (
    'REPO_ROOT', 'AI_RUN_DIR', 'MODEL_PYTHON', 'CANN_ROOT', 'TARGET_DIR', 'DRAFT_DIR',
    'RECEIVER_MODELS_DIR', 'DEPLOYMENT_MANIFEST', 'VERIFY_GDR', 'QUANT_MODE',
    'KV_CAPACITY', 'MAX_SEQUENCE_LENGTH', 'MAX_NEW_TOKENS', 'MAX_DRAFT_TOKENS',
    'BLOCK_SIZE', 'DEVICE_ID', 'TMPDIR', 'HF_HOME', 'TORCH_HOME', 'XDG_CACHE_HOME',
    'PYTHONPATH', 'PYTHONDONTWRITEBYTECODE', 'DFLASH_TEST_CANN_LOADS',
    'DRAFT_FP16_DIR', 'DRAFT_W4A16_DIR', 'DRAFT_W8A16_DIR', 'DRAFT_QUANTIZATION',
    'DRAFT_SELECTED_DIR', 'DRAFT_VARIANTS_MANIFEST', 'SELECTED_DRAFT_DEPLOYMENT_MANIFEST'
)}
values['args'] = sys.argv[1:]
values['cwd'] = os.getcwd()
if values['RECEIVER_MODELS_DIR']:
    from models import export_model_wrapper_qwen3_5
    values['receiver_marker'] = export_model_wrapper_qwen3_5.MARKER
print(json.dumps(values, ensure_ascii=False))
"""


class EnvironmentConfigTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="dflash env ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repo = self.root / "source repo's alias"
        self.repo.symlink_to(ROOT, target_is_directory=True)
        self.run = self.root / "run directory"
        self.cann = self.root / "CANN's toolkit"
        self.cann.mkdir()
        (self.cann / "set_env.sh").write_text(
            "export DFLASH_TEST_CANN_LOADS=$(( ${DFLASH_TEST_CANN_LOADS:-0} + 1 ))\n"
        )
        self.receiver = self.root / "receiver package"
        self.make_receiver(self.receiver, "first receiver")
        self.config = self.root / "dflash-env.sh"
        self.values = {
            "REPO_ROOT": str(self.repo), "AI_RUN_DIR": str(self.run),
            "MODEL_PYTHON": sys.executable, "CANN_ROOT": str(self.cann),
            "TARGET_DIR": str(self.root / "target weights"),
            "DRAFT_DIR": str(self.root / "draft weights"),
            "RECEIVER_ROOT": str(self.receiver), "VERIFY_GDR": "chunk",
            "QUANT_MODE": "enable", "DEVICE_ID": "2", "MAX_NEW_TOKENS": "512",
            "KV_CAPACITY": "2048", "MAX_DRAFT_TOKENS": "7",
            "PROMPT": "含空格、引号 ' 和字面量 $(unchanged) 的问题",
        }
        self.write_config()

    def make_receiver(self, root, marker):
        (root / "models").mkdir(parents=True)
        (root / "models/export_model_wrapper_qwen3_5.py").write_text(
            "MARKER = " + repr(marker) + "\n"
        )

    def write_config(self):
        lines = []
        for line in TEMPLATE.read_text().splitlines():
            if line.startswith("export "):
                name = line[7:].split("=", 1)[0]
                if name in self.values:
                    line = f"export {name}={shlex.quote(self.values[name])}"
            lines.append(line)
        self.config.write_text("\n".join(lines) + "\n")

    def shell(self, body, extra_env=None):
        # A fresh terminal: no inherited model settings, BASH_ENV or Python path.
        env = {key: os.environ[key] for key in ("PATH", "LANG", "LC_ALL") if key in os.environ}
        env.update(extra_env or {})
        return subprocess.run(
            ["bash", "--noprofile", "--norc", "-c", body],
            cwd=self.root, env=env, text=True, capture_output=True,
        )

    def probe(self, prefix="", extra_env=None):
        body = (
            "set -euo pipefail\n" + prefix
            + "source " + shlex.quote(str(self.config)) + "\n"
            + '"$MODEL_PYTHON" -B -c ' + shlex.quote(PROBE)
            + ' "${NPU_ARGS[@]}" --quant-boundary "${QUANT_ARGS[@]}"\n'
        )
        result = self.shell(body, extra_env)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout.splitlines()[-1])

    def test_fresh_terminal_restores_exports_arrays_and_receiver_import(self):
        values = self.probe()
        self.assertEqual(values["receiver_marker"], "first receiver")

        self.assertEqual(values["DEPLOYMENT_MANIFEST"],
                         str(self.run / "artifacts/deployment-manifest.json"))
        self.assertEqual(values["BLOCK_SIZE"], "8")
        self.assertEqual(values["MAX_SEQUENCE_LENGTH"], "2048")
        self.assertEqual(values["MAX_NEW_TOKENS"], "512")
        self.assertEqual(values["DEVICE_ID"], "2")
        self.assertIn(self.values["PROMPT"], values["args"])
        self.assertIn("npu:2", values["args"])
        self.assertEqual(values["args"][-4:],
                         ["--quant_mode", "enable", "--config", str(self.run / "qwen35-w8a8.yaml")])
        self.assertEqual(values["cwd"], str(self.root))
        self.assertEqual(values["PYTHONDONTWRITEBYTECODE"], "1")
        for name in ("TMPDIR", "HF_HOME", "TORCH_HOME", "XDG_CACHE_HOME"):
            self.assertTrue(Path(values[name]).is_relative_to(self.run))
        self.assertTrue((self.run / "reports").is_dir())

    def test_quantized_checkpoint_paths_and_selection_restore_in_fresh_shell(self):
        self.values.update(DRAFT_W4A16_DIR=str(self.root / "W4 weights"),
                           DRAFT_W8A16_DIR=str(self.root / "W8 weights"), DRAFT_QUANTIZATION="w4a16")
        self.write_config()
        values = self.probe()
        self.assertEqual(values["DRAFT_SELECTED_DIR"], str(self.root / "W4 weights"))
        self.assertEqual(values["DRAFT_FP16_DIR"], self.values["DRAFT_DIR"])
        self.assertEqual(values["DRAFT_W8A16_DIR"], str(self.root / "W8 weights"))
        self.assertEqual(values["SELECTED_DRAFT_DEPLOYMENT_MANIFEST"],
                         str(self.run / "artifacts-drafts/w4a16/chunk/deployment-manifest.json"))

    def test_repeated_source_preserves_shell_flags_and_avoids_duplicate_paths(self):
        inherited = str(self.root / "other Python modules")
        prefix = "source " + shlex.quote(str(self.config)) + "\n"
        values = self.probe(prefix, {"PYTHONPATH": inherited + ":" + inherited})
        self.assertEqual(values["DFLASH_TEST_CANN_LOADS"], "1")
        entries = values["PYTHONPATH"].split(":")
        self.assertEqual(len(entries), len(set(entries)))
        self.assertIn(inherited, entries)
        result = self.shell(
            "set -euo pipefail\nbefore=$-\nsource " + shlex.quote(str(self.config))
            + '\n[[ "$before" == "$-" ]]\n[[ "$(set -o | sed -n \'/^pipefail/p\')" == *on ]]\n'
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_switch_config_changes_route_precision_and_receiver(self):
        first = self.root / "first-env.sh"
        first.write_text(self.config.read_text())
        second_receiver = self.root / "another receiver"
        self.make_receiver(second_receiver, "second receiver")
        self.values.update(VERIFY_GDR="mtp", QUANT_MODE="disable",
                           MAX_DRAFT_TOKENS="15", RECEIVER_ROOT=str(second_receiver))
        self.write_config()
        values = self.probe("source " + shlex.quote(str(first)) + "\n")
        self.assertEqual(values["VERIFY_GDR"], "mtp")
        self.assertEqual(values["BLOCK_SIZE"], "16")
        self.assertTrue(values["DEPLOYMENT_MANIFEST"].endswith("artifacts/deployment-manifest-mtp.json"))
        self.assertEqual(values["args"][-1], "--quant-boundary")
        self.assertEqual(values["receiver_marker"], "second receiver")
        self.assertNotIn(str(self.receiver), values["PYTHONPATH"].split(":"))
        self.assertEqual(values["DFLASH_TEST_CANN_LOADS"], "1")

    def test_existing_om_does_not_require_receiver_or_built_artifacts(self):
        self.values["RECEIVER_ROOT"] = ""
        self.write_config()
        values = self.probe()
        self.assertEqual(values["RECEIVER_MODELS_DIR"], "")
        self.assertNotIn(str(self.repo / "tools/python-bootstrap"), values["PYTHONPATH"].split(":"))
        self.assertFalse(Path(values["DEPLOYMENT_MANIFEST"]).exists())

    def test_existing_bootstrap_and_reports_are_not_overwritten(self):
        old_bootstrap = self.run / "python-bootstrap/sitecustomize.py"
        old_bootstrap.parent.mkdir(parents=True)
        old_bootstrap.write_text("# existing user customization\n")
        reports = self.run / "reports"
        reports.mkdir()
        old_report = reports / "saved.json"
        old_report.write_text('{"keep":true}')
        self.probe()
        self.assertEqual(old_bootstrap.read_text(), "# existing user customization\n")
        self.assertEqual(old_report.read_text(), '{"keep":true}')

    def test_direct_execution_explains_source_and_does_not_create_run(self):
        result = subprocess.run(["bash", str(self.config)], text=True, capture_output=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("source", result.stderr)
        self.assertFalse(self.run.exists())

    def test_invalid_settings_return_to_caller_before_setup(self):
        for name, value in (("VERIFY_GDR", "bad"), ("QUANT_MODE", "bad"),
                            ("MAX_DRAFT_TOKENS", "16"),
                            ("AI_RUN_DIR", str(self.repo / "generated"))):
            with self.subTest(name=name):
                previous = self.values[name]
                self.values[name] = value
                self.write_config()
                result = self.shell(
                    "source " + shlex.quote(str(self.config))
                    + '\nstatus=$?\n[[ "$status" != 0 ]] || exit 10\nprintf "caller-alive\\n"\n'
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(name, result.stderr)
                self.assertEqual(result.stdout.strip(), "caller-alive")
                self.assertFalse(self.run.exists())
                self.values[name] = previous

    def test_failed_cann_setup_is_not_reported_as_restored(self):
        (self.cann / "set_env.sh").write_text("return 7\n")
        result = self.shell("source " + shlex.quote(str(self.config)))
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("[dflash-env]", result.stdout)
        self.assertFalse(self.run.exists())


if __name__ == "__main__":
    unittest.main()
