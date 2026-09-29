"""CPU-only dependency-wiring tests; no NPU computation or fallback is run."""
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as Namespace
import unittest
from unittest.mock import Mock, patch, sentinel

import a2_capture


class CaptureStopped(Exception):
    """Stop the real capture entry point at its mocked DraftGraph constructor."""


class ModuleStub:
    def register_buffer(self, name, value):
        setattr(self, name, value)


class CaptureRowUpdateTests(unittest.TestCase):
    def test_preflight_returns_exact_production_default_without_executing_it(self):
        operation = Mock()
        torch = Namespace(ops=Namespace(npu=Namespace(npu_scatter_nd_update=Namespace(default=operation))))
        self.assertIs(a2_capture.require_draft_row_update(torch), operation)
        operation.assert_not_called()

    def test_preflight_rejects_missing_default_noncallable_and_inplace_only(self):
        namespaces = (
            Namespace(),
            Namespace(npu_scatter_nd_update=Namespace()),
            Namespace(npu_scatter_nd_update=Namespace(default=None)),
            Namespace(npu_scatter_nd_update=Namespace(default=123)),
            Namespace(npu_scatter_nd_update_=Namespace(default=Mock())),
        )
        for npu in namespaces:
            with self.subTest(npu=npu), self.assertRaisesRegex(
                    RuntimeError, r"torch\.ops\.npu\.npu_scatter_nd_update\.default.*no fallback"):
                a2_capture.require_draft_row_update(Namespace(ops=Namespace(npu=npu)))

    @contextmanager
    def capture_environment(self, operation):
        """Replace device dependencies, then exercise capture's actual wiring."""
        config = Namespace(hidden_size=2560, intermediate_size=9728, num_hidden_layers=5,
                           block_size=16, mask_token_id=99)
        draft = Namespace(config=config)
        loader = Mock(return_value=draft)
        graph = Mock(side_effect=CaptureStopped)
        torch = Namespace(
            float16=sentinel.float16,
            npu=Namespace(set_device=Mock(), get_device_name=Mock(return_value="Ascend310P3")),
            ops=Namespace(npu=Namespace(npu_scatter_nd_update=Namespace(default=operation))),
            nn=Namespace(Module=ModuleStub, Identity=Mock(return_value=sentinel.head)))
        modules = {
            "torch": torch, "torch_npu": Namespace(),
            "models.dflash_v1.draft_quantization": Namespace(load_quantized_draft=loader),
            "models.dflash_v1.modeling_dflash": Namespace(DFlashDraftModel=sentinel.draft_class),
            "qwen35_dflash.ascend310p.incremental": Namespace(DraftGraph=graph),
            "qwen35_dflash.ascend310p.quant_factory": Namespace(AirDFlashOps=Mock()),
            "qwen35_dflash.ascend310p.utils": Namespace(require_run_output=lambda path: path),
            "qwen35_dflash.ascend310p.weight_prepack": Namespace(pack_int8_nz=Mock()),
        }
        layers = (1, 5, 8, 9, 13, 15, 17, 21, 22, 25, 29)
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, modules), \
                patch.dict("os.environ", {"ASCEND310P_SIMULATION_ONLY": "0"}), \
                patch.object(a2_capture, "frozen_inputs", return_value=([], [], layers, {"anchor": 7})) as frozen, \
                patch.object(a2_capture, "embedding_rows", return_value=(Namespace(to=Mock()), {})):
            args = Namespace(output_dir=Path(directory) / "capture", device_id=0,
                             draft_dir=sentinel.draft_dir, target_dir=sentinel.target_dir,
                             replay_report=sentinel.replay, feature_layers=layers, embedding_key="embedding.weight")
            yield args, loader, graph, frozen, draft, layers

    def test_capture_binds_production_update_to_graph(self):
        operation = Mock()
        with self.capture_environment(operation) as (args, loader, graph, frozen, draft, layers):
            with self.assertRaises(CaptureStopped):
                a2_capture.capture(args)
            loader.assert_called_once()
            frozen.assert_called_once_with(args.replay_report, draft.config, layers)
            graph.assert_called_once()
            positional, keywords = graph.call_args
            self.assertIs(positional[0], draft)
            self.assertIs(positional[2], sentinel.head)
            self.assertIs(keywords["row_update"], operation)
            self.assertIs(keywords["feature_layers"], layers)
            self.assertTrue(keywords["consume_source"])
            operation.assert_not_called()

    def test_capture_missing_update_fails_before_checkpoint_and_replay(self):
        with self.capture_environment(None) as (args, loader, graph, frozen, _, _):
            with self.assertRaisesRegex(RuntimeError, "npu_scatter_nd_update.default"):
                a2_capture.capture(args)
            loader.assert_not_called()
            frozen.assert_not_called()
            graph.assert_not_called()
            manifest = json.loads((args.output_dir / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "FAIL")
            self.assertIn("npu_scatter_nd_update.default", manifest["error"])


if __name__ == "__main__":
    unittest.main()
