"""Export only the two Target graphs, using unmodified historical factory/model code."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def run(args):
    base = args.native.resolve(strict=True)
    root = args.out.resolve(strict=True)
    cfg = json.loads(args.config.read_text())
    # The snapshot's factory verifies its own SOURCE_LOCK. Keep the receiver available only after it.
    sys.path[:0] = [str(base / "framework/python"), str(base), str(Path(cfg["receiver_models_dir"]).parent)]
    import torch, torch_npu

    sys.path.insert(0, str(Path(torch_npu.__file__).parent / "dynamo"))
    import torchair
    from qwen35_dflash.ascend310p import exporter as ex
    from qwen35_dflash.ascend310p.quant_factory import create_quant_incremental_graphs as factory
    from qwen35_dflash.ascend310p.input_manifest import build_quant_input_manifest
    from qwen35_dflash.ascend310p.utils import file_record
    from qwen35_dflash.ascend310p.runtime_input_export import canonical_runtime_input_abi
    from qwen35_dflash.ascend310p.custom_op_export import prepare_custom_op_export, audit_custom_op_export
    from qwen35_dflash.ascend310p.standard_op_export import (
        prepare_aten_softplus_export,
        audit_aten_softplus_export,
    )

    assert base in Path(ex.__file__).resolve().parents
    print(
        "Target export runtime",
        sys.executable,
        torch.__version__,
        torch_npu.__version__,
        torchair.__file__,
        flush=True,
    )
    print("Chunk dispatcher", torch.ops.npu.npu_chunk_gated_delta_rule.default._schema, flush=True)
    preflight = factory.prepare_export(dict(cfg), torchair)
    (root / "preflight.json").write_text(json.dumps(preflight, indent=2))
    ip = root / "input-manifest.json"
    build_quant_input_manifest(
        **{k: cfg[k] for k in ("target_dir", "draft_dir", "quant_config", "receiver_models_dir")}, output=ip
    )
    cfg = dict(cfg, input_manifest=str(ip))
    specs = factory(cfg)
    required = ("target_prefill", "target_verify")
    graphs = {s.name: s for s in specs if s.name in required}
    if set(graphs) != set(required):
        raise ValueError("missing Target Prefill/Chunk Verify")
    del specs
    result = {}
    for name in required:
        s = graphs.pop(name)
        meta = s.metadata
        co = meta["incremental_contract"]
        if (
            co["abi"],
            co["capacity"],
            co["cache_capacity"],
            co["feature_width"],
            co["target_feature_layers"],
            co["draft_feature_layers"],
            meta["target_quant_mode"],
        ) != (
            "qwen35-dflash-chunk-v4",
            2048,
            2112,
            28160,
            [1, 5, 8, 9, 13, 15, 17, 21, 22, 25, 29],
            [1, 8, 15, 22, 29],
            "w8a8_dynamic",
        ):
            raise ValueError("rebuilt Target ABI/layer contract differs from documented generation")
        print("Exporting", name, "from W8A8 Target weights", flush=True)
        d = root / name
        d.mkdir()
        sessions = [prepare_custom_op_export(op, torchair) for op in s.custom_ops]
        soft = prepare_aten_softplus_export(torchair) if meta.get("standard_op_export_contracts") else None
        kw = dict(s.example_kwargs, model=s.model.eval(), export_path=str(d), export_name=name, dynamic=False)
        if s.compiler_config is not None:
            kw["config"] = s.compiler_config
        with torch.inference_mode(), ex._working_directory(d), canonical_runtime_input_abi(
            torchair,
            public_inputs=s.example_args,
            public_names=s.input_names,
            require_static_shapes=True,
            public_output_names=s.output_names,
            verify_discard_output_names=(
                [t["name"] for t in co["verify_discard_states"]] if name == "target_verify" else ()
            ),
        ) as binding:
            ex._mark_input_shapes(s)
            torchair.dynamo_export(*s.example_args, **kw)
        audits = audit_custom_op_export(sessions, d, relative_to=root)
        overrides = (
            [] if soft is None else [audit_aten_softplus_export(soft, d, calls_before=0, relative_to=root)]
        )
        (air,) = d.glob("*.air")

        def file(p):
            rec = file_record(p, relative_to=root)
            rec["path"] = str(p.resolve())
            return rec

        result[name] = dict(
            air=file(air),
            payloads=[file(p) for p in sorted(d.rglob("*")) if p.is_file()],
            abi=meta["tensor_abi"],
            target_layers=co["target_feature_layers"],
            contract=co,
            target_input_identity=meta["target_input_identity"],
            source_lock=meta["quant_source_lock"],
            binding=binding,
            custom_op_audit=audits,
            standard_ops=overrides,
        )
        del s, kw
        torch._dynamo.reset()
    (root / "exported.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--native", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    run(p.parse_args())
