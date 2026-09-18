"""Scoped ATC workaround for the 310P native weight-quant transpose fusion."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re

from .utils import require_run_output, sha256_file

WEIGHT_QUANT_TRANSPOSE_PASS = "WeightQuantBatchMatmulV2TransposeNZFusionPass"
FUSION_OPTION = "--fusion_switch_file"


def _fusion_path(arguments):
    flags = [arg for arg in arguments if arg.split("=", 1)[0] == FUSION_OPTION]
    if len(flags) > 1 or (flags and ("=" not in flags[0] or not flags[0].split("=", 1)[1])):
        raise ValueError("use one --fusion_switch_file=/path/to/config")
    return Path(flags[0].split("=", 1)[1]).expanduser().resolve() if flags else None


def fusion_switch_record(arguments):
    path = _fusion_path(arguments)
    return ({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            if path is not None else None)


def normalized_atc_options(arguments):
    """Compare the fusion configuration by contents, including across bundles."""
    record = fusion_switch_record(arguments)
    return [FUSION_OPTION + "=sha256:" + record["sha256"]
            if arg.split("=", 1)[0] == FUSION_OPTION else arg for arg in arguments]


def weight_quant_fusion_args(graph, arguments, *, soc_version):
    """Disable only the failing pass; an explicit per-pass setting wins.

    AIR stays unchanged. This is a compiler compatibility candidate, not a
    claim that the private CANN pass or the resulting OM has been validated.
    """
    args = list(arguments)
    native = graph.get("name") == "draft" and any(
        item.get("ge_op_type") == "WeightQuantBatchMatmulV2"
        and item.get("status") == "PASS" and item.get("ge_node_occurrences", 0) > 0
        for item in graph.get("custom_op_audit", [])
    )
    if not native or not re.sub(r"[^a-z0-9]", "", soc_version.lower()).startswith("ascend310p"):
        return args
    source = _fusion_path(args)
    text = source.read_text(encoding="utf-8") if source else "{}"
    if text.lstrip().startswith("{"):
        data = json.loads(text)
        switch = data.setdefault("Switch", {})
        if not isinstance(switch, dict):
            raise ValueError("fusion config Switch must be an object")
        graph_switch = switch.setdefault("GraphFusion", {})
        if not isinstance(graph_switch, dict):
            raise ValueError("fusion config GraphFusion must be an object")
        graph_switch.setdefault(WEIGHT_QUANT_TRANSPOSE_PASS, "off")
        setting = graph_switch[WEIGHT_QUANT_TRANSPOSE_PASS]
        text = json.dumps(data, indent=2, sort_keys=True) + "\n"
    else:
        matches = re.findall(r"^\s*" + WEIGHT_QUANT_TRANSPOSE_PASS + r"\s*:\s*(\w+)\s*$", text, re.M)
        if len(matches) > 1:
            raise ValueError("duplicate weight-quant fusion switch")
        setting = matches[0] if matches else "off"
        if not matches:
            text = text.rstrip() + "\n" + WEIGHT_QUANT_TRANSPOSE_PASS + ":off\n"
    if setting not in ("on", "off"):
        raise ValueError("weight-quant fusion switch must be on or off")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    path = require_run_output(Path(os.environ["AI_RUN_DIR"]) / "log" / "dflash-atc" /
                              "fusion" / (digest + ".cfg"))
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(text)
    except FileExistsError:
        if sha256_file(path) != digest:
            raise ValueError("generated ATC fusion config integrity check failed")
    print(f"[compile-om] {WEIGHT_QUANT_TRANSPOSE_PASS}={setting} (quantized Draft)", flush=True)
    return [arg for arg in args if arg.split("=", 1)[0] != FUSION_OPTION] + [FUSION_OPTION + "=" + str(path)]
