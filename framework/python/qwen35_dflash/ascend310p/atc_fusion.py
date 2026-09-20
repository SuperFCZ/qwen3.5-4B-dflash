"""Content-locked provenance for explicitly supplied ATC fusion settings."""
from __future__ import annotations

from pathlib import Path

from .utils import sha256_file

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
