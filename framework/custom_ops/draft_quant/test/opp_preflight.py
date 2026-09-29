#!/usr/bin/env python3
"""Check CANN 9.0 kernel object/JSON paths before loading the isolated OPP.

This is a filesystem check, not an ACLNN or NPU correctness test. CANN v9.0.0
uses the FIRST '.o' / '.json' in a complete path when deriving sibling names.
"""
import argparse
import hashlib
import json
from pathlib import Path

OP_KERNEL = "d_flash_group_quant_linear"


def cann_json_path(binary):
    text = str(binary)
    index = text.find(".o")  # NnopbaseGetOpJsonPath, opbase v9.0.0
    if index < 0:
        raise ValueError(f"CANN cannot derive JSON from {binary}")
    return Path(text[:index + 1] + "json")


def cann_binary_path(metadata):
    text = str(metadata)
    index = text.find(".json")  # NnopbaseGetOpBinPath, opbase v9.0.0
    if index < 0:
        raise ValueError(f"CANN cannot derive object from {metadata}")
    return Path(text[:index + 1] + "o")


def check_binary_path(binary):
    """Reject ambiguous directory/filename prefixes, including symlink targets."""
    binary = Path(binary).absolute()
    if binary.suffix != ".o":
        raise ValueError(f"expected a .o kernel object: {binary}")
    for candidate in dict.fromkeys((binary, binary.resolve())):
        metadata = candidate.with_suffix(".json")
        derived = cann_json_path(candidate)
        if derived != metadata:
            raise ValueError(
                f"CANN 9.0 path collision: {candidate}; first '.o' produces {derived}, "
                f"expected {metadata}. Use an install path without '.o' before the object suffix.")
        derived_binary = cann_binary_path(metadata)
        if derived_binary != candidate:
            raise ValueError(
                f"CANN 9.0 path collision: {metadata}; first '.json' produces {derived_binary}, "
                f"expected {candidate}. Remove '.json' from the install directory path.")
    return binary.with_suffix(".json")


def check_root(root):
    root = Path(root).absolute()
    check_binary_path(root / "DFlashGroupQuantLinear_probe.o")
    return {"status": "PASS", "scope": "CANN 9.0 path prefix only; no device execution",
            "root": str(root), "resolved_root": str(root.resolve()), "npu_status": "NOT_RUN"}


def check_install(root):
    root = Path(root).absolute()
    check_root(root)
    directory = root / "vendors/customize/op_impl/ai_core/tbe/kernel/ascend310p" / OP_KERNEL
    binaries = sorted(directory.glob("*.o"))
    if not binaries:
        raise ValueError(f"no installed {OP_KERNEL} kernel objects under {directory}")
    records = []
    for binary in binaries:
        metadata = check_binary_path(binary)
        data = binary.read_bytes()
        if not data:
            raise ValueError(f"empty kernel object: {binary}")
        # Read the exact JSON path CANN would derive, without rewriting either
        # the generated metadata or the installed SDK.
        raw_json = metadata.read_bytes()
        info = json.loads(raw_json)
        if not isinstance(info, dict):
            raise ValueError(f"kernel JSON must be an object: {metadata}")
        records.append({"binary": str(binary), "json": str(metadata),
                        "binary_bytes": len(data), "json_bytes": len(raw_json),
                        "binary_sha256": hashlib.sha256(data).hexdigest(),
                        "json_sha256": hashlib.sha256(raw_json).hexdigest()})
    return {"status": "PASS", "scope": "installed object/JSON paths and JSON parsing only",
            "install_root": str(root), "kernels": records, "npu_status": "NOT_RUN"}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    target = cli.add_mutually_exclusive_group(required=True)
    target.add_argument("--root", type=Path, help="check a proposed build/install root before compilation")
    target.add_argument("--install-root", type=Path, help="check installed kernel object/JSON pairs")
    cli.add_argument("--report", type=Path)
    args = cli.parse_args()
    try:
        report = check_root(args.root) if args.root else check_install(args.install_root)
        print(f"PASS: OPP filesystem preflight ({args.root or args.install_root}); NPU NOT_RUN", flush=True)
    except (OSError, ValueError) as error:
        report = {"status": "FAIL", "scope": "OPP filesystem preflight", "npu_status": "NOT_RUN",
                  "error": f"{type(error).__name__}: {error}"}
        print(f"FAIL: {report['error']}", flush=True)
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
