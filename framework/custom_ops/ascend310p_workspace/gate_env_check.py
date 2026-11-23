"""Check the interpreter, SDK imports and actual GE protobuf compiler without touching the device."""

import importlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import subprocess


def check(prefix, cann, out):
    prefix = prefix.resolve(strict=True)
    cann = cann.resolve(strict=True)
    if (
        sys.version_info[:2] != (3, 11)
        or Path(sys.prefix).resolve() != prefix
        or Path(sys.base_prefix).resolve() != prefix
    ):
        raise ValueError("expected independent Python 3.11 Conda prefix: " + str(prefix))
    if not (prefix / "conda-meta/history").is_file():
        raise ValueError("no conda history at " + str(prefix))
    packages = {}
    for name in ("numpy", "scipy", "google.protobuf", "grpc_tools.protoc"):
        mod = importlib.import_module(name)
        path = Path(mod.__file__).resolve(strict=True)
        if prefix not in path.parents:
            raise ValueError(f"{name} escaped the new Conda environment: {path}")
        packages[name] = str(path)
    for name in ("msopgen", "tbe", "opc_tool"):
        spec = importlib.util.find_spec(name)
        if spec is None:
            raise ValueError(f"missing {name} in CANN Python path")
        paths = [Path(spec.origin)] if spec.origin else list(map(Path, spec.submodule_search_locations or []))
        if not paths or any(cann not in p.resolve(strict=True).parents for p in paths):
            raise ValueError(f"{name} did not come from {cann}: {paths}")
        packages[name] = str(paths[0])
    tools = {k: shutil.which(k) for k in ("python3", "atc", "msopgen", "cmake", "c++", "make")}
    if (
        any(v is None for v in tools.values())
        or Path(tools["python3"]).resolve() != Path(sys.executable).resolve()
    ):
        raise ValueError(f"missing build tool or wrong subprocess Python: {tools}")
    for name in ("atc", "msopgen"):
        if cann not in Path(tools[name]).resolve().parents:
            raise ValueError(f"{name} came from another CANN installation: {tools[name]}")
    schema = cann / "include/proto/ge_ir.proto"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "grpc_tools.protoc",
            "-I" + str(schema.parent),
            "--descriptor_set_out=" + str(out / "cann-ge.desc"),
            schema.name,
        ],
        cwd=schema.parent,
        check=True,
    )
    record = dict(python=sys.executable, prefix=str(prefix), cann=str(cann), packages=packages, tools=tools)
    (out / "conda-environment.json").write_text(json.dumps(record, indent=2) + "\n")
    print("CONDA READY", sys.executable, flush=True)


if __name__ == "__main__":
    check(*(Path(s) for s in sys.argv[1:]))
