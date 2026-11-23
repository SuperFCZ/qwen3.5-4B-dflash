"""Build a new HF overlay on the surviving ls_llm Python. Never install into Conda."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


WHEEL = "transformers-5.13.1-py3-none-any.whl"
WHEEL_SHA = "53f0ea8aa397e29244c2377ba981bcaf0c87adcf44fbdd447ef6306522afcacd"
HISTORICAL = {
    "torch_npu": "cd812b9e3b524a21ab37a1cd1c8542883b7b09fc65b21f75c89f2d87f28b0b77",
    "torchair": "385cc1c640b08d1dde135d9e32a28a3a6d803f2980586ffe8497e6e3139fe585",
}


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(venv, base):
    if Path(sys.prefix).resolve() != venv or Path(sys.base_prefix).resolve() != base:
        raise ValueError("unexpected Python prefix or base interpreter")
    import torch
    import torch_npu
    import transformers
    from transformers.initialization import copy_
    sys.path.insert(0, str(Path(torch_npu.__file__).parent / "dynamo"))
    import torchair
    if transformers.__version__ != "5.13.1" or not callable(copy_):
        raise ValueError("missing transformers 5.13.1 initialization.copy_")
    result = {}
    for package, prefix in ((transformers, venv), (torch, base), (torch_npu, base), (torchair, base)):
        path = Path(package.__file__).resolve(strict=True)
        if prefix not in path.parents:
            raise ValueError(f"{package.__name__} loaded from unexpected location: {path}")
        result[package.__name__] = dict(path=str(path), sha256=file_sha(path),
                                        version=str(getattr(package,"__version__","unknown")))
    result["historical_init_files_match"] = all(result[k]["sha256"] == v for k,v in HISTORICAL.items())
    result["python"] = dict(executable=sys.executable, version=sys.version, base=str(base))
    return result


def restore(run, out):
    if sys.version_info[:2] != (3,11) or sys.prefix != sys.base_prefix:
        raise ValueError("use the ls_llm base Python 3.11 (DFLASH_BASE_PYTHON)")
    base = Path(sys.prefix).resolve()
    venv = run / "venv-hf513-probe"
    if venv.is_symlink():
        raise ValueError("refusing to write through a venv directory symlink: " + str(venv))
    env = dict(os.environ, PYTHONNOUSERSITE="1", TORCH_DEVICE_BACKEND_AUTOLOAD="0",
               PIP_REQUIRE_VIRTUALENV="1", PIP_DISABLE_PIP_VERSION_CHECK="1",
               PIP_CACHE_DIR=str(run / "cache/pip"))
    for key in ("PYTHONHOME","PIP_USER","PIP_TARGET","PIP_PREFIX","PIP_ROOT"):
        env.pop(key,None)
    if not venv.exists():
        subprocess.run([sys.executable,"-m","venv","--system-site-packages",str(venv)],env=env,check=True)
    cfg = venv / "pyvenv.cfg"
    if not cfg.is_file() or "include-system-site-packages = true" not in cfg.read_text().lower():
        raise ValueError("existing directory is not the expected overlay; preserved: " + str(venv))
    py = str(venv / "bin/python")
    subprocess.run([py,"-I","-c",
        "from pathlib import Path;import sys;assert Path(sys.prefix).resolve()==Path(sys.argv[1]) and Path(sys.base_prefix).resolve()==Path(sys.argv[2])",
        str(venv),str(base)],check=True,env=env)

    pip = [py,"-m","pip"]
    offline = env.get("DFLASH_WHEELHOUSE")
    index = ["--no-index","--find-links",str(Path(offline).resolve(strict=True))] if offline else []
    cache = run / "cache/wheels"
    cache.mkdir(parents=True,exist_ok=True)
    wheel = cache / WHEEL
    if not wheel.exists():
        subprocess.run([*pip,"download","--no-deps","--only-binary=:all:",*index,"-d",str(cache),
                        "transformers==5.13.1"],env=env,check=True)
    if file_sha(wheel) != WHEEL_SHA:
        raise ValueError("wrong release wheel SHA256: " + str(wheel))
    # Force only the main package local, even if the base already exposes 5.13.1.
    subprocess.run([*pip,"install","--no-user","--no-deps","--force-reinstall",*index,str(wheel)],env=env,check=True)
    subprocess.run([*pip,"install","--no-user",*index,str(wheel)],env=env,check=True)
    subprocess.run([py,"-B",__file__,"--verify",str(venv),str(base),str(out)],env=env,check=True)
    with (out / "pip-freeze.txt").open("w") as fp:
        subprocess.run([*pip,"freeze","--all"],stdout=fp,env=env,check=True)


if __name__ == "__main__":
    if sys.argv[1] == "--verify":
        v, b, out = map(Path,sys.argv[2:])
        state = snapshot(v,b)
        (out / "python-environment.json").write_text(json.dumps(state,indent=2)+"\n")
        print("PYTHON READY",sys.executable,"transformers",state["transformers"]["version"],
              "torch",state["torch"]["version"],"torch_npu",state["torch_npu"]["version"])
        print("Historical torch_npu/TorchAir init hashes match:",state["historical_init_files_match"])
    else:
        run, out = map(Path,sys.argv[1:])
        with (run / ".restore-python.lock").open("a") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            restore(run,out)
