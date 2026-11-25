"""Read-only Target recovery inventory and native dispatcher inspection. No operator execution."""

from pathlib import Path
import hashlib, mmap, os, re, subprocess, zipfile
from history import ROOT, TARGET_SHA
from targets import HOME

PAT = re.compile(
    rb"(?:npu::)?(?:npu_chunk_gated_delta_rule|npu_cache_update_|adn_fused_infer_attention)\([^\x00\n]{0,1600}?\)\s*->[^\x00\n]{0,100}"
)


def snippets(data):
    return sorted({m.group().decode(errors="replace")[:660] for m in PAT.finditer(data)})


def inspect_file(p):
    if p.suffix == ".whl":
        with zipfile.ZipFile(p) as z:
            return [
                (k, snippets(z.read(k)))
                for k in z.namelist()
                if k.endswith(".so")
                and Path(k).name in ("libtorch_npu.so", "libop_plugin.so", "libtorch_npu_api.so")
            ]
    with p.open("rb") as fd, mmap.mmap(fd.fileno(), 0, access=mmap.ACCESS_READ) as mem:
        return snippets(mem)


def run():
    roots = [
        HOME,
        Path("/root/anaconda3/envs/torch_npu_py3.11"),
        Path("/root/anaconda3/pkgs"),
        Path("/root/.cache/pip/wheels"),
    ]
    from collections import defaultdict

    found = defaultdict(list)
    seen = set()
    for root in roots:
        for base, dirs, files in os.walk(root):
            d = Path(base)
            dirs[:] = [
                s
                for s in dirs
                if s
                not in {
                    ".git",
                    "node_modules",
                    "compiler-cache",
                    "ls_llm",
                    "cpro",
                    "gclconda_lerobot",
                    "__pycache__",
                }
                and not (s == "runs" and "qwen35-dflash-run" in d.parts)
            ]
            for name in files:
                p = d / name
                low = name.lower()
                if (
                    low.endswith(".om")
                    and any(t in low for t in ("prefill", "verify", "chunk"))
                    and "35b" not in str(p).lower()
                ):
                    found["om"].append(p)
                elif low.endswith(".air") and any(t in low for t in ("prefill", "verify")):
                    found["air"].append(p)
                elif (
                    low.startswith(("torch_npu", "torch-npu", "op_plugin")) and low.endswith(".whl")
                ) or low.startswith(("libtorch_npu.so", "libop_plugin.so")):
                    found["provider"].append(p)
                elif low.endswith((".tar", ".tgz", ".tar.gz", ".tar.zst", ".tar.xz", ".zip")) and any(
                    t in low for t in ("dflash", "w8-nz", "w8nz", "gdr", "torch_npu")
                ):
                    found["archive"].append(p)
    for tag in ("om", "air", "archive", "provider"):
        for p in sorted(found[tag]):
            try:
                real = p.resolve(strict=True)
                key = (tag, str(real))
                if key in seen:
                    continue
                seen.add(key)
                print(tag.upper(), real, p.stat().st_size, flush=True)
                if tag == "om" and p.stat().st_size > 1_000_000_000:
                    with p.open("rb") as fd:
                        sha = hashlib.file_digest(fd, "sha256").hexdigest()
                    print(
                        "  SHA", sha, "historical", [n for n, v in TARGET_SHA.items() if v == sha], flush=True
                    )
                if tag == "provider":
                    print("  NATIVE SCHEMAS", inspect_file(p), flush=True)
            except (OSError, ValueError, zipfile.BadZipFile) as e:
                print("  unavailable", e)
    print("Target vendor siblings", sorted(p.name for p in (HOME / "custom_ops_llm/vendors").glob("*")))
    py = HOME / "z50058744/qwen35-dflash-run/envs/target-export310p/bin/python"
    if py.is_file():
        src = """import importlib.util
from pathlib import Path
import torch
root=Path(importlib.util.find_spec('torch_npu').origin).parent
print('Target package',root,'torch',torch.__file__,torch.__version__)
print('Native libraries', [p.name for p in sorted((root/'lib').glob('*.so*'))])
p=root/'lib/libtorch_npu.so'
if p.is_file():
 try:torch.ops.load_library(str(p))
 except Exception as e:print('Native import',repr(e))
for n in ('npu_cache_update_','npu_chunk_gated_delta_rule','adn_fused_infer_attention','adn_rms_norm'):
 try:print(n,torch._C._dispatch_find_schema_or_throw('npu::'+n,'').schema())
 except RuntimeError:print(n,'absent')
"""
        subprocess.run([str(py), "-B", "-c", src], env=dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0"))
    try:
        from history import read_history, SUITE

        h = read_history(SUITE)
        print("Historical 20-prompt batch", h["index"]["sha256"], h["batch"]["sha256"])
    except (OSError, ValueError, KeyError) as e:
        print("Historical case admission", repr(e))


if __name__ == "__main__":
    run()
