"""Recover the four original C16 Layer0 tensors by content, even if filenames changed."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile

# Frozen, independently reported identities in draft_gate_w16_atc8/w16_source.py.
GATE = {
    "x.bin": (81920, "44fed068f079c18e95ef303c0d02cf5401470db7277ec17d9d024fbf29079256"),
    "w_nz.bin": (49807360, "c898ef7b555f6e3bcf3b183ec7742b012f0a4e40ab4a46cd34184b77218b5a2c"),
    "s_gn.bin": (778240, "1365af20b6f09734c6d02f2f3d3045fb20e4af1f53d414c9b4dd9011c7ced333"),
    "golden.bin": (622592, "353839a7a41bdd44930e91f72ae750adc2f7f15e14f2b679d5f0799e1b5da76c"),
}
PILOT = "gate-scale-reviewfix-bXR45o/compare/report.json"
OLD = Path("/home/w00949577/qwen35-dflash-run")
SKIP = {".git", "__pycache__", "node_modules", ".cache", "site-packages", "venv-hf513-probe"}
CHECKPOINT_FILES = {"config.json", "model.safetensors.index.json"}
REPLAY_FILES = {"private.json", "shared.json"}


def digest(path):
    with path.open("rb") as stream:
        h = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
        return h.hexdigest()


def matches(path, wanted):
    return path.is_file() and path.stat().st_size == wanted[0] and digest(path) == wanted[1]


def publish(source, target, wanted):
    # Verify a private copy before publishing. Parallel invocations never overwrite each other.
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".restore-", delete=False) as tmp:
        stage = Path(tmp.name)
    try:
        shutil.copyfile(source, stage)
        if not matches(stage, wanted):
            raise ValueError("source changed during recovery: " + str(source))
        try:
            os.link(stage, target)
        except FileExistsError:
            if target.is_symlink() or not matches(target, wanted):
                raise ValueError("conflicting output; preserved: " + str(target))
    finally:
        stage.unlink()


def recover(roots, dest, expected=GATE):
    dest.mkdir(parents=True, exist_ok=True)
    found = {}
    for name, identity in expected.items():
        p = dest / name
        if p.is_symlink() or (p.exists() and not matches(p, identity)):
            raise ValueError("conflicting recovered file; preserved: " + str(p))
        if p.exists():
            found[name] = str(p)
    sizes = {size: name for name, (size, _) in expected.items()}
    seen = set()
    skipped = []
    checkpoints, replays, pilots = set(), set(), set()
    nk = []
    for root in roots:
        root = root.resolve()
        if not root.is_dir():
            skipped.append(str(root) + " (directory unavailable)")
            continue
        for folder, dirs, files in os.walk(root, onerror=lambda exc: skipped.append(str(exc))):
            parent = Path(folder)
            dirs[:] = sorted(n for n in dirs if n not in SKIP and (parent/n) != dest)
            if parent == dest:
                dirs[:] = []
                continue
            if CHECKPOINT_FILES.intersection(files) and any(n.endswith(".safetensors") for n in files):
                checkpoints.add(str(parent))
            replays.update(str(parent / n) for n in REPLAY_FILES.intersection(files))
            if "report.json" in files and (parent / "report.json").as_posix().endswith("/" + PILOT):
                pilots.add(str(parent / "report.json"))
            for filename in files:
                if len(found) == len(expected) or not filename.endswith((".bin", ".raw")):
                    continue
                file = parent / filename
                try:
                    st = file.stat()
                    key = (st.st_dev, st.st_ino)
                    name = sizes.get(st.st_size)
                    if not name or name in found or not stat.S_ISREG(st.st_mode) or key in seen:
                        continue
                    seen.add(key)
                    if matches(file, expected[name]):
                        publish(file, dest / name, expected[name])
                        found[name] = str(file)
                    elif filename == "q_nk.bin" and name == "w_nz.bin":
                        nk.append(file)
                except (FileNotFoundError, PermissionError) as exc:
                    skipped.append(str(exc))
    # Historic captures sometimes retained logical NK but not physical NZ.
    if "w_nz.bin" not in found and nk:
        import numpy as np
        for file in nk:
            nz = np.fromfile(file, dtype="i1").reshape(1216,16,80,32).transpose(2,0,1,3).tobytes()
            if hashlib.sha256(nz).hexdigest() == expected["w_nz.bin"][1]:
                with tempfile.NamedTemporaryFile(dir=dest) as fp:
                    fp.write(nz)
                    fp.flush()
                    publish(Path(fp.name), dest / "w_nz.bin", expected["w_nz.bin"])
                found["w_nz.bin"] = str(file) + " (lossless NK to NZ)"
                break
    return dict(search_roots=list(map(str, roots)), destination=str(dest), recovered=found,
                identities={n: dict(bytes=v[0],sha256=v[1]) for n,v in expected.items()},
                missing=sorted(set(expected)-set(found)), checkpoint_candidates=sorted(checkpoints),
                replay_candidates=sorted(replays), archived_pilot_candidates=sorted(pilots),
                unreadable_count=len(skipped), unreadable_examples=skipped[:12])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("search", type=Path, nargs="*")
    args = p.parse_args()
    run = args.run_root.resolve()
    roots = list(dict.fromkeys([run.parent, Path(__file__).resolve().parents[3], *args.search]))
    if OLD.is_dir():
        roots.append(OLD)
    report = recover(roots, run / "inputs/c16-layer0-gate")
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print("Frozen Gate tensors (size + SHA256):")
    for name in GATE:
        print(" ", name, report["recovered"].get(name, "MISSING"))
    for key in ("archived_pilot_candidates", "checkpoint_candidates", "replay_candidates"):
        print(key, len(report[key]), *report[key][:3])
    print("INVENTORY", args.report)
    if report["missing"]:
        print("Missing frozen tensors:", ", ".join(report["missing"]))
    if report["unreadable_count"]:
        print("Unreadable or dangling paths:", report["unreadable_count"], "(examples in inventory)")


if __name__ == "__main__":
    main()
