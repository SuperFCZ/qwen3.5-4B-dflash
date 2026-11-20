"""Read-only continuation of the three authenticated ABnQDE stages."""

import hashlib
from pathlib import Path
import subprocess

from scale_contracts import HERE, c, historical, require

ORIGINAL = "6ddb9cbe1353b08707f7e85f0beb27f84b655291"
FAILED_BYTES = 49175
FAILED_SHA = "a5395a1786568096ae734d2a7c4978a2da8878dc810600cd61dba372921a0c32"
PREFIX = {
    "source": "12eed4be4b38a3000b1120f9f02adbdcdfb39d0de90d9c1c47016105324627bd",
    "build": "dabedf101c4d01372b7e17a3d45cb7b2661b0b21bb31be37b33a1a30c0d5b8ae",
    "atc": "076b6fe95512e8586eae797497916872d99962c082c84d5b3cdcfb21f8140c5b",
}
# Only these revised source bytes may be used under this specific continuation.
# Build, kernel, Host, ATC, ABI, layout, Runner and SDK probe source all compare live.
REVISED = {
    "scale_review.py": "90ca7fc40c94c789a5d16608644b41603d7f498319a37b729b92665298801f89",
    "run_scale.py": "8283f777bd7236af6beeb6da0cbe4a03c772e970de51436ceeac00dd7ce44460",
    "README.md": "d1b0202d2e1a5715757d7ee198db40f8d600beab4627105f4d89a443d1f36d66",
}
ADDED = {"scale_reuse.py", "test/test_review_fix.py"}


def original_bytes(path):
    rel = path.relative_to(c.REPO).as_posix()
    return subprocess.check_output(["git", "-C", str(c.REPO), "cat-file", "blob", ORIGINAL + ":" + rel])


def verify_transition(old_lock, current):
    reconstructed = []
    previous = {}
    revised = set()
    added = set()
    for item in current:
        path = Path(item["path"])
        rel = path.relative_to(HERE).as_posix() if HERE in path.parents else None
        if rel in ADDED:
            c.verify(item)
            added.add(rel)
            continue
        if rel in REVISED:
            require(c.verify(item) == path and item["sha256"] == REVISED[rel], "unreviewed revision: " + rel)
            blob = original_bytes(path)
            digest = hashlib.sha256(blob).hexdigest()
            reconstructed.append(dict(path=str(path), bytes=len(blob), sha256=digest))
            previous[str(path)] = (len(blob), digest)
            revised.add(rel)
        else:
            reconstructed.append(item)
    require(
        revised == REVISED.keys() and added == ADDED and reconstructed == old_lock,
        "archived source lock does not reproduce from Git and live unchanged inputs",
    )
    # verify_tree hashes every other live identity and rejects any conflicting record.
    # Only the three legitimately superseded paths use their verified ORIGINAL Git bytes.
    historical.verify_tree(old_lock, dict(previous))
    return previous


def open_prefix(old, archive, base):
    path = old / "review/report.json"
    failed_id = c.identity(path)
    require(
        failed_id == dict(path=str(path), bytes=FAILED_BYTES, sha256=FAILED_SHA),
        "different archived review failure",
    )
    failed = c.read(c.verify(failed_id))
    archived_base = dict(base, root=str(old), sources=failed["sources"])
    require(
        all(failed[k] == v for k, v in archived_base.items())
        and failed["archive"] == str(archive)
        and failed["phase"] == "review"
        and failed["folder"] == str(path.parent)
        and failed["status"] == "BLOCKED"
        and failed["error"] == "KeyError('tools')"
        and failed["integrity_error"] is None
        and all(failed[k] == "NOT_RUN" for k in ("atc", "device_execution", "benchmark")),
        "archived review has another scope, error or execution state",
    )
    previous = verify_transition(failed["sources"], base["sources"])
    refs = {}
    for stage, digest in PREFIX.items():
        original = old / stage / "report.json"
        item = c.identity(original)
        require(item["path"] == str(original) and item["sha256"] == digest, "changed archived " + stage)
        refs[stage] = item
    require(failed["predecessors"] == refs, "blocked review refers to different predecessors")
    historical.verify_tree(failed, dict(previous))
    receipt = dict(
        original_commit=ORIGINAL,
        archived_root=str(old),
        failed_review=failed_id,
        reports=refs,
        superseded_sources={
            str(Path(k).relative_to(HERE)): dict(
                original_sha256=v[1], repaired_sha256=REVISED[str(Path(k).relative_to(HERE))]
            )
            for k, v in previous.items()
        },
    )
    return archived_base, previous, receipt
