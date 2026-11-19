"""Run package assembly in a fresh process with only its future private OPP in the environment."""

from pathlib import Path
import sys
import traceback

from w16_common import c, require, sources
from w16_source import recovered
from w16_build import build_private


def main(request):
    q = c.read(request)
    folder = Path(q["dest"])
    report = dict(status="BLOCKED", variant=q["variant"], device_execution="NOT_RUN")
    try:
        require(q["sources"] == sources(), "private worker source drift")
        src, raw, _ = recovered(c.read(c.verify(q["source"]))["binding"])
        report["result"] = build_private(folder, src, raw, q["variant"])
        c.verify_tree(report["result"])
        report["status"] = "PRIVATE_W16_PACKAGE_BUILT"
    except Exception as exc:
        report["error"] = repr(exc)
        (folder.parent / "traceback.txt").write_text(traceback.format_exc())
    c.write(folder.parent / "worker.json", report)
    return int(report["status"] == "BLOCKED")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
