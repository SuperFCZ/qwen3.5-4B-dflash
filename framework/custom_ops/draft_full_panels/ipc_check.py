"""Exercise the actual CANN Python IPC path before loading the full ATC graph."""

import multiprocessing
import os
from pathlib import Path
import sys
import tempfile


def check():
    temp = Path(os.environ["TMPDIR"]).resolve(strict=True)
    if Path(tempfile.gettempdir()).resolve() != temp:
        raise ValueError(f"compiler Python did not select TMPDIR {temp}")
    candidate = temp / "pymp-XXXXXXXX" / "listener-XXXXXXXX"
    length = len(os.fsencode(candidate))
    if length > 107:
        raise ValueError(f"AF_UNIX path too long ({length} bytes, Linux maximum 107): {candidate}")
    with multiprocessing.Manager() as manager:
        address = manager.address
        if not isinstance(address, str) or Path(address).parent.parent.resolve() != temp:
            raise ValueError(f"SyncManager used an unexpected endpoint {address}")
        if manager.dict({"probe": 1}).get("probe") != 1:
            raise ValueError("SyncManager IPC round trip failed")
        print(
            f"Python {sys.version.split()[0]} SyncManager round trip PASS, address bytes={len(os.fsencode(address))}",
            flush=True,
        )


if __name__ == "__main__":
    check()
