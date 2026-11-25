"""Compute the same suite metrics in the existing CPU-only Draft frontend environment."""

from pathlib import Path
import sys
from winners import c
from measure import analyze

if __name__ == "__main__":
    f = Path(sys.argv[1]).resolve(strict=True)
    x = c.read(f)
    c.write(f.parent / "compared.json", analyze(f.parent, x["history"], x["plan"], x["capture"]))
