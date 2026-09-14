"""Download and verify one selected published quantized Draft, without remote code."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import urllib.request

from .draft_quantization import QUANTIZED_DRAFTS, require_draft_checkpoint


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-quantization", required=True, choices=tuple(QUANTIZED_DRAFTS))
    parser.add_argument("--output", required=True, help="new external checkpoint directory below AI_RUN_DIR")
    args = parser.parse_args(argv)
    root = Path(args.output).expanduser().resolve()
    if "AI_RUN_DIR" in os.environ:
        root.relative_to(Path(os.environ["AI_RUN_DIR"]).resolve())
    repository = Path(__file__).resolve().parents[2]
    if root == repository or repository in root.parents:
        raise ValueError("checkpoint output must be outside the source repository")
    root.mkdir(parents=True, exist_ok=False)
    lock = QUANTIZED_DRAFTS[args.draft_quantization]
    # Download only inert checkpoint data. No trust_remote_code or version-dependent
    # quantizer installation; W4 keeps the published GPTQ calibration result.
    for name in ("config.json", "model.safetensors", "README.md"):
        url = f'https://huggingface.co/{lock["repository"]}/resolve/{lock["revision"]}/{name}'
        temporary = root / (name + ".part")
        with urllib.request.urlopen(url, timeout=60) as source, temporary.open("xb") as target:
            while chunk := source.read(8 * 1024 * 1024):
                target.write(chunk)
        temporary.rename(root / name)
    audit = require_draft_checkpoint(root, args.draft_quantization)
    (root / "draft-checkpoint-manifest.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
