# Bound Raw review generator

`generate_raw_review.py` is the one-shot generator previously supplied outside the repository. It writes only the new review (or an UNRESOLVED diagnostic) under the explicitly selected server evidence root. It never builds, installs, runs a device, overwrites an existing review, or modifies a Preflight.

The expected Preflight SHA256 remains fixed:

```
312b1a0962ac6dce5a848243031a32f1898351b043faf60ba91aaabf54eab386
```

SDK paths, reviewed fingerprints and citation ranges are metadata. No SDK source text, snapshots or server JSON/logs are bundled here. Exact quotations are read on the server from the selected current Preflight snapshots only after live source/snapshot identities and audited content hashes match. All five static claims must validate through the unchanged `w8_sdk.reviewed()` before the final file is atomically published. Missing/mismatched evidence leaves no completed review and produces an UNRESOLVED diagnostic when run through the CLI.

The default repository root is derived from the script's location, not the current directory. `--repo` can select an explicit checkout, but the imported `w8_sdk.py` and `w8_common.py` must come from that checkout and match the current Preflight's source identities. Output records this generator's own identity separately.

## Existing Preflight remains authoritative

Creating a Preflight calls `sources()` to collect files. Reviewing it does not re-run that collection or compare the whole current directory set: the existing validator checks every file already listed in `preflight.sources`. This change only adds new files and leaves those previously bound files untouched. Therefore adding this generator and tests does not by itself require rebinding the existing Preflight. The new generator is not retrospectively claimed to be part of the old collection. If any listed source, SDK snapshot, live SDK source or tool identity has changed, validation still fails; do not rewrite hashes or automatically regenerate Preflight.

Run on the server, from any directory:

```bash
python3 -B /absolute/repository/framework/custom_ops/draft_w8a8_fixpipe/tools/generate_raw_review.py \
  --root /absolute/existing/w8a8-raw-reviewed-run
```

Inspect `<root>/review-raw.json` before any separate Build or Smoke command. The filename has a plain dot, not a literal backslash. `SOURCE_REVIEWED` covers the fixed static source contract; Host Tiling runtime, installation, NPU, Guard/readonly/repeat and numerical testing remain NOT_RUN.

CPU regressions use synthetic SDK fixture content and the real formal validator. They are not SDK compatibility or NPU evidence.
