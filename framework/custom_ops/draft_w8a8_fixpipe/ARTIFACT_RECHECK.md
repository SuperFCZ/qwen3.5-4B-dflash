# Read-only Host packaging investigation

`artifact_recheck.py` reuses an existing Raw-only compile report and CustomOp directory. It does not call subprocesses, compile, install, load a shared library, invoke CMake/strip, or launch a runner. Original reports and build/package files are only read. Every run requires a new external output directory.

Inputs are the original compile report plus its SHA256, and independently supplied SHA256 values for the canonical and packaged Host libraries. The report must record successful msopgen/build commands, no installation/NPU execution, and a verified Kernel. All previously recorded identities are rechecked, and the new Kernel result must equal the original identity list. Existing strict artifact rules are unchanged.

The new report records full program-header metadata and per-segment hashes, exact PT_LOAD equality, dynamic entries, NEEDED/SONAME and RPATH/RUNPATH strings with file offsets. Every file byte is compared; at most 128 difference ranges are displayed, with total counts and truncation flags. `rpath_string_only_candidate` means only that changed file bytes fall within original path-string slots. It is **not** an acceptance rule. Same Build ID or dependency names cannot justify changed load content.

Only generated build/install evidence under this CustomOp is collected: cmake_install.cmake, CPackConfig.cmake, install manifests, Host link commands, CMakeCache and project/cmake files. Collection has file-count/byte limits and escape checks. Source files and new snapshots have SHA256. Reports include numbered excerpts around install, RPATH and strip commands, with complete snapshots available when excerpts are truncated. Script presence alone does not prove it ran. No SDK tree is scanned.

Outputs:

- `artifact-recheck.json`: preserved command/Kernel evidence, strict artifact failures, ELF byte/segment differences and identities.
- `packaging-evidence.json` and `packaging-snapshots/`: actual generated install/strip rules for human review.
- `artifact-recheck.md`: bounded summary.

`ARTIFACT_RECHECK_UNRESOLVED` (exit 2) is expected while Host hashes differ. No SOURCE_REVIEWED is created. `BLOCKED_ARTIFACT_RECHECK` means missing, inconsistent, changed or unparseable evidence. Existing strict validation can produce `ARTIFACT_RECHECK_VERIFIED_REVIEW_UNRESOLVED` only if it independently passes; the original report remains untouched.

## Conditions for a future narrow conversion rule

No conversion exception is implemented yet. Before accepting any difference:

1. Tie the actual generated install rule to this exact source library and package destination, including conditions, old/new RPATH and any strip command/tool identity; inspect original build/install logs. Merely finding the word strip or matching Build IDs is insufficient.
2. Reproduce only the proven transformation on isolated copies or in memory and require the resulting **whole-file SHA256** to equal the packaged library. Never execute the old install script against its real destination.
3. For an RPATH change, validate the intended loader-search change explicitly. Preserve program-header mapping, code/data, symbols, relocations, dependencies and other dynamic content outside precisely documented modifications. An RPATH-only classification alone is not safe acceptance.
4. A strip claim must account for actual removed/changed metadata and its effects on file offsets/headers. Unexpected PT_LOAD changes remain FAILED. If runtime code/data or dependencies change without authorization, fix the packaging step rather than masking the difference.

Run (from the repository, Python stdlib only):

```bash
python3 -B framework/custom_ops/draft_w8a8_fixpipe/artifact_recheck.py \
  --report /absolute/old/compile-report.json \
  --report-sha256 ORIGINAL_REPORT_SHA256 \
  --build-host-sha256 OBSERVED_BUILD_HOST_SHA256 \
  --package-host-sha256 OBSERVED_PACKAGE_HOST_SHA256 \
  --output-dir /absolute/new/recheck-directory
```

Send the two JSON outputs. They identify actual transformation commands and byte differences without another compile. Source review, installation, NPU and performance are not promoted by this investigation.
