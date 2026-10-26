# Exact RPATH replay acceptance for compile-only artifacts

`rpath_recheck.py` is an independent compile-artifact finalizer. It reuses a previously compiled CustomOp and its immutable report. It does not compile, install an OPP, load a Host library, or run a Runner/NPU. Existing compile/recheck SHA-identical rules remain available unchanged; this explicit finalization entry supplies a separate, narrow transformation proof. No historical failure report is rewritten.

The only accepted transformation is the generated Host install rule removing the chosen CANN root's `/lib64:` RPATH to an empty string. The validator requires:

- Original Raw ND K128 report identity; successful msopgen/build commands; original Kernel ELF/SHA verification unchanged.
- Explicit expected hashes for canonical library, package library and `build_out/op_host/cmake_install.cmake`.
- Exactly one relevant `file(INSTALL ... TYPE SHARED_LIBRARY FILES ...)` binding the canonical Host library to the actual package destination, plus exactly one relevant `file(RPATH_CHANGE FILE ... OLD_RPATH ... NEW_RPATH ...)`. Conflicting/duplicate rules, unresolved destinations and unsupported command forms fail. Comments and quoted command names cannot act as rules. Only exact literal destinations or the known CMake install-prefix/DESTDIR path spelling are accepted; paths must bind to this package tree.
- The required package `liboptiling.so` symlink resolves to the expected aarch64 library inside this CustomOp. Canonical ELF checks and the strict build-layout dangling-alias conditions remain mandatory. No other invalid Host artifact is excused.
- The CMake executable is the SHA-bound tool in the original preflight. The original install script is never executed. A new script contains only one RPATH_CHANGE, writing a new copy in the new output directory. OLD_RPATH/NEW_RPATH use non-interpolating bracket arguments. Original source/package/script hashes are verified again after replay.
- CMake exit code zero; replay **whole-file SHA256 and size exactly equal** the packaged library. ELF program-header mapping, NEEDED/SONAME and non-RPATH dynamic tags/values are unchanged; the old path is present in the source and absent/empty in the package. This does not mask PT_LOAD differences or trust Build ID alone.

The report records original report/preflight identities, Kernel identities, script SHA and line ranges, both source/package paths, resolved CMake prefix mapping, OLD/NEW_RPATH, source script snapshot, CMake identity/argv/returncode/log, replay SHA, and ELF comparison. A byte mismatch, replay error, timeout, source change, link escape or ambiguous rule fails.

Only success may return `COMPILE_COMPLETED_REVIEW_UNRESOLVED`. `source_review=UNRESOLVED`, `host_tiling_runtime=NOT_RUN`, `installation=NOT_RUN`, `npu=NOT_RUN`, `performance=NOT_RUN` remain. This report does not replace the source-review gate for the existing build/smoke entry.

Usage (Python stdlib; on the server with the original CMake binary):

```bash
python3 -B framework/custom_ops/draft_w8a8_fixpipe/rpath_recheck.py \
  --report /absolute/old/compile-report.json \
  --report-sha256 ORIGINAL_REPORT_SHA256 \
  --build-host-sha256 OBSERVED_CANONICAL_SHA256 \
  --package-host-sha256 OBSERVED_PACKAGE_SHA256 \
  --install-script-sha256 OBSERVED_HOST_CMAKE_INSTALL_SHA256 \
  --output-dir /absolute/new/acceptance-directory
```

Return new `compile-report.json` and, on replay failure, `replay/cmake.log`. Each invocation needs a new output directory. No SDK scanning or new Preflight is required when the original bound evidence remains intact.

Local CPU tests simulate CMake effects and failure conditions; they do not establish real CANN or NPU success. The actual isolated CMake replay must succeed on the server before artifact acceptance is reported.
