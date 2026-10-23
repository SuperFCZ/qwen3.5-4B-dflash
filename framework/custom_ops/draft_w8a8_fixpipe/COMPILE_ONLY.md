# Raw INT32 compile-only diagnostic

This independent entry is for an unresolved source review. It is **not** a way to authorize an OPP installation or NPU smoke. It never creates `review-raw.json`, marks claims SUPPORTED, or invokes the reviewed build/smoke stage.

It verifies the supplied original preflight SHA256, all original source and SDK snapshot identities, the selected compiler/tools and the raw two-input candidate. It records three fixed SDK include/module files separately, without changing the old preflight or scanning the SDK tree. Only `DFlashW8A8Accumulator`, ND M16/K128/N64, is generated. The standard generated `build.sh` compiles Host/kernel and may package the result; packages remain uninstalled. No runner is built or launched. A nonempty raw kernel object and Host tiling library must exist after a successful build.

Success is `COMPILE_COMPLETED_REVIEW_UNRESOLVED`. It establishes only that the requested candidate compiled in this environment. It does not validate integer results, Host GetTiling runtime results, capacity checks, completion/lifetime or device behavior. Those remain UNRESOLVED / NOT_RUN. Compile errors are recorded as errors, not automatically classified as unsupported hardware.

The original reviewed-build and NPU gates remain unchanged. This report is intentionally not accepted by either.

After pulling, run `compile_raw.sh --preflight <existing preflight.json> --preflight-sha256 <original SHA256> --output-dir <new independent directory>`. Use the same CANN environment that created the preflight. If a bound original source changed, stop and investigate; do not rewrite its hash. Unsupported `msopgen --version` is not a compile failure; actual `msopgen gen` is executed here.

Return `compile-report.json`, `msopgen.log`, `build.log`. New diagnostics stay outside the repository; private SDK excerpts are not committed.

Local CPU tests use simulated command/artifact fixtures; CANN compilation, installation and NPU remain NOT_RUN locally.
