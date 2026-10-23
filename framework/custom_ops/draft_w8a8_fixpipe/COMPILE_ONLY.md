# Raw INT32 compile-only diagnostic

This independent entry is for an unresolved source review. It is **not** a way to authorize an OPP installation or NPU smoke. It never creates `review-raw.json`, marks claims SUPPORTED, or invokes the reviewed build/smoke stage.

It verifies the supplied original preflight SHA256, all original source and SDK snapshot identities, the selected compiler/tools and the raw two-input candidate. It records three fixed SDK include/module files separately, without changing the old preflight or scanning the SDK tree. Only `DFlashW8A8Accumulator`, ND M16/K128/N64, is generated. The standard generated `build.sh` compiles Host/kernel and may package the result; packages remain uninstalled. No runner is built or launched. A nonempty raw kernel object and Host tiling library must exist after a successful build.

Success is `COMPILE_COMPLETED_REVIEW_UNRESOLVED`. It establishes only that the requested candidate compiled in this environment. It does not validate integer results, Host GetTiling runtime results, capacity checks, completion/lifetime or device behavior. Those remain UNRESOLVED / NOT_RUN. Compile errors are recorded as errors, not automatically classified as unsupported hardware.

The original reviewed-build and NPU gates remain unchanged. This report is intentionally not accepted by either.

After pulling, run `compile_raw.sh --preflight <existing preflight.json> --preflight-sha256 <original SHA256> --output-dir <new independent directory>`. Use the same CANN environment that created the preflight. If a bound original source changed, stop and investigate; do not rewrite its hash. Unsupported `msopgen --version` is not a compile failure; actual `msopgen gen` is executed here.

Return `compile-report.json`, `msopgen.log`, `build.log`. New diagnostics stay outside the repository; private SDK excerpts are not committed.

Local CPU tests use simulated command/artifact fixtures; CANN compilation, installation and NPU remain NOT_RUN locally.

## Post-build artifact validation

Kernel candidates are named `DFlashW8A8Accumulator_*.o`; accepted originals must be directly under `CustomOp/build_out/op_kernel/binary/ascend310p/d_flash_w8_a8_accumulator/`. Host candidates use the exact names `liboptiling.so`, `libcust_opmaster_rt2.0.so`, or `libcust_optiling.so`, and must resolve inside `CustomOp/build_out/op_host/`. The isolated directory is newly created and its single raw-op source inventory is checked before build. Other directories, including CPack copies, are reported as excluded and cannot substitute for canonical outputs. Symlink escapes are rejected.

`artifact_checks.kernel` and `.host` independently record discovered/canonical/verified counts, candidate paths, checks and failure reasons. Nonempty canonical files get SHA256 and ELF validation: ELF64 little-endian, version/header/table bounds; Host must be AArch64 ET_DYN. Kernel machine/type/flags are recorded without assuming the AI Core uses the host ISA. This validates artifact format, not execution correctness or instruction semantics.

`commands.msopgen` and `commands.build` retain argv, exact return code (or timeout), and log identity. If build returns zero, `cann_compile=BUILD_RETURNED_ZERO` is saved before checking artifacts. A later validation failure produces `ARTIFACT_CHECK_FAILED_REVIEW_UNRESOLVED`, not a compiler-failure claim. A nonzero build command remains `COMPILE_FAILED_REVIEW_UNRESOLVED` / `FAILED_OR_INCOMPLETE`; consult the preserved log to distinguish compiler, linker or packaging errors. Neither state permits installation or NPU execution. Historical failed reports are never rewritten.
