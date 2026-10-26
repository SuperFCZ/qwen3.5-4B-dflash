# Raw INT32 compile-only diagnostic

This independent entry is for an unresolved source review. It is **not** a way to authorize an OPP installation or NPU smoke. It never creates `review-raw.json`, marks claims SUPPORTED, or invokes the reviewed build/smoke stage.

It verifies the supplied original preflight SHA256, all original source and SDK snapshot identities, the selected compiler/tools and the raw two-input candidate. It records three fixed SDK include/module files separately, without changing the old preflight or scanning the SDK tree. Only `DFlashW8A8Accumulator`, ND M16/K128/N64, is generated. The standard generated `build.sh` compiles Host/kernel and may package the result; packages remain uninstalled. No runner is built or launched. A nonempty raw kernel object and Host tiling library must exist after a successful build.

Success is `COMPILE_COMPLETED_REVIEW_UNRESOLVED`. It establishes only that the requested candidate compiled in this environment. It does not validate integer results, Host GetTiling runtime results, capacity checks, completion/lifetime or device behavior. Those remain UNRESOLVED / NOT_RUN. Compile errors are recorded as errors, not automatically classified as unsupported hardware.

The original reviewed-build and NPU gates remain unchanged. This report is intentionally not accepted by either.

After pulling, run `compile_raw.sh --preflight <existing preflight.json> --preflight-sha256 <original SHA256> --output-dir <new independent directory>`. Use the same CANN environment that created the preflight. If a bound original source changed, stop and investigate; do not rewrite its hash. Unsupported `msopgen --version` is not a compile failure; actual `msopgen gen` is executed here.

Return `compile-report.json`, `msopgen.log`, `build.log`. New diagnostics stay outside the repository; private SDK excerpts are not committed.

Local CPU tests use simulated command/artifact fixtures; CANN compilation, installation and NPU remain NOT_RUN locally.

## Post-build artifact validation

Kernel candidates are named `DFlashW8A8Accumulator_*.o`; accepted originals must be directly under `CustomOp/build_out/op_kernel/binary/ascend310p/d_flash_w8_a8_accumulator/`. Host candidates use the exact names `liboptiling.so`, `libcust_opmaster_rt2.0.so`, or `libcust_optiling.so`, and must resolve inside `CustomOp/build_out/op_host/`. The isolated directory is newly created and its single raw-op source inventory is checked before build. Other directories, including CPack copies, cannot substitute for canonical outputs. CPack runtime aliases are checked separately as described below. Symlink escapes are rejected.

`artifact_checks.kernel` and `.host` independently record discovered/canonical/verified counts, candidate paths, checks and failure reasons. Nonempty canonical files get SHA256 and ELF validation: ELF64 little-endian, version/header/table bounds; Host must be AArch64 ET_DYN. Kernel machine/type/flags are recorded without assuming the AI Core uses the host ISA. This validates artifact format, not execution correctness or instruction semantics.

`commands.msopgen` and `commands.build` retain argv, exact return code (or timeout), and log identity. If build returns zero, `cann_compile=BUILD_RETURNED_ZERO` is saved before checking artifacts. A later validation failure produces `ARTIFACT_CHECK_FAILED_REVIEW_UNRESOLVED`, not a compiler-failure claim. A nonzero build command remains `COMPILE_FAILED_REVIEW_UNRESOLVED` / `FAILED_OR_INCOMPLETE`; consult the preserved log to distinguish compiler, linker or packaging errors. Neither state permits installation or NPU execution. Historical failed reports are never rewritten.

## Build-layout compatibility alias versus packaged runtime alias

CANN may place `op_host/liboptiling.so -> lib/linux/aarch64/libcust_opmaster_rt2.0.so` beside the real `op_host/libcust_opmaster_rt2.0.so`. That relative link is dangling in the build layout and valid only after packaging into `op_tiling/`. This specific dangling link is not accepted merely because another library exists.

The checker records the link target and dangling state. It may mark this build-layout alias WARNING only when the canonical `libcust_opmaster_rt2.0.so` has passed ELF/SHA checks, no verified canonical Host ELF's DT_NEEDED names `liboptiling.so`, and every discovered canonical CPack runtime alias is verified. It parses dynamic dependencies without loading or executing a library. Other dangling targets, cycles, permission/I/O errors and unproven alias roles remain failures.

Each CPack `Linux/External/custom_opp_*.run/packages/vendors/customize/op_impl/ai_core/tbe/op_tiling/liboptiling.so` is treated as a required package runtime alias. It must be a symlink with the exact known relative target, resolve within this CustomOp's package to an AArch64 ELF shared library, and have identical size/SHA256 to the corresponding verified canonical build library. Missing/broken/escaped/changed package aliases fail even when the canonical build library is valid. Missing package proof never permits the dangling build-alias warning. `artifact_checks.host_runtime` preserves this independent evidence; Kernel validation remains separate.

This classification applies to compile/package artifact validation only. It does not prove that an installed OPP loads or executes, and does not replace runtime linkage, Host Tiling or NPU tests. Those remain NOT_RUN. No old link, artifact or report is repaired or overwritten by this checker.
