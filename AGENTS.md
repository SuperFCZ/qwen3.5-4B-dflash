# Development constraints

- Base custom-op work on `feature/gdr-chunk-verify`; develop on `feature/ascend310p-custom-ops`.
- Target hardware: Ascend 310P3. Target toolkit: CANN 9.0.0.
- This development machine has no NPU. Do not report local Ascend C compilation, numerical verification, or performance results as passed. Push source to GitHub; the 310P server pulls, compiles, and tests it.
- Keep experimental custom operators under `framework/custom_ops/`. Do not change the existing Qwen/DFlash model, export, or runtime path for a smoke test.
- Read `docs/OPTIMIZATION.md` before implementing production operators. Its interface, numerical, and end-to-end acceptance requirements apply when an operator is integrated.
- Keep generated `msopgen` projects and build products out of Git. Commit hand-written source and repeatable server commands.
