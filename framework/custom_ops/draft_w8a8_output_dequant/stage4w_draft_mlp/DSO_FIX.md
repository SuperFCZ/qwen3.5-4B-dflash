# Native MLP DSO isolation fix (baseline00c4147)

The old `mapped_runtime` rejected every mapped path containing `/vendors/`.
The server has demonstrated that a native-only Torch-NPU process loads three
normal libraries from the selected CANN installation even when
`ASCEND_CUSTOM_OPP_PATH` is unset. This is a DSO admission bug, not a numerical
failure or evidence that the B-NZ private operator executed in that process.

Only these exact paths **relative to the selected, resolved CANN_ROOT** are
eligible:

```
opp/vendors/customize/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so
opp/vendors/customize/op_proto/lib/linux/aarch64/libcust_opsproto_rt2.0.so
opp/vendors/customize/op_api/lib/libcust_opapi.so
```

No directory, prefix or `libcust_*` basename wildcard is allowed. Before importing
Torch, the new binding records each exact alias, resolved file, size, SHA256,
library filename and symlink chain. Every symlink hop and final file must stay
inside the chosen CANN_ROOT. These are **new installation-specific observations**,
not fabricated historical SDK hash matches. Missing entries, root escape or an
unexplained identity change BLOCK. Benchmark must use the identical Smoke binding.

`/proc/self/maps` may report the approved alias or its recorded canonical target.
Both must resolve to that same bound file. Other vendor aliases remain rejected,
including an unapproved vendor alias pointing at an approved file. Reports retain
`alias`, `realpath`, `name`, `real_name`, `file` identity, symlink chain,
classification, and actual mapped device major/minor plus inode, checked against
the file being hashed. Virtual addresses are deliberately excluded from identity
comparison because ASLR changes them. Duplicate mappings of one DSO are deduplicated;
conflicting device/inode records or deleted/replaced files are rejected.

The existing verified historical normal B-NZ Build supplies its private OPP root,
installed API/Host and all private DSO identities, Host alias and Runner identity.
They remain separately recorded under `native_dso_binding.private_bnz`. Loading
those paths, aliases, or **byte-identical copies at another path** is forbidden,
even at an otherwise eligible CANN vendor pathname. Shared ordinary CANN libraries
used by both processes are not mislabeled as B-NZ-private dependencies.

Other Ascend installations and unknown vendor paths remain rejected with actual
alias/realpath diagnostics. Ordinary non-vendor system/Torch DSOs keep the prior
record-and-freeze policy; this is not a new system-wide library allowlist. Before
Benchmark, the loaded subset must already have identical identities in Smoke;
the final inventory must match exactly. Unknown or changed Benchmark DSOs block.
Binding/loaded-map checks run before Torch loading, after import, after each
case's untimed correctness checks, and at completion/integrity recheck. No such
hashing or map inspection occurs in warmup or timed callbacks.

No weight, activation, Scale, reference, numerical tolerance, mathematical call,
output comparison, timing loop or accepted compute Kernel is changed. Status names
also stay unchanged. `ASCEND_CUSTOM_OPP_PATH` must still be unset. The old
`real-draft-mlp-s0mP4v/mlp-smoke` remains BLOCKED; its four computed cases and
reported Down errors are preserved as historical diagnostics, not promoted to PASS.

## Source lock and required reruns

Stage4W hashes the whole stage source tree. Do **not** edit hashes or statuses in
old reports, copy old PASS fields into new reports, or relax the source check.

- The four underlying **A2 `capture/manifest.json` bundles** can be reused
  read-only after their real checkpoint, input, production-source and binary
  identities are rechecked. The A2 capture implementation is unchanged. No new
  NPU capture is needed solely for this fix.
- The old Stage4W capture wrapper reports retain their old Source hashes; they
  are not treated as new-wrapper PASS reports.
- Regenerate **Prepare → Source → Build revalidation → Host → Gate Smoke →
  native MLP Smoke** in fresh directories. Prepare recomputes references from the
  same verified A2 inputs; it does not edit old Golden/data. Gate rerun is required
  by the existing source-lock chain, despite the compute binary being unchanged.
- Any new Benchmark, preprocessing timing or summary must reference this new
  chain. The historical B-NZ/FP16 pairing, Kernel, OPP and original experiment
  reports stay read-only and are reverified, not rebuilt.

## Server commands

From the repository after pulling this fix, in the existing working Torch-NPU
environment (Python3.9+):

```bash
set -euo pipefail
S=framework/custom_ops/draft_w8a8_output_dequant/stage4w_draft_mlp
OLD=/home/w00949577/qwen35-dflash-run/real-draft-mlp-s0mP4v
export CANN_ROOT=/usr/local/Ascend/cann-9.0.0
export ASCEND_OPP_PATH="$CANN_ROOT/opp"
export AI_RUN_DIR=/home/w00949577/qwen35-dflash-run
unset ASCEND_CUSTOM_OPP_PATH
R=$(mktemp -d "$AI_RUN_DIR/real-draft-mlp-dsofix-XXXXXX")
PAIR=/home/w00949577/qwen35-dflash-run/w8a8-bnz-fp16-paired-20261110-120437/benchmark/report.json

# Discover the actual checkpoint/capture paths from the old successful Prepare.
# Reverify the selected input identities; do not accept its now-old Source lock.
python3 -B - "$S" "$OLD/prepare/prepare-report.json" "$R/prepare" <<'PY'
import subprocess, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import real_mlp_common as c
old = c.read(Path(sys.argv[2]))
if (old.get('status') != 'REAL_MLP_CPU_PREPARED_NPU_NOT_RUN'
        or old.get('error') or old.get('integrity_error')):
    raise ValueError('old Prepare is not a successful source of input identities')
draft = c.verify(old['checkpoint']['files']['model.safetensors']).parent
argv = [sys.executable, '-B', str(Path(sys.argv[1])/'prepare_real.py'),
        '--draft-dir', str(draft), '--output-dir', sys.argv[3]]
for case in old['cases']:
    argv += ['--capture', str(c.verify(case['capture']['manifest']))]
subprocess.run(argv, check=True)
PY

COMMON=(--prepare-report "$R/prepare/prepare-report.json" --paired-report "$PAIR" --cann-root "$CANN_ROOT" --device 0)
python3 -B "$S/run_real_gate.py" source "${COMMON[@]}" --output-dir "$R/source"
python3 -B "$S/run_real_gate.py" build "${COMMON[@]}" --source-report "$R/source/report.json" --output-dir "$R/build"
python3 -B "$S/run_real_gate.py" host "${COMMON[@]}" --source-report "$R/source/report.json" --build-report "$R/build/report.json" --output-dir "$R/host"
python3 -B "$S/run_real_gate.py" smoke "${COMMON[@]}" --source-report "$R/source/report.json" --build-report "$R/build/report.json" --host-report "$R/host/report.json" --output-dir "$R/gate-smoke"
env -u ASCEND_CUSTOM_OPP_PATH python3 -B "$S/run_mlp_reference.py" smoke \
  --prepare-report "$R/prepare/prepare-report.json" --gate-smoke-report "$R/gate-smoke/report.json" \
  --output-dir "$R/mlp-smoke" --device 0
```

Stop on any failed stage. Successful native Smoke must be
`REAL_MLP_BOUNDARIES_RECORDED_W8_DOWN_NOT_RUN`, with no error/integrity_error,
four cases and the new `native_dso_binding` plus `runtime_libraries` evidence.
Only then may the existing optional timing command run:

```bash
env -u ASCEND_CUSTOM_OPP_PATH python3 -B "$S/run_mlp_reference.py" benchmark \
  --prepare-report "$R/prepare/prepare-report.json" --gate-smoke-report "$R/gate-smoke/report.json" \
  --smoke-report "$R/mlp-smoke/report.json" --output-dir "$R/mlp-benchmark" --device 0
```

Its success status remains `REAL_NATIVE_MLP_TIMED_W8_DOWN_NOT_RUN`.
This patch's server Smoke/Benchmark are **NOT_RUN** until executed on310P3.
Full W8A8 Down, integrated W8A8 MLP and production/quality approval remain outside
this fix. If blocked, return the new `report.json`, `native.log` if present and
the error's actual pathname. Do not replace vendor files or copy private libraries
into CANN to satisfy the checks.

## Local regression

Run the Stage4W and unchanged paired-driver tests from the README. CPU maps
fixtures cover exact normal paths, internal canonical symlinks, unknown vendor
aliases, private paths/aliases/copies, vendor/private SHA drift, symlink escape
(including out-and-back), loaded inode/device mismatch, missing entries and
Smoke/Benchmark inventory drift. Fixtures are not real DSO loading or NPU evidence.
The historical `local-validation.json` is unchanged; fix results are recorded in
the independent `dso-fix-validation.json`.
