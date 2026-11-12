# Native DSO mount identity fix (baseline e7035bf)

Scope: only `real_mlp_dso_identity.py`, CPU tests and documentation. No change to
Kernel, weight, Golden, checkpoint, W8A16/FP16 mathematical calls, output checks,
warmup/timing loops or Source Hash policy. Three precise CANN vendor paths and
all B-NZ private path/alias/content prohibitions remain unchanged.

## Why the old device comparison is not sufficient

The server reports `/` as overlay, device0:69; `libtvm.so` pathname stat is0:69,
but maps reports8:17. Both inode values are49033281. Its map_files link is readable
and names the same original path. Those observations support an overlay-view
explanation, but **do not alone prove the identity of every discrepant DSO**.

Linux documents that an overlay object can report overlay or underlying `st_dev`,
depending on configuration. Its inode behavior also depends on xino. See
[OverlayFS object identity](https://www.kernel.org/doc/html/v6.1/filesystems/overlayfs.html#overlay-objects).
The inspected upstream reference, not a claim about the server's exact kernel
build, is Linuxv6.1: `ovl_open_realfile` uses a real inode with the original path
(file.c59–60), and `ovl_mmap` switches the VMA to the real file (462–478).
See [overlayfs/file.c](https://github.com/torvalds/linux/blob/v6.1/fs/overlayfs/file.c#L462).
Meanwhile maps obtains device/inode from `file_inode(vma->vm_file)` (269–272),
then prints the file path (282–284). See
[proc/task_mmu.c](https://github.com/torvalds/linux/blob/v6.1/fs/proc/task_mmu.c#L268).
Thus the displayed pathname and the VMA backing device need not be the same view.

`map_files` exposes per-VMA links, with capability/ptrace restrictions; a readable
link does not guarantee that opening its target is permitted. See
[map_files](https://man7.org/linux/man-pages/man5/proc_pid_map_files.5.html).
Even an opened link can expose the pathname view; the fix checks its fstat result
instead of assuming it is the backing inode. The current namespace's mountinfo
provides the filesystem type, root, mountpoint, options and device. See
[mountinfo](https://man7.org/linux/man-pages/man5/proc_pid_mountinfo.5.html).
An ordinary bind mount or namespace label by itself does not justify changing a
device identity. No device numbers8:17 or0:69 are hardcoded in the admission rule.

## Admission contract

Every file retains the original vendor/private/other-installation rejection,
realpath and symlink chain, complete pathname SHA256 and exact inode checks.
Deleted mappings are rejected explicitly. For matching maps/stat device+inode,
the direct identity route remains available.

For **different devices but the same inode**, all original segments must have
readable map_files links resolving to the same bound realpath, without deleted
targets. Then:

1. Try read-only open of the actual map_files entry. Record permission errno,
   fstat identity and full-file SHA256 if readable. If its fstat equals the
   original VMA identity and its bytes match the bound file, use
   `MAP_FILES_FD_CONTENT_MATCH`.
2. If open is EPERM/EACCES, record `UNAVAILABLE`; never claim its contents were
   read. If open succeeds but fstat is only the pathname identity, record
   `PATH_VIEW_ONLY_NOT_BACKING_IDENTITY`. Do not treat that as backing-file proof.
3. In these two cases the only additional route is a measured **per-file overlay
   conversion**: the covering mount must be unambiguously overlay, with mount
   device equal to pathname stat. Open the bound pathname read-only, check fstat,
   create a temporary whole-file `MAP_PRIVATE | PROT_READ` mapping (NO_EXEC,
   NO_WRITE), and identify its unique new VMA. Its device **and inode** must equal
   the original library's VMA. Hash all bytes through the temporary mapping and
   require the original bound size/SHA256. This proves the current file maps to
   the same backing object ID; it is not merely an inode/path coincidence.
4. Close the temporary mapping and FD in all paths. Recheck pathname/fd metadata,
   original VMA records and namespace/mount table stability. Unknown translation,
   failed/ambiguous probe, different backing inode, unreadable link, byte mismatch
   or unexplained fstat remains BLOCKED with structured diagnostics in the error.

`OVERLAY_TRANSLATION_PROVEN_BY_READONLY_MMAP` records the permission limitation,
both identities, target paths, selected mount, and the actual probe's mapped
identity/content hash. It does **not** claim direct map_files content access when
denied, hash relocated live code pages, or prove arbitrary xino inode translations.
Inode differences remain BLOCKED. Ambiguous stacked mounts are also rejected.

The temporary mapping is a CPU filesystem identity test, not dlopen, executable
code, a device/NPU mapping, or a performance measurement. It neither changes nor
executes DSO contents. Files must remain frozen during the experiment.

The namespace identifier, kernel release and complete mountinfo SHA256 are frozen
in the DSO binding before Torch import and checked on every later scan, including
within Smoke. These and the relevant covering mount record are also stored in
each library's evidence. ASLR addresses are
used transiently to identify map_files/probe ranges and never become the frozen
cross-process identity. Smoke/Benchmark compare the complete library/mount
contract. The mount table is checked again at scan completion. There are no new
scans inside timed calls: all work remains at existing DSO admission points.

## Local evidence and limits

In the existing ARM64 Docker, actual `map_files` readlink succeeded and read-only
open returned EPERM for tested loaded Python/system DSOs. Its tested maps/stat
devices were equal, so this is not a reproduction of the server's8:17/0:69 pair.
The real Linux read-only mmap test verifies discovery, full-byte SHA256 and
cleanup without NPU. CPU fixtures separately exercise the discrepant-device
routes and fail-closed cases. Actual server translated identities and MLP Smoke
remain NOT_RUN_AFTER_FIX until the server run.

## Optional small server check before regenerating the full chain

This imports native Torch-NPU to obtain its loaded `libtvm.so` mapping, but calls
no device selection, tensor operation, ACLNN or NPU execution. It records only
the mount identity proof, not any Stage4W numerical or Host PASS. If libtvm is not
mapped at import, it reports BLOCKED rather than loading a guessed library.

```bash
set -euo pipefail
S=framework/custom_ops/draft_w8a8_output_dequant/stage4w_draft_mlp
export CANN_ROOT=/usr/local/Ascend/cann-9.0.0
export ASCEND_OPP_PATH="$CANN_ROOT/opp"
unset ASCEND_CUSTOM_OPP_PATH
P=$(mktemp -d /home/w00949577/qwen35-dflash-run/dso-mount-probe-XXXXXX)
python3 -B - "$S" "$P/probe.json" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import real_mlp_dso_identity as d
import torch, torch_npu
proc=Path('/proc/self'); result={'status':'BLOCKED','npu':'NOT_RUN'}
try:
    target=Path('/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64/libtvm.so').resolve(strict=True)
    rows=[r for r in d.map_rows((proc/'maps').read_text()).values() if r['path']==str(target)]
    if not rows:raise ValueError('libtvm not mapped; do not guess/load another library')
    if any(r['loaded']!=rows[0]['loaded'] for r in rows):raise ValueError('conflicting mappings')
    environment,mounts=d.mount_snapshot(proc)
    file=d.c.identity(target); loaded=rows[0]['loaded']; st=target.stat()
    if loaded==d.stat_id(st):proof={'status':'DIRECT_DEVICE_INODE_MATCH'}
    else:proof=d.mismatch_proof(target,file,loaded,rows,proc,d.covering_mount(target,mounts))
    if d.mount_snapshot(proc)[0]!=environment:raise ValueError('mount contract changed')
    result.update(status='READ_ONLY_MAPPING_PROOF_OBSERVED',file=file,environment=environment,proof=proof)
except Exception as e:result['error']=str(e)
with open(sys.argv[2],'x') as f:json.dump(result,f,indent=2)
print(result['status'],sys.argv[2])
raise SystemExit(result['status']=='BLOCKED')
PY
```

## Full server rerun and Source Hash

Keep both old directories read-only:
`real-draft-mlp-s0mP4v` and `real-draft-mlp-dsofix-1mVAM1`.
Underlying four A2 bundles can be reused after current identity revalidation.
Whole-stage Source hashes change, so regenerate Prepare, Source, Build identity
revalidation, Host, Gate Smoke and MLP Smoke. No old report is upgraded or edited.

Use the complete commands in [DSO_FIX.md](DSO_FIX.md#server-commands), changing
only the old-input source and fresh output-directory prefix:

```bash
OLD=/home/w00949577/qwen35-dflash-run/real-draft-mlp-dsofix-1mVAM1
R=$(mktemp -d /home/w00949577/qwen35-dflash-run/real-draft-mlp-mountfix-XXXXXX)
```

Run its Prepare-discovery Python block and Source→Build→Host→Gate Smoke→MLP
Smoke commands with these values. The private OPP is never rebuilt or modified.
Do not use old Prepare/Gate reports directly with the new code. New MLP success
still requires `REAL_MLP_BOUNDARIES_RECORDED_W8_DOWN_NOT_RUN`, no error or integrity
error, all numerical/safety checks, plus the per-DSO mount proof. Benchmark remains
separately gated on this new Smoke and frozen mount/library identities.

If blocked, return the new `probe.json` if run, MLP `report.json`, and `native.log`
if present. The error records errno, path, mapped/stat identities, mount record,
link/open status and failed witness where available. Do not grant broad container
privileges, modify SDK libraries, edit old report hashes, or treat readlink-only
evidence as a content check to force PASS. Unknown cases stay BLOCKED.
