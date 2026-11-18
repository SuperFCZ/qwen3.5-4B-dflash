# A-only Full Draft msprof attribution (310P3 / CANN 9)

Only the accepted **A** OM for C16 and C64. No export/build, B replay, target, decode or adoption decision. The formal comparison found identical ATC blockDims. Current uninstrumented A references are approximately 31.2262 / 32.0923 ms median for C16/C64; admission reads the exact 90 fresh samples per gear. No device profile is shipped or claimed.

## Run manually on the same server, device, environment and mount namespace

```bash
cd /home/w00949577/z50058744/qwen3.5-4B-dflash
git pull --ff-only origin feature/ascend310p-custom-ops
PY=/home/w00949577/qwen35-dflash-run/venv-hf513-probe/bin/python
S=$PWD/framework/custom_ops/draft_a_core_scheduler_profile
R=$(mktemp -d /home/w00949577/qwen35-dflash-run/a-only-draft-profile-XXXXXX)
echo "R=$R"
prof_stage() { TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$PY" -B "$S/run_profile.py" "$1" --root "$R"; }
prof_stage admit
```

| Command (one at a time) | Success status | Effect |
|---|---|---|
| `prof_stage admit` | A_OM_PROFILE_ADMITTED | Rehash/verify frozen evidence; zero ACL calls. |
| `prof_stage profile-c16` | A_C16_RUNNER_CAPTURE_VERIFIED | One instrumented A/C16 process, original 38 calls. |
| `prof_stage profile-c64` | A_C64_RUNNER_CAPTURE_VERIFIED | Independent A/C64 capture, original 38 calls. |
| `prof_stage analyze` | A_OM_PROFILE_ANALYZED | Offline, requires both verified captures. |

Stop on nonzero exit. Each stage writes a fresh directory and `profile-report.json`; failure retains logs, raw products and traceback. Do not edit or overwrite an old run. Defaults: OLD `a-core-scheduler-flow-F6ia0q`, RESUME `a-scheduler-om-resume-0foBgF` beneath the server run parent. Admission validates the five OLD predecessor reports and their inventories, the blocked pre-maps evidence, all three successful RESUME reports and their source/hash links. Only the A variants' actual OM, executable, ldd deps, private OPP, sealed env, input and Golden identities, DSO manifest plus the 22 exact extension pins enter the capture. This can require substantial read-only hashing. Run with ample space for 2 OM graph dumps and raw profiling directories.

## Launch contract

ATC mode **1** reads the frozen OM and writes JSON into the new stage; no OM compilation. Local CANN's msprof CLI is at `tools/profiler/bin/msprof`; actual tool identity and server help are checked before capture. CLI application targets an isolated Python exec shim, not a script launching a separate unmanaged inference process. The shim verifies that the inherited FD still denotes the original Unix socket, redirects application stdout/stderr to the Runner's required `runner.log`, and execs the bound executable without forking. The coordinator enables Linux SO_PASSCRED on its original socket endpoint. On the frozen Runner's '?' it requires SCM_CREDENTIALS from the shim PID after exec, rechecks /proc/PID/{exe,cmdline,fd,environ,maps}, matches the pre-maps snapshot, and uses the unchanged OM DSO admission rules before sending 'G'. It also checks post-maps and the original in-process numerical/safety evidence, including Timed Tail, independent Postcheck and unchanged actual memory sizes.

If msprof closes the inherited FD, changes loader/OPP selection, loads an unbound DSO, or fails to collect an identifiable runner trace, the result is BLOCKED. Never substitute a new gate, auto-admit a profiling DSO, silence warnings or start an unprofiled fallback. Observation of a shim PID alone does not qualify as a capture. CPU tests of FD inheritance are not proof of compatibility with the server's msprof.

## Artifacts and interpretation

Each profile stage retains the commands, CLI help/log, shim witness, Runner journal/logs, pre/post maps, frozen output comparisons, `om-graph.json`, full `profiler/` (including op_summary, timeline and Step data when present), and `windows.json`. File identities are listed in the stage report. The execution/report protocol must remain 2 correctness, 5 warmup, 30 timed, 1 postcheck. Exactly 38 paired public Execute/Sync calls on one host lane are required. Extra teardown Sync is kept separate. Trace host PID or main TID must corroborate the real Runner. An ambiguous timeline or op_summary export is rejected rather than averaged with another export.

Uses the repository's CANN window, Step, graph, Decimal origin and interval-union logic. The host/trace units are microseconds; CSV clocks are not substituted into windows. Accepted Device Step attribution is distinguished from unresolved/contradictory Step metadata. With no Step markers the result is **PROVISIONAL_HOST_ENCLOSURE**; crossing tasks have no inferred owner. Raw events, unassigned/outside records, Step checks, pre-Step transfers, errors and all 38 windows are retained. Duplicate task identity is scoped to one iteration/model/stream (and batch/subtask); reuse across iterations is not itself an error.

Analyze writes `analyze/{c16,c64}/{profile-report.json,profile-hotspots.md,operator-types.csv,operator-tasks.csv}` and a combined hotspots Markdown. Category totals and Top 20 use only owned, nonduplicate timeline device intervals in calls **7..36**. Task rows with a known owner also retain nonmeasured phase labels. Task Duration Sum/Mean/Max here are **timeline duration µs**, not a sum of op_summary, task_time or PipeUtilization counters. Per-call union is not the full Draft wall time. Categories count device tasks, not inferred GE invocation counts. Exact OM node match and op_summary/trace type underpin labels; unconfirmed MatMul remains unmapped. When kernel symbols cannot be uniquely joined, the timeline name is used and labeled accordingly. Compute-task ranking and all-device-task ranking are separate. Unavailable graph parsing, ambiguous Step metadata and unclassified tasks are reported, not silently costed as zero. Capture/analysis success can still yield provisional or unresolved attribution.

Failure handoff: stage status/error/traceback, `msprof.log`, `application/runner.log` and `application/shim.json` when present. For a successful analysis return the two gear summaries and their attribution grades. Do not interpret profiling times as new formal latency, sum overlapping pipes, or predict speedup from busy time.

CPU regression: `python3 -B -m unittest discover -s framework/custom_ops/draft_a_core_scheduler_profile/test -v`. The SCM credentials and exec/FD fixture requires Linux and a C compiler; it performs no device operations.
