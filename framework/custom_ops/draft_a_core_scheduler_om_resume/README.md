# Full OM loader continuation

Only for the A/C16 pre-launch block in `a-core-scheduler-flow-F6ia0q`. The preceding Source, Build, Host Query, projection Smoke and Export reports must still verify against the unchanged scheduler flow sources. Reuses all four OM files and both existing OM runners. The additional 22 local-install pins are observations from the supplied maps snapshot, not a vendor signature. Admit binds the file and read-only fd/VMA backing identity in the current mount namespace; changes to links, mode, size, SHA, namespace, or mapped device/inode fail closed. No wildcard, alternate installation, private vendor alias or post-launch discovery is admitted. No policy change to projection processes.

```bash
cd /home/w00949577/z50058744/qwen3.5-4B-dflash
git pull --ff-only origin feature/ascend310p-custom-ops
OLD=/home/w00949577/qwen35-dflash-run/a-core-scheduler-flow-F6ia0q
NEW=$(mktemp -d /home/w00949577/qwen35-dflash-run/a-scheduler-om-resume-XXXXXX)
PY=/home/w00949577/qwen35-dflash-run/venv-hf513-probe/bin/python
S=$PWD/framework/custom_ops/draft_a_core_scheduler_om_resume
om_stage() { TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$PY" -B "$S/resume.py" "$1" --old "$OLD" --root "$NEW"; }
echo "NEW=$NEW"
om_stage admit
```

After `OM_RESUME_ADMITTED` only: `om_stage full-smoke`. After `OM_RESUME_TOKEN_KV_MATCH` only: optionally `om_stage benchmark`. Each phase writes a fresh report, never touches the old failed stage, and never dispatches its successor. Unknown post-execute libraries still block acceptance. Timings from this run remain subject to `ACTUAL_ATC_SCHEDULES_IDENTICAL`; they are not evidence of a 7-to-8-core speedup.
