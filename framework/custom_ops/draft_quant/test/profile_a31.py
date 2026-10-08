#!/usr/bin/env python3
"""Profile one already-validated custom case in a new evidence directory."""
import argparse
import csv
import json
from pathlib import Path
import shlex
import shutil
import subprocess

from a2_common import checked_file, record, sha256, write_json


def collect_pipe_counters(root):
    """Preserve original profiler column names/units; never invent absent metrics."""
    def normalized(text):
        return "".join(c.lower() for c in text if c.isalnum())
    files, rows = [], []
    for path in sorted((root / "profiler").rglob("*.csv")):
        item = record(path, root)
        files.append(item)
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for number, row in enumerate(csv.DictReader(stream), 2):
                names = [value for key,value in row.items() if key and normalized(key) in
                         ("opname","optype","kernelname","taskname") and isinstance(value,str)]
                if not any("dflashgroupquantlinear" in normalized(name) for name in names):
                    continue
                fields = {key:value for key,value in row.items() if key and isinstance(value,str) and value and
                          any(word in normalized(key) for word in ("mte1","mte2","mte3","duration","cube","mac"))}
                if fields: rows.append(dict(file=item["path"], line=number, fields=fields))
    return dict(status="RAW_COLUMNS_FOUND" if rows else "NOT_FOUND", csv_files=files, rows=rows,
                mte2_available=any("mte2" in normalized(key) for row in rows for key in row["fields"]),
                interpretation="original strings/units; ratios are not assumed to divide Task Duration; not unprofiled timing")


def prepare(source, output):
    source, output = source.resolve(), output.resolve()
    execution = json.loads((source / "execution.json").read_text())
    command = json.loads((source / "command.json").read_text())
    if (execution.get("status") != "PASS" or execution.get("runtime") != "AscendCL ACLNN" or
            execution.get("timing", {}).get("protocol") != "continuous-v1" or
            len(command) != 9 or command[-1] != "--continuous" or Path(command[2]).resolve() != source or
            int(command[1]) != execution["device_id"] or
            tuple(map(int, command[3:6])) != tuple(execution[d] for d in ("m", "k", "n"))):
        raise ValueError("select a completed continuous-v1 real-projection <case>/custom directory")
    runner = Path(command[0]).resolve()
    real = json.loads((source.parent.parent / "suite.json").read_text())
    identity = real["runners"]["custom"]
    if real.get("status") != "PASS" or Path(identity["path"]).resolve() != runner or sha256(runner) != identity["sha256"]:
        raise ValueError("profile runner differs from the accepted run")
    vendor = runner.parent.parent / "opp/vendors/customize"
    vendor_env = vendor / "bin/set_env.bash"
    if not vendor_env.is_file():
        raise ValueError("the original isolated OPP build must still exist")
    bundle_path = Path(real["bundle"])
    if sha256(bundle_path) != real["bundle_sha256"]:
        raise ValueError("capture manifest changed after acceptance")
    bundle = json.loads(bundle_path.read_text())
    capture = next(c for c in bundle["cases"] if c["name"] == source.parent.name)
    inputs = {}
    for name in ("x.bin", "w_nz.bin", "s_gn.bin"):
        inputs[name] = checked_file(bundle_path.parent, capture["files"][name])
        if (source / name).resolve(strict=True) != inputs[name]:
            raise ValueError("case inputs differ from the captured bundle")
    output.mkdir(parents=True, exist_ok=False)
    case = output / "case"
    case.mkdir()
    for name, original in inputs.items():
        (case / name).symlink_to(original)
    command[2] = str(case)
    launcher = output / "replay.sh"
    launcher.write_text("#!/usr/bin/env bash\nset -eo pipefail\nsource " + shlex.quote(str(vendor_env)) +
                        "\nexport LD_LIBRARY_PATH=" + shlex.quote(str(vendor / "op_api/lib")) +
                        ':"${LD_LIBRARY_PATH:-}"\nexec ' + shlex.join(command) + "\n")
    return command, launcher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--msprof", default="msprof")
    args = parser.parse_args()
    profiler = shutil.which(args.msprof)
    if not profiler:
        raise ValueError("msprof is unavailable; load the existing CANN environment")
    command, launcher = prepare(args.case_dir, args.output_dir)
    root = args.output_dir.resolve()
    profile_command = [profiler, "--ai-core=on", "--aic-mode=task-based", "--aic-metrics=PipeUtilization",
                       f"--output={root / 'profiler'}", "--application=" + shlex.join(["bash", str(launcher)])]
    report = {"abi": "dflash-group-quant-profile-v1", "status": "RUNNING", "profiled": True,
              "formal_latency_evidence": False, "source_case": str(args.case_dir.resolve()),
              "profiler_command": profile_command, "runner_command": command}
    write_json(root / "profile.json", report)
    try:
        with (root / "profile.log").open("w") as log:
            subprocess.run(profile_command, stdout=log, stderr=subprocess.STDOUT, check=True)
        execution = json.loads((root / "case/execution.json").read_text())
        golden = (args.case_dir / "actual-0.bin").read_bytes()
        if execution.get("status") != "PASS" or any((root / "case" / name).read_bytes() != golden
                for name in ("actual-0.bin", "actual-1.bin", "benchmark-last.bin", "postcheck.bin")):
            raise ValueError("profiled output differs from the accepted unprofiled case")
        report.update(status="COMMAND_COMPLETED", numerical_match=True,
                      metrics="PipeUtilization requested; inspect profiler artifacts for available device counters",
                      pipe_counters=collect_pipe_counters(root))
    except Exception as error:
        report.update(status="FAIL", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(root / "profile.json", report)
    print(f"Profile command completed: {root}; profiled timings are excluded from benchmark comparisons")


if __name__ == "__main__":
    main()
