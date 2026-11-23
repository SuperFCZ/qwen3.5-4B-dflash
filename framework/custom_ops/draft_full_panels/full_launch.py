"""Run either a bare full OM or msprof with the same pre-execute socket admission."""

import os
from pathlib import Path
import select
import signal
import socket
import subprocess
import sys
import time

from joint import c, OPS, require

sys.path.insert(0, str(OPS / "draft_a_core_scheduler_profile"))
from collect_profile import capture_command, sender
from scale_contracts import host, resume


def launch(folder, runner, om, phase, env, dsos, profiler):
    app = folder / "application"
    parent, child = socket.socketpair()
    log = folder / "msprof.log" if profiler else app / "runner.log"
    with parent, child, log.open("xb") as stream:
        parent.setsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED, 1)
        fd = child.fileno()
        argv = list(
            map(str, (c.verify(runner), 0, c.verify(om), app, app / "inputs", app / "golden", phase, fd))
        )
        endpoint = os.readlink("/proc/self/fd/" + str(fd))
        cmd = argv
        if profiler:
            require(phase == "benchmark", "msprof requires exactly 38 Execute+Sync calls")
            req = folder / "shim-request.json"
            c.write(req, dict(argv=argv, endpoint=endpoint))
            cmd = capture_command(
                profiler,
                folder / "profiler",
                sys.executable,
                OPS / "draft_a_core_scheduler_profile/exec_shim.py",
                req,
            )
        c.write(folder / "command.json", dict(argv=cmd, application=argv, om=om, runner=runner, env=env))
        with subprocess.Popen(
            cmd,
            pass_fds=(fd,),
            env=env,
            cwd=folder,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        ) as proc:
            child.close()
            try:
                deadline = time.monotonic() + 900
                while not select.select([parent], [], [], 1)[0]:
                    require(
                        proc.poll() is None and time.monotonic() < deadline,
                        "process exited or timed out before pre-execute gate",
                    )
                pid = sender(parent)
                task = Path("/proc") / str(pid)
                require(pid == proc.pid or profiler, "unexpected runner PID")
                if profiler:
                    require(
                        c.read(app / "shim.json") == dict(pid=pid, fd=fd, endpoint=endpoint, argv=argv),
                        "msprof shim/FD evidence differs",
                    )
                require(
                    c.identity(task / "exe") == runner
                    and os.readlink(task / "fd" / str(fd)) == endpoint
                    and (task / "cmdline").read_bytes().split(b"\0")[:-1] == [os.fsencode(s) for s in argv],
                    "live runner, argv or gate changed",
                )
                actual = dict(
                    x.split("=", 1) for x in os.fsdecode((task / "environ").read_bytes()).split("\0") if x
                )
                for name in (
                    "ASCEND_CUSTOM_OPP_PATH",
                    "ASCEND_OPP_PATH",
                    "LD_LIBRARY_PATH",
                    "LD_PRELOAD",
                    "LD_AUDIT",
                    "ASCEND_VISIBLE_DEVICES",
                    "ASCEND_RT_VISIBLE_DEVICES",
                ):
                    require(actual.get(name) == env.get(name), "runner env changed: " + name)
                mapped = host.dsos.maps_rows((task / "maps").read_text())
                require(
                    mapped == host.dsos.maps_rows((app / "pre-maps.txt").read_text()),
                    "live pre-execute mappings differ",
                )
                resume.check_maps(mapped, dsos)
                parent.sendall(b"G")
                parent.close()
                require(proc.wait(timeout=1800) == 0, "OM execution failed: " + str(log))
                resume.check_maps(host.dsos.maps_rows((app / "post-maps.txt").read_text()), dsos)
                return dict(runner_pid=pid, launch_authorized=True)
            except BaseException:
                parent.close()
                if proc.poll() is None:
                    try:
                        proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
                for file in dict.fromkeys((log, app / "runner.log")):
                    if file.is_file():
                        with file.open("rb") as source:
                            source.seek(max(0, file.stat().st_size - 10000))
                            sys.stderr.write(source.read().decode(errors="replace"))
                raise
