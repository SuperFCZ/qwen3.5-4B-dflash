"""Per-process file limit plus monitored output budget; never deletes evidence."""
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time


def usage(root):
    files=[p.stat() for p in Path(root).rglob('*') if p.is_file() and not p.is_symlink()]
    return dict(logical_bytes=sum(s.st_size for s in files),
                accounted_bytes=sum(max(s.st_size,getattr(s,'st_blocks',0)*512) for s in files),files=len(files))


def limits(root,total,file_limit):
    measured=usage(root)
    if measured['accounted_bytes']>total:raise RuntimeError('OUTPUT_BUDGET_EXCEEDED: '+str(measured))
    if any(p.stat().st_size>file_limit for p in Path(root).rglob('*') if p.is_file()):raise RuntimeError('FILE_BUDGET_EXCEEDED')
    return measured


class BudgetCommands:
    def __init__(self,root,total,file_limit):
        if not 0<file_limit<=total:raise ValueError('invalid disk limits')
        self.root=Path(root);self.total=total;self.file_limit=file_limit

    def __call__(self,argv,log,env,cwd,timeout=1800):
        limits(self.root,self.total,self.file_limit)
        plan=dict(argv=list(map(str,argv)),cwd=str(cwd),timeout_seconds=timeout,
            total_budget_bytes=self.total,per_file_limit_bytes=self.file_limit,poll_seconds=0.25,
            environment={k:env.get(k) for k in ('ASCEND_CUSTOM_OPP_PATH','LD_LIBRARY_PATH','LD_PRELOAD','DFLASH_SIM_RUNTIME','DFLASH_SIM_DRIVER','DFLASH_PROFILER_ROOT')})
        log.with_suffix('.command.json').write_text(json.dumps(plan,indent=2)+'\n')
        def child_limits():
            resource.setrlimit(resource.RLIMIT_FSIZE,(self.file_limit,self.file_limit))
        error=None;start=time.monotonic()
        with log.open('w') as out:
            proc=subprocess.Popen(list(map(str,argv)),cwd=cwd,env=env,stdout=out,stderr=subprocess.STDOUT,
                                  start_new_session=True,preexec_fn=child_limits)
            try:
                while proc.poll() is None:
                    limits(self.root,self.total,self.file_limit)
                    if time.monotonic()-start>timeout:raise RuntimeError('SIMULATOR_TIMEOUT: no timeout escalation')
                    time.sleep(0.25)
                limits(self.root,self.total,self.file_limit)
                if proc.returncode:raise subprocess.CalledProcessError(proc.returncode,list(map(str,argv)))
            except Exception as exc:
                error=exc
                try:os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                proc.wait()
            finally:
                log.with_suffix('.budget.json').write_text(json.dumps(dict(status='STOPPED' if error else 'COMMAND_COMPLETED',
                    returncode=proc.returncode,error=str(error) if error else None,usage=usage(self.root),
                    limitation='Total limit is polled, not a filesystem quota; transient overshoot is possible. Per-file RLIMIT_FSIZE is enforced in child.',
                    truncated_output_accepted=False),indent=2)+'\n')
        if error:raise error
        return log.read_text(errors='replace')
