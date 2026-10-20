"""Per-process file limit plus monitored output budget; never deletes evidence."""
import json
import errno
import os
from pathlib import Path
import resource
import signal
import stat
import subprocess
import time


def usage(root,diagnostics=None):
    """One non-atomic snapshot. Only ENOENT for enumerated children is benign."""
    root=Path(root)
    measured=dict(logical_bytes=0,accounted_bytes=0,files=0,max_file_bytes=0,
                  max_file_path=None,symlink_targets_checked=0,
                  disappeared_count=0,disappeared_samples=[])
    if diagnostics is not None:
        diagnostics['scans']=diagnostics.get('scans',0)+1

    def disappeared(error,path,operation):
        # A missing output root is not a transient tmp_dump entry. ENOTDIR,
        # EACCES, EIO, etc. must also propagate, never become an empty scan.
        if error.errno!=errno.ENOENT or path==root:raise error
        item=dict(path=str(path),operation=operation,errno=errno.ENOENT)
        measured['disappeared_count']+=1
        if len(measured['disappeared_samples'])<20:measured['disappeared_samples'].append(item)
        if diagnostics is not None:
            diagnostics['disappeared_count']=diagnostics.get('disappeared_count',0)+1
            samples=diagnostics.setdefault('disappeared_samples',[])
            if len(samples)<20:samples.append(item)

    def scan(directory):
        try:
            with os.scandir(directory) as entries:
                paths=[Path(entry.path) for entry in entries]
        except OSError as error:
            disappeared(error,directory,'scandir');return
        for path in paths:
            try:info=path.lstat()
            except OSError as error:
                disappeared(error,path,'lstat');continue
            if stat.S_ISLNK(info.st_mode):
                # Preserve the old limits() single-file check of linked files,
                # but do not traverse linked directories or double-count their
                # payload in usage(). Only this case needs a following stat.
                try:target=path.stat()
                except OSError as error:
                    disappeared(error,path,'stat_symlink_target');continue
                measured['symlink_targets_checked']+=1
                if stat.S_ISREG(target.st_mode) and target.st_size>measured['max_file_bytes']:
                    measured.update(max_file_bytes=target.st_size,max_file_path=str(path))
                continue
            if stat.S_ISDIR(info.st_mode):scan(path)
            elif stat.S_ISREG(info.st_mode):
                measured['logical_bytes']+=info.st_size
                measured['accounted_bytes']+=max(info.st_size,getattr(info,'st_blocks',0)*512)
                measured['files']+=1
                if info.st_size>measured['max_file_bytes']:
                    measured.update(max_file_bytes=info.st_size,max_file_path=str(path))
    if not stat.S_ISDIR(root.lstat().st_mode):raise NotADirectoryError(str(root))
    scan(root)
    return measured


def limits(root,total,file_limit,diagnostics=None):
    measured=usage(root,diagnostics)
    if measured['accounted_bytes']>total:raise RuntimeError('OUTPUT_BUDGET_EXCEEDED: '+str(measured))
    if measured['max_file_bytes']>file_limit:raise RuntimeError('FILE_BUDGET_EXCEEDED: '+str(measured))
    return measured


class BudgetCommands:
    def __init__(self,root,total,file_limit):
        if not 0<file_limit<=total:raise ValueError('invalid disk limits')
        self.root=Path(root);self.total=total;self.file_limit=file_limit

    def __call__(self,argv,log,env,cwd,timeout=1800):
        diagnostics=dict(scans=0,disappeared_count=0,disappeared_samples=[])
        limits(self.root,self.total,self.file_limit,diagnostics)
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
                    limits(self.root,self.total,self.file_limit,diagnostics)
                    if time.monotonic()-start>timeout:raise RuntimeError('SIMULATOR_TIMEOUT: no timeout escalation')
                    time.sleep(0.25)
                if proc.returncode:raise subprocess.CalledProcessError(proc.returncode,list(map(str,argv)))
            except Exception as exc:
                error=exc
                try:os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                proc.wait()
            finally:
                final_usage=None;final_budget_error=None
                # Also check after a normal exit with no poll iterations. Reuse
                # this single snapshot in the audit instead of scanning again.
                try:final_usage=limits(self.root,self.total,self.file_limit,diagnostics)
                except Exception as exc:
                    final_budget_error=f'{type(exc).__name__}: {exc}'
                    if error is None:error=exc
                log.with_suffix('.budget.json').write_text(json.dumps(dict(status='STOPPED' if error else 'COMMAND_COMPLETED',
                    returncode=proc.returncode,error=str(error) if error else None,usage=final_usage,
                    final_budget_error=final_budget_error,scan_diagnostics=diagnostics,
                    limitation='Total limit is polled, not a filesystem quota; transient overshoot is possible. Per-file RLIMIT_FSIZE is enforced in child.',
                    truncated_output_accepted=False),indent=2)+'\n')
        if error:raise error
        return log.read_text(errors='replace')
