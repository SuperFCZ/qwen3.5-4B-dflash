"""Explicit env adapter for the frozen single-FC ATC invocation."""
import subprocess
from pathlib import Path
from split_atc import SplitAtcRunner
from platform_common import c
from platform_runtime_env import assert_invocation


class IsolatedAtcRunner(SplitAtcRunner):
    def __init__(self,*args,env,binding,**kwargs):
        super().__init__(*args,**kwargs);self.env=dict(env);self.binding=binding
    def __call__(self,command,cwd):
        argv=list(map(str,command));cwd=Path(cwd).resolve(strict=True)
        assert_invocation(self.env,self.binding)
        if not argv or c.identity(argv[0])!=self.atc:raise ValueError('ATC executable identity differs')
        if self.expected_command is not None and argv!=self.expected_command:raise ValueError('ATC flags changed beyond approved relocation')
        outputs=[x.split('=',1)[1] for x in argv if x.startswith('--output=')]
        if len(outputs)!=1 or self.output_root not in Path(outputs[0]).resolve().parents:raise ValueError('ATC output outside new directory')
        run=self.root/f'call-{len(self.records)}';run.mkdir();stdout=run/'stdout.log';stderr=run/'stderr.log'
        record=dict(argv=argv,cwd=str(cwd),atc=self.atc,returncode=None,environment=dict(self.env));self.records.append(record)
        try:
            with stdout.open('xb') as out,stderr.open('xb') as err:
                result=subprocess.run(argv,cwd=cwd,stdout=out,stderr=err,env=self.env,check=False)
            record['returncode']=result.returncode
        finally:
            for name,path in (('stdout',stdout),('stderr',stderr)):
                if path.is_file():record[name]=c.identity(path)
            c.write(run/'command.json',record)
        c.verify(self.atc);assert_invocation(self.env,self.binding)
        return subprocess.CompletedProcess(argv,result.returncode,stdout.read_bytes().decode(),stderr.read_bytes().decode())
