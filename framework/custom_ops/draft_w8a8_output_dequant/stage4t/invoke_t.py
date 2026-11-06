"""Independent-process continuous-v1; unchanged numerical/timing protocol."""
import shlex,shutil
from pathlib import Path
from t_common import q,bc,verify,identity,read,environment,check_execution,validate_run,COVERAGE
from run_stage2b import command

def invoke(args,folder,r,b,name,timed,profiled=False):
 folder.mkdir(parents=True);p=r['evidence']['variants'][name]
 for key in ('a_i8.bin','b_i8.bin','quant_u64_doc_candidate.bin','golden_fp16.bin'):(folder/key).symlink_to(verify(p['files'][key]))
 argv=[verify(b['variants'][name]['runner']),args.device,folder,0]+(['--benchmark'] if timed else [])
 env=environment(p,args.cann_root)
 if profiled:
  tool=shutil.which(args.msprof)
  if not tool:raise ValueError('msprof unavailable')
  launcher=folder.parent/'launch.sh';launcher.write_text('#!/usr/bin/env bash\nset -euo pipefail\nexec '+shlex.join(list(map(str,argv)))+' > '+shlex.quote(str(folder/'runner.log'))+' 2>&1\n')
  argv=[tool,'--ai-core=on','--aic-mode=task-based','--aic-metrics=PipeUtilization','--output='+str(folder.parent/'profiler'),'--application='+shlex.join(['bash',str(launcher)])]
  r.setdefault('profilers',{})[name]=dict(tool=identity(tool),argv=argv)
  command(argv,folder.parent/'profile.log',env,r,'profile_'+name,1800)
 else:command(argv,folder/'runner.log',env,r,str(folder.relative_to(Path(r['output_dir']))),600)
 ex_id=identity(folder/'execution.json');ex=read(verify(ex_id));stats=check_execution(ex,args.device,name,timed)
 row=dict(variant=name,guard_coverage=COVERAGE[name],status='FROZEN_GOLDEN_MATCH',execution=ex_id,outputs=bc.verify_outputs(p,folder,timed),
  log=identity(folder/'runner.log'),plan=q.plan((folder/'runner.log').read_text(),'N256',p),admission=identity(folder/'resource-admission.json'),
  timing=stats,workspace_bytes=ex['workspace_bytes'],profiled=profiled,formal_latency_evidence=timed and not profiled)
 validate_run(row,r['evidence'],args.device,timed,profiled);return row
