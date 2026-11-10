#!/usr/bin/env python3
"""Two safe full-shape executions under msprof, reusing the frozen normal Runner."""
import argparse
import re
import shlex
import shutil
from pathlib import Path
import profile_evidence as e
from profile_parse import collect
from profile_analysis import analyze,markdown
HERE=Path(__file__).resolve().parent
OPTIONS=['--ai-core=on','--aic-mode=task-based','--aic-metrics=PipeUtilization','--task-time=on','--task-trace=on','--type=text']


def validate_help(text):
 sections={};key=None
 for line in text.splitlines():
  m=re.match(r'\s*(--[a-z0-9-]+)\s+',line)
  if m:key=m.group(1);sections[key]=line
  elif key:sections[key]+='\n'+line
 required={'--ai-core':('on','off'),'--aic-mode':('task-based',),'--aic-metrics':('PipeUtilization',),'--task-time':('on',),'--task-trace':('on',),'--type':('text',),'--output':(),'--application':()}
 if any(k not in sections or any(v not in sections[k] for v in values) for k,values in required.items()):raise ValueError('installed msprof help does not confirm requested options/values')
 return {k:sections[k] for k in required}


def tool_check(a,root,r,bound):
 tool=shutil.which(a.msprof)
 if not tool:raise ValueError('msprof not found')
 item=e.identity(tool)
 if a.cann_root not in Path(item['path']).parents:raise ValueError('msprof outside selected CANN')
 e.c.command([tool,'--help'],root/'msprof-help.txt',e.c.environment(bound['proof'],a.cann_root),r,'msprof_help',60)
 log=e.identity(root/'msprof-help.txt');validated=validate_help(e.verify(log).read_text())
 return dict(tool=item,requested_path=str(Path(tool).absolute()),help=log,options=OPTIONS,validated_sections=validated)


def profile_run(a,root,r,ready,bound):
 folder=root/'run';folder.mkdir();proof=bound['proof'];b=bound['build'];e.verify_bound(bound)
 case=e.read(e.verify(proof['case']));maps=e.c.frozen.ref.verify_case(case)
 for n in ('a_i8.bin','b_i8.bin','quant_u64_doc_candidate.bin'):(folder/n).symlink_to(e.verify(case['files'][n]))
 expected=[]
 for i in range(2):
  f=folder/f'expected-{i}.bin';f.write_bytes(e.c.old_protocol.expected(maps,0,proof['policy']['selected'],-1,i));expected.append(e.identity(f))
 argv=[str(e.verify(b['artifacts']['runner'])),str(a.device),str(folder),'0','0','-1'] # no --benchmark: exactly two guarded runs
 launch=root/'launch-profile.sh';launch.write_text('#!/usr/bin/env bash\nset -euo pipefail\nexec '+shlex.join(argv)+' > '+shlex.quote(str(folder/'runner.log'))+' 2>&1\n')
 r.update(runner_command=argv,launcher=e.identity(launch),expected_device_executions=2)
 profile_dir=root/'profiler'
 cmd=[str(e.verify(ready['profiler']['tool']))]+OPTIONS+['--output='+str(profile_dir),'--application='+shlex.join(['bash',str(launch)])]
 r['profile_command']=cmd;r['npu']='ATTEMPTED';r['profile']='ATTEMPTED'
 row=dict(tile=-1,runner=b['artifacts']['runner'],profiled=True,expected=expected)
 r['run']=row;e.write(root/'profile-report.json',r)
 try:e.c.command(cmd,root/'profile.log',e.c.environment(proof,a.cann_root),r,'msprof',1800)
 finally:row['retained_files']=[e.identity(p) for p in sorted(folder.iterdir()) if p.is_file()]
 row.update(execution=e.identity(folder/'execution.json'),log=e.identity(folder/'runner.log'),admission=e.identity(folder/'resource-admission.json'),
  outputs=[e.identity(folder/n) for n in e.OUTPUTS[:2]],numeric=[e.identity(folder/(n+'.numeric.json')) for n in e.OUTPUTS[:2]])
 row['validation']=e.validate(row,proof,a.device,False,False)
 anchor=e.verify(bound['full_anchor']).read_bytes()
 if any(e.verify(f).read_bytes()!=anchor for f in row['outputs']):raise ValueError('profiled output differs from admitted Full Smoke')
 if any(x!=bound['full_numeric'] for x in row['validation']['numeric']):raise ValueError('profile changed frozen numeric differences')
 r['capture_status']='TWO_SAFE_FULL_EXECUTIONS_COMPLETED';r['npu']='COMPLETED';r['profile']='COMPLETED'
 r['raw_profile']=collect(profile_dir)
 r['analysis']=analyze(r['raw_profile'],r['source_counts'],bound['benchmark_statistics'])
 e.write(root/'metrics-report.json',r['raw_profile']);e.write(root/'analysis-report.json',r['analysis'])
 (root/'BOTTLENECK_REPORT.md').write_text(markdown(r['analysis'],r['raw_profile']))
 r['analysis_outputs']={name:e.identity(root/name) for name in ('metrics-report.json','analysis-report.json','BOTTLENECK_REPORT.md')}
 r['status']='FULLK_PROFILE_COMPLETED' if r['raw_profile']['task_coverage']=='TWO_IDENTIFIED_AICORE_TASKS' and not r['raw_profile']['skipped'] else 'FULLK_PROFILE_COMPLETED_METRICS_PARTIAL'


def main(a):
 root=e.fresh(a.output_dir,[HERE,a.experiment_root,a.cann_root]+([a.prepare_report.parent] if a.prepare_report else []))
 r=dict(status='NOT_RUN',sources=e.sources(),phase=a.phase,device=a.device,cann_root=str(a.cann_root),profiled=True,npu='NOT_RUN',benchmark='NOT_RUN',profile='NOT_RUN',
  numeric_status='NUMERIC_MISMATCH_UNRESOLVED',production='NOT_APPROVED',output_dir=str(root),kernel_build='NOT_RUN_REUSED_NORMAL_OPP',runner_build='NOT_RUN_REUSED_ACCEPTED_BINARY')
 try:
  bound=e.bind(a.experiment_root,a.cann_root,a.device,root,r);r['evidence']=bound;r['source_counts']=e.source_counts(bound['proof'])
  if a.phase=='prepare':
   r['profiler']=tool_check(a,root,r,bound);r['status']='FULLK_PROFILE_READY'
  else:
   ready=e.read(a.prepare_report);e.verify_tree(ready)
   if ready.get('status')!='FULLK_PROFILE_READY' or ready.get('error') or ready.get('integrity_error') or any(ready.get(k)!=r[k] for k in ('sources','device','cann_root','source_counts')):raise ValueError('Profile prepare identity differs')
   # Different npu-smi log paths are expected; all bound evidence identities and stable machine fields are fixed.
   for k in ('reports','build','proof','full_anchor','full_numeric','benchmark_statistics'):
    if ready['evidence'][k]!=bound[k]:raise ValueError('Profile history drift '+k)
   for k in ('os','version_files'):
    if ready['evidence']['machine'][k]!=bound['machine'][k]:raise ValueError('Profile machine drift')
   now=tool_check(a,root,r,bound)
   if now['tool']!=ready['profiler']['tool'] or now['validated_sections']!=ready['profiler']['validated_sections'] or now['options']!=ready['profiler']['options']:raise ValueError('msprof tool/options drift')
   r['profiler']=now;r['prepare_report']=e.identity(a.prepare_report)
   profile_run(a,root,r,ready,bound)
 except Exception as x:r.update(status='BLOCKED',error=f'{type(x).__name__}: {x}')
 finally:
  try:
   if r['sources']!=e.sources():raise ValueError('Profile source changed')
   if 'evidence' in r:e.verify_bound(r['evidence'])
   e.verify_tree(r)
  except Exception as x:r.update(status='BLOCKED',integrity_error=str(x))
  e.write(root/'profile-report.json',r)
 print(r['status']+': '+str(root/'profile-report.json'));return int(r['status']=='BLOCKED')


def parse(argv=None):
 p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False);p.add_argument('phase',choices=('prepare','collect'))
 p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--prepare-report',type=Path)
 p.add_argument('--experiment-root',type=Path,default=Path('/home/w00949577/qwen35-dflash-run/w8a8-fullk-perf-explore-20261110-022343'))
 p.add_argument('--cann-root',type=Path,default=Path('/usr/local/Ascend/cann-9.0.0'));p.add_argument('--device',type=int,default=0);p.add_argument('--msprof',default='msprof')
 a=p.parse_args(argv)
 if a.phase=='collect' and a.prepare_report is None:p.error('--prepare-report required')
 if a.device<0:p.error('negative device')
 for k,v in vars(a).items():
  if isinstance(v,Path):setattr(a,k,v.resolve())
 return a
if __name__=='__main__':raise SystemExit(main(parse()))
