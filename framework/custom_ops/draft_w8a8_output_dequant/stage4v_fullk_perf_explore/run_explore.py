#!/usr/bin/env python3
"""Frozen normal OPP only; independent safe Smoke and exploratory performance."""
import argparse
import os
from pathlib import Path
import explore_common as c
from explore_protocol import validate,summary,OUTPUTS
STATES=dict(source='FULLK_EXPLORE_SOURCE_REVIEWED',build='FULLK_EXPLORE_RUNNER_BUILT',host='FULLK_EXPLORE_HOST_TILING_ONLY',tile0='FULLK_EXPLORE_TILE0_SAFE',full='FULLK_EXPLORE_FULL_SAFE',benchmark='EXPLORATORY_PERFORMANCE_MEASURED')


def prior(path,stage,r):
 if path is None:raise ValueError('missing '+stage+' predecessor')
 old=c.read(path);c.verify_tree(old)
 if old.get('status')!=STATES[stage] or old.get('error') or old.get('integrity_error') or any(old.get(k)!=r[k] for k in ('sources','evidence','device','cann_root')):raise ValueError('predecessor state/identity differs '+stage)
 for key in ('os','version_files'):
  if old['machine'][key]!=r['machine'][key]:raise ValueError('machine drift')
 if stage=='build':c.verify_artifacts(old)
 expected_count=3 if stage=='benchmark' else 1 if stage in ('host','tile0','full') else 0
 if len(old.get('runs',[]))!=expected_count:raise ValueError('predecessor execution coverage incomplete')
 for row in old.get('runs',[]):
  if row['tile']!=(0 if stage=='tile0' else -1):raise ValueError('predecessor scope differs')
  artifacts=c.read(c.verify(old['build_report']))['artifacts']
  if row['runner']!=artifacts['runner']:raise ValueError('wrong independent Runner')
  if row['validation']!=validate(row,r['evidence'],r['device'],stage=='benchmark',stage=='host'):raise ValueError('predecessor validation changed')
 return old


def invoke(args,folder,r,b,tile,host=False,timed=False):
 folder.mkdir(parents=True);proof=r['evidence'];c.verify_artifacts(b);c.unchanged(proof);case=c.read(c.verify(proof['case']))
 maps=c.frozen.ref.verify_case(case)
 for name in ('a_i8.bin','b_i8.bin','quant_u64_doc_candidate.bin'):(folder/name).symlink_to(c.verify(case['files'][name]))
 expected=[]
 if not host:
  for i in range(2):
   f=folder/f'expected-{i}.bin';f.write_bytes(c.old_protocol.expected(maps,0,proof['policy']['selected'],tile,i));expected.append(c.identity(f))
 argv=[c.verify(b['artifacts']['runner']),args.device,folder,0,0,tile]+(['--host-only'] if host else ['--benchmark'] if timed else [])
 row=dict(tile=tile,runner=b['artifacts']['runner'],profiled=False,expected=expected)
 if timed:
  anchor=c.read(c.verify(r['full_report']))['runs'][0]['outputs'][0]
  (folder/'admitted_full.bin').symlink_to(c.verify(anchor));row['repeat_anchor']=anchor
 r.setdefault('runs',[]).append(row) # retain partial invocation identity on failure
 try:c.command(argv,folder/'runner.log',c.environment(proof,args.cann_root),r,'run_'+folder.name,900)
 finally:
  row['retained_files']=[c.identity(p) for p in sorted(folder.iterdir()) if p.is_file()]
 row.update(execution=c.identity(folder/'execution.json'),log=c.identity(folder/'runner.log'),admission=c.identity(folder/'resource-admission.json'))
 if not host:
  row['outputs']=[c.identity(folder/n) for n in OUTPUTS[:4 if timed else 2]]
  row['numeric']=[c.identity(folder/(n+'.numeric.json')) for n in OUTPUTS[:4 if timed else 2]]
 row['validation']=validate(row,proof,args.device,timed,host)
 c.write(folder/'validation.json',row['validation']);return row


def main(a):
 root=c.fresh(a.output_dir,[a.frozen_root,a.cann_root,c.HERE]+[p.parent for p in (a.source_report,a.build_report,a.host_report,a.tile_report,a.full_report) if p])
 r=dict(status='NOT_RUN',phase=a.phase,sources=c.sources(),device=a.device,cann_root=str(a.cann_root),npu='NOT_RUN',benchmark='NOT_RUN',profile='NOT_RUN',production='NOT_APPROVED',output_dir=str(root))
 try:
  r['evidence']=c.bind(a.frozen_root,a.device);r['machine']=c.machine(root,r['evidence'],a.cann_root,r)
  c.write(root/'historical-mismatches.json',r['evidence']['historical_comparison'])
  r['numeric_status']='NUMERIC_MISMATCH_UNRESOLVED'
  if a.phase!='source':
   prior(a.source_report,'source',r);r['source_report']=c.identity(a.source_report)
   if a.phase=='build':c.build(root,r['evidence'],a.cann_root,r)
   else:
    b=prior(a.build_report,'build',r);r['build_report']=c.identity(a.build_report)
    if b['source_report']!=r['source_report']:raise ValueError('source/build chain differs')
    if a.phase!='host':
     h=prior(a.host_report,'host',r);r['host_report']=c.identity(a.host_report)
     if h['build_report']!=r['build_report']:raise ValueError('Host/build chain differs')
    if a.phase in ('full','benchmark'):
     t=prior(a.tile_report,'tile0',r);r['tile_report']=c.identity(a.tile_report)
     if any(t[k]!=r[k] for k in ('build_report','host_report')):raise ValueError('Tile gate differs')
    if a.phase=='benchmark':
     f=prior(a.full_report,'full',r);r['full_report']=c.identity(a.full_report)
     if any(f[k]!=r[k] for k in ('build_report','host_report','tile_report')):raise ValueError('Full gate differs')
     r['npu']='ATTEMPTED';r['benchmark']='ATTEMPTED'
     baseline=c.verify(f['runs'][0]['outputs'][0]).read_bytes()
     for i in range(3):
      row=invoke(a,root/f'round-{i}',r,b,-1,timed=True);row['round']=i
      if any(c.verify(x).read_bytes()!=baseline for x in row['outputs']):raise ValueError('full smoke/benchmark output drift')
      c.write(root/'report.json',r)
     r['comparison']=summary(r['runs']);r['benchmark']='COMPLETED';r['npu']='COMPLETED'
    else:
     if a.phase!='host':r['npu']='ATTEMPTED'
     invoke(a,root/'run',r,b,0 if a.phase=='tile0' else -1,host=a.phase=='host')
     if a.phase!='host':r['npu']='COMPLETED'
  r['status']=STATES[a.phase]
 except Exception as x:r.update(status='BLOCKED',error=f'{type(x).__name__}: {x}')
 finally:
  try:
   c.verify_tree(r)
   if r['sources']!=c.sources():raise ValueError('new source identity changed')
   if 'evidence' in r:c.unchanged(r['evidence'])
   if a.build_report:c.verify_artifacts(c.read(a.build_report))
  except Exception as x:r.update(status='BLOCKED',integrity_error=str(x))
  if r['status']=='BLOCKED':r.pop('comparison',None)
  c.write(root/'report.json',r)
  if a.phase=='benchmark':c.write(root/'benchmark-report.json',r);c.write(root/'comparison.json',r.get('comparison',dict(status='NOT_RUN_OR_BLOCKED')))
 print(r['status']+': '+str(root/'report.json'));return int(r['status']=='BLOCKED')


def parse(argv=None):
 p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False);p.add_argument('phase',choices=STATES);p.add_argument('--output-dir',type=Path,required=True)
 p.add_argument('--frozen-root',type=Path,default=Path('/home/w00949577/qwen35-dflash-run/w8a8-full-k-l1fix-20261106-073850'))
 p.add_argument('--cann-root',type=Path,default=Path(os.environ.get('CANN_ROOT','/usr/local/Ascend/cann-9.0.0')));p.add_argument('--device',type=int,default=0)
 for k in ('source','build','host','tile','full'):p.add_argument('--'+k+'-report',type=Path)
 a=p.parse_args(argv);needed=[]
 if a.phase!='source':needed+=['source']
 if a.phase not in ('source','build'):needed+=['build']
 if a.phase in ('tile0','full','benchmark'):needed+=['host']
 if a.phase in ('full','benchmark'):needed+=['tile']
 if a.phase=='benchmark':needed+=['full']
 for k in needed:
  if getattr(a,k+'_report') is None:p.error('--'+k+'-report required')
 if a.device<0:p.error('negative device')
 for k,v in vars(a).items():
  if isinstance(v,Path):setattr(a,k,v.resolve())
 return a
if __name__=='__main__':raise SystemExit(main(parse()))
