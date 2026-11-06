#!/usr/bin/env python3
"""Gated weaker Guard diagnostic. Never a production or full Guard acceptance. All previous artifacts read-only."""
import argparse,copy,os
from pathlib import Path
from t_common import (q,p,rp,pr,identity,verify,read,write,fresh,sources,evidence,immutable,environment,
 materialize,checked_build,ORDER,comparison,COVERAGE,LIMITATION)
from build_t import build as build_op
from build_bench_t import build as build_bench
from run_stage2b import command
from profile_rows import collect
from invoke_t import invoke
from t_common import validate_run
STATES=dict(build='STAGE4T_BUILT_NPU_NOT_RUN',**{'host-only':'STAGE4T_HOST_ONLY','bounded':'STAGE4T_DIAGNOSTIC_BOUNDED_MATCH','full':'STAGE4T_DIAGNOSTIC_FULL_MATCH','benchmark':'STAGE4T_PAIRED_MEASURED','profile':'STAGE4T_PROFILE_COMPLETED'})

def sequence(stage):
 return rp.sequence(stage)

def prior(path,stage,r,b):
 if path is None:raise ValueError('missing '+stage+' proof')
 old=read(path)
 if old.get('status')!=STATES[stage] or old.get('phase')!='complete' or any(old.get(k)!=r[k] for k in ('sources','evidence','device','build_report','guard_coverage')):raise ValueError('prior stage binding mismatch')
 if old['machine']['os']!=r['machine']['os'] or old['machine']['version_files']!=r['machine']['version_files']:raise ValueError('environment drift')
 predecessors={'bounded':'host-only','full':'bounded','benchmark':'full'}
 if stage in predecessors:prior(verify(old['prior_report']),predecessors[stage],r,b)
 if stage=='benchmark':
  for row in old['runs']:validate_bench(row,r,b,False)
  if comparison(old['runs'])!=old['comparison']:raise ValueError('summary mismatch')
 else:
  if [(x['core_limit'],x['mode'],x['tile_id']) for x in old['runs']]!=sequence(stage):raise ValueError('correctness/fault coverage incomplete')
  for row in old['runs']:
   if row.get('variant')!='candidate' or row.get('runner')!=b['variants']['candidate']['build']['runner']:raise ValueError('candidate Runner binding differs')
   result=check_correct(row,old,stage)
   if result!=row['validation']:raise ValueError('prior correctness evidence changed')
   if result['comparison']['status'] not in ('HOST_TILING_ONLY','BITWISE_MATCH'):raise ValueError('prior numerical failure')
 if stage=='full':
  if [(x['core_limit'],x['mode'],x['tile_id']) for x in old.get('baseline_runs',[])]!=sequence('full'):raise ValueError('baseline numerical preflight missing')
  for row in old['baseline_runs']:
   if row.get('variant')!='baseline' or row.get('runner')!=r['evidence']['baseline']['runner']:raise ValueError('baseline Runner binding differs')
   if check_correct(row,old,'full')!=row['validation'] or row['validation']['comparison']['status']!='BITWISE_MATCH':raise ValueError('baseline correctness failure')
 return old

def check_correct(row,r,stage):
 if row.get('variant') not in COVERAGE or row.get('guard_coverage')!=COVERAGE[row['variant']]:raise ValueError('correctness coverage mislabeled')
 ex=read(verify(row['execution']))
 if row['variant']=='candidate' and stage!='host-only' and ex.get('guard_coverage')!=COVERAGE['candidate']:raise ValueError('Runner diagnostic label absent')
 maps=pr.verify_case(read(verify(r['evidence']['case'])))
 plan,result=rp.evaluate_row(r,row,maps,stage=='host-only')
 return dict(plan=plan,comparison=result)

def correct(args,root,r,b,name="candidate"):
 stage=args.stage;proof=r['evidence'];variant=b['variants']['candidate']['build']
 case=read(verify(proof['case']));r['runs']=[]
 for limit,mode,tile in sequence(stage):
  folder=root/'execute'/f'l{limit}_t{tile}_m{mode}';folder.mkdir(parents=True)
  for key in ('a_i8.bin','b_i8.bin','quant_u64_doc_candidate.bin'):(folder/key).symlink_to(verify(case['files'][key]))
  e=environment(dict(proof['baseline'],opp=variant['opp']),args.cann_root)
  r.update(phase=f'l{limit}_t{tile}_m{mode}',npu='NOT_RUN' if stage=='host-only' else 'RUNNING');write(root/'report.json',r)
  command([verify(variant['runner']),args.device,folder,mode,limit,tile]+(['--host-only'] if stage=='host-only' else []),folder/'runner.log',e,r,r['phase'],600)
  row=dict(variant=name,guard_coverage=COVERAGE[name],runner=variant['runner'],core_limit=limit,mode=mode,tile_id=tile,execution=identity(folder/'execution.json'),log=identity(folder/'runner.log'),resource_admission=identity(folder/'resource-admission.json'))
  if stage!='host-only':row['outputs']=[identity(folder/f'actual-{i}.bin') for i in range(2)]
  r['runs'].append(row);row['validation']=check_correct(row,r,stage);write(root/'report.json',r)
  if row['validation']['comparison']['status'] not in ('HOST_TILING_ONLY','BITWISE_MATCH'):raise ValueError('correctness failure; raw outputs and comparison retained')

def bench_proof(r,b,name):
 v=copy.deepcopy(r['evidence']['baseline'])
 if name=='candidate':v.update(opp=b['variants']['candidate']['build']['opp'],installed=b['variants']['candidate']['build']['installed'])
 return dict(variants={name:v})

def validate_bench(row,r,b,profiled):
 validate_run(row,bench_proof(r,b,row['variant']),r['device'],True,profiled)

def measured(args,root,r,b,profiled=False):
 r['runs']=[]
 for i,order in enumerate((('baseline','candidate'),) if profiled else ORDER):
  for position,name in enumerate(order):
   shadow=dict(device=args.device,output_dir=str(root),evidence=bench_proof(r,b,name))
   runner=dict(variants={name:b['bench_runners'][name]})
   r.update(phase=f'round{i}_{name}',npu='RUNNING');write(root/'report.json',r)
   row=invoke(args,root/f'round-{i}'/name/'case',shadow,runner,name,True,profiled)
   row.update(variant=name,round=i,position=position,commands=shadow.get('commands',{}),profiler=shadow.get('profilers'))
   r['runs'].append(row);validate_bench(row,r,b,profiled);write(root/'report.json',r)
 if profiled:
  r['raw_profile']={name:collect(root/'round-0'/name/'profiler',p.NAME) for name in ('baseline','candidate')}
 else:r['comparison']=comparison(r['runs'])


def main(args):
 root=fresh(args.output_dir,[args.cann_root,args.stage4p_root,args.stage4n_root,args.stage4o_root,args.stage4q_root]+[f.parent for f in (args.build_report,args.prior_report) if f])
 r=dict(stage='4T',status='NOT_RUN',phase='evidence',sources=sources(),device=args.device,npu='NOT_RUN',benchmark='NOT_RUN',profile='NOT_RUN',scope=q.SCOPE,guard_coverage=COVERAGE,safety_limitation=LIMITATION,production_adoption='PROHIBITED',fault_coverage='NOT_EQUIVALENT_NO_FULL_GUARD_CERTIFICATION',online_preprocessing='NOT_RUN',draft='NOT_RUN')
 try:
  r['evidence']=evidence(args);r['machine']=rc_machine(args,root,r)
  if args.stage=='build':
   r['variants']={}
   for name in ('candidate',):
    src=root/('source-'+name);audit=materialize(src);dest=root/name;dest.mkdir();item=dict(evidence=r['evidence'],device=args.device,guard_coverage=COVERAGE['candidate'])
    r['variants'][name]=dict(audit=audit,build=item);r['phase']='build_'+name;write(root/'report.json',r)
    sub=copy.copy(args);sub.runner_kind='correctness'
    try:build_op(sub,dest,item,src)
    finally:write(dest/'report.json',item)
    r['variants'][name]=dict(audit=audit,build=item)
   r['bench_runners']={}
   for name in ('baseline','candidate'):
    proof=bench_proof(r,r,name)['variants'][name];dest=root/('bench-'+name);dest.mkdir();item=dict(evidence=proof,device=args.device)
    sub=copy.copy(args);sub.diagnostic=name=='candidate'
    build_bench(sub,dest,item);r['bench_runners'][name]=item
   r.update(status=STATES['build'],phase='complete');checked_build(r,r)
  else:
   b=read(args.build_report);checked_build(b,r);r['build_report']=identity(args.build_report)
   prerequisites={'host-only':None,'bounded':'host-only','full':'bounded','benchmark':'full','profile':'benchmark'}
   required=prerequisites[args.stage]
   if required:prior(args.prior_report,required,r,b);r['prior_report']=identity(args.prior_report)
   if args.stage in ('benchmark','profile'):
    measured(args,root,r,b,args.stage=='profile');r[args.stage]='COMPLETED'
   else:
    correct(args,root,r,b)
    if args.stage=='full':
     baseline=copy.deepcopy(b);base=r['evidence']['baseline']
     baseline['variants']['candidate']['build'].update(opp=base['opp'],runner=base['runner'])
     shadow=dict(r);shadow['runs']=[]
     try:correct(args,root/'baseline',shadow,baseline,'baseline')
     finally:
      r['baseline_runs']=shadow['runs']
      if (root/'baseline').exists():write(root/'baseline/report.json',shadow)
   r.update(status=STATES[args.stage],phase='complete',npu='NOT_RUN' if args.stage=='host-only' else 'CHECKED')
 except Exception as e:
  r.update(status='BLOCKED',error=f'{type(e).__name__}: {e}')
  if r['npu']=='RUNNING':r['npu']='FAILED_OR_INCOMPLETE'
 finally:
  try:
   for item in r['sources']:verify(item)
   if 'evidence' in r:immutable(r['evidence'])
   if args.build_report:checked_build(read(args.build_report),r)
   elif r.get('status')==STATES['build']:checked_build(r,r)
   q.verify_tree(r.get('machine',{}))
  except Exception as e:r.update(status='BLOCKED',integrity_error=str(e))
  if r['status']=='BLOCKED':r.pop('comparison',None)
  r['artifacts']=[identity(f) for f in sorted(root.rglob('*')) if f.is_file() and f.name in ('runner.log','execution.json','resource-admission.json','actual-0.bin','actual-1.bin','benchmark-last.bin','postcheck.bin')]
  write(root/'report.json',r);write(root/'comparison.json',r.get('comparison',dict(status='NOT_RUN')))
 print(r['status']+': '+str(root/'report.json'));return int(r['status']=='BLOCKED')

def rc_machine(args,root,r):return q.rc.machine(root,r['evidence']['baseline'],args.cann_root,r)

def parse(argv=None):
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('stage',choices=tuple(STATES));ap.add_argument('--output-dir',type=Path,required=True)
 ap.add_argument('--build-report',type=Path);ap.add_argument('--prior-report',type=Path);ap.add_argument('--device',type=int,default=0);ap.add_argument('--msprof',default='msprof')
 ap.add_argument('--cann-root',type=Path,default=Path(os.environ.get('CANN_ROOT','/usr/local/Ascend/cann-9.0.0')))
 base=Path('/home/w00949577/qwen35-dflash-run')
 for name,folder in [('stage4n','w8a8-n128-xor-guard-20261102-063409'),('stage4o','w8a8-n256-host-20261102-081401'),('stage4p','w8a8-n256-correctness-20261103-012123'),('stage4q','w8a8-xor-width-pair-20261103-054336')]:ap.add_argument('--'+name+'-root',type=Path,default=base/folder)
 a=ap.parse_args(argv);a.cann_root=a.cann_root.resolve()
 if a.stage!='build' and a.build_report is None:ap.error('--build-report required')
 if a.stage not in ('build','host-only') and a.prior_report is None:ap.error('--prior-report required')
 if a.device<0:ap.error('negative device')
 return a
if __name__=='__main__':raise SystemExit(main(parse()))
