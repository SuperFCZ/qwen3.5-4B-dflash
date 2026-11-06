#!/usr/bin/env python3
"""Full-K per-channel scheme; strict phase gates, independent numerical contract."""
import argparse,copy,os,shlex,shutil
from pathlib import Path
import numpy as np
from v_common import q,p,rr,NAME,COVERAGE,POLICY,identity,verify,verify_tree,read,write,fresh,sources,evidence,immutable,environment,materialize
from reference_v import prepare,verify_case
from protocol_v import schedule,expected,validate_row,verify_build
from host_v import query
from build_v import build as build_op
from run_stage2b import command
import run_stage4q as rq
from profile_rows import collect
STATES=dict(prepare='STAGE4V_CPU_PREPARED_QUALITY_UNRESOLVED',host='STAGE4V_FULL_K_HOST_FEASIBLE',build='STAGE4V_BUILT_NPU_NOT_RUN',bounded='STAGE4V_BOUNDED_MATCH',full='STAGE4V_FULL_MATCH',fault='STAGE4V_TILE_FAULT_COVERAGE_MATCH',benchmark='STAGE4V_PAIRED_MEASURED',profile='STAGE4V_PROFILE_COMPLETED')
ORDER=(('Group128','FullK','FP16'),('FullK','FP16','Group128'),('FP16','Group128','FullK'))

def same(old,r,keys=('sources','evidence','device','policy')):
 if any(old.get(k)!=r[k] for k in keys):raise ValueError('phase source/device/input binding differs')
 if old['machine']['os']!=r['machine']['os'] or old['machine']['version_files']!=r['machine']['version_files']:raise ValueError('environment drift')

def prior(path,phase,r,b=None):
 if path is None:raise ValueError('missing '+phase+' report')
 old=read(path);same(old,r)
 if old.get('status')!=STATES[phase] or old.get('phase')!='complete':raise ValueError('incomplete '+phase)
 if phase=='prepare':
  case=read(verify(old['case']))
  if case['baseline_case']!=r['evidence']['baseline']['case'] or case['native_reference']!=r['evidence']['native_reference']:raise ValueError('reference/input provenance drift')
  verify_case(case);return old
 if old['prepare_report']!=r['prepare_report'] or old['case']!=r['case']:raise ValueError('Prepare proof differs')
 if phase=='host':
  from resources_v import validate
  from dependency_identity_v import verify_dependency_binding
  raw=read(verify(old['host_query']));row=raw['cases'][0]
  if validate(raw,row)!=old['resources'] or old['resources']['status']!='HOST_RESOURCE_CHECKS_PASS':raise ValueError('Host resources differ')
  wanted=dict(fields={k:row['fields'][k] for k in p.FIELDS},capacity=raw['capacity'],system_workspace_bytes=raw['system_workspace_bytes'],scratch=202240)
  if old['t1']!=wanted:raise ValueError('T1 identity differs')
  verify_dependency_binding(old['sdk']['dependency_binding']);return old
 if old['host_report']!=r['host_report'] or old['t1']!=r['t1']:raise ValueError('Host proof differs')
 if phase=='build':verify_build(old,r);validate_row(old['op_host'],r,host=True);return old
 if old['build_report']!=r['build_report']:raise ValueError('OPP differs')
 previous={'bounded':None,'full':'bounded','fault':'full','benchmark':'fault'}[phase]
 if previous:prior(verify(old['prior_report']),previous,r,b)
 if phase=='benchmark':
  validate_measurements(old,r)
 else:
  if [(x['limit'],x['mode'],x['tile']) for x in old['runs']]!=schedule(phase):raise ValueError('coverage incomplete')
  for row in old['runs']:
   if row['runner']!=b['variants']['fault' if phase=='fault' else 'normal']['build']['runner']:raise ValueError('wrong variant Runner')
   if validate_row(row,r)!=row['validation'] or row['validation']['status']!='MATCH':raise ValueError('prior numerical/fault check invalid')
 return old

def variant_proof(r,b):
 normal=b['variants']['normal']['build']
 return dict(opp=normal['opp'],installed=normal['installed'],files=read(verify(r['case']))['files'],policy=r['evidence']['formal_core_policy'])

def invoke(args,folder,r,b,mode,limit,tile,timed=False,host=False,fault=False,profiled=False):
 if timed and (fault or host or mode!=0 or limit!=0 or tile!=-1):raise ValueError('only normal full Auto OPP eligible for timing')
 folder.mkdir(parents=True);case=read(verify(r['case']));maps=verify_case(case)
 for name in ('a_i8.bin','b_i8.bin','quant_u64_doc_candidate.bin'):(folder/name).symlink_to(verify(case['files'][name]))
 if not host:
  for i in range(2):(folder/f'expected-{i}.bin').write_bytes(expected(maps,mode,limit or r['evidence']['formal_core_policy']['selected'],tile,i))
 variant=b['variants']['fault' if fault else 'normal']['build'];proof=dict(opp=variant['opp'],policy=r['evidence']['formal_core_policy'])
 argv=[verify(variant['runner']),args.device,folder,mode,limit,tile]+(['--benchmark'] if timed else ['--host-only'] if host else [])
 env=environment(proof,args.cann_root)
 if profiled:
  tool=shutil.which(args.msprof)
  if not tool:raise ValueError('msprof missing')
  launcher=folder.parent/'launch.sh';launcher.write_text('#!/usr/bin/env bash\nset -euo pipefail\nexec '+shlex.join(list(map(str,argv)))+' > '+shlex.quote(str(folder/'runner.log'))+' 2>&1\n')
  argv=[tool,'--ai-core=on','--aic-mode=task-based','--aic-metrics=PipeUtilization','--output='+str(folder.parent/'profiler'),'--application='+shlex.join(['bash',str(launcher)])]
  log=folder.parent/'profile.log';r.setdefault('profilers',[]).append(dict(tool=identity(tool),argv=argv))
 else:log=folder/'runner.log'
 try:command(argv,log,env,r,'run_'+folder.parent.name+'_'+folder.name,1800 if profiled else 900)
 except Exception:
  diagnostics=dict(status='RUNNER_FAILED',mode=mode,limit=limit,tile=tile,outputs=[])
  for name in q.bc.OUTPUTS:
   f=folder/name
   if not f.is_file():continue
   raw=f.read_bytes();item=dict(file=identity(f),bytes=len(raw))
   target=expected(maps,mode,limit or r['evidence']['formal_core_policy']['selected'],tile,0 if name=='actual-0.bin' else 1)
   if len(raw)==len(target):
    a=np.frombuffer(raw,'<u2');g=np.frombuffer(target,'<u2');bad=np.flatnonzero(a!=g);item['bit_mismatches']=int(bad.size)
    if bad.size:
     m,n=divmod(int(bad[0]),19456);item['first']=dict(row=m,channel=n,tile=n//256,actual_bits=int(a[bad[0]]),reference_bits=int(g[bad[0]]),active=tile<0 or n//256==tile)
     if mode==0:item['first'].update(dot_int32=int(maps['dot_i32.bin'][m,n]),effective_scale=float(maps['effective_f32.bin'][n]),scale_word=int(maps['quant_u64_doc_candidate.bin'][n//256,n%256]))
   diagnostics['outputs'].append(item)
  write(folder/'failure-diagnostics.json',diagnostics);r.setdefault('failure_diagnostics',[]).append(identity(folder/'failure-diagnostics.json'))
  raise
 row=dict(mode=mode,core_limit=limit,tile_id=tile,limit=limit,tile=tile,runner=variant['runner'],execution=identity(folder/'execution.json'),log=identity(folder/'runner.log'),admission=identity(folder/'resource-admission.json'),profiled=profiled,fault=fault)
 if not host:row['outputs']=[identity(folder/n) for n in (q.bc.OUTPUTS if timed else q.bc.OUTPUTS[:2])]
 row['validation']=validate_row(row,r,timed,host);write(folder/'validation.json',row['validation'])
 if row['validation']['status'] not in ('MATCH','HOST_TILING_ONLY'):raise ValueError('strict new-reference mismatch; see validation.json and raw outputs')
 return row

def validate_native(row,r,timed):
 ex=read(verify(row['execution']));rr.fc.check_execution(ex,'fp16',r['device'],timed)
 if ex.get('independent_cpu_reference') is not True or ex.get('timed_log_stable') is not True or ex.get('poison_verified') is not True or ex.get('verify_calls')!=(4 if timed else 2):raise ValueError('FP16 checks incomplete')
 native=read(verify(r['evidence']['native_reference']));ref=np.fromfile(verify(native['files']['reference_f64.bin']),'<f8').reshape(16,19456)
 raw=[verify(x).read_bytes() for x in row['outputs']]
 if len(raw)!=(4 if timed else 2) or any(x!=raw[0] for x in raw) or timed and raw[0]!=verify(row['frozen_native']).read_bytes():raise ValueError('FP16 repeat/tail differs')
 for x in raw:rr.check(ref,np.frombuffer(x,'<f2').reshape(16,19456))
 return {k:q.statistics_ms(ex['timing'][k]['samples_ms'],30) for k in ('execute_sync','prepare')} if timed else None

def comparison(rows):
 if [(x['round'],x['position'],x['scheme']) for x in rows]!=[(i,j,n) for i,o in enumerate(ORDER) for j,n in enumerate(o)]:raise ValueError('three-way pairing incomplete')
 if any(x['profiled'] for x in rows):raise ValueError('profiled samples excluded')
 for n in ORDER[0]:
  if len({x['workspace_bytes'] for x in rows if x['scheme']==n})!=1:raise ValueError('Workspace drift')
 stats_={n:{k:q.statistics_ms([v for row in rows if row['scheme']==n for v in row['timing'][k]['samples_ms']],90) for k in ('execute_sync','prepare')} for n in ORDER[0]}
 med=lambda n:stats_[n]['execute_sync']['median_ms']
 ratios=[next(x['timing']['execute_sync']['median_ms'] for x in rows if x['round']==i and x['scheme']=='Group128')/next(x['timing']['execute_sync']['median_ms'] for x in rows if x['round']==i and x['scheme']=='FullK') for i in range(3)]
 return dict(statistics=stats_,group128_over_full_k=med('Group128')/med('FullK'),round_ratios=ratios,full_k_over_fp16=med('FullK')/med('FP16'),performance_decision='GO_FURTHER_REVIEW' if med('Group128')/med('FullK')>=1.03 and all(x>1 for x in ratios) else 'NO_GO_NO_MATERIAL_IMPROVEMENT',quality_go='UNRESOLVED',production='NOT_APPROVED',scope='RESIDENT_INPUT_DIFFERENT_QUANTIZATION_AND_GUARD_LIFETIMES',historical_samples_used=False)

def validate_measurements(old,r):
 validate_native(old['prechecks']['FP16'],r,False)
 q.validate_run(old['prechecks']['Group128'],r['evidence']['qproof'],r['device'],False)
 if verify(old['native_frozen']).read_bytes()!=verify(old['prechecks']['FP16']['outputs'][0]).read_bytes():raise ValueError('fresh Native admission binding differs')
 for row in old['runs']:
  if row['scheme']=='FullK':
   val=validate_row(row,r,True)
   if val['status']!='MATCH' or val['timing']!=row['timing']:raise ValueError('Full-K timing/numerics invalid')
  elif row['scheme']=='Group128':q.validate_run(row,r['evidence']['qproof'],r['device'],True)
  elif row['scheme']!='FP16' or row['frozen_native']!=old['native_frozen'] or validate_native(row,r,True)!=row['timing']:raise ValueError('native samples differ')
 expected_summary=comparison(old['runs']);expected_summary['quality_report']=read(verify(r['case']))['quality']
 if expected_summary!=old['comparison']:raise ValueError('paired summary differs')

def measured(args,root,r,b,profiled):
 proof=r['evidence'];native=read(verify(proof['native_reference']));nb=dict(native=proof['native_build']);r['runs']=[]
 # Both historical schemes receive fresh untimed numerical admission before ANY formal timing.
 shadow=dict(device=args.device,evidence=proof['qproof'],output_dir=str(root))
 rb=read(verify(read(verify(proof['stage4r_benchmark']))['build_report']))
 wb=dict(variants={'N256':rb['w8_runner_build']})
 if not profiled:
  npass=rr.invoke_native(args,root/'precheck-fp16',shadow,nb,native,False);validate_native(npass,r,False)
  frozen=root/'native-admitted.bin';frozen.write_bytes(verify(npass['outputs'][0]).read_bytes());r['native_frozen']=identity(frozen)
  r['prechecks']=dict(FP16=npass,Group128=rq.invoke(args,root/'precheck-group128',shadow,wb,'N256',False))
 else:r['native_frozen']=read(args.prior_report)['native_frozen']
 for i,order in enumerate((ORDER[0],) if profiled else ORDER):
  for pos,name in enumerate(order):
   folder=root/f'round-{i}'/name/'case';r['phase']=f'round-{i}-{name}';write(root/'report.json',r)
   if name=='FullK':
    row=invoke(args,folder,r,b,0,0,-1,True,profiled=profiled);row['timing']=row['validation']['timing'];row['workspace_bytes']=row['validation']['workspace_bytes']
   elif name=='Group128':row=rq.invoke(args,folder,shadow,wb,'N256',True,profiled)
   elif not profiled:
    row=rr.invoke_native(args,folder,shadow,nb,native,True,r['native_frozen']);row['frozen_native']=r['native_frozen']
   else:
    folder.mkdir(parents=True)
    for key,item in native['files'].items():(folder/key).symlink_to(verify(item))
    (folder/'golden_fp16.bin').symlink_to(verify(r['native_frozen']))
    argv=[verify(nb['native']['runner']),args.device,folder,5,30,'--continuous']
    tool=shutil.which(args.msprof)
    if not tool:raise ValueError('msprof missing')
    launch=folder.parent/'launch.sh';launch.write_text('#!/usr/bin/env bash\nset -euo pipefail\nexec '+shlex.join(list(map(str,argv)))+' > '+shlex.quote(str(folder/'runner.log'))+' 2>&1\n')
    argv=[tool,'--ai-core=on','--aic-mode=task-based','--aic-metrics=PipeUtilization','--output='+str(folder.parent/'profiler'),'--application='+shlex.join(['bash',str(launch)])]
    r.setdefault('profilers',[]).append(dict(tool=identity(tool),argv=argv))
    command(argv,folder.parent/'profile.log',rr.native_env(args.cann_root),r,'profile-native',1800)
    row=dict(execution=identity(folder/'execution.json'),log=identity(folder/'runner.log'),outputs=[identity(folder/n) for n in rr.fc.OUTPUTS],frozen_native=r['native_frozen'])
    row['timing']=validate_native(row,r,True);row['workspace_bytes']=read(verify(row['execution']))['workspace_bytes']
   row.update(scheme=name,round=i,position=pos,profiled=profiled)
   if name!='FullK':row.update(commands=copy.deepcopy(shadow.get('commands',{})),profilers=copy.deepcopy(shadow.get('profilers',{})))
   r['runs'].append(row);write(root/'report.json',r)
 if profiled:
  r['raw_profile']={n:collect(root/'round-0'/n/'profiler',NAME if n=='FullK' else p.NAME) for n in ('Group128','FullK')}
  r['raw_profile']['FP16']=collect(root/'round-0'/'FP16'/'profiler','MatMul')
  r['native_profile_note']='Normalized MatMul substring filter; preserved names/tasks must be inspected separately. Missing or auxiliary counters remain UNAVAILABLE; raw CSV retained. Not mixed with unprofiled samples.'
 else:
  r['comparison']=comparison(r['runs']);r['comparison']['quality_report']=read(verify(r['case']))['quality']

def main(args):
 root=fresh(args.output_dir,[args.cann_root,args.stage4p_root,args.stage4r_root]+[x.parent for x in (args.prepare_report,args.host_report,args.build_report,args.prior_report) if x])
 r=dict(stage='4V',phase='evidence',status='NOT_RUN',sources=sources(),device=args.device,policy=POLICY,npu='NOT_RUN',benchmark='NOT_RUN',profile='NOT_RUN',quality_go='UNRESOLVED',production='NOT_APPROVED',online_preprocessing='NOT_RUN',online_activation_quantization='NOT_RUN',online_scale_generation='NOT_RUN',draft='NOT_RUN',decode='NOT_RUN',output_dir=str(root))
 try:
  r['evidence']=evidence(args);r['machine']=q.rc.machine(root,r['evidence']['baseline'],args.cann_root,r)
  r['phase']=args.stage
  if args.stage=='prepare':prepare(root/'data',r['evidence']);r['case']=identity(root/'data/case.json')
  else:
   pr=prior(args.prepare_report,'prepare',r);r.update(prepare_report=identity(args.prepare_report),case=pr['case'])
   if args.stage=='host':
    verify_case(read(verify(r['case'])),numerical=True);query(args,root,r)
   else:
    hr=prior(args.host_report,'host',r);r.update(host_report=identity(args.host_report),t1=hr['t1'])
    if args.stage=='build':
     r['variants']={}
     for name,diag in (('normal',False),('fault',True)):
      r['phase']='build_'+name
      audit=materialize(root/('source-'+name),diag);dest=root/name;dest.mkdir();item=dict(evidence=dict(r['evidence'],t1=r['t1']),device=args.device)
      r['variants'][name]=dict(audit=audit,build=item);sub=copy.copy(args);sub.runner_kind='runner'
      try:build_op(sub,dest,item,root/('source-'+name))
      finally:write(dest/'report.json',item)
     r['npu']='NOT_RUN';r['op_host']=invoke(args,root/'op-host',r,r,0,0,-1,host=True)
    else:
     b=prior(args.build_report,'build',r);r['build_report']=identity(args.build_report)
     previous={'bounded':None,'full':'bounded','fault':'full','benchmark':'fault','profile':'benchmark'}[args.stage]
     if previous:prior(args.prior_report,previous,r,b);r['prior_report']=identity(args.prior_report)
     r['npu']='RUNNING'
     if args.stage in ('benchmark','profile'):measured(args,root,r,b,args.stage=='profile');r[args.stage]='COMPLETED'
     else:
      r['runs']=[]
      for limit,mode,tile in schedule(args.stage):
       row=invoke(args,root/f'l{limit}-t{tile}-m{mode}',r,b,mode,limit,tile,fault=args.stage=='fault');r['runs'].append(row);write(root/'report.json',r)
      if args.stage=='fault':r['fault_coverage']=dict(protocol=COVERAGE,boundaries=3,words_per_boundary=384,single_bit_injections_per_sweep=36864,repeats=2,total_injections=73728,abort_witnesses=3,not_group128_equivalent=True)
     r['npu']='NEW_REFERENCE_AND_SCOPED_SAFETY_CHECKED'
  r.update(status=STATES[args.stage],phase='complete')
 except Exception as e:
  r.update(status='BLOCKED',error=f'{type(e).__name__}: {e}')
  if r['npu']=='RUNNING':r['npu']='FAILED_OR_INCOMPLETE'
 finally:
  try:
   for x in r['sources']:verify(x)
   if 'evidence' in r:immutable(r['evidence'])
   if 'case' in r:verify_case(read(verify(r['case'])))
   if args.build_report:verify_build(read(args.build_report),r)
   elif args.stage=='build' and r['status']==STATES['build']:verify_build(r,r)
   verify_tree(r.get('machine',{}))
  except Exception as e:r.update(status='BLOCKED',integrity_error=str(e))
  if r['status']=='BLOCKED':r.pop('comparison',None)
  if args.stage=='host':
   write(root/'tiling-report.json',r);write(root/'resource-report.json',r.get('resources',dict(status='NOT_RUN')));write(root/'sdk-binding.json',r.get('sdk',dict(status='NOT_RUN')))
  write(root/'report.json',r);write(root/'comparison.json',r.get('comparison',dict(status='NOT_RUN')))
  write(root/'preprocessing-inventory.json',dict(status='ONLINE_PREPROCESSING_NOT_RUN',FullK=dict(A=40960,B=49807360,scale=155648,output=622592),Group128=dict(A=40960,B=49807360,scale=3112960,output=622592),FP16=dict(A=81920,B=99614720,output=622592),excluded=['offline W_eff/requantization/packing','online Activation quantization','combined Scale/encoding','H2D/D2H','Draft/Decode'],memory_note='payload bytes, not peak process memory or measured DDR traffic'))
 print(r['status']+': '+str(root/'report.json'));return int(r['status']=='BLOCKED')

def parse(argv=None):
 ap=argparse.ArgumentParser(description=__doc__,allow_abbrev=False);ap.add_argument('stage',choices=tuple(STATES));ap.add_argument('--output-dir',type=Path,required=True)
 for n in ('prepare-report','host-report','build-report','prior-report'):ap.add_argument('--'+n,type=Path)
 ap.add_argument('--device',type=int,default=0);ap.add_argument('--msprof',default='msprof');ap.add_argument('--cann-root',type=Path,default=Path(os.environ.get('CANN_ROOT','/usr/local/Ascend/cann-9.0.0')))
 base=Path('/home/w00949577/qwen35-dflash-run')
 for n,d in [('stage4n','w8a8-n128-xor-guard-20261102-063409'),('stage4o','w8a8-n256-host-20261102-081401'),('stage4p','w8a8-n256-correctness-20261103-012123'),('stage4r','fp16-integrity-fix-20261103-085750')]:ap.add_argument('--'+n+'-root',type=Path,default=base/d)
 a=ap.parse_args(argv);a.cann_root=a.cann_root.resolve()
 needed=[]
 if a.stage!='prepare':needed+=['prepare_report']
 if a.stage not in ('prepare','host'):needed+=['host_report']
 if a.stage not in ('prepare','host','build'):needed+=['build_report']
 if a.stage in ('full','fault','benchmark','profile'):needed+=['prior_report']
 for n in needed:
  if getattr(a,n) is None:ap.error('--'+n.replace('_','-')+' required')
 if a.device<0:ap.error('negative device')
 return a
if __name__=='__main__':raise SystemExit(main(parse()))
