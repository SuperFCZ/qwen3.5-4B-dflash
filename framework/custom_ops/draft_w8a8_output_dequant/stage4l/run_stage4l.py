#!/usr/bin/env python3
"""Crossed width/Guard experiment; only N128 Weak is a new compute OPP."""
import argparse,copy,os,shlex,shutil
from pathlib import Path
from l_common import (ORDER,SCOPE,bc,rc,identity,verify,write,fresh,read,sources,evidence,
 environment,immutable,check_execution,plan,validate_run,comparison,NAMES,POLICY,materialize,verify_audit,k)
from build_runner_l import build
from build_weak import build as build_weak
from profile_rows import collect, review_historical
from run_stage2b import command

STATES=dict(build='STAGE4L_RUNNERS_BUILT',smoke='STAGE4L_SMOKE_MATCH',benchmark='STAGE4L_PAIRED_MEASURED',profile='STAGE4L_PROFILE_COMPLETED')

def verify_new_weak(b):
 verify_audit(b['source_audit']);w=b['weak_opp'];k.verify_tree(w)
 if w.get('status')!='STAGE4L_WEAK_OPP_BUILT' or k.jb.installed(Path(w['opp']))!=w['installed']:raise ValueError('new weak OPP identity differs')
 expected={Path(x['generated']['path']).parent.name+'/'+Path(x['generated']['path']).name:verify(x['generated']).read_bytes() for x in b['source_audit']['files'] if Path(x['generated']['path']).suffix in ('.cpp','.h')}
 expected['op_host/t1_contract.h']=k.jc.contract_header(w['evidence']['t1']).encode()
 actual={Path(x['path']).parent.name+'/'+Path(x['path']).name:verify(x).read_bytes() for x in w['generated_sources']}
 if (Path(w['opp']).parent/'operator.json').read_bytes()!=verify(next(x['generated'] for x in b['source_audit']['files'] if Path(x['generated']['path']).name=='operator.json')).read_bytes():raise ValueError('operator ABI changed')
 if expected!=actual:raise ValueError('compiled weak source does not equal exact audit')
 full=b['evidence']['variants']['N128Full']['installed']['kernels']
 if {x['binary']['sha256'] for x in full}=={x['binary']['sha256'] for x in w['installed']['kernels']}:raise ValueError('identical full/weak binaries: contrast unestablished')

def active(proof,b):
 verify_new_weak(b);result=copy.deepcopy(proof)
 result['variants']['N128Weak'].update(opp=b['weak_opp']['opp'],installed=b['weak_opp']['installed'])
 immutable(result);return result

def prior(path,stage,r):
 if path is None:raise ValueError('missing '+stage+' proof')
 p=read(path)
 if p.get('status')!=STATES[stage] or p.get('phase')!='complete' or any(p.get(k)!=r[k] for k in ('sources','evidence','device')):raise ValueError('prior phase binding mismatch')
 if p.get('machine',{}).get('os')!=r.get('machine',{}).get('os') or p.get('machine',{}).get('version_files')!=r.get('machine',{}).get('version_files'):raise ValueError('server/driver/CANN environment changed between stages')
 if stage=='build':
  if set(p.get('variants',{}))!=set(NAMES):raise ValueError('four isolated Runner builds required')
  for name,b in p['variants'].items():
   if b.get('status')!='STAGE4L_RUNNER_BUILT' or b.get('evidence')!=r['active']['variants'][name]:raise ValueError('Runner variant binding mismatch')
   if b['linked_libraries']['LIB_custom']!=r['active']['variants'][name]['installed']['api']:raise ValueError('Runner links wrong private OPP')
 if stage!='build':
  if p.get('build_report')!=r['build_report']:raise ValueError('Runner build mismatch')
  expected=list(NAMES) if stage=='smoke' else [v for row in ORDER for v in row]
  if [x['variant'] for x in p['runs']]!=expected:raise ValueError('prior coverage incomplete')
  for row in p['runs']:validate_run(row,r['active'],r['device'],stage=='benchmark')
  if stage=='benchmark' and comparison(p['runs'])!=p['comparison']:raise ValueError('paired summary mismatch')
 return p

def invoke(args,folder,r,b,name,timed,profiled=False):
 folder.mkdir(parents=True);p=r['active']['variants'][name]
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
 row=dict(variant=name,safety_protocol=POLICY[name],status='FROZEN_GOLDEN_MATCH',execution=ex_id,outputs=bc.verify_outputs(p,folder,timed),
  log=identity(folder/'runner.log'),plan=plan((folder/'runner.log').read_text(),name,p),admission=identity(folder/'resource-admission.json'),
  timing=stats,workspace_bytes=ex['workspace_bytes'],profiled=profiled,formal_latency_evidence=timed and not profiled)
 validate_run(row,r['active'],args.device,timed,profiled);return row

def main(args):
 protected=[args.cann_root,args.stage4b_root,args.stage4j_root,args.stage4i_root,args.stage4c_report.parent.parent,args.stage4d_root,args.stage4k_root]
 protected += [p.parent for p in (args.runner_build_report,args.smoke_report,args.benchmark_report) if p]
 root=fresh(args.output_dir,protected)
 r=dict(stage='4L',safety_protocols=POLICY,weak_adoption='NOT_ELIGIBLE_FOR_PRODUCTION',status='NOT_RUN',phase='evidence',device=args.device,sources=sources(),output_dir=str(root),scope=SCOPE,
  kernel_build='NOT_RUN_REUSED_ACCEPTED_OPPS',opp_installation='NOT_RUN',npu='NOT_RUN',benchmark='NOT_RUN',profile='NOT_RUN',
  online_activation_quantization='NOT_RUN',combined_scale_generation='NOT_RUN',draft='NOT_RUN',decode='NOT_RUN')
 inv=dict(status='ONLINE_PREPROCESSING_NOT_RUN');cmp=dict(status='NOT_RUN');profile=dict(status='NOT_RUN')
 reportname={'build':'build-report.json','smoke':'smoke-report.json','benchmark':'benchmark-report.json','profile':'profile-report.json'}[args.stage]
 try:
  r['evidence']=evidence(args);r['active']=r['evidence'];r['historical_csv_review']=review_historical(r['evidence']['stage4k_profile']);inv=bc.preprocessing(r['evidence']['variants']['N64Full']);inv.update(scope=SCOPE,packing_by_variant={n:p['files'] for n,p in r['evidence']['variants'].items()})
  inv['safety_protocols']=POLICY
  for component in inv['components']:
   if component['name']=='W8A8_Cube_VDEQ16_FP32_reduction':component['status']='Full/Weak coverage differs; see safety_protocols'
  r['machine']=rc.machine(root,r['evidence']['variants']['N64Full'],args.cann_root,r)
  if args.stage=='build':
   r['variants']={}
   audit=materialize(root/'weak-source');r['source_audit']=audit
   weakdir=root/'weak-opp';weakdir.mkdir();w=dict(evidence=r['evidence']['variants']['N128Full'],device=args.device,safety_protocol=POLICY['N128Weak'])
   try:build_weak(args,weakdir,w,root/'weak-source')
   finally:write(weakdir/'build-report.json',w)
   r['weak_opp']=w;r['kernel_build']='N128_WEAK_ONLY_COMPLETED';r['opp_installation']='NEW_N128_WEAK_PRIVATE_ONLY'
   verify_new_weak(r)
   r['active']=active(r['evidence'],r)
   for name,p in r['active']['variants'].items():
    dest=root/name;dest.mkdir();local=dict(evidence=p,device=args.device);sub=copy.copy(args);sub.variant=name
    build(sub,dest,local);write(dest/'runner-build-report.json',local)
    r['variants'][name]=local
   r.update(status=STATES['build'],phase='complete',runner_compile='COMPLETED')
  else:
   b=read(args.runner_build_report);r['active']=active(r['evidence'],b);b=prior(args.runner_build_report,'build',r);r['build_report']=identity(args.runner_build_report);r['runs']=[]
   if args.stage=='benchmark':prior(args.smoke_report,'smoke',r);r['smoke_report']=identity(args.smoke_report)
   if args.stage=='profile':
    prior(args.benchmark_report,'benchmark',r);r['benchmark_report']=identity(args.benchmark_report)
   orders=ORDER if args.stage=='benchmark' else (NAMES,)
   for i,order in enumerate(orders):
    for pos,name in enumerate(order):
     r.update(phase=f'round{i}_{name}',npu='RUNNING');write(root/reportname,r)
     row=invoke(args,root/f'round-{i}'/name/'case',r,b,name,args.stage!='smoke',args.stage=='profile')
     row.update(round=i,position=pos);r['runs'].append(row);write(root/reportname,r)
     immutable(r.get('active',r['evidence']))
   if args.stage=='benchmark':
    cmp=comparison(r['runs']);r['comparison']=cmp;r['benchmark']='MEASURED_UNPROFILED'
   if args.stage=='profile':
    profile={name:collect(root/'round-0'/name/'profiler', 'DFlashW8A8FullGate' if name.startswith('N64') else 'DFlashW8A8N128Gate') for name in NAMES}
    r.update(raw_profile=profile,profile='COLLECTED_SEPARATELY',formal_latency_evidence=False)
   r.update(status=STATES[args.stage],phase='complete',npu='GOLDEN_MATCH_UNDER_SCOPED_GUARDS')
 except Exception as e:
  r.update(status='FAILED',error=f'{type(e).__name__}: {e}')
  if r['npu']=='RUNNING':r['npu']='FAILED_OR_INCOMPLETE'
 finally:
  try:
   for item in r['sources']:verify(item)
   if 'evidence' in r:immutable(r.get('active',r['evidence']))
   if 'weak_opp' in r:verify_new_weak(r)
   for item in r.get('machine',{}).get('version_files',[]):verify(item['file'])
  except Exception as e:r.update(status='FAILED',integrity_error=str(e))
  if r['status']=='FAILED':cmp=dict(status='INVALID_OR_NOT_RUN',reason=r.get('error',r.get('integrity_error')))
  r['artifacts']=[identity(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name in ('runner.log','execution.json','actual-0.bin','actual-1.bin','benchmark-last.bin','postcheck.bin')]
  write(root/reportname,r);write(root/'comparison.json',cmp);write(root/'preprocessing-inventory.json',inv)
 print(r['status']+': '+str(root/reportname));return int(r['status']=='FAILED')


def parse():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('build','smoke','benchmark','profile'))
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--runner-build-report',type=Path);p.add_argument('--smoke-report',type=Path);p.add_argument('--benchmark-report',type=Path)
    p.add_argument('--msprof',default='msprof')
    for name,default in [('stage4b','w8a8-full-gate-20261027-093617'),('stage4a','w8a8-core-policy-20261027-083223'),('stage3b4','w8a8-twenty-group-20261027-055348'),('stage3b3','w8a8-two-group-20261027-042044'),('stage3b2','w8a8-local-nz-20261027-031856'),('stage3b1','w8a8-stage3b-20261027-024844'),('stage3a','w8a8-stage3a-20261027-011523'),('stage2b','w8a8-stage2b-20261026-100533')]:
        p.add_argument('--'+name+'-root',type=Path,default=Path('/home/w00949577/qwen35-dflash-run')/default)
    for name in ('build','basic','zero','nonpower'):p.add_argument('--'+name+'-report',type=Path)
    p.add_argument('--stage2c-report',type=Path,default=Path('/home/w00949577/qwen35-dflash-run/w8a8-stage2c-20261026-104943/report.json'))
    p.add_argument('--device',type=int,default=0);p.add_argument('--cann-root',type=Path,default=Path(os.environ.get('CANN_ROOT','/usr/local/Ascend/cann-9.0.0')))
    p.add_argument('--stage4j-root',type=Path,default=Path('/home/w00949577/qwen35-dflash-run/w8a8-n128-t1-20261030-092118'))
    p.add_argument('--stage4i-root',type=Path,default=Path('/home/w00949577/qwen35-dflash-run/w8a8-n128-host-shareub-20261030-083337'))
    p.add_argument('--stage4c-report',type=Path,default=Path('/home/w00949577/qwen35-dflash-run/w8a8-baseline-20261027-111748/benchmark/benchmark-report.json'))
    p.add_argument('--stage4d-root',type=Path,default=Path('/home/w00949577/qwen35-dflash-run/w8a8-guard-ab-20261029-012635'))
    p.add_argument('--stage4k-root',type=Path,default=Path('/home/w00949577/qwen35-dflash-run/w8a8-n64-n128-paired-20261102-013734'))
    args=p.parse_args();args.cann_root=args.cann_root.resolve()
    for name in ('build','basic','zero','nonpower'):
        if getattr(args,name+'_report') is None:setattr(args,name+'_report',args.stage2b_root/name/'report.json')
    if args.device<0:p.error('negative device')
    return args

if __name__=='__main__':raise SystemExit(main(parse()))
