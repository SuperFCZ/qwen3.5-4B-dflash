#!/usr/bin/env python3
"""New FP16 numerical admission and paired resident-input experiment, no legacy Profile dependency."""
import argparse,copy,os,shutil
from pathlib import Path
import numpy as np
from reference_r import HERE,q,fc,POLICY,prepare,check
import run_stage4q as qq
from build_q import build as build_w8
from run_stage2b import command
STATES={'prepare':'STAGE4R_CPU_PREPARED','build':'STAGE4R_RUNNERS_BUILT','smoke':'STAGE4R_NUMERICS_ADMITTED','benchmark':'STAGE4R_PAIRED_MEASURED'}

def sources():
 rows={x['path']:x for x in q.sources()}
 for folder in (HERE,fc.HERE):
  for f in folder.rglob('*'):
   if f.is_file() and '__pycache__' not in f.parts and '.pytest_cache' not in f.parts:rows[str(f.resolve())]=q.identity(f)
 return [rows[x] for x in sorted(rows)]

def prior(path,stage,r):
 if path is None:raise ValueError('missing '+stage+' report')
 p=q.read(path)
 if p.get('status')!=STATES[stage] or p.get('phase')!='complete' or any(p.get(k)!=r[k] for k in ('sources','evidence','device','candidate','policy')):raise ValueError('prior admission differs')
 return p

def native_env(cann):return q.common.env(cann)

def build_native(args,root,r):
 candidates={p.resolve() for p in [args.cann_root/rel for rel in ('include/aclnnop/aclnn_mm.h','aarch64-linux/include/aclnnop/aclnn_mm.h','opp/built-in/op_impl/ai_core/tbe/op_api/include/aclnn_mm.h')] if p.is_file()}
 if len(candidates)!=1:raise ValueError('selected SDK aclnn_mm.h missing/ambiguous in fixed public locations')
 header=candidates.pop()
 if args.cann_root not in header.parents:raise ValueError('native header outside selected SDK')
 r['native_header']=q.identity(header)
 r['native_tools']={name:q.identity(shutil.which(name)) for name in ('cmake','c++')}
 e=native_env(args.cann_root)
 command(['cmake','-S',HERE,'-B',root/'native-build','-DCANN_ROOT='+str(args.cann_root),'-DNATIVE_MM_HEADER='+str(header.resolve()),'-DCOMMON_INCLUDE='+str(q.common.QUANT/'test'),'-DCMAKE_EXPORT_COMPILE_COMMANDS=ON'],root/'cmake.log',e,r,'native_configure')
 command(['cmake','--build',root/'native-build','--parallel'],root/'compile.log',e,r,'native_build')
 runner=root/'native-build/stage4r_fp16';command(['ldd',runner],root/'ldd.log',e,r,'native_ldd')
 if 'not found' in (root/'ldd.log').read_text():raise ValueError('native unresolved library')
 links={}
 for line in (root/'native-build/CMakeCache.txt').read_text().splitlines():
  if line.startswith('LIB_') and ':FILEPATH=' in line:
   key,value=line.split(':FILEPATH=',1);item=q.identity(value)
   if args.cann_root not in Path(item['path']).parents:raise ValueError('native library outside SDK')
   links[key]=item
 if set(links)!={'LIB_opapi','LIB_ascendcl','LIB_nnopbase'}:raise ValueError('native libraries differ')
 return dict(runner=q.identity(runner),libraries=links,header=r['native_header'],tools=r['native_tools'],compile_commands=q.identity(root/'native-build/compile_commands.json'))

def invoke_native(args,folder,r,b,prepared,timed,golden=None):
 folder.mkdir(parents=True)
 for name,item in prepared['files'].items():(folder/name).symlink_to(q.verify(item))
 if timed:
  if golden is None:raise ValueError('independently admitted Golden required before timing')
  (folder/'golden_fp16.bin').symlink_to(q.verify(golden))
 argv=[q.verify(b['native']['runner']),args.device,folder]+(['5','30','--continuous'] if timed else [])
 command(argv,folder/'runner.log',native_env(args.cann_root),r,str(folder.relative_to(Path(r['output_dir']))),600)
 ex_id=q.identity(folder/'execution.json');ex=q.read(q.verify(ex_id));fc.check_execution(ex,'fp16',args.device,timed)
 if ex.get('independent_cpu_reference') is not True or ex.get('poison_verified') is not True or ex.get('timed_log_stable') is not True or ex.get('prepare_calls')!=(38 if timed else 2) or ex.get('verify_calls')!=(4 if timed else 2):raise ValueError('native protocol incomplete')
 ref=np.fromfile(q.verify(prepared['files']['reference_f64.bin']),'<f8').reshape(16,19456)
 names=fc.OUTPUTS if timed else fc.OUTPUTS[:2];out=[q.identity(folder/n) for n in names];metrics=[];first=None
 for item in out:
  raw=q.verify(item).read_bytes();metrics.append(check(ref,np.frombuffer(raw,'<f2').reshape(16,19456)))
  if first is None:first=raw
  if raw!=first or golden is not None and raw!=q.verify(golden).read_bytes():raise ValueError('native strict repeat/frozen Golden drift')
 return dict(status='INDEPENDENT_CPU_NUMERICS_PASS',execution=ex_id,outputs=out,numerics=metrics,workspace_bytes=ex['workspace_bytes'],
  timing={k:q.statistics_ms(ex['timing'][k]['samples_ms'],30) for k in ('execute_sync','prepare')} if timed else None,profiled=False)

def validate_smoke(path,r,b,prepared):
 s=prior(path,'smoke',r)
 if s['machine']['os']!=r['machine']['os'] or s['machine']['version_files']!=r['machine']['version_files']:raise ValueError('smoke environment changed')
 if s['build_report']!=r['build_report'] or s['prepare_report']!=r['prepare_report']:raise ValueError('smoke build/input mismatch')
 ref=np.fromfile(q.verify(prepared['files']['reference_f64.bin']),'<f8').reshape(16,19456)
 native=s['native'];ex=q.read(q.verify(native['execution']));fc.check_execution(ex,'fp16',r['device'],False)
 if ex.get('independent_cpu_reference') is not True or ex.get('poison_verified') is not True or ex.get('prepare_calls')!=2 or ex.get('verify_calls')!=2:raise ValueError('no independent native admission')
 outputs=[q.verify(x).read_bytes() for x in native['outputs']]
 if len(outputs)!=2 or outputs[0]!=outputs[1]:raise ValueError('native repeat proof missing')
 for raw in outputs:check(ref,np.frombuffer(raw,'<f2').reshape(16,19456))
 if q.verify(s['frozen_native']).read_bytes()!=outputs[0]:raise ValueError('frozen native identity changed')
 if s['w8']['variant']!=r['candidate']:raise ValueError('wrong W8 candidate in smoke')
 q.validate_run(s['w8'],r['evidence'],r['device'],False)
 return s

def main(args):
 root=q.fresh(args.output_dir,[args.cann_root,args.stage4n_root,args.stage4p_root]+[p.parent for p in (args.prepare_report,args.build_report,args.smoke_report) if p])
 r=dict(stage='4R',status='NOT_RUN',phase='evidence',sources=sources(),device=args.device,candidate=args.candidate,policy=POLICY,output_dir=str(root),
  npu='NOT_RUN',benchmark='NOT_RUN',profile='NOT_RUN',draft='NOT_RUN',decode='NOT_RUN',online_preprocessing='NOT_RUN',scope='RESIDENT_INPUT_IMPLEMENTATIONS_DIFFERENT_NUMERICAL_SEMANTICS')
 try:
  r['evidence']=q.evidence(args);r['machine']=q.rc.machine(root,r['evidence']['variants'][args.candidate],args.cann_root,r)
  if args.stage=='prepare':
   data=prepare(root/'data',r['evidence']);q.write(root/'data/reference.json',data);r['reference']=q.identity(root/'data/reference.json')
  else:
   p=prior(args.prepare_report,'prepare',r);data=q.read(q.verify(p['reference']))
   if data['policy']!=POLICY:raise ValueError('reference policy changed')
   r.update(prepare_report=q.identity(args.prepare_report),reference=p['reference'])
   if args.stage=='build':
    r['native']=build_native(args,root,r);sub=copy.copy(args);sub.variant=args.candidate;dest=root/'w8-build';dest.mkdir();w=dict(evidence=r['evidence']['variants'][args.candidate],device=args.device)
    build_w8(sub,dest,w);r['w8_runner_build']=w
   else:
    b=prior(args.build_report,'build',r)
    if b['w8_runner_build']['evidence']!=r['evidence']['variants'][args.candidate] or b['w8_runner_build']['linked_libraries']['LIB_custom']!=r['evidence']['variants'][args.candidate]['installed']['api']:raise ValueError('W8 private OPP binding differs')
    if b['prepare_report']!=r['prepare_report']:raise ValueError('build uses another reference')
    if b['machine']['os']!=r['machine']['os'] or b['machine']['version_files']!=r['machine']['version_files']:raise ValueError('environment drift')
    r['build_report']=q.identity(args.build_report);wrapper=dict(variants={args.candidate:b['w8_runner_build']})
    if args.stage=='smoke':
     r['npu']='RUNNING';r['native']=invoke_native(args,root/'fp16',r,b,data,False)
     # Freeze only after independent full-array checks and both repeat proofs.
     path=root/'admitted_fp16.bin';path.write_bytes(q.verify(r['native']['outputs'][0]).read_bytes());r['frozen_native']=q.identity(path)
     r['w8']=qq.invoke(args,root/'w8',r,wrapper,args.candidate,False)
    else:
     smoke=validate_smoke(args.smoke_report,r,b,data);r['smoke_report']=q.identity(args.smoke_report);r['runs']=[]
     for i,order in enumerate((('FP16','W8'),('W8','FP16'),('FP16','W8'))):
      for pos,name in enumerate(order):
       r.update(phase=f'round{i}_{name}',npu='RUNNING');q.write(root/'benchmark-report.json',r)
       row=invoke_native(args,root/f'round-{i}'/'fp16',r,b,data,True,smoke['frozen_native']) if name=='FP16' else qq.invoke(args,root/f'round-{i}'/'w8',r,wrapper,args.candidate,True)
       row.update(scheme=name,round=i,position=pos);r['runs'].append(row)
     for scheme in ('FP16','W8'):
      if len({row['workspace_bytes'] for row in r['runs'] if row['scheme']==scheme})!=1:raise ValueError('Workspace drift across rounds')
     stats={name:{key:q.statistics_ms([v for row in r['runs'] if row['scheme']==name for v in row['timing'][key]['samples_ms']],90) for key in ('execute_sync','prepare')} for name in ('FP16','W8')}
     ref=np.fromfile(q.verify(data['files']['reference_f64.bin']),'<f8').reshape(16,19456)
     w8=np.fromfile(q.verify(smoke['w8']['outputs'][0]),'<f2').reshape(16,19456).astype(np.float64)
     diff=w8-ref
     ratios=[next(row['timing']['execute_sync']['median_ms'] for row in r['runs'] if row['round']==i and row['scheme']=='W8')/next(row['timing']['execute_sync']['median_ms'] for row in r['runs'] if row['round']==i and row['scheme']=='FP16') for i in range(3)]
     r['comparison']=dict(round_w8_over_fp16=ratios,statistics=stats,w8_over_fp16=stats['W8']['execute_sync']['median_ms']/stats['FP16']['execute_sync']['median_ms'],candidate=args.candidate,
      fp16_error=smoke['native']['numerics'],w8_vs_fp64_diagnostic=dict(rmse=float(np.sqrt(np.mean(diff*diff))),max_abs=float(np.abs(diff).max()),relative_l2=float(np.linalg.norm(diff)/np.linalg.norm(ref))),
      scope=r['scope'],historical_samples_used=False,numerical_schemes_not_bitwise_equivalent=True,internal_guard_protocols_differ=True)
     r['benchmark']='MEASURED_UNPROFILED'
  r.update(status=STATES[args.stage],phase='complete',npu='NOT_RUN' if args.stage in ('prepare','build') else 'NUMERICS_CHECKED')
 except Exception as e:r.update(status='BLOCKED',error=f'{type(e).__name__}: {e}')
 finally:
  try:
   for item in r['sources']:q.verify(item)
   if 'evidence' in r:q.immutable(r['evidence'])
   if 'reference' in r:q.read(q.verify(r['reference']))
   if args.build_report:q.read(args.build_report)
   for item in r.get('machine',{}).get('version_files',[]):q.verify(item)
  except Exception as e:r.update(status='BLOCKED',integrity_error=str(e))
  if r['status']=='BLOCKED':r.pop('comparison',None)
  r['artifacts']=[q.identity(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name in ('runner.log','execution.json','actual-0.bin','actual-1.bin','benchmark-last.bin','postcheck.bin')]
  q.write(root/(args.stage+'-report.json'),r);q.write(root/'comparison.json',r.get('comparison',dict(status='NOT_RUN')))
  q.write(root/'preprocessing-inventory.json',dict(status='ONLINE_PREPROCESSING_NOT_RUN',excluded=['FP16 effective weight generation','INT8 activation quantization','combined scale and encoding','H2D/D2H','validation','Draft/Decode'],resident_input_bytes=dict(FP16=81920+99614720,W8=40960+49807360+3112960),output_bytes=622592,measured_memory='workspace in each execution; payload arithmetic is not process peak or DDR traffic'))
 print(r['status']+': '+str(root/(args.stage+'-report.json')));return int(r['status']=='BLOCKED')

def parse():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('prepare','build','smoke','benchmark'))
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--build-report',type=Path);p.add_argument('--prepare-report',type=Path);p.add_argument('--candidate',choices=('N128','N256'),default='N128');p.add_argument('--smoke-report',type=Path);p.add_argument('--benchmark-report',type=Path)
    p.add_argument('--msprof',default='msprof')
    for name,default in [('stage4b','w8a8-full-gate-20261027-093617'),('stage4a','w8a8-core-policy-20261027-083223'),('stage3b4','w8a8-twenty-group-20261027-055348'),('stage3b3','w8a8-two-group-20261027-042044'),('stage3b2','w8a8-local-nz-20261027-031856'),('stage3b1','w8a8-stage3b-20261027-024844'),('stage3a','w8a8-stage3a-20261027-011523'),('stage2b','w8a8-stage2b-20261026-100533')]:
        p.add_argument('--'+name+'-root',type=Path,default=Path('/home/w00949577/qwen35-dflash-run')/default)
    for name in ('build','basic','zero','nonpower'):p.add_argument('--'+name+'-report',type=Path)
    p.add_argument('--stage2c-report',type=Path,default=Path('/home/w00949577/qwen35-dflash-run/w8a8-stage2c-20261026-104943/report.json'))
    p.add_argument('--device',type=int,default=0);p.add_argument('--cann-root',type=Path,default=Path(os.environ.get('CANN_ROOT','/usr/local/Ascend/cann-9.0.0')))
    p.add_argument('--stage4j-root',type=Path,default=Path('/home/w00949577/qwen35-dflash-run/w8a8-n128-t1-20261030-092118'))
    p.add_argument('--stage4i-root',type=Path,default=Path('/home/w00949577/qwen35-dflash-run/w8a8-n128-host-shareub-20261030-083337'))
    p.add_argument('--stage4c-report',type=Path,default=Path('/home/w00949577/qwen35-dflash-run/w8a8-baseline-20261027-111748/benchmark/benchmark-report.json'))
    p.add_argument('--stage4n-root',type=Path,default=Path('/home/w00949577/qwen35-dflash-run/w8a8-n128-xor-guard-20261102-063409'))
    p.add_argument('--stage4p-root',type=Path,default=Path('/home/w00949577/qwen35-dflash-run/w8a8-n256-correctness-20261103-012123'))
    p.add_argument('--stage4o-root',type=Path,default=Path('/home/w00949577/qwen35-dflash-run/w8a8-n256-host-20261102-081401'))
    args=p.parse_args();args.cann_root=args.cann_root.resolve()
    for name in ('build','basic','zero','nonpower'):
        if getattr(args,name+'_report') is None:setattr(args,name+'_report',args.stage2b_root/name/'report.json')
    if args.device<0:p.error('negative device')
    return args

if __name__=='__main__':raise SystemExit(main(parse()))
