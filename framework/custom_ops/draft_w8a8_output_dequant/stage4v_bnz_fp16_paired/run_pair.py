#!/usr/bin/env python3
"""Fresh native FP16 / accepted B-NZ resident-input pairing, no Kernel edits."""
import argparse
from pathlib import Path
import shutil,re
import numpy as np
import pair_evidence as e
from pair_contract import ORDER,validate_row,comparison,inventory
from run_b_nz import invoke as invoke_nz
from raw_probe_runtime_identity import collect,verify_runtime
from run_stage2b import command
STATES=dict(source='BNZ_FP16_SOURCE_REVERIFIED',build='BNZ_FP16_RUNNERS_REVERIFIED',smoke='BNZ_FP16_FRESH_SMOKE_ADMITTED',benchmark='BNZ_FP16_FRESH_PAIRED_MEASURED')


def build_bind(a,root,r):
 bound=r['evidence'];nb=bound['native']['build'];wb=bound['bnz']['build']['variants']['candidate']['build'];p=bound['bnz']['evidence']['baseline']
 e.rr.verify_binding(nb['header_binding'],a.cann_root)
 for name,f in nb['libraries'].items():
  if a.cann_root not in e.verify(f).parents:raise ValueError('native library outside chosen CANN '+name)
 dp=dict(p,normal=dict(p['normal'],opp=wb['opp'],installed=wb['installed']))
 history=[dict(label='native_build',report=bound['native']['build_report'],content=e.read(e.verify(bound['native']['build_report']))),dict(label='preflight',report=p['original_evidence']['preflight'],content=e.read(e.verify(p['original_evidence']['preflight'])))]
 r['runtime_bindings']={};r['symbol_evidence']={}
 runs=[('FP16',nb['runner'],e.rr.native_env(a.cann_root)),('W8A8_NZ',wb['runner'],e.nz.env(bound['bnz']['evidence'],a.cann_root,wb['opp'])),('W8_HOST',wb['installed']['host'],e.nz.env(bound['bnz']['evidence'],a.cann_root,wb['opp']))]
 r['inspection_tools']={name:e.identity(shutil.which(name)) for name in ('ldd','nm')}
 for name,item,env in runs:
  log=root/(name+'-ldd.log');command(['ldd',e.verify(item)],log,env,r,name+'_ldd')
  binding=collect(log,a.cann_root,history) if name=='FP16' else e.c.runtime(log,a.cann_root,dp)
  if name=='FP16':
   if any(x['needed'].startswith('libcust_opapi') or '/vendors/' in x['file']['path'] for category in ('sdk_libraries','driver_libraries','system_libraries') for x in binding[category]):raise ValueError('native process resolved a custom OPP dependency')
   resolved={x['needed']:x['file'] for x in binding['sdk_libraries']}
   for short,key in [('opapi','LIB_opapi'),('ascendcl','LIB_ascendcl'),('nnopbase','LIB_nnopbase')]:
    if not any(n.startswith('lib'+short+'.so') and f==nb['libraries'][key] for n,f in resolved.items()):raise ValueError('native runtime link differs '+short)
  r['runtime_bindings'][name]=binding
  if name=='W8_HOST':continue
  symbols=root/(name+'-symbols.log');command(['nm','-u',e.verify(item)],symbols,env,r,name+'_symbols')
  expected=['aclnnMmGetWorkspaceSize','aclnnMm'] if name=='FP16' else [bound['bnz']['operator']['prepare_symbol'],bound['bnz']['operator']['execute_symbol']]
  text=symbols.read_text()
  if any(not re.search(r'\b'+x+r'\b',text) for x in expected):raise ValueError('frozen Runner ABI symbols differ '+name)
  r['symbol_evidence'][name]=dict(file=e.identity(symbols),expected=expected,runner=item)
 r['build_kind']='READ_ONLY_EXISTING_BINARIES_DEPENDENCIES_NO_RECOMPILE'
 r['kernel_build']=r['opp_install']='NOT_RUN'


def verify_build(b,bound,cann):
 if b.get('status')!=STATES['build'] or b.get('build_kind')!='READ_ONLY_EXISTING_BINARIES_DEPENDENCIES_NO_RECOMPILE':raise ValueError('Build identity gate incomplete')
 e.verify_tree(b)
 if set(b['runtime_bindings'])!={'FP16','W8A8_NZ','W8_HOST'}:raise ValueError('runtime evidence incomplete')
 for dep in b['runtime_bindings'].values():
  verify_runtime(dep)
  for private in dep.get('private_opp_libraries',[]):
   if e.c.trace(private['requested_path'])!={k:private[k] for k in ('requested_path','real_path','symlink_chain')}:raise ValueError('private runtime alias changed')
 expected={'FP16':bound['native']['build']['runner'],'W8A8_NZ':bound['bnz']['build']['variants']['candidate']['build']['runner']}
 if any(b['symbol_evidence'][n]['runner']!=x for n,x in expected.items()):raise ValueError('Build admitted wrong Runner')
 e.unchanged(bound,cann)


def prior(path,stage,r):
 if path is None:raise ValueError('missing '+stage+' report')
 old=e.read(path);e.verify_tree(old)
 if old.get('status')!=STATES[stage] or old.get('error') or old.get('integrity_error') or any(old.get(k)!=r[k] for k in ('sources','evidence','device','cann_root')):raise ValueError('paired predecessor differs '+stage)
 e.compare_machine(old['machine'],r['machine'])
 if stage=='build':verify_build(old,r['evidence'],Path(r['cann_root']))
 if stage=='smoke':
  if [x['scheme'] for x in old['runs']]!=['FP16','W8A8_NZ']:raise ValueError('fresh Smoke coverage incomplete')
  if old['npu']!='COMPLETED' or old['build_report']!=r['build_report'] or old['source_report']!=r['source_report']:raise ValueError('Smoke chain differs')
  for row in old['runs']:
   if row['validation']!=validate_row(row,r['evidence'],r['device'],False):raise ValueError('Smoke validation changed')
  if e.verify(old['native_repeat_anchor']).read_bytes()!=e.verify(old['runs'][0]['outputs'][0]).read_bytes():raise ValueError('native repeat anchor not independently admitted')
 return old


def invoke(a,root,r,b,scheme,timed,anchor=None):
 verify_build(b,r['evidence'],a.cann_root)
 bound=r['evidence'];shadow=dict(device=a.device,output_dir=str(Path(r['output_dir'])),evidence=bound['bnz']['evidence'],packing=bound['bnz']['packing'],t1=bound['bnz']['t1'])
 row=dict(scheme=scheme,profiled=False,timed=timed)
 r.setdefault('runs',[]).append(row)
 try:
  if scheme=='FP16':
   value=e.rr.invoke_native(a,root,shadow,dict(native=bound['native']['build']),bound['native']['data'],timed,anchor)
   row.update(value,runner=bound['native']['build']['runner'],log=e.identity(root/'runner.log'))
  elif scheme=='W8A8_NZ':
   value=invoke_nz(a,root,shadow,bound['bnz']['build'],'candidate',timed=timed)
   row.update(value)
  else:raise ValueError('invalid scheme')
 finally:
  row['commands']=shadow.get('commands',{})
  row['retained_files']=[e.identity(p) for p in sorted(root.iterdir()) if p.is_file()] if root.exists() else []
 row['validation']=validate_row(row,bound,a.device,timed,anchor if scheme=='FP16' else None)
 e.write(root/'paired-validation.json',row['validation'])
 return row


def cross_numeric(smoke,bound):
 ref=np.fromfile(e.verify(bound['native']['data']['files']['reference_f64.bin']),'<f8').reshape(16,19456)
 from reference_v import stats
 result={}
 for row in smoke['runs']:
  values=np.fromfile(e.verify(row['outputs'][0]),'<f2').reshape(16,19456)
  result[row['scheme']]=dict(full=stats(ref,values),gate=stats(ref[:,:9728],values[:,:9728]),up=stats(ref[:,9728:],values[:,9728:]))
 return dict(metrics_vs_same_fp64_reference=result,scope='DIAGNOSTIC_CROSS_SCHEME_NOT_BITWISE_EQUIVALENCE',w8_numeric_status='NUMERIC_MISMATCH_UNRESOLVED',fp16_acceptance_policy=e.rr.POLICY)


def main(a):
 root=e.fresh(a.output_dir,[e.HERE,a.bnz_root,a.cann_root]+[p.parent for p in (a.source_report,a.build_report,a.smoke_report) if p])
 r=dict(status='NOT_RUN',phase=a.phase,sources=e.sources(),device=a.device,cann_root=str(a.cann_root),output_dir=str(root),npu='NOT_RUN',benchmark='NOT_RUN',profile='NOT_RUN',production='NOT_APPROVED',w8_numeric_status='NUMERIC_MISMATCH_UNRESOLVED')
 try:
  r['evidence']=e.bind(a,root,r)
  if a.phase!='source':prior(a.source_report,'source',r);r['source_report']=e.identity(a.source_report)
  if a.phase=='build':build_bind(a,root,r)
  elif a.phase in ('smoke','benchmark'):
   b=prior(a.build_report,'build',r);r['build_report']=e.identity(a.build_report)
   if b['source_report']!=r['source_report']:raise ValueError('Build uses another Source')
   if a.phase=='smoke':
    r['npu']='ATTEMPTED'
    for scheme in ('FP16','W8A8_NZ'):invoke(a,root/scheme,r,b,scheme,False)
    f=root/'native-repeat-anchor.bin';f.write_bytes(e.verify(r['runs'][0]['outputs'][0]).read_bytes());r['native_repeat_anchor']=e.identity(f)
    r['numeric_diagnostics']=cross_numeric(r,r['evidence']);r['npu']='COMPLETED'
   else:
    smoke=prior(a.smoke_report,'smoke',r);r['smoke_report']=e.identity(a.smoke_report);r['native_repeat_anchor']=smoke['native_repeat_anchor']
    r['npu']='ATTEMPTED';r['benchmark']='ATTEMPTED'
    for i,order in enumerate(ORDER):
     for j,scheme in enumerate(order):
      row=invoke(a,root/f'round-{i}'/scheme,r,b,scheme,True,r['native_repeat_anchor']);row.update(round=i,position=j)
      e.write(root/'report.json',r)
    r['comparison']=comparison(r['runs'],r['evidence']['bnz']['packing']);r['comparison']['numeric_diagnostics']=smoke['numeric_diagnostics']
    r['comparison']['implementations']={'FP16':{'op':'aclnnMm','runner':r['evidence']['native']['build']['runner'],'libraries':r['evidence']['native']['build']['libraries'],'core_count':'UNAVAILABLE_INTERNAL_ACLNN'},
     'W8A8_NZ':{'operator':r['evidence']['bnz']['operator'],'runner':r['evidence']['bnz']['build']['variants']['candidate']['build']['runner'],'opp':r['evidence']['bnz']['build']['variants']['candidate']['build']['opp'],'core_policy':r['evidence']['bnz']['evidence']['formal_core_policy']}}
    r['npu']='COMPLETED';r['benchmark']='MEASURED_UNPROFILED'
  r['status']=STATES[a.phase]
 except Exception as x:r.update(status='BLOCKED',error=f'{type(x).__name__}: {x}')
 finally:
  try:
   e.verify_tree(r)
   if r['sources']!=e.sources():raise ValueError('new source changed')
   if 'evidence' in r:e.unchanged(r['evidence'],a.cann_root)
   if a.build_report:verify_build(e.read(a.build_report),r['evidence'],a.cann_root)
  except Exception as x:r.update(status='BLOCKED',integrity_error=str(x))
  if r['status']=='BLOCKED':r.pop('comparison',None)
  e.write(root/'report.json',r)
  if 'evidence' in r:e.write(root/'preprocessing-inventory.json',inventory(r['evidence']))
  if a.phase=='benchmark':e.write(root/'benchmark-report.json',r);e.write(root/'comparison.json',r.get('comparison',dict(status='NOT_RUN_OR_BLOCKED')))
 print(r['status']+': '+str(root/'report.json'));return int(r['status']=='BLOCKED')


def parse(argv=None):
 p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False);p.add_argument('phase',choices=STATES)
 p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--bnz-root',type=Path,default=Path('/home/w00949577/qwen35-dflash-run/w8a8-fullk-bnz-20261110-085944'))
 p.add_argument('--cann-root',type=Path,default=Path('/usr/local/Ascend/cann-9.0.0'));p.add_argument('--device',type=int,default=0)
 for x in ('source','build','smoke'):p.add_argument('--'+x+'-report',type=Path)
 a=p.parse_args(argv)
 for x in ('source','build','smoke')[:list(STATES).index(a.phase)]:
  if getattr(a,x+'_report') is None:p.error('--'+x+'-report required')
 if a.device<0:p.error('negative device')
 for k,v in vars(a).items():
  if isinstance(v,Path):setattr(a,k,v.resolve())
 return a
if __name__=='__main__':raise SystemExit(main(parse()))
