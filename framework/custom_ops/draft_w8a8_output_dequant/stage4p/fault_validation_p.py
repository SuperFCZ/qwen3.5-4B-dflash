from pathlib import Path
"""Separate instrumented OPP: exhaustive per-boundary sweep plus20 abort witnesses."""
import struct
from p_common import read,verify,identity,write,common,check_execution,plan_from_log
from p_reference import verify_case,compare
from run_stage4p import admission_expected
from run_stage2b import command

def fault_env(args,b,p):
 env=common.env(args.cann_root,Path(b['fault_build']['opp']))
 env['DFLASH_STAGE4P_EXPECTED_CORE_POLICY']=','.join(str(p['formal_core_policy'][k]) for k in ('aic','aiv','all','selected','arch'))
 return env

def validate_row(row,r):
 mode=row['mode'];p=r['evidence'];ex=read(verify(row['execution']))
 if ex.get('fault_diagnostic') is not True or ex.get('ub_guard_protocol')!='FAULT_INSTRUMENTED_ABORT_WITNESS':raise ValueError('fault artifact protocol mismatch')
 check_execution(dict(ex,ub_guard_protocol='UB_AND_CORE_ARENA_POISON_ON_FAILURE'),r['device'],mode,1,0)
 plan=plan_from_log(verify(row['log']).read_text(),mode,1,0,p['t1'])
 if plan['core_policy']!=p['formal_core_policy'] or ex['workspace_bytes']<plan['total'] or read(verify(row['admission']))!=admission_expected(ex,plan,False):raise ValueError('fault resource admission mismatch')
 data=[verify(x).read_bytes() for x in row['outputs']]
 if mode==0:
  case=read(verify(r['case']));maps=verify_case(case)
  result=compare(maps,0,1,0,data)
  if result['status']!='BITWISE_MATCH':raise ValueError('exhaustive diagnostic sweep did not reach correct final output')
 else:
  expected=struct.pack('<8I',0x534e4641,mode-1,0x80000001,640,4,1,63,mode)
  for i,raw in enumerate(data):
   if len(raw)!=622592 or raw[:32]!=expected or raw[32:]!=bytes([127 if i==0 else 255])*(622592-32):raise ValueError('fault did not abort at exact expected group, or wrote outside witness')
 return dict(status='FAULT_SWEEP_GOLDEN_MATCH' if mode==0 else 'EXACT_GROUP_ABORT_WITNESS',group=None if mode==0 else mode-1,plan=plan)

def faults(args,root,r,b):
 for mode in range(21):
  folder=root/f'mode-{mode}';folder.mkdir();p=r['evidence']
  for name in ('a_i8.bin','b_i8.bin','quant_u64_doc_candidate.bin'):(folder/name).symlink_to(verify(read(verify(r['case']))['files'][name]))
  r['phase']='fault_mode_'+str(mode);r['npu']='RUNNING';write(root/'fault-report.json',r)
  command([verify(b['fault_build']['runner']),args.device,folder,mode,1,0],folder/'runner.log',fault_env(args,b,p),r,'fault_'+str(mode),600)
  row=dict(mode=mode,execution=identity(folder/'execution.json'),log=identity(folder/'runner.log'),admission=identity(folder/'resource-admission.json'),outputs=[identity(folder/f'actual-{i}.bin') for i in range(2)])
  row['validation']=validate_row(row,r);r['runs'].append(row);write(root/'fault-report.json',r)
 r['coverage']=dict(words=640,bits=32,group_boundaries=20,injections_per_sweep=409600,repeats=2,abort_groups=list(range(20)),performance_eligible=False,
  limits='Fault artifact intentionally injects and restores guard storage. Timing candidate contains no injection, witness or extra synchronization.')

def verify_faults(path,r):
 if path is None:raise ValueError('accepted --fault-report required before benchmark')
 p=read(path)
 if p.get('status')!='STAGE4P_DEVICE_FAULT_COVERAGE_MATCH' or p.get('phase')!='complete' or any(p.get(key)!=r[key] for key in ('sources','evidence','device','build_report','full_report','case')):raise ValueError('fault proof binding differs')
 if [x['mode'] for x in p['runs']]!=list(range(21)):raise ValueError('incomplete fault coverage')
 for row in p['runs']:
  if validate_row(row,p)!=row['validation']:raise ValueError('fault witness changed')
