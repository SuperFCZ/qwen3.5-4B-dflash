"""Optional read-only historical FP16 admission audit; no benchmark or regenerated weights."""
import sys
import numpy as np
from o_layout import n,identity,verify
FP=n.common.QUANT/'fp16_compare';sys.path.insert(0,str(FP))
import fp16_gate_common as fc

def audit(path,freeze):
 r=dict(status='UNRESOLVED',historical_report=str(path),fresh_paired_measurement='NOT_RUN',online_preprocessing='NOT_RUN',native_golden_semantics='UNRESOLVED_NOT_REAUDITED',
  source_review=dict(api='aclnnMm',cube_math_type=0,weights='RN16(INT8 NK * FP16 GN); offline ND_KN',shape=[16,2560,19456],bias=False,activation=False,
  timing='continuous-v1 execute+stream-sync; preparation/upload/readback outside execute samples',
  runner_gap='Historical Runner checks repeat/guards/readonly, but Golden comparison is in outer Python after process exit. Future Runner must check its own admitted Golden before warmup and at tail/postcheck, verify Poison readback and timed-log stability.',
  source_files=[identity(FP/x) for x in ('native_mm.cpp','fp16_gate_common.py','fp16_gate_evidence.py','experiment.py')]))
 try:
  old=n.read(path)
  if old.get('status')!='EXPERIMENT_COMPLETED_DIAGNOSTIC_ONLY' or old.get('phase')!='complete' or old.get('shape')!=[16,2560,19456] or old.get('projection')!='layer-0-gate_up':raise ValueError('historical FP16 experiment not accepted')
  case=n.read(verify(freeze['frozen_inputs']));src=n.k.bc.reference.load_source(case['source']);frozen=case['source']['source']['frozen'];w=old['weights']
  if w['status']!='CPU_EFFECTIVE_WEIGHT_VERIFIED' or w['group_size']!=128:raise ValueError('unverified effective weights')
  for key,item in frozen.items():
   actual=w['source_files'][key]
   if any(actual[k]!=item[k] for k in ('bytes','sha256')):raise ValueError('frozen source differs '+key)
  kn=np.memmap(verify(w['files']['w_fp16_kn.bin']),mode='r',dtype='<f2',shape=(2560,19456))
  nk=np.memmap(verify(w['files']['w_fp16_nk.bin']),mode='r',dtype='<f2',shape=(19456,2560))
  for g in range(20):
   codes=src['nz'][g*4:g*4+4].transpose(1,2,0,3).reshape(19456,128)
   if not np.array_equal(codes,src['q'][:,g*128:(g+1)*128]):raise ValueError('NZ inverse changed')
   expected=(codes.astype(np.float64)*src['sw'][g,:,None].astype(np.float64)).astype('<f2')
   if not np.isfinite(expected).all() or expected.tobytes()!=nk[:,g*128:(g+1)*128].tobytes() or expected.T.tobytes()!=kn[g*128:(g+1)*128].tobytes():raise ValueError('effective FP16 weight mismatch')
  baseline=old['prechecks']['fp16'];first=None;samples=[]
  rows=[(baseline,False)]+[(x['results']['fp16'],True) for x in old['rounds']]
  if len(old['rounds'])!=6 or [x['order'] for x in old['rounds']]!=[list(x) for x in fc.ORDERS]:raise ValueError('historical rotation coverage differs')
  for row,timed in rows:
   if row['mode']!='fp16' or row['profiled'] or row['formal_latency']!=timed:raise ValueError('profile/formal evidence differs')
   ex=n.read(verify(row['execution_file']))
   if ex!=row['execution']:raise ValueError('execution record changed')
   fc.check_execution(ex,'fp16',freeze['device'],timed)
   if row['input_hashes']!={'x.bin':frozen['x.bin']['sha256'],'w_fp16_kn.bin':w['files']['w_fp16_kn.bin']['sha256']}:raise ValueError('runtime input identities differ')
   names=fc.OUTPUTS if timed else fc.OUTPUTS[:2]
   if set(row['outputs'])!=set(names):raise ValueError('output check coverage incomplete')
   for item in row['outputs'].values():
    data=verify(item).read_bytes()
    if len(data)!=622592 or not np.isfinite(np.frombuffer(data,'<f2')).all():raise ValueError('invalid FP16 output')
    if first is None:first=data
    if data!=first:raise ValueError('historical repeat/tail/postcheck mismatch')
   if timed:samples+=ex['timing']['execute_sync']['samples_ms']
  stats=fc.stats(samples)
  if len(samples)!=180 or stats!=old['timing_summary']['fp16']:raise ValueError('historical samples/summary differ')
  runner_sources=old['native_build']['sources']
  for item in r['source_review']['source_files'][:1]:
   if item not in runner_sources:raise ValueError('native Runner source identity differs')
  r.update(status='HISTORICAL_INPUT_WEIGHTS_AND_TIMING_VERIFIED_ONLY',report=identity(path),weights=w['files'],inputs=frozen,native_build=old['native_build'],historical_samples=stats,
   historical_golden_claim=baseline['golden_comparison'],golden_reference='Separate admitted FP16 baseline required; do not substitute W8A8 Golden or assert Native OM graph semantics',
   current_device_driver_equivalence='UNRESOLVED_FUTURE_PAIRED_RUN_MUST_RECORD',new_speedup='NOT_RUN')
 except (OSError,ValueError,KeyError,TypeError) as error:r.update(status='UNRESOLVED',reason=f'{type(error).__name__}: {error}')
 return r
