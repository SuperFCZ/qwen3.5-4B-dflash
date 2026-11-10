"""Separate numeric contracts; only fresh unprofiled samples enter ratios."""
import numpy as np
from pair_evidence import rr,verify,read,amortization
from explore_numeric import stats
ORDER=(('FP16','W8A8_NZ'),('W8A8_NZ','FP16'),('FP16','W8A8_NZ'))


def native_validate(row,e,device,timed,anchor=None):
 ex=read(verify(row['execution']));rr.fc.check_execution(ex,'fp16',device,timed)
 expected=dict(independent_cpu_reference=True,poison_verified=True,timed_log_stable=True,prepare_calls=38 if timed else 2,verify_calls=4 if timed else 2)
 if any(type(ex.get(k))!=type(v) or ex[k]!=v for k,v in expected.items()):raise ValueError('FP16 checks incomplete')
 if row.get('profiled') is not False or row['runner']!=e['native']['build']['runner']:raise ValueError('FP16 Runner/profile identity differs')
 data=e['native']['data'];ref=np.fromfile(verify(data['files']['reference_f64.bin']),'<f8').reshape(16,19456)
 if len(row['outputs'])!=(4 if timed else 2):raise ValueError('FP16 output coverage incomplete')
 values=[verify(x).read_bytes() for x in row['outputs']]
 if any(len(x)!=622592 or x!=values[0] for x in values):raise ValueError('FP16 repeat/size differs')
 if anchor is not None and any(x!=verify(anchor).read_bytes() for x in values):raise ValueError('FP16 independently admitted repeat anchor drift')
 metrics=[rr.check(ref,np.frombuffer(x,'<f2').reshape(16,19456)) for x in values]
 if timed:timing={k:stats(ex['timing'][k]['samples_ms'],30) for k in ('prepare','execute_sync')}
 else:
  if ex['timing']['execute_sync']['samples_ms'] or ex['timing']['prepare']['samples_ms']:raise ValueError('Smoke must be untimed')
  timing=None
 return dict(status='FP16_INDEPENDENT_REFERENCE_AND_SAFETY_MATCH',numerics=metrics,workspace_bytes=ex['workspace_bytes'],
  timing=timing,core_count='UNAVAILABLE_INTERNAL_ACLNN_NOT_EXPOSED_BY_REUSED_RUNNER',tracked_device_allocation_bytes=ex['tracked_device_allocation_bytes'])


def w8_validate(row,e,device,timed):
 from nz_protocol import validate
 b=e['bnz'];x=dict(row)
 if row['runner']!=b['build']['variants']['candidate']['build']['runner'] or row['variant']!='candidate' or row['mode']!=0 or row['limit']!=0 or row['tile']!=-1 or row.get('profiled') is not False:raise ValueError('wrong W8 normal Runner/scope')
 if row['accepted_t1']!=b['t1'] or row['b_input']!=b['packing']['packed_b']:raise ValueError('W8 input/resource identity changed')
 result=validate(x,b['evidence'],device,timed=timed)
 # All bytes bind the historical, already safe W8 output. Never compare to native as a Golden.
 if any(verify(out).read_bytes()!=verify(b['full_anchor']).read_bytes() for out in row['outputs']):raise ValueError('W8 frozen baseline output changed')
 return dict(result,workspace_bytes=result['plan']['total'],core_count=result['plan']['blocks'])


def validate_row(row,e,device,timed,anchor=None):
 if row['scheme']=='FP16':return native_validate(row,e,device,timed,anchor)
 if row['scheme']=='W8A8_NZ':return w8_validate(row,e,device,timed)
 raise ValueError('unknown scheme')


def comparison(rows,packing):
 if [(r['round'],r['position'],r['scheme']) for r in rows]!=[(i,j,n) for i,o in enumerate(ORDER) for j,n in enumerate(o)]:raise ValueError('fresh paired schedule/coverage differs')
 if any(r.get('profiled') is not False for r in rows):raise ValueError('profiled samples forbidden')
 for r in rows:
  status='FP16_INDEPENDENT_REFERENCE_AND_SAFETY_MATCH' if r['scheme']=='FP16' else 'BASELINE_BITS_AND_SAFETY_MATCH'
  if r['validation']['status']!=status:raise ValueError('unadmitted samples')
 for n in ('FP16','W8A8_NZ'):
  if len({r['validation']['workspace_bytes'] for r in rows if r['scheme']==n})!=1:raise ValueError('per-scheme Workspace drift')
 statistics={n:{k:stats([v for r in rows if r['scheme']==n for v in r['validation']['timing'][k]['samples_ms']],90) for k in ('execute_sync','prepare')} for n in ('FP16','W8A8_NZ')}
 fp=statistics['FP16']['execute_sync']['median_ms'];w8=statistics['W8A8_NZ']['execute_sync']['median_ms']
 ratios=[next(r['validation']['timing']['execute_sync']['median_ms'] for r in rows if r['round']==i and r['scheme']=='FP16')/next(r['validation']['timing']['execute_sync']['median_ms'] for r in rows if r['round']==i and r['scheme']=='W8A8_NZ') for i in range(3)]
 return dict(status='BNZ_FP16_FRESH_PAIRED_MEASURED',statistics=statistics,fp16_over_w8a8_nz=fp/w8,w8a8_nz_over_fp16=w8/fp,
  round_fp16_over_w8a8_nz=ratios,round_w8a8_nz_over_fp16=[1/x for x in ratios],
  per_round=[dict(round=r['round'],scheme=r['scheme'],timing=r['validation']['timing'],workspace_bytes=r['validation']['workspace_bytes'],core_count=r['validation']['core_count']) for r in rows],
  workspace_by_scheme={n:next(r['validation']['workspace_bytes'] for r in rows if r['scheme']==n) for n in ('FP16','W8A8_NZ')},
  kernel_device_time='UNAVAILABLE_NO_DEVICE_TIMESTAMPS',numeric_status_w8='NUMERIC_MISMATCH_UNRESOLVED',production='NOT_APPROVED',historical_samples_used=False,
  scope='SAME_SHAPE_RESIDENT_IMPLEMENTATIONS_DIFFERENT_INPUTS_WEIGHTS_SCALES_GUARDS',prepack_cost=packing['layout'],
  amortization_permutation=amortization(packing['layout']['permutation_ms'],fp,w8),amortization_complete_prepare=amortization(packing['layout']['prepare_wall_ms'],fp,w8),
  amortization_scope='B_NZ_LAYOUT_INCREMENTAL_ONLY; NOT_ALL_W8_INITIALIZATION; FP16_WEIGHT_PREPARATION_AND_ONLINE_QUANT_UPLOAD_EXCLUDED')


def inventory(e):
 n=e['native']['data'];b=e['bnz'];case=read(verify(b['evidence']['baseline']['case']))
 return dict(status='COSTS_SEPARATE_FROM_RESIDENT_EXECUTION',
  FP16=dict(activation=dict(dtype='FP16',shape=[16,2560],file=n['files']['x.bin']),weight=dict(dtype='FP16',layout='ND_KN',shape=[2560,19456],file=n['files']['w_fp16_kn.bin'],meaning='RN16(original Group128 INT8*FP16 scale), not original checkpoint FP16'),
   scale='NO_SEPARATE_INPUT; incorporated offline into FP16 weights',input_payload_bytes=99696640,output_bytes=622592,core_count='UNAVAILABLE_INTERNAL_ACLNN',
   guard_protocol='Runner GM guards/readonly/poison/repeat; builtin internal UB/Core guards not exposed',numeric_policy=rr.POLICY),
  W8A8_NZ=dict(activation=dict(dtype='INT8',shape=[16,2560],file=case['files']['a_i8.bin']),weight=dict(dtype='INT8',layout='TRANSPOSED_STORAGE_NZ',shape=[76,80,16,16,32],file=b['packing']['packed_b'],meaning='Full-K per-channel requantization of same original W_eff; different effective weights'),
   scale=dict(dtype='UINT64',shape=[76,256],file=case['files']['quant_u64_doc_candidate.bin'],mechanism='experimental encoded combined Scale / SDK VDEQ16'),input_payload_bytes=50003968,output_bytes=622592,
   core_policy=b['evidence']['formal_core_policy'],guard_protocol='Full-K three UB384-word boundaries per Tile plus Arena XOR/OR; Runner GM checks',numeric_policy='Frozen W8 bitwise + all201 CPU mismatches retained UNRESOLVED'),
  prepack=dict(report=b['reports']['prepare'],record=b['packing'],remeasured_this_pair=False,physical_B_bytes=49807360,padding_bytes=0,original_plus_NZ_disk_payload_bytes=99614720),
  excluded=['FP16 weight reconstruction','Full-K W8 weight requantization','online Activation quantization','combined Scale generation/encoding','B NZ prepack','H2D/D2H','validation','Draft/Decode'],
  transfer_note='payload bytes per initial upload, not measured traffic/bandwidth; no per-call transfer in resident timing',
  process_peak_memory='UNAVAILABLE; Workspace is measured per scheme; tracked payload/guards are not allocator peak',
  numerical_equivalence='NOT_CLAIMED',model_quality='UNRESOLVED')
