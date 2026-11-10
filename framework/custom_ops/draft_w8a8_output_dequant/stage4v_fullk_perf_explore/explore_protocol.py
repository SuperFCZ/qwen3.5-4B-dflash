from explore_common import verify,read,old_protocol,frozen
from explore_numeric import observe,timing,stats
OUTPUTS=('actual-0.bin','actual-1.bin','benchmark-last.bin','postcheck.bin')


def validate(row,proof,device,timed=False,host=False):
 tile=row['tile'];ex=read(verify(row['execution']))
 pl=old_protocol.plan(verify(row['log']).read_text(),0,0,tile,proof['t1'],proof['policy'])
 adm=read(verify(row['admission']))
 want=dict(status='ADMITTED_BEFORE_EXECUTION',workspace_bytes=proof['workspace'],declared_total=pl['total'],blocks=pl['blocks'],cores=pl['cores'],core_policy=pl['core_policy'],host_proof_checked=not host)
 if adm!=want or ex.get('workspace_bytes')!=proof['workspace']:raise ValueError('Workspace admission differs')
 if host:
  if tile!=-1 or ex!=dict(status='HOST_TILING_ONLY',npu='NOT_RUN',device=device,workspace_bytes=proof['workspace'],output_mode=0,core_limit=0,tile_id=-1):raise ValueError('Host protocol differs')
  return dict(status='HOST_TILING_ONLY',plan=pl)
 required=dict(status='EXPLORATORY_EXECUTION_SAFE',op='DFlashW8A8FullKChannel',soc='Ascend310P3',shape=[16,2560,19456],tile_count=76,group_count=1,guards=True,readonly=True,repeat_equal=True,repetitions=2,device=device,cpu_fallback=False,output_mode=0,core_limit=0,tile_id=tile,guard_coverage='FULL_K_TILE_384_WORD_THREE_BOUNDARIES',prepare_calls=38 if timed else 2,verify_calls=4 if timed else 2,poison_verified=True,poison_patterns=[127,255],timed_prepare_io=False,kernel_guards_unchanged=True,timed_log_checked=timed,timed_log_stable=True,strict_golden_match=False,sync_completed=True,finite_active_output=True,inactive_poison_intact=True,active_coverage_checked=True,production='NOT_APPROVED',kernel_guard_observation='POISON_ON_FAILURE_AND_COMPLETE_FINITE_OUTPUT',reference_kind='CPU_FULL_K_REFERENCE')
 if any(type(ex.get(k))!=type(v) or ex[k]!=v for k,v in required.items()):raise ValueError('exploration execution/safety protocol differs')
 if timed and tile!=-1:raise ValueError('full only benchmark')
 t=timing(ex,timed);count=4 if timed else 2
 if len(row['outputs'])!=count or len(row['numeric'])!=count:raise ValueError('all output/number records required')
 if len(row['expected'])!=2:raise ValueError('two poison-specific expected files required')
 g=[verify(x).read_bytes() for x in row['expected']]
 maps=frozen.ref.verify_case(read(verify(proof['case'])))
 if any(g[i]!=old_protocol.expected(maps,0,proof['policy']['selected'],tile,i) for i in range(2)):raise ValueError('reference differs from original Golden/scope')
 raw=[verify(x).read_bytes() for x in row['outputs']]
 if timed and any(x!=verify(row['repeat_anchor']).read_bytes() for x in raw):raise ValueError('full Smoke anchor drift')
 results=[]
 for i,x in enumerate(raw):
  result=observe(x,g[min(i,1)],tile,min(i,1),raw[0] if i else None)
  if result!=read(verify(row['numeric'][i])):raise ValueError('C++/Python numeric contract differs')
  results.append(result)
 numeric='NUMERIC_MISMATCH_UNRESOLVED' if any(x['mismatch_count'] for x in results) else 'BITWISE_REFERENCE_OBSERVATION'
 if ex['numeric_status']!=numeric:raise ValueError('numeric status differs')
 return dict(status='EXPLORATORY_SAFETY_AND_DATA_COMPLETE',numeric_status=numeric,numeric=results,plan=pl,timing=t,workspace_bytes=proof['workspace'],production='NOT_APPROVED')


def summary(rows):
 if len(rows)!=3 or [x['round'] for x in rows]!=[0,1,2]:raise ValueError('exactly three new rounds')
 if any(x['tile']!=-1 or x.get('profiled') is not False or x['validation']['status']!='EXPLORATORY_SAFETY_AND_DATA_COMPLETE' for x in rows):raise ValueError('wrong scope/instrumentation/safety')
 if len({x['validation']['workspace_bytes'] for x in rows})!=1:raise ValueError('Workspace drift across rounds')
 values={k:stats([n for row in rows for n in row['validation']['timing'][k]['samples_ms']],90) for k in ('execute_sync','prepare')}
 return dict(status='EXPLORATORY_PERFORMANCE_MEASURED',statistics=values,
  per_round=[dict(round=x['round'],timing=x['validation']['timing']) for x in rows],numeric_status='NUMERIC_MISMATCH_UNRESOLVED' if any(x['validation']['numeric_status']=='NUMERIC_MISMATCH_UNRESOLVED' for x in rows) else 'BITWISE_REFERENCE_OBSERVATION',
  device_kernel_time='UNAVAILABLE_NO_DEVICE_TIMESTAMPS',scope='FULL_M16_K2560_N19456_PREPACKED_RESIDENT_NORMAL_OPP',
  production='NOT_APPROVED',historical_reference_only=dict(fp16_median_ms=.78698,fp16_p95_ms=.79286,group128_median_ms=2.77505,group128_p95_ms=2.78094,fresh_paired=False),
  historical_ratios_not_paired=dict(full_k_over_fp16=values['execute_sync']['median_ms']/.78698,group128_over_full_k=2.77505/values['execute_sync']['median_ms']),
  limitations=['different quantization and Scale scheme','kernel Guard/control overhead included','historical baselines not current paired samples','not pure INT8 versus FP16 throughput','no online preprocessing or full Draft'])
