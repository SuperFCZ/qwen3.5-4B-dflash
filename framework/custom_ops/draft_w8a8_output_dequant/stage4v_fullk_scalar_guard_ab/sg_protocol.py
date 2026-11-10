import struct
from sg_common import c,verify,read,NAMES
from explore_numeric import observe,timing,stats
ORDER=(('baseline','candidate'),('candidate','baseline'),('baseline','candidate'))


def fault_expected(mode,repeat):
 y=bytearray([127 if repeat==0 else 255]*622592)
 if mode==2:words=(0x53474753,3,384,32,36864,16384,2,1)
 elif mode in (3,4,5):words=(0x53564641,mode-3,0x80000001,384,2,1,63,mode-2)
 elif mode in (6,7):words=(0x53474146,mode-6,0x80000001,256,1,127,mode-6,1)
 else:raise ValueError('invalid fault mode')
 y[:32]=struct.pack('<8I',*words);return bytes(y)


def validate(row,evidence,device,host=False,timed=False):
 proof=evidence['baseline'];kind=row['variant'];mode=row['mode'];limit=row['limit'];tile=row['tile']
 ex=read(verify(row['execution']));pl=c.old_protocol.plan(verify(row['log']).read_text(),mode,limit,tile,proof['t1'],proof['policy'])
 if ex['workspace_bytes']!=pl['total']:raise ValueError('Workspace not exact declared total')
 adm=read(verify(row['admission']))
 if adm!=dict(status='ADMITTED_BEFORE_EXECUTION',workspace_bytes=pl['total'],declared_total=pl['total'],blocks=pl['blocks'],cores=pl['cores'],core_policy=pl['core_policy'],host_proof_checked=not host):raise ValueError('admission differs')
 if host:
  if ex!=dict(status='HOST_TILING_ONLY',npu='NOT_RUN',device=device,workspace_bytes=pl['total'],output_mode=0,core_limit=0,tile_id=-1):raise ValueError('Host state differs')
  return dict(status='FORMAL_HOST_ADMITTED',plan=pl)
 common=dict(soc='Ascend310P3',shape=[16,2560,19456],tile_count=76,group_count=1,guards=True,readonly=True,repeat_equal=True,repetitions=2,device=device,cpu_fallback=False,output_mode=mode,core_limit=limit,tile_id=tile,guard_coverage='FULL_K_TILE_384_WORD_THREE_BOUNDARIES',prepare_calls=38 if timed else 2,verify_calls=4 if timed else 2,poison_verified=True,poison_patterns=[127,255],timed_prepare_io=False,timed_log_stable=True,sync_completed=True,resource_admission='BEFORE_EXECUTION',ub_guard_protocol='UB_AND_CORE_ARENA_POISON_ON_FAILURE',layout='nd',input_count=3)
 expected_op='DFlashW8A8FullKChannel' if kind=='baseline' else NAMES[kind][0]
 common['op']=expected_op
 if kind=='fault':
  if timed or mode not in range(2,8) or tile!=0 or limit!=1:raise ValueError('fault cannot be timed')
  common.update(status='FAULT_PROTOCOL_MATCH',reference_kind='FAULT_WITNESS',strict_golden_match=False,fault_witness_match=True)
 else:
  if mode!=0 or limit!=0 or tile not in (0,-1):raise ValueError('normal scope')
  common.update(status='EXPLORATORY_EXECUTION_SAFE',sync_completed=True,finite_active_output=True,inactive_poison_intact=True,active_coverage_checked=True,kernel_guards_unchanged=kind=='baseline',strict_golden_match=False,timed_log_checked=timed,reference_kind='CPU_FULL_K_REFERENCE',production='NOT_APPROVED')
 if kind=='candidate' and not host:common.update(guard_coverage_unchanged=True,arena_detector='XOR_OR_U32')
 if any(type(ex.get(k))!=type(v) or ex[k]!=v for k,v in common.items()):raise ValueError('execution safety/variant contract differs')
 tt=timing(ex,timed);raw=[verify(x).read_bytes() for x in row['outputs']]
 if len(raw)!=(4 if timed else 2):raise ValueError('missing repeats/tail/postcheck')
 if kind=='fault':
  if any(x!=fault_expected(mode,i) for i,x in enumerate(raw)):raise ValueError('fault sweep/abort witness differs')
  return dict(status='FAULT_WITNESS_MATCH_NOT_NUMERICAL_PASS',mode=mode,plan=pl)
 maps=c.frozen.ref.verify_case(read(verify(proof['case'])));base=verify(evidence['anchor']).read_bytes();numeric=[]
 for i,x in enumerate(raw):
  repeat=min(i,1);gold=c.old_protocol.expected(maps,0,pl['blocks'],tile,repeat)
  num=observe(x,gold,tile,repeat,raw[0] if i else None)
  if num!=read(verify(row['numeric'][i])):raise ValueError('numeric report differs')
  match=observe(x,base,tile,repeat)
  if match['mismatch_count']!=0:raise ValueError('candidate changed frozen baseline output')
  if num['mismatch_count']!=(6 if tile==0 else 201) or num['max_ulp']!=1:raise ValueError('frozen CPU differences changed')
  numeric.append(num)
 if ex['numeric_status']!='NUMERIC_MISMATCH_UNRESOLVED':raise ValueError('numeric status changed')
 return dict(status='BASELINE_BITS_AND_SAFETY_MATCH',numeric_status='NUMERIC_MISMATCH_UNRESOLVED',numeric=numeric,plan=pl,timing=tt)


def paired(rows):
 if [(x['round'],x['position'],x['variant']) for x in rows]!=[(i,j,n) for i,o in enumerate(ORDER) for j,n in enumerate(o)]:raise ValueError('A/B order/coverage differs')
 if any(x.get('profiled') is not False or x['validation']['status']!='BASELINE_BITS_AND_SAFETY_MATCH' for x in rows):raise ValueError('invalid performance row')
 ws={x['validation']['plan']['total'] for x in rows}
 if len(ws)!=1:raise ValueError('workspace drift')
 stats_={n:{k:stats([v for x in rows if x['variant']==n for v in x['validation']['timing'][k]['samples_ms']],90) for k in ('execute_sync','prepare')} for n in ('baseline','candidate')}
 speed=stats_['baseline']['execute_sync']['median_ms']/stats_['candidate']['execute_sync']['median_ms']
 ratios=[next(x['validation']['timing']['execute_sync']['median_ms'] for x in rows if x['round']==i and x['variant']=='baseline')/next(x['validation']['timing']['execute_sync']['median_ms'] for x in rows if x['round']==i and x['variant']=='candidate') for i in range(3)]
 return dict(status='GUARD_AB_PAIRED_MEASURED',statistics=stats_,speedup=speed,round_speedups=ratios,
  decision='GO_FURTHER_REVIEW' if speed>=1.03 and all(x>1 for x in ratios) else 'NO_GO_NO_MATERIAL_IMPROVEMENT',
  numeric_status='NUMERIC_MISMATCH_UNRESOLVED',production='NOT_APPROVED',device_kernel_time='UNAVAILABLE_NO_DEVICE_TIMESTAMPS',
  interpretation='Net compiled implementation effect, not pure Guard instruction time; no historical samples used')
