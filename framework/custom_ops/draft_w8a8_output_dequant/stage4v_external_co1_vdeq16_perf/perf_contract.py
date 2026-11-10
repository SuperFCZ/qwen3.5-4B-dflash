"""CPU specification for exploratory admission. Never authorizes a device launch."""
import math
import statistics
import numpy as np

SAFETY=('raw_matches','readonly','gm_guards','ub_guards','arena_guards',
        'workspace_admitted','sync_completed','no_cpu_fallback','witness_valid')
KEYS=((0,84),(1,53),(1,201),(5,138),(7,235),(15,210))


def half_order(bits):
 x=bits.astype(np.int32)
 return np.where(x & 0x8000,0x8000-(x & 0x7fff),0x8000+x)


def compare_half(actual,reference):
 if actual.dtype!=np.dtype('<f2') or reference.dtype!=np.dtype('<f2') or actual.shape!=(16,256) or reference.shape!=(16,256):
  raise ValueError('exact compact FP16 [16,256] required')
 if not np.isfinite(actual).all() or not np.isfinite(reference).all():raise ValueError('nonfinite/Poison is not admitted')
 a=actual.view('<u2');b=reference.view('<u2');bad=np.argwhere(a!=b)
 ulp=np.abs(half_order(a)-half_order(b))
 return dict(bit_mismatches=int(len(bad)),max_ulp=int(ulp.max()),
  max_abs=float(np.max(np.abs(actual.astype(np.float64)-reference.astype(np.float64)))),
  differences=[dict(row=int(m),channel=int(n),actual_bits=int(a[m,n]),reference_bits=int(b[m,n]),ulp=int(ulp[m,n])) for m,n in bad])


def smoke_observation(raw,fp16,raw_reference,historical_actual,cpu_reference,safety):
 """Offline analysis of already supplied bytes; safety is an external prerequisite.
 This function cannot create or validate Runner/OPP/source/runtime identities.
 """
 if set(safety)!=set(SAFETY) or any(type(v)is not bool or not v for v in safety.values()):raise ValueError('basic safety not satisfied')
 if len(raw)!=2 or len(fp16)!=2:raise ValueError('two independent executions required')
 if raw_reference.dtype!=np.dtype('<i4') or raw_reference.shape!=(16,256):raise ValueError('raw reference layout')
 for a in raw:
  if a.dtype!=np.dtype('<i4') or a.shape!=(16,256) or a.tobytes()!=raw_reference.tobytes():raise ValueError('all4096 Raw integers must match')
 if fp16[0].tobytes()!=fp16[1].tobytes():raise ValueError('FP16 Repeat mismatch')
 results=[dict(actual=compare_half(x,historical_actual),cpu=compare_half(x,cpu_reference)) for x in fp16]
 return dict(status='OFFLINE_OBSERVATION_ANALYZED_NOT_DEVICE_ADMISSION',comparisons=results,
  numeric_cpu_bitwise=all(x['cpu']['bit_mismatches']==0 for x in results),
  quality='UNRESOLVED',exploration_only=True,
  key_positions=[dict(row=m,channel=n,raw_i32=int(raw[0][m,n]),fp16_bits=int(fp16[0].view('<u2')[m,n])) for m,n in KEYS])


def historical_tile0(raw):
 a=np.frombuffer(raw,dtype='<f2')
 if a.size!=16*19456:raise ValueError('historical full output length')
 return a.reshape(16,19456)[:,:256].copy()


def summary_ms(samples):
 if not samples or any(type(x) not in (int,float) or not math.isfinite(x) or x<=0 for x in samples):raise ValueError('finite positive timing samples required')
 ordered=sorted(samples)
 return dict(samples_ms=list(samples),count=len(samples),median_ms=statistics.median(samples),
  p95_ms=ordered[math.ceil(.95*len(ordered))-1],p95_definition='nearest_rank',min_ms=min(samples),max_ms=max(samples))


def benchmark_summary(rounds):
 if len(rounds)!=3:raise ValueError('exactly three rounds for first experiment')
 samples=[];reports=[]
 for i,r in enumerate(rounds):
  if r.get('round')!=i or r.get('warmup')!=5 or r.get('repetitions')!=30 or len(r.get('execute_sync_ms',[]))!=30:raise ValueError('5/30/3 protocol differs')
  if r.get('instrumented') is not False or r.get('host_logs_during_timing')!=0 or r.get('workspace_drift') is not False:raise ValueError('timing contaminated')
  if any(r.get(k) is not True for k in ('two_poison_prechecks','timed_tail','poison_postcheck','safety','identities_bound')):raise ValueError('timing admission missing')
  reports.append(dict(round=i,execute_sync=summary_ms(r['execute_sync_ms']),prepare=summary_ms(r['prepare_ms'])))
  samples.extend(r['execute_sync_ms'])
 return dict(status='CPU_SAMPLE_STATISTICS_ONLY_NOT_BENCHMARK_PASS',execute_sync=summary_ms(samples),rounds=reports,
  device_execution_time='UNAVAILABLE_NO_DEVICE_TIMESTAMPS',scope='SINGLE_TILE_M16_K2560_N256_WITH_ECHO_AND_DIAGNOSTICS',
  layer_projection='PROHIBITED',numeric_acceptance='SEPARATE_FROM_TIMING')


def admission():
 return dict(status='BLOCKED_NORMAL_PATH_EVENT_BUDGET',device_execution_allowed=False,
  producer='EXTERNAL_CO1_HISTORICAL_ECHO_PATH',selected_design='B',ub_accounted=240384,reserve=8192,
  deferred=['production-level hardware-fault recovery','six-position arithmetic root cause','production numerical acceptance'],
  blockers=['selected input-copy helper call graph peak per-HardEvent usage with application reservations has not been fully bounded',
   'compile interface checks do not prove same-pool Fetch-use completion at transition from SDK Iterate to application fences',
   'new B actual TPipe/Formal Host and Workspace admission NOT_RUN'],
  minimal_fix=['finish fixed IsNeedUB ND helper event ledger; preserve SDK-private M_MTE1/MTE1_M IDs',
   'reserve only proven-safe normal completion tickets before work; retain until observed completion; no app S_V in conversion window',
   'preserve End after completed consumers; ordinary check failures converge on the same completed terminal path; no unsafe early return',
   'implement and re-admit new B ABI only after normal event contract closes'],
  numerical_standard='UNCHANGED; FP16 differences may be reported in exploration, never converted to numerical PASS',
  kernel='NOT_CREATED',opp='NOT_CREATED',host='NOT_RUN',build='NOT_RUN',smoke='NOT_RUN',benchmark='NOT_RUN',profile='NOT_RUN')
