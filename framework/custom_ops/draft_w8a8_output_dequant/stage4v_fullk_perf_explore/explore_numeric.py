"""Exact numeric observations, independently of exploratory timing admission."""
import math
import statistics
import numpy as np
KEYS={(0,84),(1,53),(1,201),(5,138),(7,235),(15,210)}
BYTES=622592


def observe(actual,expected,tile,repeat,first=None):
 if len(actual)!=BYTES or len(expected)!=BYTES or first is not None and len(first)!=BYTES:raise ValueError('full ABI requires622592 bytes')
 if tile not in (0,-1) or repeat not in (0,1):raise ValueError('invalid observation scope')
 a=np.frombuffer(actual,'<u2').reshape(16,19456);g=np.frombuffer(expected,'<u2').reshape(16,19456)
 active=np.ones(a.shape,bool)
 if tile==0:
  active[:,256:]=False
  if np.any(a[~active]!=(0x7f7f if repeat==0 else 0xffff)):raise ValueError('inactive poison changed')
 x=a[active];y=g[active]
 if np.any((x&0x7c00)==0x7c00) or np.any((y&0x7c00)==0x7c00):raise ValueError('active nonfinite/Poison')
 if first is not None and not np.array_equal(a[active],np.frombuffer(first,'<u2').reshape(a.shape)[active]):raise ValueError('active Repeat drift')
 def ordered(z):
  z=z.astype(np.int32);return np.where(z&32768,32768-(z&32767),32768+z)
 ulps=np.abs(ordered(a)-ordered(g));err=np.zeros(a.shape,dtype=np.float64);err[active]=np.abs(x.view('<f2').astype(np.float64)-y.view('<f2').astype(np.float64))
 bad=np.argwhere(active&(a!=g));rows=[]
 for m,n in bad:rows.append(dict(row=int(m),channel=int(n),actual_bits=int(a[m,n]),expected_bits=int(g[m,n]),ulp=int(ulps[m,n]),absolute_error=float(err[m,n])))
 return dict(numeric_status='NUMERIC_MISMATCH_UNRESOLVED' if len(bad) else 'BITWISE_REFERENCE_OBSERVATION',active_elements=int(active.sum()),mismatch_count=int(len(bad)),max_ulp=int(ulps[active].max()),max_absolute_error=float(err[active].max()),differences=rows)


def known_six(result):
 rows=result['differences']
 if len(rows)!=6 or {(d['row'],d['channel']) for d in rows}!=KEYS or any(d['ulp']!=1 for d in rows):raise ValueError('historical mismatch set differs from frozen six1-ULP observations')
 if any(abs(np.array(d['actual_bits'],dtype='<u2').view('<f2').item())>=abs(np.array(d['expected_bits'],dtype='<u2').view('<f2').item()) for d in rows):raise ValueError('historical mismatch direction differs')
 return 'SIX_FROZEN_COORDINATES_ONE_ULP_TOWARD_ZERO'


def stats(samples,count):
 if len(samples)!=count or any(type(x) not in (float,int) or not math.isfinite(x) or x<=0 for x in samples):raise ValueError('timing count/value invalid')
 s=sorted(samples)
 return dict(samples_ms=samples,count=count,median_ms=statistics.median(s),p95_ms=s[math.ceil(count*.95)-1],min_ms=s[0],max_ms=s[-1],p95_method='nearest_rank')


def timing(ex,timed):
 t=ex['timing']
 want=dict(status='MEASURED' if timed else 'NOT_RUN',protocol='continuous-v1' if timed else 'checked-v1',warmup=5 if timed else 0,repetitions=30 if timed else 0,per_timed_call_readback=not timed,per_timed_call_poison=not timed,correctness_before_calls=2,correctness_after_calls=1 if timed else 0,timed_tail_checked=timed)
 if any(type(t.get(k))!=type(v) or t[k]!=v for k,v in want.items()):raise ValueError('timing protocol drift')
 if timed:return {k:stats(t[k]['samples_ms'],30) for k in ('execute_sync','prepare')}
 if t['execute_sync']['samples_ms'] or t['prepare']['samples_ms']:raise ValueError('Smoke must be untimed')
 return None
