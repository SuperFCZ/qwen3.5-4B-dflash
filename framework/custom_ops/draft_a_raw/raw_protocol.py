"""Complete Token/KV checks and fresh per-gear paired statistics."""
import math,statistics
import raw_common as c
OUTPUTS=['draft_top1']+[f'd{i}_{x}' for i in range(5) for x in ('key','value')]


def stats(values,count):
    if len(values)!=count or any(type(x) not in (float,int) or not math.isfinite(x) or x<=0 for x in values):raise ValueError('invalid sample inventory')
    s=sorted(values)
    return dict(samples_ms=values,count=count,median_ms=statistics.median(s),p95_ms=s[math.ceil(count*.95)-1],min_ms=s[0],max_ms=s[-1])


def verify_run(row,gear,device,timed):
    e=c.read(c.verify(row['execution']))
    expected=dict(status='PASS',guards=True,readonly=True,repeat=True,cpu_fallback=False,golden_match=True,poison_verified=True,sync_completed=True,
        timed_log_stable=True,device=device,verification_calls=4 if timed else 2,row_permutation_checked=False)
    if any(type(e.get(k))!=type(v) or e[k]!=v for k,v in expected.items()):raise ValueError('OM Runner safety/Golden contract differs')
    if set(row['outputs'])!=set(OUTPUTS) or set(gear['golden'])!=set(OUTPUTS):raise ValueError('all11 outputs required')
    for name,files in row['outputs'].items():
        if len(files)!=(4 if timed else 2):raise ValueError('repeat/tail/postcheck inventory')
        wanted=c.verify(gear['golden'][name]).read_bytes()
        for file in files:
            if c.verify(file).read_bytes()!=wanted:raise ValueError('full logical output mismatch: '+name)
    t=e['timing'];wanted=dict(status='MEASURED' if timed else 'NOT_RUN',protocol='continuous-v1' if timed else 'checked-v1',warmup=5 if timed else 0,repetitions=30 if timed else 0,
        correctness_before_calls=2,correctness_after_calls=1 if timed else 0,timed_tail_checked=timed,per_timed_call_readback=not timed,per_timed_call_poison=not timed)
    if any(t.get(k)!=v for k,v in wanted.items()):raise ValueError('timing protocol differs')
    if t['prepare']['samples_ms']:raise ValueError('model load must not be sampled')
    for key in ('workspace_bytes','model_weight_bytes'):
        if type(e.get(key)) is not int or e[key]<0:raise ValueError('actual model memory missing')
    if not timed and t['execute_sync']['samples_ms']:raise ValueError('Smoke timed unexpectedly')
    return dict(status='ALL_TOKEN_KV_BITS_AND_SAFETY_MATCH',timing=stats(t['execute_sync']['samples_ms'],30) if timed else None,
        workspace_bytes=e['workspace_bytes'],model_weight_bytes=e['model_weight_bytes'])


def order(round_id,gear):
    return ('B','R') if (round_id+(gear==64))%2==0 else ('R','B')


def summarize(rows):
    wanted=[(i,g,j,v) for i in range(3) for g in (16,64) for j,v in enumerate(order(i,g))]
    if [(r['round'],r['gear'],r['position'],r['variant']) for r in rows]!=wanted:raise ValueError('paired schedule incomplete')
    result={}
    for g in (16,64):
        values={}
        for v in ('B','R'):
            selected=[r for r in rows if r['gear']==g and r['variant']==v]
            if any(r['validation']['status']!='ALL_TOKEN_KV_BITS_AND_SAFETY_MATCH' or r.get('profiled') is not False for r in selected):raise ValueError('unadmitted/profiled samples')
            if len({(r['validation']['workspace_bytes'],r['validation']['model_weight_bytes']) for r in selected})!=1:raise ValueError('per-OM memory drift')
            values[v]=stats([s for r in selected for s in r['validation']['timing']['samples_ms']],90)
        ratios=[next(r['validation']['timing']['median_ms'] for r in rows if r['round']==i and r['gear']==g and r['variant']=='B')/next(r['validation']['timing']['median_ms'] for r in rows if r['round']==i and r['gear']==g and r['variant']=='R') for i in range(3)]
        result[str(g)]=dict(statistics=values,median_speedup=values['B']['median_ms']/values['R']['median_ms'],round_speedups=ratios,
            p95_regression=values['R']['p95_ms']>values['B']['p95_ms'])
    return dict(gears=result,scope='FULL_FIVE_LAYER_DRAFT_OM_EXECUTE_SYNC',historical_samples_used=False,
        aggregate_speedup='NOT_DEFINED_WITHOUT_GEAR_MIX',adoption='NOT_DECIDED',decode='NOT_RUN',device_only_time='UNAVAILABLE')
