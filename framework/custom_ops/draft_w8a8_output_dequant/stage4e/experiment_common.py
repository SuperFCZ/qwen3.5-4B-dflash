"""Explicitly unequal guard protocols; one immutable Golden and paired fresh timing."""
import json,math,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'stage4c'))
import bench_common as baseline
from bench_common import full,stage4b,identity,verify,verify_tree,statistics_ms
from guard_variants import POLICIES,COVERAGE,verify_audit
import pair_common as d_common
import run_stage4d as d_run
import build_pair as d_build
c4=d_common.c4;common=baseline.common;write=baseline.write;fresh=baseline.fresh
ORDER=(('B0','B1','B2'),('B1','B2','B0'),('B2','B0','B1'))
NAME='DFlashW8A8FullGate';SNAKE='d_flash_w8_a8_full_gate'


def sources():
    own=[p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts]
    return [identity(p) for p in sorted(own)]+d_common.sources()


def evidence(args):
    # Revalidate accepted D reports with their original validator and original identities.
    prior=d_common.evidence(args)
    build_path=args.stage4d_root/'build/build-report.json'
    bp=args.stage4d_root/'benchmark/benchmark-report.json'
    sp=args.stage4d_root/'smoke/smoke-report.json'
    pp=args.stage4d_root/'profile/profile-report.json'
    b=json.loads(build_path.read_text())
    context=dict(sources=d_common.sources(),evidence=prior,device=args.device,pair_build_report=identity(build_path))
    d_common.validate_build(b,context,d_build.installed)
    d_run.accepted_report(sp,'STAGE4D_SCOPED_CORRECTNESS_MATCH',context,b)
    d_run.accepted_report(bp,'STAGE4D_AB_MEASURED_DIAGNOSTIC_ONLY',context,b)
    prof=json.loads(pp.read_text());verify_tree(prof)
    if prof.get('status')!='STAGE4D_PROFILES_COLLECTED_DIAGNOSTIC_ONLY' or prof.get('phase')!='complete' or prof.get('benchmark_report')!=identity(bp) or any(prof.get(k)!=v for k,v in context.items()):raise ValueError('Stage4D Profile binding differs')
    if set(prof.get('profiles',{}))!={'A','B'}:raise ValueError('Stage4D profiles incomplete')
    rows=[]
    for v,entry in prof['profiles'].items():
        row=entry['execution'];folder=Path(row['execution']['path']).parent;ex=json.loads(verify(row['execution']).read_text())
        if row.get('variant')!=v or row.get('safety_protocol')!=d_common.POLICIES[v] or row.get('status')!='SCOPED_GOLDEN_MATCH' or row.get('profiled') is not True or row.get('formal_latency_evidence') is not False:raise ValueError('old profile scope differs')
        if d_common.check_execution(ex,args.device,v,True)!=row['timing'] or baseline.verify_outputs(prior,folder,True)!=row['outputs']:raise ValueError('old profile correctness/timing differs')
        plan=full.plan_from_log((folder/'runner.log').read_text(),0,0)
        if plan!=row['plan'] or plan['core_policy']!=prior['policy'] or json.loads(verify(row['admission']).read_text())!=stage4b.admission_expected(ex,plan,False):raise ValueError('old profile resource evidence differs')
        rows.append(row)
    d_run.same_resources(rows)
    return dict(**prior,stage4d_sources=d_common.sources(),stage4d_build=identity(build_path),stage4d_smoke=identity(sp),
        stage4d_benchmark=identity(bp),stage4d_profile=identity(pp),stage4d_b_kernel=b['source_audit']['variants']['B']['kernel'],
        stage4d_source_audit=b['source_audit'],historical_samples='NOT_INCLUDED_IN_NEW_MEASUREMENTS')


def check_execution(ex,device,variant,timed):
    expected=dict(status='EXECUTION_COMPLETED_GOLDEN_MATCH',op=NAME,soc='Ascend310P3',shape=[16,2560,19456],tile_count=304,group_count=20,
        output_mode=0,core_limit=0,resource_admission='BEFORE_EXECUTION',ub_guard_protocol='VARIANT_SCOPED_UB_AND_ARENA_CHECKS',
        guards=True,readonly=True,repeat_equal=True,repetitions=2,device=device,cpu_fallback=False,layout='nd',input_count=3,
        poison_verified=True,poison_patterns=[127,255],stage4e=True,variant=variant,safety_protocol=POLICIES[variant],
        strict_golden_match=True,timed_prepare_io=False,kernel_guards_unchanged=False,timed_log_checked=timed,timed_log_stable=True,
        prepare_calls=38 if timed else 2,verify_calls=4 if timed else 2)
    if any(type(ex.get(k)) is not type(v) or ex[k]!=v for k,v in expected.items()):raise ValueError('variant/guard/Golden/protocol mismatch')
    t=ex.get('timing',{})
    if not timed:
        if t.get('status')!='NOT_RUN' or t.get('warmup')!=0 or t.get('repetitions')!=0:raise ValueError('untimed Smoke required')
        return None
    expected_t=dict(status='MEASURED',protocol='continuous-v1',warmup=5,repetitions=30,per_timed_call_readback=False,
        per_timed_call_poison=False,correctness_before_calls=2,correctness_after_calls=1,timed_tail_checked=True)
    if any(t.get(k)!=v for k,v in expected_t.items()):raise ValueError('exact continuous-v1 5/30 required')
    stats={name:statistics_ms(t.get(name,{}).get('samples_ms',[]),30) for name in ('execute_sync','prepare')}
    for name,row in stats.items():
        for key in ('median_ms','p95_ms'):
            if not math.isclose(row[key],t[name].get(key,float('nan')),rel_tol=1e-7,abs_tol=1e-9):raise ValueError('timing summary differs from samples')
    return stats


def summarize(runs):
    if [(e['round'],e['variant'],e['position']) for e in runs]!=[(i,v,j) for i,row in enumerate(ORDER) for j,v in enumerate(row)]:raise ValueError('three exact rotating rounds required')
    for e in runs:
        if e['status']!='SCOPED_GOLDEN_MATCH' or e['profiled'] or not e['formal_latency_evidence'] or e['safety_protocol']!=POLICIES[e['variant']]:raise ValueError('invalid scoped/unprofiled run')
    stats={v:{kind:statistics_ms([x for e in runs if e['variant']==v for x in e['timing'][kind]['samples_ms']],90) for kind in ('execute_sync','prepare')} for v in POLICIES}
    def contrasts(medians):
        return {v:dict(delta_B0_minus_variant_ms=medians['B0']-medians[v],ratio_B0_over_variant=medians['B0']/medians[v]) for v in ('B1','B2')}
    rounds=[]
    for i,order in enumerate(ORDER):
        medians={e['variant']:e['timing']['execute_sync']['median_ms'] for e in runs if e['round']==i}
        rounds.append(dict(round=i,order=list(order),medians_ms=medians,contrasts=contrasts(medians)))
    return dict(variants=stats,rounds=rounds,contrasts=contrasts({v:stats[v]['execute_sync']['median_ms'] for v in POLICIES}),
        interpretation='Net unprofiled effect of each exact guard predicate removal, including codegen. Independent contrasts must not be added or called pure Scalar time.',
        safety='ALL_VARIANTS_DIAGNOSTIC_ONLY_WEAKER_THAN_STAGE4B',coverage=COVERAGE)


def validate_build(report,current,installed):
    verify_tree(report)
    if report.get('status')!='STAGE4E_BUILT_NPU_NOT_RUN' or any(report.get(k)!=current[k] for k in ('sources','evidence','device')):raise ValueError('paired build/evidence binding differs')
    verify_audit(report['source_audit'],verify)
    if set(report['variants'])!=set(POLICIES):raise ValueError('all three private variants required')
    for v,b in report['variants'].items():
        if b['safety_protocol']!=POLICIES[v] or installed(Path(b['opp']))!=b['installed']:raise ValueError('variant private OPP/coverage changed')
