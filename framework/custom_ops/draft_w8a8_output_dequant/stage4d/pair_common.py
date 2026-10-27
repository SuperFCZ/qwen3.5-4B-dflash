"""Explicitly unequal guard protocols; one immutable Golden and paired fresh timing."""
import json,math,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'stage4c'))
import bench_common as baseline
import run_stage4c as c4
from bench_common import common,full,stage4b,identity,verify,verify_tree,write,fresh,statistics_ms,OUTPUTS
from guard_variant import FROZEN,KERNEL,POLICIES,verify_audit
ORDER=(('A','B'),('B','A'),('A','B'))
NAME='DFlashW8A8FullGate';SNAKE='d_flash_w8_a8_full_gate'


def sources():
    own=[p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts]
    return [identity(p) for p in sorted(own)]+[identity(FROZEN/KERNEL),identity(common.QUANT/'test/runner_common.h'),identity(HERE.parent/'stage4c/bench_common.py'),identity(HERE.parent/'stage4c/run_stage4c.py'),identity(HERE.parent/'stage4c/profile_metrics.py'),identity(FROZEN/'build_full.py')]


def evidence(args):
    proof=baseline.evidence(args)
    bp=args.stage4c_root/'benchmark/benchmark-report.json';br=json.loads(bp.read_text());verify_tree(br)
    if br.get('evidence')!=proof or br.get('sources')!=baseline.sources() or br.get('device')!=args.device:raise ValueError('accepted Stage4C baseline identity differs')
    c4.prior_report(bp,'STAGE4C_BASELINE_MEASURED',br)
    p=args.stage4c_root/'profile/profile-report.json';prof=json.loads(p.read_text());verify_tree(prof)
    main_id=identity(args.stage4c_root/'profile/benchmark-report.json');main=json.loads(verify(main_id).read_text());verify_tree(main)
    if prof.get('status')!='STAGE4C_PROFILE_COMPLETED' or prof.get('benchmark_report')!=identity(bp) or prof.get('formal_latency_evidence') is not False or main.get('status')!='STAGE4C_PROFILE_COMPLETED' or any(main.get(k)!=br[k] for k in ('sources','evidence','device','runner_build_report')):raise ValueError('accepted Stage4C profiling binding differs')
    row=prof['execution'];ex=json.loads(verify(row['execution']).read_text())
    if row.get('core_limit')!=0 or row.get('profiled') is not True or row.get('formal_latency_evidence') is not False or baseline.check_execution(ex,args.device,0,True)!=row['timing']:raise ValueError('old profile scope differs')
    if baseline.verify_outputs(proof,Path(row['execution']['path']).parent,True)!=row['outputs']:raise ValueError('old profile outputs differ')
    plan=full.plan_from_log((Path(row['execution']['path']).parent/'runner.log').read_text(),0,0)
    if plan!=row['plan'] or plan['core_policy']!=proof['policy'] or json.loads(verify(row['admission']).read_text())!=stage4b.admission_expected(ex,plan,False):raise ValueError('old profile resource proof differs')
    frozen_build=json.loads(verify(proof['build_report']).read_text())
    return dict(stage4b=proof,stage4b_sources=frozen_build['sources'],stage4c_sources=br['sources'],
        stage4c_baseline=identity(bp),stage4c_profile=identity(p),stage4c_profile_main=main_id,
        policy=proof['policy'],preflight=proof['preflight'],files=proof['files'],prior_measurements='IDENTITY_AND_CORRECTNESS_ONLY; not fresh A/B samples')


def check_execution(ex,device,variant,timed):
    expected=dict(status='EXECUTION_COMPLETED_GOLDEN_MATCH',op=NAME,soc='Ascend310P3',shape=[16,2560,19456],tile_count=304,group_count=20,
        output_mode=0,core_limit=0,resource_admission='BEFORE_EXECUTION',ub_guard_protocol='UB_AND_CORE_ARENA_POISON_ON_FAILURE',
        guards=True,readonly=True,repeat_equal=True,repetitions=2,device=device,cpu_fallback=False,layout='nd',input_count=3,
        poison_verified=True,poison_patterns=[127,255],stage4d=True,variant=variant,safety_protocol=POLICIES[variant],
        strict_golden_match=True,timed_prepare_io=False,kernel_guards_unchanged=variant=='A',timed_log_checked=timed,timed_log_stable=True,
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
    if [(e['round'],e['variant']) for e in runs]!=[(i,v) for i,row in enumerate(ORDER) for v in row]:raise ValueError('three alternating A/B rounds required')
    if [e['position'] for e in runs]!=[0,1]*3:raise ValueError('round positions differ')
    for e in runs:
        if e['status']!='SCOPED_GOLDEN_MATCH' or e['profiled'] or not e['formal_latency_evidence'] or e['safety_protocol']!=POLICIES[e['variant']]:raise ValueError('invalid scoped/unprofiled run')
    stats={v:{kind:statistics_ms([x for e in runs if e['variant']==v for x in e['timing'][kind]['samples_ms']],90) for kind in ('execute_sync','prepare')} for v in POLICIES}
    rounds=[]
    for i,row in enumerate(ORDER):
        a=next(e for e in runs if e['round']==i and e['variant']=='A')['timing']['execute_sync']['median_ms']
        b=next(e for e in runs if e['round']==i and e['variant']=='B')['timing']['execute_sync']['median_ms']
        rounds.append(dict(round=i,order=list(row),A_median_ms=a,B_median_ms=b,delta_A_minus_B_ms=a-b,speedup_A_over_B=a/b))
    a=stats['A']['execute_sync']['median_ms'];b=stats['B']['execute_sync']['median_ms']
    return dict(variants=stats,rounds=rounds,delta_A_minus_B_ms=a-b,speedup_A_over_B=a/b,
        all_round_medians_improved=all(x['delta_A_minus_B_ms']>0 for x in rounds),
        interpretation='Net unprofiled effect of removing group scan conditional, including codegen/scheduling consequences; not pure GetValue or Scalar time.',
        safety='B remains DIAGNOSTIC_ONLY_WEAKER_GUARD_COVERAGE regardless of speed or numerical match')


def validate_build(report,current,installed):
    verify_tree(report)
    if report.get('status')!='STAGE4D_BUILT_NPU_NOT_RUN' or any(report.get(k)!=current[k] for k in ('sources','evidence','device')):raise ValueError('paired build/evidence binding differs')
    verify_audit(report['source_audit'],verify)
    if set(report['variants'])!=set(POLICIES):raise ValueError('both private variants required')
    for v,b in report['variants'].items():
        if b['safety_protocol']!=POLICIES[v] or installed(Path(b['opp']))!=b['installed']:raise ValueError('variant private OPP/coverage changed')
