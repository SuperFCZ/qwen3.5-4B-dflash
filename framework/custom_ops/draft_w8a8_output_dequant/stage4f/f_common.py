"""Explicitly unequal guard protocols; one immutable Golden and paired fresh timing."""
import json,math,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'stage4c'))
import bench_common as baseline
from bench_common import full,stage4b,identity,verify,verify_tree,statistics_ms
from workspace_variant import POLICIES,COVERAGE,verify_audit
sys.path.insert(0,str(HERE.parent/"stage4e"))
import experiment_common as e_common
import run_stage4e as e_run
import build_variants as e_build
import pair_common as d_common
c4=d_common.c4;common=baseline.common;write=baseline.write;fresh=baseline.fresh
ORDER=(('A0','A1'),('A1','A0'),('A0','A1'),('B0','B1'),('B1','B0'),('B0','B1'))
SMOKE_ORDER=[(v,limit) for pair in ('A','B') for limit in (0,2) for v in (pair+'0',pair+'1')]
NAME='DFlashW8A8FullGate';SNAKE='d_flash_w8_a8_full_gate'


def sources():
    own=[p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts]
    return [identity(p) for p in sorted(own)]+e_common.sources()


def evidence(args):
    prior=e_common.evidence(args)
    build_path=args.stage4e_root/'build/build-report.json'
    bp=args.stage4e_root/'benchmark/benchmark-report.json';sp=args.stage4e_root/'smoke/smoke-report.json'
    pp=args.stage4e_root/'profile/profile-report.json'
    b=json.loads(build_path.read_text());context=dict(sources=e_common.sources(),evidence=prior,device=args.device,pair_build_report=identity(build_path))
    e_common.validate_build(b,context,e_build.installed)
    e_run.accepted_report(sp,'STAGE4E_SCOPED_CORRECTNESS_MATCH',context,b)
    e_run.accepted_report(bp,'STAGE4E_CONTRASTS_MEASURED_DIAGNOSTIC_ONLY',context,b)
    prof=json.loads(pp.read_text());verify_tree(prof)
    if prof.get('status')!='STAGE4E_PROFILES_COLLECTED_DIAGNOSTIC_ONLY' or prof.get('benchmark_report')!=identity(bp) or any(prof.get(k)!=v for k,v in context.items()):raise ValueError('accepted Stage4E profile binding differs')
    if set(prof.get('profiles',{}))!=set(e_common.POLICIES):raise ValueError('incomplete Stage4E profile coverage')
    for v,item in prof['profiles'].items():
        row=item['execution'];folder=Path(verify(row['execution'])).parent;ex=json.loads(verify(row['execution']).read_text())
        if row['variant']!=v or row['safety_protocol']!=e_common.POLICIES[v] or row['profiled'] is not True or row['formal_latency_evidence'] is not False:raise ValueError('old profiling scope differs')
        if e_common.check_execution(ex,args.device,v,True)!=row['timing'] or baseline.verify_outputs(prior,folder,True)!=row['outputs']:raise ValueError('old profile numerical/timing proof differs')
        plan=full.plan_from_log((folder/'runner.log').read_text(),0,0)
        if plan!=row['plan'] or plan['core_policy']!=prior['policy'] or json.loads(verify(row['admission']).read_text())!=stage4b.admission_expected(ex,plan,False):raise ValueError('old profile resource evidence differs')
    return dict(**prior,stage4e_sources=e_common.sources(),stage4e_build=identity(build_path),stage4e_benchmark=identity(bp),stage4e_smoke=identity(sp),stage4e_profile=identity(pp))


def check_execution(ex,device,variant,timed,limit=0):
    if limit not in (0,2) or (timed and limit!=0):raise ValueError("invalid smoke/timing Core scope")
    expected=dict(status='EXECUTION_COMPLETED_GOLDEN_MATCH',op=NAME,soc='Ascend310P3',shape=[16,2560,19456],tile_count=304,group_count=20,
        output_mode=0,core_limit=limit,resource_admission='BEFORE_EXECUTION',ub_guard_protocol='VARIANT_SCOPED_UB_AND_ARENA_CHECKS',
        guards=True,readonly=True,repeat_equal=True,repetitions=2,device=device,cpu_fallback=False,layout='nd',input_count=3,
        poison_verified=True,poison_patterns=[127,255],stage4f=True,variant=variant,safety_protocol=POLICIES[variant],
        strict_golden_match=True,timed_prepare_io=False,kernel_guards_unchanged=variant.startswith('A'),timed_log_checked=timed,timed_log_stable=True,
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
    if [(e['round'],e['variant'],e['position']) for e in runs]!=[(i,v,j) for i,row in enumerate(ORDER) for j,v in enumerate(row)]:raise ValueError('three alternating rounds per pair required')
    for e in runs:
        if e['status']!='SCOPED_GOLDEN_MATCH' or e['profiled'] or not e['formal_latency_evidence'] or e['core_limit']!=0 or e['safety_protocol']!=POLICIES[e['variant']]:raise ValueError('invalid formal run')
    stats={v:{kind:statistics_ms([x for e in runs if e['variant']==v for x in e['timing'][kind]['samples_ms']],90) for kind in ('execute_sync','prepare')} for v in POLICIES}
    pairs={}
    for pair in ('A','B'):
        a,b=pair+'0',pair+'1';am=stats[a]['execute_sync']['median_ms'];bm=stats[b]['execute_sync']['median_ms']
        rounds=[]
        for i,row in enumerate(ORDER):
            if row[0][0]!=pair:continue
            m={e['variant']:e['timing']['execute_sync']['median_ms'] for e in runs if e['round']==i}
            rounds.append(dict(round=i,order=list(row),medians_ms=m,delta_baseline_minus_candidate_ms=m[a]-m[b],speedup=m[a]/m[b]))
        pairs[pair]=dict(baseline=a,candidate=b,safety_protocol=POLICIES[a],rounds=rounds,delta_baseline_minus_candidate_ms=am-bm,speedup=am/bm)
    return dict(variants=stats,pairs=pairs,coverage=COVERAGE,
        interpretation='A0/A1 and B0/B1 are separate matched-safety contrasts. No A0/B1 causal comparison. No automatic improvement/adoption claim.')


def binary_summary(variants):
    result={}
    for pair in ('A','B'):
        a,b=pair+'0',pair+'1'
        left={x['binary']['sha256'] for x in variants[a]['installed']['kernels']}
        right={x['binary']['sha256'] for x in variants[b]['installed']['kernels']}
        if not left or not right:raise ValueError('missing kernel binary')
        equal=left==right
        result[pair]=dict(baseline=a,candidate=b,exact_sha_equal=equal,
            status='IDENTICAL_BINARY_NO_COMPILED_CHANGE' if equal else 'DIFFERENT_BINARY_NOT_PROOF_OF_INSTRUCTION_SAVING',
            optimization_success='NOT_ESTABLISHED',
            interpretation='Identical binaries cannot establish a compiled optimization; different ELF hashes may include debug/path changes.')
    return result


def validate_build(report,current,installed):
    verify_tree(report)
    if report.get('status')!='STAGE4F_BUILT_NPU_NOT_RUN' or any(report.get(k)!=current[k] for k in ('sources','evidence','device','sdk_binding')):raise ValueError('paired build/evidence binding differs')
    verify_audit(report['source_audit'],verify)
    if report.get('binary_comparison')!=binary_summary(report['variants']):raise ValueError('binary comparison differs from verified identities')
    if set(report['variants'])!=set(POLICIES):raise ValueError('all four private variants required')
    for v,b in report['variants'].items():
        if b['safety_protocol']!=POLICIES[v] or installed(Path(b['opp']))!=b['installed']:raise ValueError('variant private OPP/coverage changed')


def sdk_binding(cann):
    contract=json.loads((HERE/'sdk-contract.json').read_text());records=[]
    for row in contract['files']:
        p=(cann/row['relative']).resolve(strict=True)
        if cann.resolve() not in p.parents:raise ValueError('SDK path escaped selected CANN root')
        item=identity(p)
        if item['sha256']!=row['sha256']:raise ValueError('approved SDK identity changed: '+row['relative'])
        records.append(dict(file=item,lines=row['lines']))
    return dict(contract=identity(HERE/'sdk-contract.json'),files=records,status='APPROVED_IDENTITIES_MATCHED_NOT_NPU_VERIFIED')
