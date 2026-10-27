"""Stage4C evidence, timing validation, and explicit scope of resident-input latency."""
import json,math,statistics,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'stage4b'))
import full_common as full
import full_reference as reference
import run_stage4b as stage4b
from full_common import common,identity,verify,verify_tree,write,fresh,metrics,np
from build_full import installed
LIMITS=(1,2,0)
ORDER=((1,2,0),(2,0,1),(0,1,2))
OUTPUTS=('actual-0.bin','actual-1.bin','benchmark-last.bin','postcheck.bin')


def sources():
    paths=[p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts]
    paths += [HERE.parent/'stage4b/full_common.py',HERE.parent/'stage4b/full_reference.py',HERE.parent/'stage4b/run_stage4b.py',
              HERE.parent/'stage4b/build_full.py',common.QUANT/'test/runner_common.h']
    return [identity(p) for p in sorted(set(paths))]


def evidence(args):
    upstream=full.evidence(args)
    path=args.stage4b_root/'smoke/report.json';r=json.loads(path.read_text());verify_tree(r)
    if r.get('status')!='STAGE4B_FULL_GATE_MATCH' or r.get('phase')!='complete' or r.get('device')!=args.device or r.get('settings')!=full.SETTINGS or r.get('sources')!=full.sources() or r.get('evidence')!=upstream or r.get('performance')!='NOT_RUN':raise ValueError('accepted unchanged Stage4B full-Gate evidence required')
    b=json.loads(verify(r['multi_build_report']).read_text());verify_tree(b)
    if b.get('status')!='STAGE4B_BUILT_NPU_NOT_RUN' or any(b.get(k)!=r[k] for k in ('settings','sources','evidence','device','prepare_report','case')):raise ValueError('Stage4B Build binding differs')
    if installed(Path(b['opp']))!=b['installed']:raise ValueError('accepted OPP identity changed')
    policy=stage4b.validate_host_probe(verify(r['host_probe_report']),r['multi_build_report'],r)
    case=json.loads(verify(r['case']).read_text());maps=reference.verify_case(case)
    limits=[0,2 if min(304,policy['selected'])>2 else 1];modes=[21,0]+[g+1 for g in r.get('diagnostic_groups',[])]
    if [(x['core_limit'],x['mode']) for x in r.get('runs',[])]!=[(l,m) for l in limits for m in modes] or r.get('core_policy')!=policy:raise ValueError('Stage4B run/Core coverage incomplete')
    for row in r['runs']:
        l,m=row['core_limit'],row['mode'];ex=json.loads(verify(row['execution']).read_text());full.check_execution(ex,args.device,m,l)
        plan=full.plan_from_log(verify(r['commands'][f'l{l}_m{m}']['log_identity']).read_text(),m,l)
        if row.get('plan')!=plan or plan['core_policy']!=policy or json.loads(verify(row['resource_admission']).read_text())!=stage4b.admission_expected(ex,plan,False):raise ValueError('Stage4B resource proof differs')
        checked=reference.compare(case,maps,m,plan['blocks'],[verify(x).read_bytes() for x in row['outputs']])
        if row.get('status')!='BITWISE_MATCH' or checked!=row['comparison'] or checked['status']!='BITWISE_MATCH':raise ValueError('Stage4B numeric evidence differs')
    return dict(report=identity(path),build_report=r['multi_build_report'],case=r['case'],opp=b['opp'],installed=b['installed'],
        policy=policy,preflight=upstream['preflight'],files=case['files'],frozen_inputs=case['source']['source']['frozen'],
        scope='same frozen Layer0 Gate/Up, preprocessed INT8 A and encoded Scale resident on NPU',earlier_runners_executed=False)


def statistics_ms(samples,count):
    if len(samples)!=count or any(type(x) not in (int,float) or not math.isfinite(x) or x<=0 for x in samples):raise ValueError('missing/invalid positive timing samples')
    return dict(samples_ms=samples,median_ms=statistics.median(samples),p95_ms=sorted(samples)[math.ceil(count*.95)-1],min_ms=min(samples),max_ms=max(samples))


def check_execution(ex,device,limit,timed):
    expected=dict(status='EXECUTION_COMPLETED_GOLDEN_MATCH',op='DFlashW8A8FullGate',soc='Ascend310P3',shape=[16,2560,19456],tile_count=304,group_count=20,
        output_mode=0,core_limit=limit,resource_admission='BEFORE_EXECUTION',ub_guard_protocol='UB_AND_CORE_ARENA_POISON_ON_FAILURE',
        guards=True,readonly=True,repeat_equal=True,repetitions=2,device=device,cpu_fallback=False,layout='nd',input_count=3,
        poison_verified=True,poison_patterns=[127,255],stage4c=True,strict_golden_match=True,timed_prepare_io=False,kernel_guards_unchanged=True,timed_log_checked=timed,timed_log_stable=True,
        prepare_calls=38 if timed else 2,verify_calls=4 if timed else 2)
    if any(type(ex.get(k)) is not type(v) or ex[k]!=v for k,v in expected.items()):raise ValueError('benchmark safety/Golden/protocol gate failed')
    t=ex.get('timing',{})
    if not timed:
        if t.get('status')!='NOT_RUN' or t.get('warmup')!=0 or t.get('repetitions')!=0:raise ValueError('Smoke must not measure latency')
        return None
    flags=dict(status='MEASURED',protocol='continuous-v1',warmup=5,repetitions=30,per_timed_call_readback=False,
        per_timed_call_poison=False,correctness_before_calls=2,correctness_after_calls=1,timed_tail_checked=True)
    if any(t.get(k)!=v for k,v in flags.items()):raise ValueError('requires exact continuous-v1 5/30 and tail/postcheck')
    result={name:statistics_ms(t.get(name,{}).get('samples_ms',[]),30) for name in ('execute_sync','prepare')}
    for name,item in result.items():
        for key in ('median_ms','p95_ms'):
            if not math.isclose(item[key],t[name].get(key,float('nan')),rel_tol=1e-7,abs_tol=1e-9):raise ValueError('timing summary does not match samples')
    return result


def verify_outputs(proof,folder,timed):
    golden=verify(proof['files']['golden_fp16.bin']).read_bytes()
    names=OUTPUTS if timed else OUTPUTS[:2];items=[identity(folder/n) for n in names]
    for item in items:
        if verify(item).read_bytes()!=golden:raise ValueError('strict frozen Golden mismatch: '+item['path'])
    return items


def preprocessing(proof):
    return dict(status='ONLINE_PREPROCESSING_NOT_RUN',scope='resident preprocessed Kernel latency, not complete FP16-input forward latency',
        components=[dict(name='weight_tile_pack',kind='STATIC_OFFLINE_INITIALIZATION',input_bytes=49807360,output_bytes=49807360,included=False,status='CPU_LAYOUT_VALIDATED_STAGE4B'),
          dict(name='activation_FP16_to_INT8',kind='DYNAMIC_PER_INPUT',input_bytes=81920,output_bytes=40960,included=False,status='ONLINE_NPU_NOT_RUN',contract='saved full-X binary64 scale, RNE/clip[-127,127], zero special case'),
          dict(name='combined_scale',kind='DYNAMIC_PER_INPUT',weight_scale_bytes=778240,output_fp32_bytes=1556480,included=False,status='ONLINE_NPU_NOT_RUN',contract='FP32(FP64(scale_A)*FP64(original FP16 GN Scale))'),
          dict(name='uint64_encode_and_layout',kind='DYNAMIC_PER_INPUT',output_bytes=3112960,included=False,status='ONLINE_NPU_NOT_RUN',contract='existing mask13/bit46 candidate, tile/group/n'),
          dict(name='input_upload',kind='TRANSFER_BEFORE_MEASUREMENT',bytes=52961280,included=False,status='Runner copies inputs once'),
          dict(name='W8A8_Cube_VDEQ16_FP32_reduction',kind='KERNEL_EXECUTION',included=True,status='frozen Stage4B; all in-kernel guards remain'),
          dict(name='final_FP16_cast_and_GM_output',kind='KERNEL_EXECUTION',output_bytes=622592,included=True),
          dict(name='output_readback_and_correctness',kind='CHECKS_OUTSIDE_TIMING',output_bytes=622592,included=False)],
        online_plan='Separate future NPU preprocessing experiment must prove the existing scale/RNE/zero/encoding contract first; no implementation or latency claim here.',
        inputs={n:proof['files'][n] for n in ('a_i8.bin','b_i8.bin','quant_u64_doc_candidate.bin')})


def a31_readiness(path,proof):
    result=dict(status='NOT_RUN',measurement='NOT_RUN',historical_numbers_used_as_samples=False,
        reason='first baseline measures W8A8 only; W8A16 needs independently admitted fresh paired execution')
    try:
        sys.path.insert(0,str(common.QUANT/'test'))
        from compare_a3 import load_run
        from a2_common import load_bundle,validate_layout,checked_file,sha256
        from run_a2 import validate_om_manifest
        report,real,root=load_run(path,'multi-core')
        if report['build'].get('dequant_mode')!='batched' or report.get('timing_protocol')!='continuous-v1':raise ValueError('requires accepted A3.1 batched continuous run')
        bundle_path=Path(real['bundle']);om_path=Path(real['native_om_manifest'])
        if sha256(bundle_path)!=real['bundle_sha256'] or sha256(om_path)!=real['native_om_manifest_sha256']:raise ValueError('historical manifest changed')
        bundle=load_bundle(bundle_path,'a2');native=validate_om_manifest(om_path,bundle_path,bundle,'a2')
        case=next(c for c in bundle['cases'] if c['name']=='layer-0-gate_up');validate_layout(bundle_path.parent,case,'a2')
        ids={}
        for name,original in proof['frozen_inputs'].items():
            p=checked_file(bundle_path.parent,case['files'][name]);item=identity(p)
            if any(item[k]!=original[k] for k in ('bytes','sha256')):raise ValueError('A3.1 capture differs from this workload: '+name)
            ids[name]=item
        compiled=next(c for c in native['cases'] if c['name']=='layer-0-gate_up')
        result.update(readiness='INPUT_AND_OM_IDENTITIES_MATCHED_ONLY',suite=identity(path),inputs=ids,
            om=identity(checked_file(om_path.parent,compiled['om'])),
            limitations='custom Runner/build/OPP and fresh 3-way timing remain unadmitted; no comparison speedup is reported')
    except (OSError,ValueError,KeyError,StopIteration) as error:result.update(readiness='BLOCKED_OR_UNAVAILABLE',error=f'{type(error).__name__}: {error}',requested_suite=str(path))
    return result
