"""Stage3B-1 fixed reduction contract and read-only Stage3A evidence."""
import json
from pathlib import Path
import sys
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'stage3a'))
import s3a_reference as s3a
from s3a_reference import e2c,common,identity,verify,write,verify_tree
from w8_common import fresh
from w8_reference import metrics
NAME='DFlashW8A8PartialReduce'
SNAKE='d_flash_w8_a8_partial_reduce'
BUDGET=dict(input_ub_bytes=2048,cast_fp32_bytes=4096,accumulator_fp32_bytes=4096,output_ub_bytes=2048,
    explicit_ub_bytes=12288,conservative_reserve_bytes=8192,minimum_platform_ub_bytes=20480,
    user_l1_bytes=0,user_l0a_bytes=0,user_l0b_bytes=0,user_l0c_bytes=0,requested_workspace_bytes=0,
    gm_input_bytes=40960,gm_output_bytes=2048,block_dim=1,queue_depth=1)
SETTINGS=dict(op=NAME,input_shape=[20,16,64],output_shape=[16,64],dtype='FP16',group_order=list(range(20)),
    accumulation='FP32 sequential from +0',output_cast='CAST_NONE',budget=BUDGET)


def resolve_report(root,explicit):
    if explicit is not None:return explicit
    candidates=[p for p in (root/'npu/report.json',root/'report.json') if p.is_file()]
    if len(candidates)!=1:raise ValueError('missing/ambiguous Stage3A report; set --stage3a-report explicitly')
    return candidates[0]


def sources():
    paths=[p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts]
    paths += [common.QUANT/'test/runner_common.h',common.QUANT/'test/opp_preflight.py',common.FIX/'compile_raw.py',
              common.FIX/'w8_common.py',common.FIX/'w8_reference.py',HERE.parent/'stage3a/s3a_reference.py',
              HERE.parent/'stage2b/s2b_reference.py',HERE.parent/'stage2b/run_stage2b.py']
    return [identity(p) for p in sorted(set(paths))]


def reduce_cpu(partials):
    if partials.shape!=(20,16,64) or partials.dtype!=np.dtype('<f2') or not np.isfinite(partials).all():raise ValueError('finite FP16 [20,16,64] required')
    total=np.zeros((16,64),'<f4')
    for g in range(20):total=np.add(total,partials[g].astype('<f4'),dtype=np.float32)
    with np.errstate(over='ignore',under='ignore'):half=total.astype('<f2')
    if not np.isfinite(half).all():raise ValueError('nonfinite final reference; no saturation or Golden adjustment')
    return total,half


def evidence(path,device,cann):
    r=json.loads(path.read_text())
    expected=dict(stage='3A',status='STAGE3A_20_GROUP_MATCH',phase='complete',shape=[16,2560,64],group_count=20,
                  device=device,npu='ALL_20_GROUPS_EXECUTED',performance='NOT_RUN')
    if any(r.get(k)!=v for k,v in expected.items()) or r.get('summary')!=dict(passed=20,failed=0,not_run=0):raise ValueError('complete accepted Stage3A 20-group report required')
    groups=r.get('groups',[])
    if len(groups)!=20 or any(type(c.get('group_index')) is not int for c in groups) or [c['group_index'] for c in groups]!=list(range(20)):raise ValueError('unique ordered 20 groups required')
    required=[]
    def watch(item):verify(item);required.append(item);return Path(item['path'])
    for item in r['sources']:watch(item)
    prepared=json.loads(watch(r['cpu_manifest']).read_text())
    if prepared.get('group_count')!=20 or prepared.get('group_order')!=list(range(20)) or len(prepared.get('cases',[]))!=20:raise ValueError('CPU partial manifest scope differs')
    stacks=[[],[]]
    for g,c in enumerate(groups):
        if c.get('status')!='PASS' or c['case']!=prepared['cases'][g]:raise ValueError('Stage3A group was not accepted or uses a different reference')
        case=json.loads(watch(c['case']).read_text())
        if case.get('group_index')!=g or case.get('k_range')!=[g*128,(g+1)*128] or case.get('n_range')!=[0,64]:raise ValueError('group slice identity differs')
        for item in case['files'].values():watch(item)
        common.check_execution(json.loads(watch(c['execution']).read_text()),device)
        payload=[watch(item).read_bytes() for item in c['outputs']]
        comparison=e2c.reference.compare(case,payload)
        if comparison!=c['comparison'] or comparison['status']!='CANDIDATE_BITWISE_MATCH':raise ValueError('Stage3A partial numerical evidence changed')
        for i,data in enumerate(payload):stacks[i].append(np.frombuffer(data,'<f2').reshape(16,64).copy())
    stacks=[np.stack(s) for s in stacks]
    if stacks[0].tobytes()!=stacks[1].tobytes():raise ValueError('Stage3A partial repeat drift')
    total,half=reduce_cpu(stacks[0])
    refs={k:prepared['references'][k] for k in ('partial_f32','partial_f16')}
    if watch(refs['partial_f32']).read_bytes()!=total.tobytes() or watch(refs['partial_f16']).read_bytes()!=half.tobytes():raise ValueError('saved NPU partials differ from Stage3A CPU partial Reference')
    aggregate=r.get('aggregation',{})
    if aggregate.get('status')!='MATCH' or aggregate.get('repeat_equal') is not True or aggregate.get('order')!=list(range(20)) or len(aggregate.get('outputs',[]))!=2:raise ValueError('Stage3A host aggregation not accepted')
    for out in aggregate['outputs']:
        if watch(out['fp32']).read_bytes()!=total.tobytes() or watch(out['fp16']).read_bytes()!=half.tobytes():raise ValueError('Stage3A saved sum changed')
    # Bind the real SDK/tool environment without executing Stage1 or rechecking optional Native files.
    pf_id=r['baseline']['stage2b']['upstream']['stage1']['preflight'];pf=json.loads(watch(pf_id).read_text())
    if Path(pf['cann_root']).resolve()!=cann or pf.get('soc')!='Ascend310P3':raise ValueError('SDK/device scope differs from prior accepted execution')
    for pair in pf['files']:watch(pair['source']);watch(pair['snapshot'])
    for tool in pf['tools'].values():watch(tool['file'])
    for item in r['baseline']['stage2b']['sources']:watch(item)
    bound={item['path']:item for item in required}
    return stacks[0],total,half,dict(report=identity(path),references=refs,required_files=list(bound.values()),
        groups=[dict(group_index=c['group_index'],outputs=c['outputs']) for c in groups],preflight=pf_id,
        upstream_rerun=False,native_dependency='NONE; previous Native semantics audit is not a reduction prerequisite'),pf


def prepare(root,stack,total,half,proof):
    root.mkdir();stack.tofile(root/'partials_fp16.bin');total.tofile(root/'reference_fp32.bin');half.tofile(root/'reference_fp16.bin')
    files={p.name:identity(p) for p in sorted(root.glob('*.bin'))}
    if files['reference_fp16.bin']['sha256']!=proof['references']['partial_f16']['sha256'] or files['reference_fp32.bin']['sha256']!=proof['references']['partial_f32']['sha256']:raise ValueError('reference copy identity mismatch')
    bits=stack.view('<u2')
    case=dict(settings=SETTINGS,files=files,source=proof,status='PREPARED_NPU_NOT_RUN',selected_partial_repeat=0,
        input_statistics=dict(subnormal_count=int(np.sum(((bits&0x7fff)>0)&((bits&0x7fff)<0x400))),
                              negative_zero_count=int(np.sum(bits==0x8000)),positive_zero_count=int(np.sum(bits==0))))
    write(root/'case.json',case);return case


def verify_case(case):
    if case.get('settings')!=SETTINGS or set(case['files'])!={'partials_fp16.bin','reference_fp32.bin','reference_fp16.bin'}:raise ValueError('reduction case scope differs')
    for name,size in [('partials_fp16.bin',40960),('reference_fp32.bin',4096),('reference_fp16.bin',2048)]:
        verify(case['files'][name])
        if case['files'][name]['bytes']!=size:raise ValueError('case byte count differs')
    verify_tree(case['source'])
    group_ids=case['source']['groups']
    if len(group_ids)!=20 or [g['group_index'] for g in group_ids]!=list(range(20)):raise ValueError('partial source order differs')
    data=verify(case['files']['partials_fp16.bin']).read_bytes()
    for repeat in (0,1):
        if b''.join(verify(g['outputs'][repeat]).read_bytes() for g in group_ids)!=data:raise ValueError('packed input differs from actual Stage3A partials')
    total,half=reduce_cpu(np.frombuffer(data,'<f2').reshape(20,16,64))
    if verify(case['files']['reference_fp32.bin']).read_bytes()!=total.tobytes() or verify(case['files']['reference_fp16.bin']).read_bytes()!=half.tobytes():raise ValueError('ordered reduction reference differs')
    for name,key in [('reference_fp32.bin','partial_f32'),('reference_fp16.bin','partial_f16')]:
        if case['files'][name]['sha256']!=case['source']['references'][key]['sha256']:raise ValueError('Golden is not Stage3A partial Reference')
    return half


def check_execution(ex,device):
    expected=dict(status='EXECUTION_COMPLETED_NUMERICS_UNCHECKED',op=NAME,soc='Ascend310P3',input_shape=[20,16,64],output_shape=[16,64],group_count=20,
        guards=True,readonly=True,repeat_equal=True,repetitions=2,device=device,cpu_fallback=False,layout='nd',input_count=1,poison_verified=True,poison_patterns=[127,255])
    if any(type(ex.get(k)) is not type(v) or ex.get(k)!=v for k,v in expected.items()):raise ValueError('reduction execution/guard/readonly/poison/repeat mismatch')
    t=ex.get('timing',{})
    if t.get('status')!='NOT_RUN' or t.get('warmup')!=0 or t.get('repetitions')!=0:raise ValueError('performance measurement not enabled')


def compare(case,outputs):
    half=verify_case(case)
    if len(outputs)!=2 or any(len(x)!=2048 for x in outputs):raise ValueError('two full reduction output repeats required')
    checks=[metrics(half,np.frombuffer(x,'<f2').reshape(16,64)) for x in outputs]
    passed=outputs[0]==outputs[1] and all(c['bitwise_equal'] and not c['actual_nonfinite'] for c in checks)
    return dict(status='REDUCTION_BITWISE_MATCH' if passed else 'FAIL_NUMERICAL',reference='unchanged Stage3A CPU partial Reference',comparisons=checks,repeat_equal=outputs[0]==outputs[1])
