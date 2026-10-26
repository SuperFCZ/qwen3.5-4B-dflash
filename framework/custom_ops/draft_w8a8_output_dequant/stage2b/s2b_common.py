"""Stage 2B identities and immutable upstream evidence; no device calls."""
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
OPS=HERE.parents[1]
FIX=OPS/'draft_w8a8_fixpipe'
QUANT=OPS/'draft_quant'
sys.path.insert(0,str(FIX));sys.path.insert(0,str(HERE.parent))
from w8_common import identity,verify,write,fresh
from w8_run import verify_tree,env
import stage2a
if Path(stage2a.__file__).resolve()!=HERE.parent/'stage2a.py':raise RuntimeError('wrong Stage 2A module')
NAME='DFlashW8A8OutputDequant'
SNAKE='d_flash_w8_a8_output_dequant'
CODEC='EXPERIMENTAL_CODEC_CANDIDATE'
GROUPS={'basic':('scale_one','scale_half','per_channel_powers','per_channel_blocks'),
        'zero':('zeros_and_ones','scale_zero'),'nonpower':('mantissa_probe',)}
OLD_CASES={'scale_one','scale_half','per_channel_powers','zeros_and_ones','mantissa_probe'}
FILES={'a_i8.bin':2048,'b_i8.bin':8192,'z_i32.bin':4096,'scale_f32.bin':256,
       'quant_u64_doc_candidate.bin':512,'scale_effective_f32.bin':256,'ideal_fp16.bin':2048,'doc_candidate_fp16.bin':2048}


def sources():
    paths=[p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts]
    paths += [HERE.parent/'stage2a.py',QUANT/'test/runner_common.h',QUANT/'test/opp_preflight.py',
              FIX/'w8_common.py',FIX/'w8_run.py',FIX/'w8_raw.py',FIX/'w8_reference.py']
    return [identity(p) for p in sorted(set(paths))]


def check_execution(ex,device):
    expected=dict(status='EXECUTION_COMPLETED_NUMERICS_UNCHECKED',op=NAME,soc='Ascend310P3',shape=[16,128,64],
        guards=True,readonly=True,repeat_equal=True,repetitions=2,device=device,cpu_fallback=False,
        layout='nd',input_count=3,poison_verified=True,poison_patterns=[127,255])
    if any(type(ex.get(k)) is not type(v) or ex.get(k)!=v for k,v in expected.items()):raise ValueError('Stage2B execution/guard/readonly/poison/repeat mismatch')
    t=ex.get('timing',{})
    if t.get('status')!='NOT_RUN' or t.get('warmup')!=0 or t.get('repetitions')!=0:raise ValueError('no timing permitted')


def stage1_evidence(path,device,cann):
    from w8_raw import compare
    from w8_run import check_execution as raw_execution
    r=json.loads(path.read_text())
    names={'signed_dense','zero','k_boundaries','max_positive','max_negative','cancellation'}
    if r.get('status')!='RAW_INT32_ND_PASS' or r.get('phase')!='complete' or r.get('raw_only') is not True or r.get('layout')!='nd':raise ValueError('Stage1 Raw ND PASS required')
    if r.get('device')!=device or len(r.get('cases',[]))!=6 or {c['name'] for c in r['cases']}!=names:raise ValueError('Stage1 device/cases differ')
    verify_tree(r)
    built=json.loads(verify(r['build_report']).read_text())
    if built.get('status')!='BUILT_NOT_NPU_VERIFIED' or built.get('raw_only') is not True or built.get('layout')!='nd':raise ValueError('Stage1 build identity/scope differs')
    verify_tree(built)
    for c in r['cases']:
        case=json.loads(verify(c['case_file']).read_text())
        actual=compare(case,[verify(f).read_bytes() for f in c['raw']['outputs']])
        if actual!=c['comparison'] or actual['status']!='RAW_INT32_MATCH':raise ValueError('Stage1 numeric evidence changed')
        ex=json.loads(verify(c['raw']['execution_file']).read_text());raw_execution(ex,'raw','nd',device)
    pf=json.loads(verify(r['preflight']).read_text())
    if Path(pf['cann_root']).resolve()!=cann or pf.get('soc')!='Ascend310P3':raise ValueError('Stage1 SDK differs')
    verify_tree(pf)
    return dict(report=identity(path),preflight=r['preflight'],build=r['build_report'],status='RAW_INT32_ND_PASS',device=device,rerun=False)


def checked_cases(path):
    m=json.loads(path.read_text())
    if m.get('status')!='CPU_MODEL_ONLY_NOT_NPU_VERIFIED' or m.get('shape')!=[16,128,64] or len(m.get('cases',[]))!=5 or {c['name'] for c in m['cases']}!=OLD_CASES:raise ValueError('Stage2A CPU manifest differs')
    result={}
    for c in m['cases']:
        if set(c['files'])!=set(FILES):raise ValueError('Stage2A file set differs')
        result[c['name']]={}
        for name,bytes_ in FILES.items():
            p=(path.parent/c['name']/name).resolve(strict=True)
            if path.parent.resolve() not in p.parents:raise ValueError('Stage2A data escape')
            item=identity(p);expected=c['files'][name]
            if item['bytes']!=bytes_ or expected!={'bytes':item['bytes'],'sha256':item['sha256']}:raise ValueError('Stage2A data identity changed: '+name)
            result[c['name']][name]=item
    return result


def upstream(args):
    one=stage1_evidence(args.stage1,args.device,args.cann_root)
    index=json.loads(args.stage2a_index.read_text())
    asc=(args.cann_root/'aarch64-linux/asc').resolve(strict=True)
    if index.get('stage')!='2A' or index.get('status')!='UNRESOLVED_SOURCE_REVIEW_REQUIRED' or Path(index['source_root']).resolve()!=asc:raise ValueError('Stage2A SDK index scope differs')
    bound=[];missing=[];seen=set()
    for f in index['files']:
        rel=Path(f['relative'])
        if rel.is_absolute() or '..' in rel.parts or str(rel) in seen:raise ValueError('invalid SDK index path')
        seen.add(str(rel));p=asc/rel
        if f['state']=='MISSING_UNRESOLVED':missing.append(str(rel));continue
        if f['state']!='INDEXED_NOT_REVIEWED':raise ValueError('unexpected SDK index state')
        real=p.resolve(strict=True)
        if asc not in real.parents:raise ValueError('SDK index path escape')
        item=identity(real)
        if item['sha256']!=f['sha256'] or item['bytes']!=f['bytes']:raise ValueError('Stage2A SDK changed: '+str(rel))
        bound.append(item)
    required={
       'impl/adv_api/detail/matmul/matmul_impl_base.h',
       'impl/adv_api/detail/matmul/stage/copy_cube_out/copy_cube_out_datacopy.h',
       'impl/adv_api/detail/matmul/stage/copy_cube_out/quant/quant_processor_datacopy.h',
       'impl/adv_api/detail/matmul/stage/copy_cube_out/quant/quant_processor_utils.h'}
    if not required.issubset(seen-set(missing)):raise ValueError('required Stage2A output path evidence missing')
    cases=checked_cases(args.stage2a_cases)
    from s2b_reference import verify_case
    for name,files in cases.items():verify_case(dict(name=name,files=files))
    return dict(stage1=one,stage2a_index=identity(args.stage2a_index),stage2a_manifest=identity(args.stage2a_cases),
        stage2a_case_files=cases,sdk_files=bound,missing_sdk_files=missing,
        stage2a_review_status=index['status'],codec=CODEC,
        codec_formula='uint64(float32_bits & 0xFFFFE000) | (1 << 46)',
        host_ctype='DT_FLOAT16_CANDIDATE_NOT_YET_TILING_VERIFIED')
