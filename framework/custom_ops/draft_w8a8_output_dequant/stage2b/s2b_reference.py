"""Independent Stage2B goldens, reusing the unchanged Stage2A experimental model."""
import json
from pathlib import Path
import shutil
import numpy as np
from s2b_common import stage2a,identity,verify,write,FILES,GROUPS,CODEC
from w8_reference import metrics


def verify_case(case):
    files=case['files']
    if set(files)!=set(FILES):raise ValueError('wrong case file set')
    for name,item in files.items():
        verify(item)
        if item['bytes']!=FILES[name]:raise ValueError('wrong input/output size')
    load=lambda name,dtype,shape:np.fromfile(files[name]['path'],dtype).reshape(shape)
    a=load('a_i8.bin','i1',(16,128));b=load('b_i8.bin','i1',(128,64));acc=stage2a.dot(a,b)
    scale=load('scale_f32.bin','<f4',(64,));words,effective=stage2a.encode_doc_candidate(scale)
    ideal=stage2a.convert(acc,scale);candidate=stage2a.convert(acc,effective)
    expected={'z_i32.bin':acc.astype('<i4'),'quant_u64_doc_candidate.bin':words,'scale_effective_f32.bin':effective,
              'ideal_fp16.bin':ideal,'doc_candidate_fp16.bin':candidate}
    for name,value in expected.items():
        if Path(files[name]['path']).read_bytes()!=value.tobytes():raise ValueError('CPU model/Golden differs: '+name)
    if not np.isfinite(ideal).all() or not np.isfinite(candidate).all():raise ValueError('Stage2B cases must stay finite; no inferred overflow policy')
    return acc,scale,effective,ideal,candidate


def verify_origin(case,upstream):
    name=case['name'];old=upstream['stage2a_case_files']
    if case.get('shape')!=[16,128,64] or case.get('codec')!=CODEC or name not in GROUPS.get(case.get('group'),()):
        raise ValueError('case scope/codec/group differs')
    if name in old:
        for fn,item in old[name].items():
            if any(case['files'][fn][k]!=item[k] for k in ('bytes','sha256')):raise ValueError('case differs from frozen Stage2A data')
    else:
        for fn in ('a_i8.bin','b_i8.bin','z_i32.bin'):
            if case['files'][fn]['sha256']!=old['scale_one'][fn]['sha256']:raise ValueError('supplement inputs differ')
        scale=np.fromfile(verify(case['files']['scale_f32.bin']),'<f4')
        expected=np.array([2.**(-(j//8)) for j in range(64)],'<f4') if name=='per_channel_blocks' else np.zeros(64,'<f4')
        if scale.tobytes()!=expected.tobytes():raise ValueError('supplement Scale pattern differs')


def prepare(root,upstream,group):
    root.mkdir();result=[];old=upstream['stage2a_case_files']
    for name in GROUPS[group]:
        folder=root/name;folder.mkdir()
        if name in old:
            for fn,item in old[name].items():shutil.copyfile(verify(item),folder/fn)
            origin=dict(stage2a_manifest=upstream['stage2a_manifest'],name=name)
        else:
            base=old['scale_one'];a=np.fromfile(verify(base['a_i8.bin']),'i1').reshape(16,128)
            b=np.fromfile(verify(base['b_i8.bin']),'i1').reshape(128,64)
            scale=np.array([2.**(-(j//8)) for j in range(64)],'<f4') if name=='per_channel_blocks' else np.zeros(64,'<f4')
            words,effective=stage2a.encode_doc_candidate(scale);acc=stage2a.dot(a,b)
            values={'a_i8.bin':a,'b_i8.bin':b,'z_i32.bin':acc.astype('<i4'),'scale_f32.bin':scale,
              'quant_u64_doc_candidate.bin':words,'scale_effective_f32.bin':effective,
              'ideal_fp16.bin':stage2a.convert(acc,scale),'doc_candidate_fp16.bin':stage2a.convert(acc,effective)}
            for fn,value in values.items():value.tofile(folder/fn)
            origin=dict(stage2a_manifest=upstream['stage2a_manifest'],inputs_from='scale_one',supplemental_fixture=name)
        case=dict(name=name,shape=[16,128,64],codec=CODEC,group=group,origin=origin,
                  files={fn:identity(folder/fn) for fn in FILES},status='CPU_GOLDENS_READY_NPU_NOT_RUN')
        acc,scale,effective,ideal,candidate=verify_case(case)
        if group=='basic' and ideal.tobytes()!=candidate.tobytes():raise ValueError('power-of-two references must agree')
        case.update(integer_accumulator_min=int(acc.min()),integer_accumulator_max=int(acc.max()),
            ideal_vs_candidate=metrics(ideal,candidate),scale_words=[f'0x{int(w):016x}' for w in np.fromfile(folder/'quant_u64_doc_candidate.bin','<u8')])
        verify_origin(case,upstream)
        write(folder/'case.json',case);result.append(case)
    return result


def compare(case,outputs):
    acc,scale,effective,ideal,candidate=verify_case(case)
    if len(outputs)!=2 or any(len(o)!=2048 for o in outputs):raise ValueError('two complete FP16 output repeats required')
    actual=[np.frombuffer(o,'<f2').reshape(16,64) for o in outputs]
    cand=[metrics(candidate,a) for a in actual];ideal_results=[metrics(ideal,a) for a in actual]
    repeat=outputs[0]==outputs[1]
    passed=repeat and all(c['bitwise_equal'] and c['actual_nonfinite']==0 for c in cand)
    result=dict(status='CANDIDATE_BITWISE_MATCH' if passed else 'FAIL_NUMERICAL',repeat_equal=repeat,
       candidate_reference=cand,ideal_fp32_reference=ideal_results,codec_status=CODEC,
       scope='selected finite K128 cases only; not general codec/rounding/overflow validation')
    if case['name'] in ('per_channel_powers','per_channel_blocks'):
        unscaled=stage2a.convert(acc,np.ones(64,dtype='<f4'))
        diagnostic=metrics(unscaled,actual[0]);expected=metrics(unscaled,candidate)
        if expected['bit_mismatches']==0:raise ValueError('uninformative per-channel fixture')
        result['per_channel']=dict(expected_vs_unscaled=expected,actual_vs_unscaled=diagnostic,
                                  observation='MATCHED_PER_CHANNEL_REFERENCE' if passed else 'NOT_CONFIRMED')
    return result
