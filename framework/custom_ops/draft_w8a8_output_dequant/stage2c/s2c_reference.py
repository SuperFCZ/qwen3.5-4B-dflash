"""Global-X quantization and three independent first-group CPU references."""
from pathlib import Path
import numpy as np
from s2c_evidence import identity,verify,write,verify_tree,CODEC,common,reference
from w8_reference import activation,metrics

CORE=common.FILES
EXTRA={'x_full_fp16.bin':16*2560*2,'activation_full_i8.bin':16*2560,'weight_nk_i8.bin':64*128,
       'weight_scale_fp16.bin':64*2,'weight_effective_fp16.bin':64*128*2,'w8a16_k128_fp16.bin':16*64*2}


def distribution(expected,actual):
    e=np.asarray(expected,dtype=np.float64);a=np.asarray(actual,dtype=np.float64)
    if e.shape!=a.shape or not np.isfinite(e).all() or not np.isfinite(a).all():raise ValueError('finite equal-shape Scale vectors required')
    delta=a-e;absolute=np.abs(delta);nonzero=e!=0;relative=absolute[nonzero]/np.abs(e[nonzero])
    def summary(values):
        if not values.size:return dict(count=0,min=None,max=None,mean=None,p50=None,p95=None,p99=None)
        return dict(count=int(values.size),min=float(values.min()),max=float(values.max()),mean=float(values.mean()),
                    p50=float(np.percentile(values,50)),p95=float(np.percentile(values,95)),p99=float(np.percentile(values,99)))
    denom=float(np.linalg.norm(e));num=float(np.linalg.norm(delta))
    return dict(count=int(e.size),changed=int(np.sum(e!=a)),signed_min=float(delta.min()),signed_max=float(delta.max()),
        mean_signed=float(delta.mean()),rmse=float(np.sqrt(np.mean(delta**2))),absolute=summary(absolute),
        relative_on_nonzero_reference=summary(relative),zero_reference_count=int(np.sum(~nonzero)),
        nonzero_actual_at_zero_reference=int(np.sum((~nonzero)&(a!=0))),
        relative_l2_error=num/denom if denom else (0. if num==0 else None))


def derive(x,nk,sw):
    if x.dtype!=np.dtype('<f2') or x.shape!=(16,2560):raise ValueError('full FP16 X[16,2560] required')
    if nk.dtype!=np.dtype('i1') or nk.shape!=(64,128):raise ValueError('signed INT8 N64/K128 required')
    if sw.dtype!=np.dtype('<f2') or sw.shape!=(64,) or not np.isfinite(sw).all() or np.any(sw<0):raise ValueError('finite nonnegative original group Scale required')
    aq,sa,quant=activation(x) # established whole-X binary64/RNE/clip[-127,127] contract
    a=aq[:,:128].copy();b=nk.T.copy();acc=common.stage2a.dot(a,b)
    combined64=np.float64(sa)*sw.astype(np.float64)
    combined=combined64.astype('<f4')
    words,effective=common.stage2a.encode_doc_candidate(combined)
    ideal=common.stage2a.convert(acc,combined);candidate=common.stage2a.convert(acc,effective)
    with np.errstate(over='ignore',under='ignore',invalid='ignore'):
        w16=(nk.astype(np.float64)*sw[:,None].astype(np.float64)).astype('<f2')
    if not np.isfinite(w16).all():raise ValueError('nonfinite W8A16 effective weight; reference not silently saturated')
    partial=np.zeros((16,64),np.float32)
    for k in range(128):
        partial=np.add(partial,x[:,k,None].astype(np.float32)*w16[None,:,k].astype(np.float32),dtype=np.float32)
    with np.errstate(over='ignore',under='ignore'):w8a16=partial.astype('<f2')
    if not all(np.isfinite(v).all() for v in (ideal,candidate,w8a16)):raise ValueError('nonfinite CPU reference; no automatic saturation or Golden adjustment')
    quant['k128_clipped_ratio']=float(np.mean(np.abs(np.rint(x[:,:128].astype(np.float64)/sa))>127))
    quant['full_reconstruction']=distribution(x.astype(np.float64),aq.astype(np.float64)*sa)
    scales=dict(scale_A=sa,combined_arithmetic='FP32(FP64(scale_A)*FP64(original FP16 scale_W))',
        effective_vs_combined_fp32=distribution(combined,effective),
        combined_fp32_vs_product_fp64=distribution(combined64,combined),
        per_channel=[dict(n=i,weight_scale_fp16=float(sw[i]),weight_scale_bits=f'0x{int(sw.view("<u2")[i]):04x}',
            combined_fp64=float(combined64[i]),combined_fp32=float(combined[i]),effective_fp32=float(effective[i]),
            encoded_u64=f'0x{int(words[i]):016x}',signed_error=float(np.float64(effective[i])-np.float64(combined[i]))) for i in range(64)])
    arrays={'a_i8.bin':a,'b_i8.bin':b,'z_i32.bin':acc.astype('<i4'),'scale_f32.bin':combined,
        'quant_u64_doc_candidate.bin':words,'scale_effective_f32.bin':effective,
        'ideal_fp16.bin':ideal,'doc_candidate_fp16.bin':candidate,
        'x_full_fp16.bin':x,'activation_full_i8.bin':aq,'weight_nk_i8.bin':nk,'weight_scale_fp16.bin':sw,
        'weight_effective_fp16.bin':w16,'w8a16_k128_fp16.bin':w8a16}
    return arrays,dict(activation=quant,scale_error=scales,integer_accumulator_min=int(acc.min()),integer_accumulator_max=int(acc.max()))


def read_frozen(files):
    sizes={'x.bin':16*2560*2,'q_nk.bin':19456*2560,'w_nz.bin':19456*2560,'s_gn.bin':20*19456*2}
    for name,size in sizes.items():
        verify(files[name])
        if files[name]['bytes']!=size:raise ValueError('frozen full tensor size mismatch: '+name)
    x=np.fromfile(files['x.bin']['path'],'<f2').reshape(16,2560)
    q=np.memmap(files['q_nk.bin']['path'],mode='r',dtype='i1',shape=(19456,2560))
    nz=np.memmap(files['w_nz.bin']['path'],mode='r',dtype='i1',shape=(80,1216,16,32))
    sw=np.memmap(files['s_gn.bin']['path'],mode='r',dtype='<f2',shape=(20,19456))[0,:64].copy()
    nk=nz[:4,:4].transpose(1,2,0,3).reshape(64,128).copy()
    if not np.array_equal(nk,q[:64,:128]):raise ValueError('NZ inverse differs from original signed weight slice')
    return x,nk,sw


def prepare(folder,frozen):
    files=frozen['files'];x,nk,sw=read_frozen(files)
    sizes={'x.bin':16*2560*2,'q_nk.bin':19456*2560,'w_nz.bin':19456*2560,'s_gn.bin':20*19456*2}
    arrays,detail=derive(x,nk,sw);folder.mkdir()
    for name,a in arrays.items():a.tofile(folder/name)
    case=dict(name='layer-0-gate_up-group0-n0',shape=[16,128,64],original_shape=[16,2560,19456],
        group=0,k_range=[0,128],n_range=[0,64],codec_status=CODEC,files={name:identity(folder/name) for name in arrays},
        source_files={name:files[name] for name in sizes},nz_roundtrip='BIT_EXACT',**detail,
        references=dict(candidate='FP32(INT64 dot)*candidate effective FP32 Scale, then FP16 RNE',
            ideal_w8a8='FP32(INT64 dot)*original combined FP32 Scale, then FP16 RNE',
            w8a16_k128='RN16(q*original group Scale), original FP16 X, sequential FP32 K128 reduction, final FP16 RNE; CPU partial reference, not Native partial Golden'),
        status='CPU_REFERENCES_READY_NPU_NOT_RUN')
    if case['files']['x_full_fp16.bin']['sha256']!=files['x.bin']['sha256']:raise ValueError('full X copy changed')
    verify_tree(frozen);write(folder/'case.json',case);return case


def verify_case(case):
    if case.get('shape')!=[16,128,64] or case.get('original_shape')!=[16,2560,19456] or case.get('group')!=0 or case.get('k_range')!=[0,128] or case.get('n_range')!=[0,64] or case.get('codec_status')!=CODEC:
        raise ValueError('Stage2C must remain first K128/N64 with the experimental codec')
    sizes={**CORE,**EXTRA}
    if set(case['files'])!=set(sizes):raise ValueError('Stage2C reference file set differs')
    for name,item in case['files'].items():
        verify(item)
        if item['bytes']!=sizes[name]:raise ValueError('Stage2C payload size differs')
    load=lambda n,d,s:np.fromfile(case['files'][n]['path'],d).reshape(s)
    x=load('x_full_fp16.bin','<f2',(16,2560));nk=load('weight_nk_i8.bin','i1',(64,128));sw=load('weight_scale_fp16.bin','<f2',(64,))
    original=read_frozen(case['source_files'])
    if any(a.tobytes()!=b.tobytes() for a,b in zip((x,nk,sw),original)):raise ValueError('derived inputs differ from frozen source slices')
    arrays,detail=derive(x,nk,sw)
    for name,a in arrays.items():
        if a.tobytes()!=Path(case['files'][name]['path']).read_bytes():raise ValueError('Stage2C model/reference mismatch: '+name)
    for name,value in detail.items():
        if case[name]!=value:raise ValueError('Stage2C quantization/Scale evidence mismatch')
    reference.verify_case(dict(files={name:case['files'][name] for name in CORE}))
    return arrays


def compare(case,outputs):
    arrays=verify_case(case)
    core=dict(name=case['name'],files={name:case['files'][name] for name in CORE})
    base=reference.compare(core,outputs)
    actual=[np.frombuffer(v,'<f2').reshape(16,64) for v in outputs]
    result=dict(status='CANDIDATE_BITWISE_MATCH' if base['status']=='CANDIDATE_BITWISE_MATCH' else 'FAIL_NUMERICAL',
        codec_status=CODEC,repeat_equal=base['repeat_equal'],npu_vs_candidate=base['candidate_reference'],
        npu_vs_ideal_w8a8=base['ideal_fp32_reference'],
        npu_vs_w8a16_k128=[metrics(arrays['w8a16_k128_fp16.bin'],v) for v in actual],
        cpu_candidate_vs_ideal=metrics(arrays['ideal_fp16.bin'],arrays['doc_candidate_fp16.bin']),
        cpu_ideal_vs_w8a16=metrics(arrays['w8a16_k128_fp16.bin'],arrays['ideal_fp16.bin']),
        scale_error=case['scale_error'],activation_quantization=case['activation'],
        acceptance='only candidate reference bits=0 and repeat/Guard/Readonly/Poison; other schemes are diagnostics, no tolerance changes')
    return result
