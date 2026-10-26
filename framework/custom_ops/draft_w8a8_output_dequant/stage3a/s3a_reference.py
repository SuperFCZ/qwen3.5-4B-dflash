"""Fixed 20-group CPU references. Never rounds ideal/effective groups to half."""
from pathlib import Path
import json
import sys
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'stage2c'))
import s2c_evidence as e2c
import s2c_reference as r2c
from s2c_evidence import identity,verify,write,verify_tree,common,reference,CODEC
from w8_reference import activation,metrics
COUNT=20


def fp32_metrics(expected,actual):
    x=np.asarray(expected,dtype='<f4');y=np.asarray(actual,dtype='<f4')
    if x.shape!=y.shape:raise ValueError('FP32 diagnostic shape mismatch')
    finite=np.isfinite(x).all() and np.isfinite(y).all()
    if not finite:return dict(finite=False,expected_nonfinite=int((~np.isfinite(x)).sum()),actual_nonfinite=int((~np.isfinite(y)).sum()))
    a=x.astype(np.float64);b=y.astype(np.float64);diff=b-a;denom=float(np.linalg.norm(a));bn=float(np.linalg.norm(b))
    return dict(finite=True,bit_mismatches=int(np.sum(x.view('<u4')!=y.view('<u4'))),max_abs=float(np.abs(diff).max()),
        rmse=float(np.sqrt(np.mean(diff**2))),relative_l2_error=float(np.linalg.norm(diff))/denom if denom else (0. if bn==0 else None),
        cosine=float(np.vdot(a,b)/(denom*bn)) if denom and bn else (1. if denom==bn==0 else None))


def load_inputs(files):
    sizes={'x.bin':16*2560*2,'q_nk.bin':19456*2560,'w_nz.bin':19456*2560,'s_gn.bin':20*19456*2}
    for name,size in sizes.items():
        verify(files[name])
        if files[name]['bytes']!=size:raise ValueError('full frozen size mismatch: '+name)
    x=np.fromfile(files['x.bin']['path'],'<f2').reshape(16,2560)
    q=np.memmap(files['q_nk.bin']['path'],mode='r',dtype='i1',shape=(19456,2560))
    nz=np.memmap(files['w_nz.bin']['path'],mode='r',dtype='i1',shape=(80,1216,16,32))
    sw=np.memmap(files['s_gn.bin']['path'],mode='r',dtype='<f2',shape=(20,19456))[:,:64].copy()
    nk=q[:64,:].copy()
    for g in range(COUNT):
        unpack=nz[g*4:g*4+4,:4].transpose(1,2,0,3).reshape(64,128)
        if not np.array_equal(unpack,nk[:,g*128:(g+1)*128]):raise ValueError(f'group {g} NZ inverse mismatch')
    if not np.isfinite(sw).all() or (sw<0).any():raise ValueError('finite nonnegative GN Scales required')
    return x,nk,sw


def references(x,nk,sw):
    if x.shape!=(16,2560) or x.dtype!=np.dtype('<f2') or nk.shape!=(64,2560) or nk.dtype!=np.dtype('i1') or sw.shape!=(20,64) or sw.dtype!=np.dtype('<f2'):
        raise ValueError('fixed full-X / N64 K2560 / 20 GN scales required')
    if not np.isfinite(sw).all() or (sw<0).any():raise ValueError('invalid GN Scale')
    aq,sa,quant=activation(x) # exactly once over full X, never per group
    sums={name:np.zeros((16,64),'<f4') for name in ('ideal','effective','partial','w8a16')}
    groups=[];scale_original=[];scale_effective=[]
    for g in range(COUNT):
        lo,hi=g*128,(g+1)*128;a=aq[:,lo:hi].copy();b=nk[:,lo:hi].T.copy()
        acc=common.stage2a.dot(a,b)
        scale=(np.float64(sa)*sw[g].astype(np.float64)).astype('<f4')
        words,eff=common.stage2a.encode_doc_candidate(scale)
        ideal=acc.astype('<f4')*scale[None,:];effective=acc.astype('<f4')*eff[None,:]
        with np.errstate(over='ignore',under='ignore'):
            part=effective.astype('<f2');ideal_half=ideal.astype('<f2')
            w16=(nk[:,lo:hi].astype(np.float64)*sw[g,:,None].astype(np.float64)).astype('<f2')
        if not all(np.isfinite(v).all() for v in (ideal,effective,part,ideal_half,w16)):raise ValueError(f'nonfinite group {g} reference; no saturation repair')
        sums['ideal']=np.add(sums['ideal'],ideal,dtype=np.float32)
        sums['effective']=np.add(sums['effective'],effective,dtype=np.float32)
        sums['partial']=np.add(sums['partial'],part.astype('<f4'),dtype=np.float32)
        # W8A16 is one continuous K2560 FP32 reduction, not 20 rounded partials.
        for k in range(128):
            sums['w8a16']=np.add(sums['w8a16'],x[:,lo+k,None].astype('<f4')*w16[None,:,k].astype('<f4'),dtype=np.float32)
        groups.append(dict(index=g,k_range=[lo,hi],a=a,b=b,acc=acc.astype('<i4'),scale=scale,effective=eff,words=words,
                           ideal_f32=ideal,effective_f32=effective,partial_f16=part,ideal_f16=ideal_half))
        scale_original.append(scale);scale_effective.append(eff)
    result={name+'_f32':v for name,v in sums.items()}
    with np.errstate(over='ignore',under='ignore'):
        result.update({name+'_f16':v.astype('<f2') for name,v in sums.items()})
    if not all(np.isfinite(v).all() for v in result.values()):raise ValueError('nonfinite full-K CPU reference')
    return aq,quant,groups,result,r2c.distribution(np.stack(scale_original),np.stack(scale_effective))


def prepare(root,files):
    x,nk,sw=load_inputs(files);aq,quant,groups,refs,scale_error=references(x,nk,sw);root.mkdir()
    for name,a in [('x_full_fp16.bin',x),('activation_full_i8.bin',aq),('weight_nk_i8.bin',nk),('scale_gn_fp16.bin',sw)]:a.tofile(root/name)
    cases=[]
    for g in groups:
        folder=root/f'group-{g["index"]:02d}';folder.mkdir()
        values={'a_i8.bin':g['a'],'b_i8.bin':g['b'],'z_i32.bin':g['acc'],'scale_f32.bin':g['scale'],
            'quant_u64_doc_candidate.bin':g['words'],'scale_effective_f32.bin':g['effective'],
            'ideal_fp16.bin':g['ideal_f16'],'doc_candidate_fp16.bin':g['partial_f16']}
        for name,a in values.items():a.tofile(folder/name)
        g['ideal_f32'].tofile(folder/'ideal_partial_f32.bin');g['effective_f32'].tofile(folder/'effective_partial_f32.bin')
        case=dict(name=f'group-{g["index"]:02d}',group_index=g['index'],k_range=g['k_range'],n_range=[0,64],shape=[16,128,64],
            codec_status=CODEC,files={name:identity(folder/name) for name in values},
            fp32_partials={name:identity(folder/name) for name in ('ideal_partial_f32.bin','effective_partial_f32.bin')},
            nz_roundtrip='BIT_EXACT',scale_A=quant['scale'])
        reference.verify_case(case);write(folder/'case.json',case);cases.append(identity(folder/'case.json'))
    paths={}
    for name,a in refs.items():p=root/(name+'.bin');a.tofile(p);paths[name]=identity(p)
    r=dict(status='CPU_20_GROUP_REFERENCES_READY',shape=[16,2560,64],group_count=20,group_order=list(range(20)),
        codec_status=CODEC,activation=quant,source_files=files,cases=cases,references=paths,
        prepared_inputs={name:identity(root/name) for name in ('x_full_fp16.bin','activation_full_i8.bin','weight_nk_i8.bin','scale_gn_fp16.bin')},
        scale_error=scale_error,
        arithmetic='g=0..19 FP32 sequential sum; ideal/effective use FP32 scaled groups without FP16 intermediate; partial uses RN16 per group; final outputs RN16',
        w8a16_reference='RN16(original q*group Scale), original FP16 X, continuous sequential FP32 K2560, one final RN16; not asserted Native bitwise')
    write(root/'references.json',r);return r


def verify_prepared(r):
    if r.get('codec_status')!=CODEC or r.get('shape')!=[16,2560,64] or r.get('group_count')!=20 or r.get('group_order')!=list(range(20)) or len(r.get('cases',[]))!=20:raise ValueError('complete ordered 20-group reference required')
    verify_tree(r);x,nk,sw=load_inputs(r['source_files']);aq,quant,groups,refs,scale=references(x,nk,sw)
    if quant!=r['activation'] or scale!=r['scale_error']:raise ValueError('global activation/Scale reference differs')
    for name,a in [('x_full_fp16.bin',x),('activation_full_i8.bin',aq),('weight_nk_i8.bin',nk),('scale_gn_fp16.bin',sw)]:
        if verify(r['prepared_inputs'][name]).read_bytes()!=a.tobytes():raise ValueError('prepared full input differs')
    for g,item in zip(groups,r['cases']):
        c=json.loads(verify(item).read_text())
        if c.get('codec_status')!=CODEC or type(c.get('group_index')) is not int or c.get('group_index')!=g['index'] or c.get('k_range')!=g['k_range'] or c.get('n_range')!=[0,64] or c.get('scale_A')!=quant['scale']:raise ValueError('group identity/Scale drift')
        reference.verify_case(c)
        for name,key in [('a_i8.bin','a'),('b_i8.bin','b'),('scale_f32.bin','scale')]:
            if verify(c['files'][name]).read_bytes()!=g[key].tobytes():raise ValueError('group slice/combined Scale differs')
        for name,key in [('ideal_partial_f32.bin','ideal_f32'),('effective_partial_f32.bin','effective_f32')]:
            if verify(c['fp32_partials'][name]).read_bytes()!=g[key].tobytes():raise ValueError('FP32 partial reference changed')
    for name,a in refs.items():
        if verify(r['references'][name]).read_bytes()!=a.tobytes():raise ValueError('full-K reference mismatch: '+name)
    return refs


def effect_report(refs):
    def pair(a,b):return dict(fp32=fp32_metrics(refs[a+'_f32'],refs[b+'_f32']),fp16=metrics(refs[a+'_f16'],refs[b+'_f16']))
    return dict(scale_encoding=pair('ideal','effective'),group_fp16_rounding=pair('effective','partial'),
                ideal_w8a8_vs_w8a16=pair('w8a16','ideal'),effective_w8a8_vs_w8a16=pair('w8a16','effective'),
                fp16_partial_scheme_vs_w8a16=pair('w8a16','partial'),
                interpretation='Differences between separately computed schemes; norms/relative errors must not be added as independent costs.')


def accumulate_saved(cases):
    if len(cases)!=20 or any(type(c['group_index']) is not int for c in cases) or [c['group_index'] for c in cases]!=list(range(20)):raise ValueError('20 unique ordered groups required for aggregation')
    sums=[np.zeros((16,64),'<f4') for _ in range(2)]
    for c in cases:
        if len(c['outputs'])!=2:raise ValueError('two output repeats per group required')
        for i,item in enumerate(c['outputs']):
            p=verify(item)
            if item['bytes']!=2048:raise ValueError('FP16 partial size differs')
            v=np.fromfile(p,'<f2').reshape(16,64)
            if not np.isfinite(v).all():raise ValueError('nonfinite NPU partial; no silent repair')
            sums[i]=np.add(sums[i],v.astype('<f4'),dtype=np.float32)
    with np.errstate(over='ignore',under='ignore'):halves=[s.astype('<f2') for s in sums]
    return sums,halves
