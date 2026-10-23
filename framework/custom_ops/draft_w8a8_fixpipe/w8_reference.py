"""Independent W8A8 contract. No W8A16 Golden is written or treated as this oracle."""
import numpy as np
from w8_common import identity,verify,write


def check_codec(codec):
    # A constrained, source-reviewed recipe, NOT a default hardware encoding.
    keys={'float_field','constant_u64','stored_mantissa_bits','effective_mantissa_bits',
          'coefficient_rounding','product_rounding','overflow','subnormal','sign','vector_storage'}
    if set(codec)!=keys:raise ValueError('complete reviewed codec recipe required')
    if codec['float_field']!='low32_ieee_float32' or codec['vector_storage']!='64_contiguous_u64':
        raise ValueError('BLOCKED_UNSUPPORTED: codec adapter does not implement this field/layout')
    if codec['coefficient_rounding']!='truncate_low_mantissa_bits' or codec['sign']!='positive_finite_scale_signed_accumulator':
        raise ValueError('BLOCKED_UNSUPPORTED: unimplemented coefficient contract')
    for key in ('stored_mantissa_bits','effective_mantissa_bits'):
        if type(codec[key]) is not int or codec[key] not in (10,23):raise ValueError('unsupported reviewed precision')
    if codec['effective_mantissa_bits']>codec['stored_mantissa_bits']:raise ValueError('effective precision exceeds stored precision')
    if type(codec['constant_u64']) is not int or not 0<=codec['constant_u64']<2**64 or codec['constant_u64']&0xffffffff:
        raise ValueError('fixed flags overlap scale low32 field')
    if codec['product_rounding'] not in ('fp32_then_fp16_rne','exact_product_then_fp16_rne'):
        raise ValueError('unsupported reviewed product rounding')
    if codec['overflow'] not in ('ieee','saturate_finite') or codec['subnormal'] not in ('preserve','flush_signed_zero'):
        raise ValueError('unsupported reviewed FP16 conversion')


def encode_scales(scales,codec):
    check_codec(codec);s=np.asarray(scales,dtype='<f4')
    if s.shape!=(64,) or not np.isfinite(s).all() or not (s>0).all():raise ValueError('positive finite N64 scale required')
    bits=s.view('<u4').copy()
    stored=bits & np.uint32((0xffffffff << (23-codec['stored_mantissa_bits'])) & 0xffffffff)
    effective=stored & np.uint32((0xffffffff << (23-codec['effective_mantissa_bits'])) & 0xffffffff)
    if not (effective.view('<f4')>0).all():raise ValueError('scale truncates to zero; unsupported by this PoC contract')
    words=stored.astype('<u8') | np.uint64(codec['constant_u64'])
    return words,effective.view('<f4').copy()


def dot(a,b):
    if a.dtype!=np.dtype('i1') or b.dtype!=np.dtype('i1') or a.shape!=(16,128) or b.shape!=(128,64):
        raise ValueError('exact signed INT8 M16/K128/N64 required')
    acc=a.astype(np.int64)@b.astype(np.int64)
    if np.abs(acc).max()>128*128*128:raise ValueError('integer accumulator exceeds mathematical bound')
    return acc


def convert(acc,scale,codec):
    check_codec(codec)
    # INT32 accumulators in this shape are exactly representable in FP32.
    if codec['product_rounding']=='fp32_then_fp16_rne':value=acc.astype(np.float32)*scale[None,:].astype(np.float32)
    else:value=acc.astype(np.float64)*scale[None,:].astype(np.float64)
    if codec['overflow']=='saturate_finite':value=np.clip(value,-65504,65504)
    with np.errstate(over='ignore',under='ignore'):half=value.astype('<f2')
    if codec['subnormal']=='flush_signed_zero':
        bits=half.view('<u2');mask=((bits&0x7fff)>0)&((bits&0x7fff)<0x400)
        bits[mask]&=0x8000
    return half


def nz_pack(b):
    if b.shape!=(128,64) or b.dtype!=np.dtype('i1'):raise ValueError('INT8 KN required')
    nk=b.T.copy();nz=nk.reshape(4,16,4,32).transpose(2,0,1,3).copy()
    if not np.array_equal(nz.transpose(1,2,0,3).reshape(64,128),nk):raise ValueError('NZ roundtrip')
    return nz


def activation(x):
    if x.shape!=(16,2560) or x.dtype!=np.dtype('<f2') or not np.isfinite(x).all():raise ValueError('whole frozen FP16 X required')
    maximum=float(np.max(np.abs(x.astype(np.float64))))
    # Contract: one FP64 scale from ALL 16x2560 inputs; no row/group-specific scales.
    scale=maximum/127.0 if maximum else 1.0
    unrounded=x.astype(np.float64)/scale
    rounded=np.rint(unrounded)
    q=np.clip(rounded,-127,127).astype('i1')
    return q,scale,dict(scale=scale,scale_arithmetic='binary64 max_abs(X_full)/127; RNE division result then clip[-127,127]',
        scale_scope=[16,2560],max_abs=maximum,all_zero=maximum==0,
        clipped_ratio=float(np.mean((rounded < -127)|(rounded > 127))),
        endpoint_ratio=float(np.mean(np.abs(q.astype(np.int16))==127)),
        k128_endpoint_ratio=float(np.mean(np.abs(q[:,:128].astype(np.int16))==127)))


def metrics(expected,actual):
    e=np.asarray(expected,dtype='<f2');a=np.asarray(actual,dtype='<f2')
    if e.shape!=a.shape:raise ValueError('shape mismatch')
    eb=e.view('<u2');ab=a.view('<u2')
    ordered=lambda b:np.where(b&0x8000,0x8000-(b.astype(np.int32)&0x7fff),0x8000+b.astype(np.int32))
    finite=np.isfinite(e)&np.isfinite(a);ulp_valid=~np.isnan(e)&~np.isnan(a)
    with np.errstate(invalid='ignore'):delta=a.astype(np.float64)-e.astype(np.float64)
    same=bool(np.array_equal(eb,ab));indices=np.argwhere(eb!=ab)
    if finite.all():
        ef=e.astype(np.float64).ravel();af=a.astype(np.float64).ravel();norm=float(np.linalg.norm(ef));an=float(np.linalg.norm(af))
        relative=float(np.linalg.norm(af-ef)/norm) if norm else (0.0 if an==0 else None)
        cosine=float(np.dot(af,ef)/(norm*an)) if norm and an else (1.0 if norm==an==0 else None)
    else:relative=cosine=None
    first=None if same else tuple(int(v) for v in indices[0])
    return dict(bitwise_equal=same,bit_mismatches=int(np.sum(eb!=ab)),
        max_ulp=int(np.max(np.abs(ordered(eb)[ulp_valid]-ordered(ab)[ulp_valid]))) if ulp_valid.any() else (0 if same else None),
        max_abs=float(np.max(np.abs(delta))) if finite.all() else None,
        rmse=float(np.sqrt(np.mean(delta**2))) if finite.all() else None,
        cosine=cosine,relative_l2_error=relative,
        expected_nonfinite=int(np.sum(~np.isfinite(e))),actual_nonfinite=int(np.sum(~np.isfinite(a))),
        first_difference=None if first is None else dict(index=list(first),expected_bits=f'0x{int(eb[first]):04x}',actual_bits=f'0x{int(ab[first]):04x}'))


def save_case(folder,name,a,b,scale,codec,w8a16=None,extra=None):
    folder.mkdir();words,effective=encode_scales(scale,codec);acc=dot(a,b)
    encoded=convert(acc,effective,codec)
    with np.errstate(over='ignore',under='ignore'):
        ideal=(acc.astype(np.float32)*np.asarray(scale,dtype=np.float32)[None,:]).astype('<f2')
    arrays={'a.bin':a,'b_nd.bin':b,'b_nz.bin':nz_pack(b),'scale_u64.bin':words,
        'scale_ideal_f32.bin':np.asarray(scale,dtype='<f4'),'scale_effective_f32.bin':effective,
        'acc_int64.bin':acc.astype('<i8'),'encoded_ref.bin':encoded,'ideal_fp32_ref.bin':ideal}
    if w8a16 is not None:arrays['w8a16_k128_ref.bin']=w8a16
    for filename,data in arrays.items():data.tofile(folder/filename)
    report=dict(name=name,shape=[16,128,64],status='CPU_REFERENCES_READY_NPU_NOT_RUN',
        files={n:identity(folder/n) for n in arrays},codec=codec,extra=extra or {},
        w8a16_golden='UNTOUCHED; full K2560 Golden is never sliced as a K128 output',
        scale_ideal_bits=[f'0x{int(v):08x}' for v in np.asarray(scale,dtype='<f4').view('<u4')],
        scale_words=[f'0x{int(v):016x}' for v in words])
    write(folder/'case.json',report);return report


def synthetic(root,codec):
    root.mkdir();a=((np.arange(16*128)*17)%33-16).astype('i1').reshape(16,128)
    b=((np.arange(128*64)*13)%31-15).astype('i1').reshape(128,64)
    cases=[]
    for name,aa,bb,scale in (
        ('scale_one',a,b,np.ones(64)),('scale_half',a,b,np.full(64,.5)),
        ('channel_powers',a,b,np.array([2.0**(-i%5-4) for i in range(64)])),
        ('zero',np.zeros_like(a),b,np.ones(64)),
        ('signed_boundaries',np.where(np.indices(a.shape)[0]%2,-128,127).astype('i1'),
            np.where(np.indices(b.shape)[1]%2,-128,127).astype('i1'),np.full(64,2**-8)),
        ('precision_probe',a,b,(np.arange(64,dtype=np.float32)+1025.25)/8192),
        ('overflow_probe',np.full_like(a,127),np.full_like(b,127),np.ones(64)),
        ('subnormal_probe',np.pad(np.ones((16,1),dtype='i1'),((0,0),(0,127))),
            np.ones_like(b),np.full(64,2**-24))):
        cases.append(save_case(root/name,name,aa,bb,scale,codec))
    return cases


def frozen(root,codec,a51,pair):
    import sys
    from w8_common import QUANT
    sys.path.insert(0,str(QUANT/'fp16_compare'))
    from fp16_gate_evidence import accepted
    src=accepted(a51,pair);files=src['files']
    x=np.memmap(verify(files['x.bin']),mode='r',dtype='<f2',shape=(16,2560))
    q=np.memmap(verify(files['q_nk.bin']),mode='r',dtype='i1',shape=(19456,2560))
    nz=np.memmap(verify(files['w_nz.bin']),mode='r',dtype='i1',shape=(80,1216,16,32))
    scales=np.memmap(verify(files['s_gn.bin']),mode='r',dtype='<f2',shape=(20,19456))
    slice_nk=nz[:4,:4].transpose(1,2,0,3).reshape(64,128).copy()
    if not np.array_equal(slice_nk,q[:64,:128]):raise ValueError('frozen NZ inverse mismatch')
    aq,sa,info=activation(x)
    sw=scales[0,:64].copy();combined=(sa*sw.astype(np.float64)).astype('<f4')
    effective=(slice_nk.astype(np.float64)*sw[:,None].astype(np.float64)).astype('<f2')
    if not np.isfinite(effective).all():raise ValueError('nonfinite W8A16 effective weight')
    # Diagnostic K128 reference, independent of the immutable full-K Golden.
    acc=np.zeros((16,64),np.float32)
    for k in range(128):acc=np.add(acc,x[:,k,None].astype(np.float32)*effective[None,:,k].astype(np.float32),dtype=np.float32)
    wref=acc.astype('<f2')
    case=save_case(root,'frozen_layer0_k128',aq[:,:128].copy(),slice_nk.T.copy(),combined,codec,wref,
        dict(activation=info,combined_scale='FP32(FP64(scale_A)*FP64(scale_W_half))',
             w8a16_reference='RN16(q*scale_W), sequential FP32 K128, final RN16; CPU diagnostic, not Native Golden'))
    case['watched']=src['watched'];case['frozen_identity']=src['identity']
    for item in src['watched'].values():verify(item)
    write(root/'case.json',case);return [case]


def compare_case(case,raw_outputs,quant_outputs):
    for f in case['files'].values():verify(f)
    if len(raw_outputs)!=2 or len(quant_outputs)!=2:raise ValueError('two repeats required')
    if any(len(v)!=4096 for v in raw_outputs) or any(len(v)!=2048 for v in quant_outputs):raise ValueError('wrong output bytes')
    oracle=np.fromfile(case['files']['acc_int64.bin']['path'],'<i8').reshape(16,64)
    raw=[np.frombuffer(v,'<i4').reshape(16,64) for v in raw_outputs]
    quant=[np.frombuffer(v,'<f2').reshape(16,64) for v in quant_outputs]
    encoded=np.fromfile(case['files']['encoded_ref.bin']['path'],'<f2').reshape(16,64)
    ideal=np.fromfile(case['files']['ideal_fp32_ref.bin']['path'],'<f2').reshape(16,64)
    r=dict(accumulator=[dict(equal=bool(np.array_equal(a,oracle)),mismatches=int(np.sum(a!=oracle)),
        max_abs_integer=int(np.max(np.abs(a.astype(np.int64)-oracle)))) for a in raw],
        encoded=[metrics(encoded,a) for a in quant],ideal_fp32=[metrics(ideal,a) for a in quant],
        repeat_equal=raw_outputs[0]==raw_outputs[1] and quant_outputs[0]==quant_outputs[1])
    if 'w8a16_k128_ref.bin' in case['files']:
        w=np.fromfile(case['files']['w8a16_k128_ref.bin']['path'],'<f2').reshape(16,64)
        r['w8a8_vs_w8a16']=dict(ideal=metrics(w,ideal),npu=[metrics(w,a) for a in quant],
                              scope='cross-semantics diagnostic; never a W8A16 acceptance gate')
    passed=r['repeat_equal'] and all(x['equal'] for x in r['accumulator']) and all(x['bitwise_equal'] for x in r['encoded'])
    r['status']='W8A8_REFERENCE_MATCH' if passed else 'FAIL_NUMERICAL'
    return r
