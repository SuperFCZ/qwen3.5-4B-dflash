"""Tile-streaming preparation. Real A and scale_A are read, never re-quantized."""
import hashlib,json
import numpy as np
from full_common import *
LAYOUT=dict(A='group,row,k (copied saved full-X quantization)',B='tile,group,k,n',Scale='tile,group,n',Output='row,tile*64+n',row_stride_elements=N)

class ReferenceFailure(ValueError):
    def __init__(self,kind,tile,group=None,row=None,channel=None,k=None):
        self.detail=dict(kind=kind,tile=tile,group=group,row=row,channel=channel,k=k)
        super().__init__(json.dumps(self.detail,sort_keys=True))


def finite(value,kind,tile,group=None,weight=False):
    bad=np.argwhere(~np.isfinite(value))
    if bad.size:
        index=bad[0].tolist()
        if weight:raise ReferenceFailure(kind,tile,group,channel=tile*64+index[0],k=group*128+index[1])
        if value.ndim==1:raise ReferenceFailure(kind,tile,group,channel=tile*64+index[0])
        raise ReferenceFailure(kind,tile,group,row=index[0],channel=tile*64+index[-1])


def load_source(proof):
    source=proof['source'];files=source['frozen']
    for name,size in [('x.bin',81920),('q_nk.bin',49807360),('w_nz.bin',49807360),('s_gn.bin',778240)]:
        verify(files[name])
        if files[name]['bytes']!=size:raise ValueError('frozen size mismatch: '+name)
    x=np.fromfile(files['x.bin']['path'],'<f2').reshape(16,2560)
    if not np.isfinite(x).all():raise ValueError('nonfinite original X')
    prepared=source['prepared'];saved=verify(prepared['activation_full_i8.bin'])
    if prepared['activation_full_i8.bin']['bytes']!=40960:raise ValueError('saved full INT8 A size mismatch')
    aq=np.fromfile(saved,'i1').reshape(16,2560)
    if np.any(aq==-128):raise ValueError('saved symmetric Activation violates [-127,127]')
    if verify(prepared['x_full_fp16.bin']).read_bytes()!=x.tobytes():raise ValueError('saved X identity differs')
    sa=float(source['activation']['scale'])
    if not np.isfinite(sa) or sa<=0 or source['activation'].get('scale_scope')!=[16,2560]:raise ValueError('invalid bound full-X activation scale')
    a=aq.reshape(16,20,128).transpose(1,0,2).copy()
    if a.tobytes()!=verify(proof['history']['files']['a_i8.bin']).read_bytes():raise ValueError('saved Activation differs from accepted Stage4A input')
    q=np.memmap(files['q_nk.bin']['path'],mode='r',dtype='i1',shape=(19456,2560))
    nz=np.memmap(files['w_nz.bin']['path'],mode='r',dtype='i1',shape=(80,1216,16,32))
    sw=np.memmap(files['s_gn.bin']['path'],mode='r',dtype='<f2',shape=(20,19456))
    return dict(x=x,a=a,sa=sa,q=q,nz=nz,sw=sw)


def compute_tile(src,tile,with_w8a16=False):
    if tile not in range(304):raise ValueError('tile out of range')
    a=src['a'];n0=64*tile;b=np.empty((20,128,64),'i1');scales=np.empty((20,64),'<f4')
    words=np.empty((20,64),'<u8');eff=np.empty((20,64),'<f4');parts=np.empty((20,16,64),'<f2')
    total=np.zeros((16,64),'<f4');wacc=np.zeros((16,64),'<f4');boundaries=[];werror=None
    for g in range(20):
        nk=src['nz'][g*4:g*4+4,tile*4:tile*4+4].transpose(1,2,0,3).reshape(64,128)
        expected=src['q'][n0:n0+64,g*128:(g+1)*128]
        bad=np.argwhere(nk!=expected)
        if bad.size:raise ReferenceFailure('NZ_INVERSE_MISMATCH',tile,g,channel=n0+int(bad[0,0]),k=g*128+int(bad[0,1]))
        sw=src['sw'][g,n0:n0+64];finite(sw,'WEIGHT_SCALE_NONFINITE',tile,g)
        neg=np.flatnonzero(sw<0)
        if neg.size:raise ReferenceFailure('NEGATIVE_WEIGHT_SCALE',tile,g,channel=n0+int(neg[0]))
        b[g]=nk.T
        with np.errstate(over='ignore',under='ignore'):scales[g]=(np.float64(src['sa'])*sw.astype(np.float64)).astype('<f4')
        finite(scales[g],'COMBINED_SCALE_NONFINITE',tile,g)
        words[g],eff[g]=common.stage2a.encode_doc_candidate(scales[g])
        zero=np.flatnonzero((scales[g]>0)&(eff[g]==0));sub=np.flatnonzero((eff[g]>0)&(eff[g]<np.finfo(np.float32).tiny))
        if zero.size or sub.size:boundaries.append(dict(group=g,zero_after_encoding_channels=(zero+n0).tolist(),subnormal_effective_channels=(sub+n0).tolist()))
        z=common.stage2a.dot(a[g],b[g]) # exact INT64 oracle; no full-N INT64 tensor
        with np.errstate(over='ignore',under='ignore'):parts[g]=(z.astype('<f4')*eff[g][None,:]).astype('<f2')
        finite(parts[g],'CANDIDATE_PARTIAL_NONFINITE',tile,g)
        total=np.add(total,parts[g].astype('<f4'),dtype=np.float32);finite(total,'CANDIDATE_SUM_NONFINITE',tile,g)
        if with_w8a16:
            with np.errstate(over='ignore',under='ignore',invalid='ignore'):
                w16=(nk.astype(np.float64)*sw[:,None].astype(np.float64)).astype('<f2')
            try:finite(w16,'W8A16_EFFECTIVE_WEIGHT_NONFINITE',tile,g,weight=True)
            except ReferenceFailure as error:
                if werror is None:werror=error.detail
            with np.errstate(over='ignore',invalid='ignore'):
                for k in range(128):
                    product=src['x'][:,g*128+k,None].astype('<f4')*w16[None,:,k].astype('<f4')
                    wacc=np.add(wacc,product,dtype=np.float32)
    with np.errstate(over='ignore',under='ignore'):out=total.astype('<f2')
    finite(out,'CANDIDATE_FINAL_NONFINITE',tile,19)
    result=dict(b=b,scale=scales,words=words,effective=eff,parts=parts,total=total,out=out,boundaries=boundaries)
    if with_w8a16:
        with np.errstate(over='ignore',under='ignore'):wout=wacc.astype('<f2')
        if werror is None:
            try:finite(wout,'W8A16_FINAL_NONFINITE',tile,19)
            except ReferenceFailure as error:werror=error.detail
        result.update(w8a16=wout,w8a16_error=werror)
    return result


def regions(reference,actual):
    results={}
    for name,lo,hi in [('full',0,N),('gate',0,9728),('up',9728,N)]:
        x=reference[:,lo:hi];y=actual[:,lo:hi]
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            results[name]=dict(status='SKIPPED_NONFINITE_REFERENCE_OR_OUTPUT',reference_nonfinite=int((~np.isfinite(x)).sum()),actual_nonfinite=int((~np.isfinite(y)).sum()))
        else:
            m=metrics(x,y);results[name]=dict(status='COMPLETED',rmse=m['rmse'],max_abs=m['max_abs'],relative_l2_error=m['relative_l2_error'],cosine=m['cosine'])
    return results


def history_views(proof):
    h=proof['history'];verify_tree(h)
    files=h['files']
    return dict(b=np.memmap(verify(files['b_i8.bin']),mode='r',dtype='i1',shape=(8,20,128,64)),
        words=np.memmap(verify(files['quant_u64_doc_candidate.bin']),mode='r',dtype='<u8',shape=(8,20,64)),
        parts=np.memmap(verify(files['partials_fp16.bin']),mode='r',dtype='<f2',shape=(8,20,16,64)),
        out=np.fromfile(verify(files['golden_fp16.bin']),'<f2').reshape(16,512))


def check_history(tile,d,history):
    if tile not in HISTORY:return
    i=HISTORY.index(tile)
    for key,target in [('b',history['b'][i]),('words',history['words'][i]),('parts',history['parts'][i]),('out',history['out'][:,i*64:(i+1)*64])]:
        if d[key].tobytes()!=target.tobytes():raise ReferenceFailure('HISTORICAL_STAGE4A_'+key.upper()+'_MISMATCH',tile)


def prepare(root,proof):
    root.mkdir();src=load_source(proof);hist=history_views(proof)
    files={name:np.memmap(root/name,mode='w+',dtype=dt,shape=shape) for name,(dt,shape) in SPECS.items()}
    files['a_i8.bin'][:]=src['a'];rows=[];diagnostic_errors=[]
    write(root/'progress.json',dict(status='INCOMPLETE',completed_tiles=0))
    for tile in range(TILES):
        d=compute_tile(src,tile,True);check_history(tile,d,hist);lo=tile*64;hi=lo+64
        for name,key in [('b_i8.bin','b'),('quant_u64_doc_candidate.bin','words'),('scale_f32.bin','scale'),('scale_effective_f32.bin','effective'),('partials_fp16.bin','parts')]:files[name][tile]=d[key]
        files['sum_fp32.bin'][:,lo:hi]=d['total'];files['golden_fp16.bin'][:,lo:hi]=d['out'];files['w8a16_cpu_fp16.bin'][:,lo:hi]=d['w8a16']
        if d['w8a16_error']:diagnostic_errors.append(d['w8a16_error'])
        rows.append(dict(tile=tile,n_range=[lo,hi],status='CPU_REFERENCE_VALIDATED',nz_groups=20,
            b_sha256=hashlib.sha256(d['b'].tobytes()).hexdigest(),scale_sha256=hashlib.sha256(d['words'].tobytes()).hexdigest(),
            golden_sha256=hashlib.sha256(d['out'].tobytes()).hexdigest(),historical_stage4a='BITWISE_MATCH' if tile in HISTORY else 'NOT_APPLICABLE',scale_boundaries=d['boundaries']))
        write(root/'progress.json',dict(status='INCOMPLETE',completed_tiles=tile+1,last_tile=tile))
    for a in files.values():a.flush()
    accuracy=regions(files['w8a16_cpu_fp16.bin'],files['golden_fp16.bin'])
    case=dict(status='CPU_304_TILES_READY',settings=SETTINGS,source=proof,tiles=rows,files={n:identity(root/n) for n in SPECS},
        layout=LAYOUT,
        w8a16_reference=dict(kind='CPU_DIAGNOSTIC_NOT_NATIVE_GOLDEN',arithmetic='RN16(INT8 q * FP16 GN scale); original FP16 X; sequential FP32 products/additions k=0..2559; final FP16 RNE; no saturation',errors=diagnostic_errors,accuracy=accuracy),
        native_comparison=proof.get('native_comparison',dict(status='SKIPPED',reason='Native semantic evidence not re-audited')))
    write(root/'case.json',case);write(root/'progress.json',dict(status='COMPLETE',completed_tiles=304,nz_groups=6080));return case


def verify_case(case,numerical=True):
    if case.get('status')!='CPU_304_TILES_READY' or case.get('settings')!=SETTINGS or case.get('layout')!=LAYOUT or set(case['files'])!=set(SPECS):raise ValueError('incomplete full-Gate CPU contract')
    if len(case.get('tiles',[]))!=304 or [r['tile'] for r in case['tiles']]!=list(range(304)):raise ValueError('304 ordered tiles required')
    for name,(dt,shape) in SPECS.items():
        verify(case['files'][name])
        if case['files'][name]['bytes']!=int(np.prod(shape))*np.dtype(dt).itemsize:raise ValueError('file size differs: '+name)
    verify_tree(case['source'])
    maps={name:np.memmap(case['files'][name]['path'],mode='r',dtype=dt,shape=shape) for name,(dt,shape) in SPECS.items()}
    src=load_source(case['source']);hist=history_views(case['source'])
    if src['a'].tobytes()!=maps['a_i8.bin'].tobytes():raise ValueError('saved full A changed')
    for tile,row in enumerate(case['tiles']):
        lo=tile*64;hi=lo+64
        if row.get('status')!='CPU_REFERENCE_VALIDATED' or row.get('n_range')!=[lo,hi] or row.get('nz_groups')!=20 or row.get('historical_stage4a')!=('BITWISE_MATCH' if tile in HISTORY else 'NOT_APPLICABLE'):raise ValueError('tile validation record differs')
        for field,value in [('b_sha256',maps['b_i8.bin'][tile]),('scale_sha256',maps['quant_u64_doc_candidate.bin'][tile]),('golden_sha256',maps['golden_fp16.bin'][:,lo:hi])]:
            if hashlib.sha256(value.tobytes()).hexdigest()!=row[field]:raise ValueError('tile digest differs')
        if numerical:
            d=compute_tile(src,tile);check_history(tile,d,hist)
            for name,key in [('b_i8.bin','b'),('quant_u64_doc_candidate.bin','words'),('scale_f32.bin','scale'),('scale_effective_f32.bin','effective'),('partials_fp16.bin','parts')]:
                if maps[name][tile].tobytes()!=d[key].tobytes():raise ReferenceFailure('PACKED_'+key.upper()+'_DIFFERS',tile)
            if maps['sum_fp32.bin'][:,lo:hi].tobytes()!=d['total'].tobytes() or maps['golden_fp16.bin'][:,lo:hi].tobytes()!=d['out'].tobytes():raise ReferenceFailure('STRICT_GOLDEN_DIFFERS',tile)
            if d['boundaries']!=row['scale_boundaries']:raise ValueError('boundary record differs')
    return maps


def target_tile(maps,tile,mode,blocks):
    if mode==0:return maps['golden_fp16.bin'][:,tile*64:(tile+1)*64]
    if mode==21:return np.full((16,64),(tile%blocks)+1+tile/16,'<f2')
    if mode in range(1,21):return maps['partials_fp16.bin'][tile,mode-1]
    raise ValueError('invalid output mode')


def compare(case,maps,mode,blocks,outputs):
    if len(outputs)!=2 or any(len(x)!=622592 for x in outputs):raise ValueError('two complete full outputs required')
    actual=[np.frombuffer(x,'<f2').reshape(16,N) for x in outputs];rows=[];matched=True
    for tile in range(TILES):
        expected=target_tile(maps,tile,mode,blocks);checks=[metrics(expected,a[:,tile*64:(tile+1)*64]) for a in actual]
        ok=all(x['bitwise_equal'] and not x['actual_nonfinite'] for x in checks);matched &= ok
        first=[]
        for result in checks:
            bad=result['first_difference']
            first.append(None if bad is None else dict(row=bad['index'][0],channel=tile*64+bad['index'][1],
                group=mode-1 if mode in range(1,21) else None,expected_bits=bad['expected_bits'],actual_bits=bad['actual_bits']))
        rows.append(dict(tile=tile,n_range=[tile*64,(tile+1)*64],status='BITWISE_MATCH' if ok else 'FAIL_NUMERICAL',comparisons=checks,first_global_differences=first))
    repeated=outputs[0]==outputs[1]
    r=dict(status='BITWISE_MATCH' if matched and repeated else 'FAIL_NUMERICAL',tiles=rows,repeat_equal=repeated,
        total_elements=16*N,total_bit_mismatches=[sum(row['comparisons'][i]['bit_mismatches'] for row in rows) for i in range(2)],
        output_coverage='ALL_311296_ELEMENTS_MATCHED' if matched and repeated else 'FAILED',mode=mode)
    if mode==0:
        hist=history_views(case['source'])
        r['historical_stage4a']=[dict(tile=t,comparisons=[metrics(hist['out'][:,i*64:(i+1)*64],a[:,t*64:(t+1)*64]) for a in actual]) for i,t in enumerate(HISTORY)]
        if any(not c['bitwise_equal'] for row in r['historical_stage4a'] for c in row['comparisons']):r['status']='FAIL_NUMERICAL'
        r['cross_scheme_accuracy']=dict(reference=case['w8a16_reference'],npu_vs_w8a16_cpu=regions(maps['w8a16_cpu_fp16.bin'],actual[0]),
            native=case['native_comparison'],acceptance='DIAGNOSTIC_ONLY; not part of candidate bitwise gate')
    return r
