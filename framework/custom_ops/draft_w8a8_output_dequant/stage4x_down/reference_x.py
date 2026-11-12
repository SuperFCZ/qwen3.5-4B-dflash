"""Real NPU nonlinear inputs only; independent Down references, no new quantizer."""
import time
import numpy as np
from x_common import w,verify,read,identity
M,K,N,T=16,9728,2560,10


def comparison(reference,actual):
    metrics=w.errors(reference,actual)
    with np.errstate(over='ignore'):rounded=np.asarray(reference).astype('<f2')
    if np.isfinite(rounded).all():
        a=np.asarray(actual,dtype='<f2').view('<u2');b=rounded.view('<u2')
        order=lambda v:np.where(v&32768,32768-(v.astype('i4')&32767),32768+v.astype('i4'))
        metrics.update(bit_mismatches=int(np.count_nonzero(a!=b)),max_ulp_vs_rn16_reference=int(np.max(np.abs(order(a)-order(b)))))
    else:metrics.update(bit_mismatches=None,max_ulp_vs_rn16_reference=None)
    return metrics


def pack(b):
    if b.dtype!=np.int8 or b.shape!=(K,256):raise ValueError('Down B KN Tile shape')
    return np.ascontiguousarray(b.reshape(304,32,16,16).transpose(0,2,3,1))


def unpack(z):
    if z.dtype!=np.int8 or z.shape!=(304,16,16,32):raise ValueError('Down B NZ Tile shape')
    return np.ascontiguousarray(z.transpose(0,3,1,2)).reshape(K,256)


def numeric(actual,expected,tile=-1,repeat=0,first=None):
    if len(actual)!=81920 or len(expected)!=81920 or first is not None and len(first)!=81920:raise ValueError('complete Down output bytes')
    a=np.frombuffer(actual,'<u2').reshape(M,N);b=np.frombuffer(expected,'<u2').reshape(M,N)
    mask=np.ones((M,N),bool)
    if tile==0:
        mask[:,256:]=False
        if np.any(a[~mask]!=(0xffff if repeat else 0x7f7f)):raise ValueError('inactive output Poison changed')
    elif tile!=-1:raise ValueError('scope')
    if not np.isfinite(a[mask].view('<f2')).all() or not np.isfinite(b[mask].view('<f2')).all():raise ValueError('active Poison/nonfinite')
    if first is not None and not np.array_equal(a[mask],np.frombuffer(first,'<u2').reshape(M,N)[mask]):raise ValueError('Repeat mismatch')
    ordered=lambda v:np.where(v&32768,32768-(v.astype('i4')&32767),32768+v.astype('i4'))
    ulp=np.abs(ordered(a)-ordered(b));err=np.zeros((M,N),'f8');err[mask]=np.abs(a[mask].view('<f2').astype('f8')-b[mask].view('<f2').astype('f8'))
    indices=np.argwhere(mask&(a!=b))
    return dict(numeric_status='NUMERIC_MISMATCH_UNRESOLVED' if len(indices) else 'BITWISE_REFERENCE_OBSERVATION',active_elements=int(mask.sum()),mismatch_count=len(indices),max_ulp=int(ulp[mask].max()),max_absolute_error=float(err.max()),
        differences=[dict(row=int(m),channel=int(n),actual_bits=int(a[m,n]),expected_bits=int(b[m,n]),ulp=int(ulp[m,n]),absolute_error=float(err[m,n])) for m,n in indices])


def expected(case,tile,repeat):
    raw=verify(case['files']['golden_fp16.bin']).read_bytes()
    if tile==-1:return raw
    out=np.full((M,N),0xffff if repeat else 0x7f7f,'<u2');out[:,:256]=np.frombuffer(raw,'<u2').reshape(M,N)[:,:256]
    return out.tobytes()


def prepare(root,e):
    p=read(verify(e['prepare']));m=read(verify(e['mlp']));files=p['weights']['down']['files']
    q=w.array(files['q_fullk_nk.bin'],'i1',(N,K));sw=w.array(files['weight_scale_f64.bin'],'<f8',(N,))
    weight=w.array(files['w_effective_nk.bin'],'<f2',(N,K))
    if np.any(q==-128) or not np.isfinite(sw).all() or np.any(sw<=0):raise ValueError('weight scale/range')
    root.mkdir();path=root/'b_nz.bin';pack_ns=0;start=time.perf_counter_ns()
    with path.open('xb') as f:
        for t in range(T):
            lo=t*256;refq,refs,_=w.quantize_weight(weight[lo:lo+256])
            if not np.array_equal(refq,q[lo:lo+256]) or refs.tobytes()!=sw[lo:lo+256].tobytes():raise ValueError('frozen Full-K checkpoint weight rule differs')
            b=q[lo:lo+256].T.copy();now=time.perf_counter_ns();z=pack(b);pack_ns+=time.perf_counter_ns()-now
            if not np.array_equal(unpack(z),b):raise ValueError('lossless B mapping failed')
            z.tofile(f)
    result=dict(packed_b=identity(path),shape=[10,304,16,16,32],padding_bytes=0,logical_bytes=N*K,
        prepack_cpu_ms=pack_ns/1e6,verified_prepare_wall_ms=(time.perf_counter_ns()-start)/1e6,weight_sources=files,cases=[])
    result['cost_scope']=dict(weight_conversion='REUSED_CHECKED_FROZEN_FULLK_WEIGHTS_NO_REQUANTIZATION',
        packing='STATIC_CPU_NOT_FREE',initial_input_payload_bytes=25079808,output_payload_bytes=81920,
        online_activation_and_scale='CPU_PER_SAMPLE_PREPARE_MEASURED_SEPARATELY; NPU_NOT_RUN',
        transfer='Runner initial CopyIn and output CopyOut wall measurements, outside resident Execute+Sync; includes wrapper overhead')
    for case in m['cases']:
        mixed=case['boundaries']['MIXED_BNZ_GATE_W8A16_DOWN'][0];native=case['boundaries']['W8A16'][0]
        x=w.array(mixed['multiply'],'<f2',(M,K));now=time.perf_counter_ns()
        v=x.astype('f8');peak=float(np.max(np.abs(v)));sa=peak/127 if peak else 1.
        a=np.clip(np.rint(v/sa),-127,127).astype('i1');qms=(time.perf_counter_ns()-now)/1e6
        checked,checked_s,aq=w.shared_activation(x)
        if a.tobytes()!=checked.tobytes() or sa!=checked_s:raise ValueError('shared activation rule changed')
        now=time.perf_counter_ns();product=np.float64(sa)*sw;combined=product.astype('<f4')
        if np.any((product>0)&(combined==0)):raise ValueError('FP32 Scale underflow')
        words,eff=w.encode(combined);sms=(time.perf_counter_ns()-now)/1e6
        dot=np.empty((M,N),'<i4');y=np.empty((M,N),'<f2')
        for t in range(T):dot[:,t*256:(t+1)*256],y[:,t*256:(t+1)*256]=w.compute(a,q[t*256:(t+1)*256].T.copy(),words[t*256:(t+1)*256])
        ref=w.fp64_mm(x,weight);dest=root/case['case'];dest.mkdir()
        arrays={'a_i8.bin':a,'quant_u64_doc_candidate.bin':words,'golden_fp16.bin':y,'dot_i32.bin':dot,'dot_i64.bin':dot.astype('<i8'),
            'effective_f32.bin':eff,'combined_f32.bin':combined,'reference_f64.bin':ref}
        record=dict(id=case['case'],context_rows=case['context_rows'],input=mixed['multiply'],same_input_w8a16_down=mixed['down'],
            original_w8a16_mlp=native['down'],gate_output=mixed['gate_up'],scale_a=sa,activation=aq,
            cpu_online_quant_ms=qms,cpu_scale_encode_ms=sms,cpu_timing_scope='ONE_PREPARE_OBSERVATION_NOT_RESIDENT_NPU_OR_STEADY_STATE',online_npu_preprocessing='NOT_RUN',
            files={k:w.save(dest,k,v) for k,v in arrays.items()},cpu_error_vs_fp64=w.errors(ref,y))
        result['cases'].append(record)
    return result
