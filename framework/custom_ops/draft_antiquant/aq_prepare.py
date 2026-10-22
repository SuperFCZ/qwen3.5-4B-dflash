"""K128 frozen slice and independent CPU references; never a new full-Gate golden."""
from pathlib import Path
import sys
import numpy as np
from aq_common import HERE, identity, verify, write


def slice_arrays(x, q, nz, scale):
    m,k = x.shape; n=q.shape[0]
    if m!=16 or q.shape!=(n,k) or k%128 or n%16 or n<64 or k<128:
        raise ValueError('invalid frozen logical shapes')
    if x.dtype!=np.dtype('<f2') or q.dtype!=np.dtype('i1') or nz.dtype!=q.dtype or scale.dtype!=x.dtype:
        raise ValueError('original FP16/signed INT8 dtype required')
    if nz.shape!=(k//32,n//16,16,32) or scale.shape!=(k//128,n):
        raise ValueError('NZ/GN physical shape mismatch')
    compact=np.ascontiguousarray(nz[:4,:4,:,:])
    restored=compact.transpose(1,2,0,3).reshape(64,128)
    if not np.array_equal(restored,q[:64,:128]):
        raise ValueError('frozen NZ inverse mismatch')
    a=np.ascontiguousarray(x[:,:128]);s=np.ascontiguousarray(scale[:1,:64])
    if not np.isfinite(a).all() or not np.isfinite(s).all() or not (s>0).all():
        raise ValueError('nonfinite input or invalid scale')
    # signed INT8 -> FP16 is exact; check half multiply against independent binary64.
    with np.errstate(over='ignore',invalid='ignore'):
        effective=np.multiply(restored.astype('<f2'),s.T,dtype=np.float16)
        oracle=(restored.astype(np.float64)*s.T.astype(np.float64)).astype('<f2')
    if not np.isfinite(effective).all() or not np.array_equal(effective.view('<u2'),oracle.view('<u2')):
        raise ValueError('FP16 effective-weight RN16 disagreement')
    accum=np.zeros((16,64),dtype=np.float32)
    exact=np.zeros((16,64),dtype=np.float64)
    for index in range(128):
        accum=np.add(accum,a[:,index,None].astype(np.float32)*effective[None,:,index].astype(np.float32),dtype=np.float32)
        exact+=a[:,index,None].astype(np.float64)*effective[None,:,index].astype(np.float64)
    return {'x.bin':a,'w_nz.bin':compact,'q_nk.bin':restored,'scale.bin':s,
            'offset.bin':np.zeros((1,64),dtype='<f2'),'w_fp16_kn.bin':np.ascontiguousarray(effective.T),
            'cpu_fp32.bin':accum.astype('<f2'),'cpu_fp64.bin':exact.astype('<f2')}


def prepare(a51, pair, folder):
    sys.path.insert(0,str(HERE.parent/'draft_quant/fp16_compare'))
    from fp16_gate_evidence import accepted
    source=accepted(a51,pair)  # original hashes, signed NZ inverse, native + repeat golden, Off/Raw OPP identity
    files=source['files'];m,k,n=16,2560,19456
    def load(name,dtype,shape):
        p=verify(files[name])
        if p.stat().st_size!=int(np.prod(shape))*np.dtype(dtype).itemsize:
            raise ValueError('frozen byte count differs')
        return np.memmap(p,mode='r',dtype=dtype,shape=shape)
    arrays=slice_arrays(load('x.bin','<f2',(m,k)),load('q_nk.bin','i1',(n,k)),
        load('w_nz.bin','i1',(k//32,n//16,16,32)),load('s_gn.bin','<f2',(k//128,n)))
    folder.mkdir()
    for name,array in arrays.items():array.tofile(folder/name)
    for item in source['watched'].values():verify(item)
    report=dict(status='CPU_SLICE_VERIFIED_DEVICE_NOT_RUN',shape=[16,128,64],group=0,n_tile=0,
        source_identity=source['identity'],source_files=files,watched=source['watched'],
        full_gate_golden=source['golden'],full_gate_golden_usage='identity only; K128 is NOT a column slice of K2560 output',
        slice_rule='x[:,0:128], q[0:64,0:128], scale[0,0:64], nz[0:4,0:4,:,:]',
        nz_layout='[K/32,N/16,16,32], compact [4,4,16,32]; same axis order, reduced N stride',
        prep_bytes=dict(x=4096,w_nz=8192,scale=128,offset=128,fp16_reference_weight=16384),
        references='RN16(q * scale), FP32 sequential accumulation and independent FP64 sum; CPU order is not presumed identical to Cube',
        files={name:identity(folder/name) for name in arrays},npu='NOT_RUN')
    write(folder/'slice.json',report)
    return report
