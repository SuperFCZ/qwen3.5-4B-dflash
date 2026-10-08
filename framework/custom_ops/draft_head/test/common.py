"""Head fixed ABI, byte-bound evidence, and CPU *checking* only. No fallback."""
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parents[1]
REPO=HERE.parents[2]
QUANT=HERE.parent/'draft_quant/test'
sys.path.append(str(QUANT))
from a2_common import checked_file, record, sha256, write_json

K,N,MAX_M,TILE_N=2560,248320,15,64
ABI='dflash-draft-head-v1'
WEIGHT_BYTES=N*K*2


def rows(m):
    if type(m) is not int or not 1<=m<=15: raise ValueError('Head requires M=1..15')
    return m


def load_capture(path):
    path=Path(path).resolve(); x=json.loads(path.read_text())
    if (x.get('abi')!=ABI or x.get('status')!='PASS' or x.get('runtime')!='native NPU DraftGraph final norm capture' or
        x.get('cpu_fallback') is not False or x.get('input_readonly') is not True or
        x.get('repeat_equal') is not True or set(x.get('gears',{}))!={'c16','c64'} or
        x.get('weight',{}).get('shape')!=[N,K] or x['weight'].get('dtype')!='float16' or x['weight'].get('layout')!='fp16_nk'):
        raise ValueError('need complete real C16/C64 post-norm capture and full FP16 head')
    checked_file(path.parent,x['weight']['file'],WEIGHT_BYTES)
    for gear in (16,64):
        item=x['gears'][f'c{gear}']
        if item.get('source',{}).get('context_rows')!=gear: raise ValueError('frozen gear differs')
        if item.get('checkpoint',{}).get('variant')!='w8a16' or item['checkpoint'].get('status')!='PASS':
            raise ValueError('capture checkpoint audit missing')
        checked_file(path.parent,item['hidden'],15*K*2)
        if item.get('hidden_repeat_sha256')!=item['hidden']['sha256']: raise ValueError('capture drift')
    a,b=x['gears']['c16'],x['gears']['c64']
    for key in ('model_sha256','config_sha256'):
        if not a['checkpoint'].get(key) or a['checkpoint'][key]!=b['checkpoint'].get(key): raise ValueError('checkpoint differs between gears')
    if not a['source'].get('feature_layers') or a['source']['feature_layers']!=b['source'].get('feature_layers'):
        raise ValueError('feature order differs between gears')
    return x


def argmax_half(raw,m):
    import numpy as np
    x=np.frombuffer(raw,dtype='<f2').reshape(rows(m),N)
    # Validation oracle only; runtime IDs always come from NPU custom/native.
    return np.argmax(x,axis=1).astype('<i8').tobytes()


def bits(expected,actual,width):
    import numpy as np
    if len(expected)!=len(actual) or len(expected)%width: raise ValueError('output byte size differs')
    a=np.frombuffer(expected,dtype=f'<u{width}'); b=np.frombuffer(actual,dtype=f'<u{width}')
    mismatch=np.flatnonzero(a!=b)
    result=dict(equal=not len(mismatch),bit_mismatches=int(len(mismatch)),
                first_index=int(mismatch[0]) if len(mismatch) else None)
    if width==2:
        nan=lambda x:((x&0x7c00)==0x7c00)&((x&0x3ff)!=0)
        finite_pair=~(nan(a)|nan(b))
        rank=lambda x:np.where(x&0x8000,0x8000-(x&0x7fff).astype('i4'),0x8000+(x&0x7fff).astype('i4'))
        result['max_ulp']=int(np.max(np.abs(rank(a[finite_pair])-rank(b[finite_pair])))) if finite_pair.any() else 0
        result['nan_payload_mismatches']=int(np.count_nonzero((nan(a)|nan(b))&(a!=b)))
    return result


def verify_launch(path,m,audit,cap):
    prefix='DFLASH_HEAD_LAUNCH '
    records=[json.loads(line.partition(prefix)[2]) for line in Path(path).read_text().splitlines() if prefix in line]
    if not records or any(r!=records[0] for r in records): raise ValueError('missing/inconsistent Head launch')
    x=records[0]
    for key,value in dict(version=1,m=m,k=K,n=N,tile_n=64,padded_m=16,audit=audit,core_limit=cap,batchmode=1).items():
        if x.get(key)!=value: raise ValueError(f'wrong Head launch: {key}')
    available=x['available_cores']; p=min(available,64,cap or available)
    if p<1 or x['partitions']!=p or x['user_workspace_bytes']!=p*192: raise ValueError('partition/workspace mismatch')
    if x['user_ub_bytes']!=16*K*2+16*64*2+160+p*32 or x['matmul_ub_bytes']<=0:
        raise ValueError('Head UB accounting mismatch')
    if x.get('single_k')!=K or type(x.get('base_k')) is not int or not 0<x['base_k']<=K or x['base_k']%16:
        raise ValueError('incomplete K reduction plan')
    return x


def source_hashes():
    out={str(p.relative_to(HERE)):sha256(p) for p in sorted(HERE.rglob('*'))
         if p.is_file() and p.suffix in ('.py','.h','.cpp','.json','.sh') and not any(a.startswith('.') for a in p.relative_to(HERE).parts)}
    for name in ('runner_common.h','a2_common.py','a2_capture.py','opp_preflight.py'):
        out['../draft_quant/test/'+name]=sha256(QUANT/name)
    return out
