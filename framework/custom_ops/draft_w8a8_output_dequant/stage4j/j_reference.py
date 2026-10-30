"""Repack frozen data, recompute strict partials, never replace Stage4B Golden."""
import hashlib
import numpy as np
from j_common import SETTINGS,identity,verify,write,read,reference,layout_i,metrics

SPECS={'a_i8.bin':('i1',(20,16,128)), 'b_i8.bin':('i1',(152,20,128,128)),
 'quant_u64_doc_candidate.bin':('<u8',(152,20,128)), 'partials_fp16.bin':('<f2',(152,20,16,128)),
 'golden_fp16.bin':('<f2',(16,19456))}

def compute(a,b,words):
    if a.dtype!=np.int8 or b.dtype!=np.int8 or a.shape!=(20,16,128) or b.shape!=(20,128,128) or words.shape!=(20,128):raise ValueError('N128 shape mismatch')
    # This is the unchanged experimental encoding, not a new hardware assertion.
    if np.any((words & np.uint64(0xffffffff00000000))!=np.uint64(1<<46)) or np.any(words & np.uint64(0x1fff)):
        raise ValueError('candidate Scale control bits differ')
    effective=(words & np.uint64(0xffffffff)).astype('<u4').view('<f4')
    if not np.isfinite(effective).all() or np.any(effective<0):raise ValueError('invalid effective Scale')
    parts=np.empty((20,16,128),'<f2');total=np.zeros((16,128),'<f4')
    for g in range(20):
        dot=a[g].astype(np.int64) @ b[g].astype(np.int64)
        parts[g]=(dot.astype('<f4')*effective[g]).astype('<f2')
        if not np.isfinite(parts[g]).all():raise ValueError('nonfinite partial group='+str(g))
        total=np.add(total,parts[g].astype('<f4'),dtype=np.float32)
    final=total.astype('<f2')
    if not np.isfinite(final).all():raise ValueError('nonfinite final')
    return parts,final

def digest(a):return hashlib.sha256(a.tobytes()).hexdigest()

def prepare(root,evidence):
    root.mkdir();oldcase=read(verify(evidence['stage4b_case']))
    oldmaps=reference.verify_case(oldcase,numerical=False);src=reference.load_source(oldcase['source'])
    maps={n:np.memmap(root/n,mode='w+',dtype=dt,shape=shape) for n,(dt,shape) in SPECS.items() if n!='golden_fp16.bin'}
    maps['a_i8.bin'][:]=src['a'];records=[]
    if maps['a_i8.bin'].tobytes()!=oldmaps['a_i8.bin'].tobytes():raise ValueError('Activation changed')
    write(root/'progress.json',dict(status='INCOMPLETE',tiles=0))
    for tile in range(152):
        b,words=layout_i.verify_tile(src,oldmaps['b_i8.bin'],oldmaps['quant_u64_doc_candidate.bin'],tile)
        parts,final=compute(src['a'],b,words)
        expected=np.concatenate([oldmaps['partials_fp16.bin'][2*tile],oldmaps['partials_fp16.bin'][2*tile+1]],axis=2)
        if parts.tobytes()!=expected.tobytes():raise ValueError('historical Partial differs tile='+str(tile))
        if final.tobytes()!=oldmaps['golden_fp16.bin'][:,tile*128:(tile+1)*128].tobytes():raise ValueError('Stage4B Golden differs tile='+str(tile))
        maps['b_i8.bin'][tile]=b;maps['quant_u64_doc_candidate.bin'][tile]=words;maps['partials_fp16.bin'][tile]=parts
        records.append(dict(tile=tile,n64_tiles=[tile*2,tile*2+1],n_range=[tile*128,(tile+1)*128],groups=20,b_sha256=digest(b),scale_sha256=digest(words),partial_sha256=digest(parts),golden_sha256=digest(final)))
        write(root/'progress.json',dict(status='INCOMPLETE',tiles=tile+1))
    for m in maps.values():m.flush()
    files={n:identity(root/n) for n in maps}
    files['golden_fp16.bin']=oldcase['files']['golden_fp16.bin'] # bind original, never generate a replacement
    if files['b_i8.bin']['sha256']!=evidence['layout']['packed_B_sha256'] or files['quant_u64_doc_candidate.bin']['sha256']!=evidence['layout']['packed_scale_sha256']:raise ValueError('packed SHA differs from Stage4I')
    case=dict(status='STAGE4J_CPU_152_TILES_READY',settings=SETTINGS,files=files,tiles=records,stage4b_case=evidence['stage4b_case'],
        layout=layout_i.symbolic(),activation_requantized=False,golden_policy='ORIGINAL_STAGE4B_IDENTITY; N128 recomputation is only a check',scale_codec='UNCHANGED_EXPERIMENTAL_CODEC_CANDIDATE')
    write(root/'case.json',case);write(root/'progress.json',dict(status='COMPLETE',tiles=152,validated_tile_groups=3040));return case

def verify_case(case,numerical=False):
    if case.get('status')!='STAGE4J_CPU_152_TILES_READY' or case.get('settings')!=SETTINGS or case.get('layout')!=layout_i.symbolic() or case.get('activation_requantized') is not False:raise ValueError('N128 preparation scope differs')
    if set(case['files'])!=set(SPECS) or [x['tile'] for x in case['tiles']]!=list(range(152)):raise ValueError('incomplete N128 inputs')
    maps={}
    for n,(dt,shape) in SPECS.items():
        p=verify(case['files'][n])
        if p.stat().st_size!=int(np.prod(shape))*np.dtype(dt).itemsize:raise ValueError('input bytes differ: '+n)
        maps[n]=np.memmap(p,mode='r',dtype=dt,shape=shape)
    oldcase=read(verify(case['stage4b_case']))
    if case['files']['golden_fp16.bin']!=oldcase['files']['golden_fp16.bin'] or case['files']['a_i8.bin']['sha256']!=oldcase['files']['a_i8.bin']['sha256']:raise ValueError('historical Golden/Activation binding differs')
    for tile,row in enumerate(case['tiles']):
        if row['groups']!=20 or row['n_range']!=[tile*128,(tile+1)*128] or row['n64_tiles']!=[2*tile,2*tile+1]:raise ValueError('index record differs')
        for key,data in [('b_sha256',maps['b_i8.bin'][tile]),('scale_sha256',maps['quant_u64_doc_candidate.bin'][tile]),('partial_sha256',maps['partials_fp16.bin'][tile]),('golden_sha256',maps['golden_fp16.bin'][:,tile*128:(tile+1)*128])]:
            if row[key]!=digest(data):raise ValueError('tile digest differs: '+key)
        if numerical:
            parts,final=compute(maps['a_i8.bin'],maps['b_i8.bin'][tile],maps['quant_u64_doc_candidate.bin'][tile])
            if parts.tobytes()!=maps['partials_fp16.bin'][tile].tobytes() or final.tobytes()!=maps['golden_fp16.bin'][:,tile*128:(tile+1)*128].tobytes():raise ValueError('strict numerical reference changed')
    return maps

def expected(maps,tile,mode,blocks):
    if mode==0:return maps['golden_fp16.bin'][:,tile*128:(tile+1)*128]
    if mode==21:return np.full((16,128),tile%blocks+1+tile/16,'<f2')
    if mode in range(1,21):return maps['partials_fp16.bin'][tile,mode-1]
    raise ValueError('invalid mode')

def compare(maps,mode,blocks,tile_id,outputs):
    if tile_id not in range(-1,152) or len(outputs)!=2 or any(len(x)!=622592 for x in outputs):raise ValueError('invalid scope/output count/size')
    active=list(range(152)) if tile_id<0 else [tile_id];rows=[];good=True;poison=True
    arrays=[np.frombuffer(x,'<f2').reshape(16,19456) for x in outputs]
    for t in range(152):
        lo=t*128;hi=lo+128
        if t not in active:
            for i,a in enumerate(arrays):poison &= bool(np.all(a[:,lo:hi].view('<u2')==(0x7f7f if i==0 else 0xffff)))
            continue
        target=expected(maps,t,mode,blocks);checks=[metrics(target,a[:,lo:hi]) for a in arrays]
        ok=all(x['bitwise_equal'] and x['actual_nonfinite']==0 for x in checks);good &= ok
        mismatch=[]
        for c in checks:
            d=c['first_difference']
            if d:
                m,n=d['index'];groups=[mode-1] if mode in range(1,21) else list(range(20))
                mismatch.append(dict(row=m,channel=lo+n,tile=t,group=(mode-1 if mode in range(1,21) else 'UNRESOLVED_FINAL_SUM'),
                    actual_bits=d['actual_bits'],expected_bits=d['expected_bits'],
                    evidence=[dict(group=g,scale_word=int(maps['quant_u64_doc_candidate.bin'][t,g,n]),cpu_partial_bits=int(maps['partials_fp16.bin'][t,g,m,n].view('<u2'))) for g in groups]))
            else:mismatch.append(None)
        rows.append(dict(tile=t,status='BITWISE_MATCH' if ok else 'FAIL_NUMERICAL',comparisons=checks,first_differences=mismatch))
    repeated=all(arrays[0][:,t*128:(t+1)*128].tobytes()==arrays[1][:,t*128:(t+1)*128].tobytes() for t in active)
    return dict(status='BITWISE_MATCH' if good and poison and repeated else 'FAIL_NUMERICAL',tiles=rows,
        scope='FULL_152_TILES' if tile_id<0 else 'BOUNDED_ONE_TILE',checked_elements=len(active)*2048,
        unchecked_columns='NONE' if tile_id<0 else 'POISON_ONLY_NOT_NUMERICAL_PASS',inactive_poison_unchanged=poison,repeat_equal=repeated,
        total_bit_mismatches=[sum(row['comparisons'][i]['bit_mismatches'] for row in rows) for i in range(2)])
