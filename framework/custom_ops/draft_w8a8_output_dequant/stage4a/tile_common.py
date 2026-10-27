"""Stage4A: eight immutable tile identities, compact row stride, strict references."""
import hashlib,json,re
from pathlib import Path
import sys
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'stage3b4'))
import twenty_common as prev
from twenty_common import common,local,reduction,identity,verify,write,verify_tree,fresh,metrics
from w8_reference import activation
NAME='DFlashW8A8MultiTile';SNAKE='d_flash_w8_a8_multi_tile'
TILES=[0,1,15,16,151,152,302,303]
SETTINGS=dict(op=NAME,tiles=TILES,A=[20,16,128],B=[8,20,128,64],scale=[8,20,64],output=[16,512],
    tile_n=64,groups=20,base=[16,64,128],scratch=239104,user_ub=12288,ub_guards=2560,reserve=8192,
    scheduling='slot=GetBlockIdx()+i*GetBlockNum(); reset FP32 sum for every slot',
    workspace='one full platform-sized SDK arena per core, 512B GM guards, plus framework system reservation',
    workspace_cap=512*1024*1024,performance='NOT_RUN')
SIZES={'a_i8.bin':40960,'b_i8.bin':1310720,'quant_u64_doc_candidate.bin':81920,
       'scale_f32.bin':40960,'scale_effective_f32.bin':40960,'partials_fp16.bin':327680,
       'sum_fp32.bin':32768,'golden_fp16.bin':16384}


def sources():
    files=[p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts]
    files += [HERE.parent/'stage3b4/twenty_common.py',HERE.parent/'stage2b/run_stage2b.py',common.FIX/'compile_raw.py']
    return [identity(p) for p in sorted(set(files))]


def evidence(args):
    upstream=prev.evidence(args)
    path=args.stage3b4_root/'smoke/report.json';r=json.loads(path.read_text());verify_tree(r)
    if r.get('status')!='STAGE3B4_20_GROUP_FUSED_MATCH' or r.get('phase')!='complete' or r.get('device')!=args.device or r.get('settings')!=prev.SETTINGS or r.get('sources')!=prev.sources() or r.get('evidence')!=upstream or r.get('performance')!='NOT_RUN':raise ValueError('unchanged accepted Stage3B4 required')
    b=json.loads(verify(r['fused_build_report']).read_text());verify_tree(b)
    if b.get('status')!='STAGE3B4_BUILT_NPU_NOT_RUN' or any(b.get(k)!=r[k] for k in ('evidence','sources','settings','device')):raise ValueError('Stage3B4 build binding differs')
    if tuple(c['name'] for c in r.get('cases',[]))!=prev.CASE_NAMES:raise ValueError('all four Stage3B4 cases required')
    for row in r['cases']:
        if row.get('status')!='PASS' or [m['mode'] for m in row.get('modes',[])]!=prev.MODES:raise ValueError('Stage3B4 modes incomplete')
        case=json.loads(verify(row['case']).read_text());actual={}
        if case['name']!=row['name']:raise ValueError('Stage3B4 case name mismatch')
        if case['name']!=row['name']:raise ValueError('old case name mismatch')
        for m in row['modes']:
            prev.check_execution(json.loads(verify(m['execution']).read_text()),args.device,m['mode'])
            actual[m['mode']]=[verify(x).read_bytes() for x in m['outputs']]
            checked=prev.compare(case,m['mode'],actual[m['mode']])
            if m.get('status')!='BITWISE_MATCH' or checked!=m['comparison'] or checked['status']!='BITWISE_MATCH':raise ValueError('Stage3B4 numerical proof changed')
        checks=[]
        for repeat in (0,1):
            _,out=prev.ordered([np.frombuffer(actual[m][repeat],'<f2').reshape(16,64) for m in range(1,21)])
            c=metrics(out,np.frombuffer(actual[0][repeat],'<f2').reshape(16,64));checks.append(c)
            if not c['bitwise_equal']:raise ValueError('old partial-final inconsistency')
        if checks!=row['partial_final_consistency']:raise ValueError('old consistency report changed')
    if r.get('summary',{}).get('matched_modes')!=84:raise ValueError('84 accepted modes required')
    rotated=[next(m for m in row['modes'] if m['mode']==0)['outputs'][0] for row in r['cases'][1:3]]
    rotation=metrics(np.fromfile(verify(rotated[0]),'<f2'),np.fromfile(verify(rotated[1]),'<f2'))
    if rotation!=r['scale_swap_observation'] or not rotation['bit_mismatches']:raise ValueError('Stage3B4 rotation proof differs')
    real=r['cases'][0];rc=json.loads(verify(real['case']).read_text())
    final=next(x for x in real['modes'] if x['mode']==0)
    a=json.loads(verify(upstream['two_group_proof']['stage3a']['report']).read_text())
    manifest=json.loads(verify(a['cpu_manifest']).read_text());verify_tree(manifest)
    return dict(previous_report=identity(path),upstream=upstream,manifest=a['cpu_manifest'],
        frozen=manifest['source_files'],prepared=manifest['prepared_inputs'],activation=manifest['activation'],
        tile0=dict(case=real['case'],golden=rc['files']['golden_fp16.bin'],outputs=final['outputs'],inputs=rc['files']),
        preflight=upstream['preflight'],earlier_runners_executed=False)


def slot_metadata():
    return [dict(slot=i,original_tile=t,n_range=[64*t,64*(t+1)],nz_n16_range=[4*t,4*t+4],
                 compact_columns=[64*i,64*(i+1)],b_offset_bytes=i*20*8192,scale_offset_bytes=i*20*512) for i,t in enumerate(TILES)]


def compact(tiles):return np.concatenate(list(tiles),axis=1)


def reference(a,b,scale):
    if a.shape!=(20,16,128) or a.dtype!=np.dtype('i1') or b.shape!=(8,20,128,64) or b.dtype!=np.dtype('i1') or scale.shape!=(8,20,64):raise ValueError('fixed packed layout required')
    refs=[prev.reference(a,b[i],scale[i]) for i in range(8)]
    return dict(words=np.stack([d['words'] for d in refs]),effective=np.stack([d['effective'] for d in refs]),
        parts=np.stack([d['parts'] for d in refs]),total=np.stack([d['total'] for d in refs]),
        out=compact([d['out'] for d in refs]))


def derive(proof):
    sizes={'x.bin':81920,'q_nk.bin':19456*2560,'w_nz.bin':19456*2560,'s_gn.bin':20*19456*2}
    for name,size in sizes.items():
        verify(proof['frozen'][name])
        if proof['frozen'][name]['bytes']!=size:raise ValueError('frozen shape/bytes mismatch')
    x=np.fromfile(proof['frozen']['x.bin']['path'],'<f2').reshape(16,2560)
    aq,sa,quant=activation(x) # verify ONCE over the entire X; no tile-local scale
    if quant!=proof['activation']:raise ValueError('full-X quantization contract differs')
    if verify(proof['prepared']['x_full_fp16.bin']).read_bytes()!=x.tobytes() or verify(proof['prepared']['activation_full_i8.bin']).read_bytes()!=aq.tobytes():raise ValueError('saved full Activation differs')
    a=aq.reshape(16,20,128).transpose(1,0,2).copy()
    q=np.memmap(proof['frozen']['q_nk.bin']['path'],mode='r',dtype='i1',shape=(19456,2560))
    nz=np.memmap(proof['frozen']['w_nz.bin']['path'],mode='r',dtype='i1',shape=(80,1216,16,32))
    scales=np.memmap(proof['frozen']['s_gn.bin']['path'],mode='r',dtype='<f2',shape=(20,19456))
    b=np.empty((8,20,128,64),'i1');s=np.empty((8,20,64),'<f4')
    for slot,t in enumerate(TILES):
        n=64*t
        for g in range(20):
            nk=nz[g*4:g*4+4,t*4:t*4+4].transpose(1,2,0,3).reshape(64,128)
            if not np.array_equal(nk,q[n:n+64,g*128:(g+1)*128]):raise ValueError(f'NZ inverse differs slot={slot} tile={t} group={g}')
            sw=scales[g,n:n+64]
            if not np.isfinite(sw).all() or np.any(sw<0):raise ValueError('invalid GN Scale')
            b[slot,g]=nk.T;s[slot,g]=(np.float64(sa)*sw.astype(np.float64)).astype('<f4')
    ref=reference(a,b,s)
    old=proof['tile0']
    for name,data in [('a_i8.bin',a),('b_i8.bin',b[0]),('quant_u64_doc_candidate.bin',ref['words'][0])]:
        if verify(old['inputs'][name]).read_bytes()!=data.tobytes():raise ValueError('Tile0 packed data differs from accepted Stage3B4')
    gold=verify(old['golden']).read_bytes()
    if ref['out'][:,:64].tobytes()!=gold or any(verify(x).read_bytes()!=gold for x in old['outputs']):raise ValueError('Tile0 reference differs from bound Stage3B4 Golden')
    # Retain literal old Golden bytes for Tile0, after independent equivalence verification.
    ref['out'][:,:64]=np.frombuffer(gold,'<f2').reshape(16,64)
    return a,b,s,ref


def prepare(root,proof):
    root.mkdir();a,b,s,d=derive(proof)
    arrays={'a_i8.bin':a,'b_i8.bin':b,'scale_f32.bin':s,'quant_u64_doc_candidate.bin':d['words'],
        'scale_effective_f32.bin':d['effective'],'partials_fp16.bin':d['parts'],'sum_fp32.bin':d['total'],'golden_fp16.bin':d['out']}
    for name,data in arrays.items():data.tofile(root/name)
    slots=slot_metadata()
    for i,meta in enumerate(slots):
        meta.update(b_sha256=hashlib.sha256(b[i].tobytes()).hexdigest(),scale_sha256=hashlib.sha256(d['words'][i].tobytes()).hexdigest(),
            golden_sha256=hashlib.sha256(d['out'][:,i*64:(i+1)*64].tobytes()).hexdigest(),nz_inverse='BIT_EXACT_ALL_20_GROUPS')
    case=dict(settings=SETTINGS,files={n:identity(root/n) for n in SIZES},source=proof,slots=slots,
        layout=dict(A='group,row,k',B='slot,group,k,n',Scale='slot,group,n',Output='row,slot*64+n',row_stride_elements=512),
        status='CPU_PREPARED_NPU_NOT_RUN')
    write(root/'case.json',case);return case


def verify_case(case):
    if case.get('settings')!=SETTINGS or set(case['files'])!=set(SIZES):raise ValueError('Stage4A contract differs')
    for n,size in SIZES.items():
        verify(case['files'][n])
        if case['files'][n]['bytes']!=size:raise ValueError('packed size differs')
    a,b,s,d=derive(case['source'])
    expected={'a_i8.bin':a,'b_i8.bin':b,'scale_f32.bin':s,'quant_u64_doc_candidate.bin':d['words'],
        'scale_effective_f32.bin':d['effective'],'partials_fp16.bin':d['parts'],'sum_fp32.bin':d['total'],'golden_fp16.bin':d['out']}
    for n,value in expected.items():
        if verify(case['files'][n]).read_bytes()!=value.tobytes():raise ValueError('packed source/reference mismatch: '+n)
    if len(case['slots'])!=8:raise ValueError('tile count differs')
    for i,(row,meta) in enumerate(zip(case['slots'],slot_metadata())):
        if any(row.get(k)!=v for k,v in meta.items()) or row.get('nz_inverse')!='BIT_EXACT_ALL_20_GROUPS':raise ValueError('original/compact tile identity differs')
        for key,value in [('b_sha256',b[i]),('scale_sha256',d['words'][i]),('golden_sha256',d['out'][:,i*64:(i+1)*64])]:
            if row[key]!=hashlib.sha256(value.tobytes()).hexdigest():raise ValueError('slot hash differs')
    return d


def schedule(cores,limit):
    if type(cores) is not int or cores<1 or type(limit) is not int or limit not in range(9):raise ValueError('invalid core count/limit')
    blocks=min(8,cores) if limit==0 else limit
    if blocks>cores:raise ValueError('more blocks than real cores')
    return blocks,[[slot for slot in range(core,8,blocks)] for core in range(blocks)]


def plan_from_log(text,mode,limit):
    pattern=r'STAGE4A_SCHEDULE coreLimit=(\d+) blocks=(\d+) cores=(\d+) system=(\d+) arena=(\d+) stride=(\d+) user=(\d+) total=(\d+) mode=(\d+)'
    rows=set(re.findall(pattern,text))
    if len(rows)!=1:raise ValueError('missing/ambiguous actual workspace/core plan')
    p=dict(zip(('limit','blocks','cores','system','arena','stride','user','total','mode'),map(int,rows.pop())))
    blocks,owners=schedule(p['cores'],limit)
    if p['limit']!=limit or p['mode']!=mode or p['blocks']!=blocks or p['system']<1 or p['arena']!=(p['system']+511)//512*512 or p['stride']!=p['arena']+1024 or p['user']!=blocks*p['stride'] or p['total']!=p['system']+p['user'] or p['total']>SETTINGS['workspace_cap']:
        raise ValueError('workspace/core plan violates disjoint full-arena contract')
    if 'scratch=239104' not in text or 'userUB=12288 ubGuards=2560 reserve=8192' not in text or 'single=16/64/128 base=16/64/128' not in text:raise ValueError('resource/Tiling validation log missing')
    p['owners']=owners;p['arena_intervals']=[[i*p['stride']+512,i*p['stride']+512+p['arena']] for i in range(blocks)]
    return p


def target(d,mode,blocks):
    if mode==0:return d['out']
    if mode==21:return compact([np.full((16,64),(i%blocks)+1+i/16,'<f2') for i in range(8)])
    if mode in range(1,21):return compact(d['parts'][:,mode-1])
    raise ValueError('invalid mode')


def compare(case,d,mode,blocks,outputs):
    if len(outputs)!=2 or any(len(x)!=16384 for x in outputs):raise ValueError('two full compact outputs required')
    ref=target(d,mode,blocks);rows=[]
    for i,t in enumerate(TILES):
        actual=[np.frombuffer(x,'<f2').reshape(16,512)[:,i*64:(i+1)*64] for x in outputs]
        checks=[metrics(ref[:,i*64:(i+1)*64],x) for x in actual]
        rows.append(dict(slot=i,original_tile=t,comparisons=checks))
    ok=outputs[0]==outputs[1] and all(x['bitwise_equal'] and not x['actual_nonfinite'] for r in rows for x in r['comparisons'])
    return dict(status='BITWISE_MATCH' if ok else 'FAIL_NUMERICAL',tiles=rows,repeat_equal=outputs[0]==outputs[1],
        output_coverage='ALL_8192_ELEMENTS_MATCHED' if ok else 'FAILED',mode=mode,
        tile0_reference=case['source']['tile0']['golden'] if mode==0 else None)


def check_execution(ex,device,mode,limit):
    if ex.get('op')!=NAME or ex.get('shape')!=[16,2560,512] or ex.get('tile_count')!=8 or ex.get('group_count')!=20 or ex.get('output_mode')!=mode or ex.get('core_limit')!=limit or ex.get('resource_admission')!='BEFORE_EXECUTION' or ex.get('ub_guard_protocol')!='UB_AND_CORE_ARENA_POISON_ON_FAILURE':raise ValueError('Stage4A execution scope differs')
    old=dict(ex);old['op']=common.NAME;old['shape']=[16,128,64];common.check_execution(old,device)
