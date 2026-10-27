"""Twenty-group identity chain, immutable real Golden and independent synthetic witnesses."""
import json
from pathlib import Path
import sys
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'stage3b3'))
import fused_common as prior
from fused_common import common,local,reduction,identity,verify,write,verify_tree,fresh,metrics
NAME='DFlashW8A8TwentyGroup';SNAKE='d_flash_w8_a8_twenty_group'
COUNT=20
MODES=list(range(1,21))+[0]
SETTINGS=dict(op=NAME,shape=[16,2560,64],groups=20,group_k=128,base=[16,64,128],block_dim=1,
    A=[20,16,128],B=[20,128,64],scale=[20,64],output=[16,64],C='VECIN_FP16_NZ',dequant='TENSOR',
    user_ub=12288,ub_guard_bytes=2560,reserve=8192,scratch=239104,
    lifecycle=prior.SETTINGS['lifecycle'],modes={str(i):('ordered_sum' if i==0 else f'group{i-1}_after_all20') for i in range(21)},
    arithmetic='FP32 +0; add FP32(partial[g]) for g=0..19; final FP16',performance='NOT_RUN')
SIZES={'a_i8.bin':40960,'b_i8.bin':163840,'scale_f32.bin':5120,'quant_u64_doc_candidate.bin':10240,
       'scale_effective_f32.bin':5120,'partials_fp16.bin':40960,'sum_fp32.bin':4096,'golden_fp16.bin':2048}
CASE_NAMES=('real_20_groups','synthetic_distinct','synthetic_rotated_scale','synthetic_order_rounding')


def sources():
    paths=[p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts]
    paths += [HERE.parent/'stage3b3/fused_common.py',HERE.parent/'stage3b2/local_common.py',
              HERE.parent/'stage3b/s3b_common.py',HERE.parent/'stage2b/run_stage2b.py',common.FIX/'compile_raw.py']
    return [identity(p) for p in sorted(set(paths))]


def ordered(parts):
    total=np.zeros((16,64),'<f4')
    for part in parts:total=np.add(total,part.astype('<f4'),dtype=np.float32)
    return total,total.astype('<f2')


def validate_two_group(r,device):
    if len(r.get('cases',[]))!=3 or [c['name'] for c in r['cases']]!=['real_g0_g1','synthetic_distinct','synthetic_swapped']:
        raise ValueError('complete ordered Stage3B3 cases required')
    for row in r['cases']:
        if row.get('status')!='PASS' or [m['mode'] for m in row.get('modes',[])]!=[1,2,0]:raise ValueError('Stage3B3 modes incomplete')
        case=json.loads(verify(row['case']).read_text());actual={}
        if case['name']!=row['name']:raise ValueError('Stage3B3 case name mismatch')
        for mode in row['modes']:
            prior.check_execution(json.loads(verify(mode['execution']).read_text()),device,mode['mode'])
            actual[mode['mode']]=[verify(x).read_bytes() for x in mode['outputs']]
            result=prior.compare(case,mode['mode'],actual[mode['mode']])
            if mode.get('status')!='BITWISE_MATCH' or result!=mode['comparison'] or result['status']!='BITWISE_MATCH':
                raise ValueError('Stage3B3 partial/final evidence changed')
        consistency=[]
        for repeat in range(2):
            _,out=ordered([np.frombuffer(actual[m][repeat],'<f2').reshape(16,64) for m in (1,2)])
            check=metrics(out,np.frombuffer(actual[0][repeat],'<f2').reshape(16,64));consistency.append(check)
            if not check['bitwise_equal']:raise ValueError('Stage3B3 partial-to-sum inconsistency')
        if consistency!=row['partial_final_consistency']:raise ValueError('Stage3B3 sum record changed')
    rows=r['cases'][1:];out=[next(m for m in c['modes'] if m['mode']==0)['outputs'][0] for c in rows]
    swap=metrics(np.fromfile(verify(out[0]),'<f2'),np.fromfile(verify(out[1]),'<f2'))
    if swap!=r['scale_swap_observation'] or not swap['bit_mismatches']:raise ValueError('Stage3B3 Scale swap not observed')


def evidence(args):
    upstream=prior.evidence(args) # validates previous stages offline; never executes their runners
    rp=args.stage3b3_root/'smoke/report.json';r=json.loads(rp.read_text());verify_tree(r)
    if r.get('status')!='STAGE3B3_TWO_GROUP_MATCH' or r.get('phase')!='complete' or r.get('device')!=args.device or r.get('settings')!=prior.SETTINGS or r.get('sources')!=prior.sources() or r.get('evidence')!=upstream or r.get('performance')!='NOT_RUN':
        raise ValueError('accepted unchanged Stage3B3 evidence required')
    b=json.loads(verify(r['fused_build_report']).read_text());verify_tree(b)
    if b.get('status')!='STAGE3B3_BUILT_NPU_NOT_RUN' or any(b.get(k)!=r[k] for k in ('evidence','sources','settings','device')):
        raise ValueError('Stage3B3 build binding differs')
    validate_two_group(r,args.device)
    a=json.loads(verify(upstream['stage3a']['report']).read_text())
    groups=a['groups']
    if len(groups)!=20 or [g['group_index'] for g in groups]!=list(range(20)):raise ValueError('complete Stage3A group order required')
    # Read Golden identities from the accepted reduction report/case, never a guessed data filename.
    reduce_id=upstream['local_proof']['reduction_report'];reduce=json.loads(verify(reduce_id).read_text())
    rc=json.loads(verify(reduce['case']).read_text());reduction.verify_case(rc)
    final=dict(report=reduce_id,case=reduce['case'],fp16=rc['files']['reference_fp16.bin'],
               fp32=rc['files']['reference_fp32.bin'],npu_outputs=reduce['outputs'])
    payload=verify(final['fp16']).read_bytes()
    if len(payload)!=2048 or any(verify(x).read_bytes()!=payload for x in final['npu_outputs']):raise ValueError('Stage3B1 bound Golden/NPU outputs differ')
    return dict(two_group_report=identity(rp),two_group_proof=upstream,groups=groups,final=final,
                preflight=upstream['preflight'],earlier_kernels_executed=False)


def reference(a,b,scale):
    if a.dtype!=np.dtype('i1') or a.shape!=(20,16,128) or b.dtype!=np.dtype('i1') or b.shape!=(20,128,64) or scale.shape!=(20,64):
        raise ValueError('fixed twenty-group reference shapes required')
    encoded=[common.stage2a.encode_doc_candidate(scale[g]) for g in range(20)]
    words=np.stack([x[0] for x in encoded]);effective=np.stack([x[1] for x in encoded])
    z=np.stack([common.stage2a.dot(a[g],b[g]) for g in range(20)])
    parts=np.stack([common.stage2a.convert(z[g],effective[g]) for g in range(20)])
    total,out=ordered(parts)
    if not np.isfinite(parts).all() or not np.isfinite(out).all():raise ValueError('nonfinite reference; no saturation fallback')
    return dict(words=words,effective=effective,z=z,parts=parts,total=total,out=out)


def synthetic(name):
    if name=='synthetic_order_rounding':
        a=np.zeros((20,16,128),'i1');b=np.zeros((20,128,64),'i1');s=np.zeros((20,64),'<f4')
        a[0,:,0]=1;a[1,:,0]=1;a[2,:,0]=-1;a[3,:,0]=1
        b[0,0,:]=1;b[1,0,:32]=1;b[2,0,:]=1;b[3,0,32:]=1
        s[0,:32]=2048;s[0,32:]=65504;s[1]=1;s[2]=s[0];s[3]=2.**-12
        return a,b,s
    if name not in ('synthetic_distinct','synthetic_rotated_scale'):raise ValueError('unknown synthetic fixture')
    rng=np.random.Generator(np.random.PCG64(20261027))
    a=rng.integers(-9,10,size=(20,16,128),dtype=np.int8);b=rng.integers(-11,12,size=(20,128,64),dtype=np.int8)
    s=np.array([[(g+1)/32*2.**(-(n%4)) for n in range(64)] for g in range(20)],'<f4')
    return a,b,np.roll(s,1,axis=0).copy() if name=='synthetic_rotated_scale' else s


def controls(a,b,s,d=None):
    d=reference(a,b,s) if d is None else d;eff=d['effective'];z=d['z'];rows=[]
    for g in range(1,20):
        wrong={'scale_previous':common.stage2a.convert(z[g],eff[g-1]),'scale_first':common.stage2a.convert(z[g],eff[0]),
               'a_previous':common.stage2a.convert(common.stage2a.dot(a[g-1],b[g]),eff[g]),
               'b_previous':common.stage2a.convert(common.stage2a.dot(a[g],b[g-1]),eff[g]),
               'l0c_carry':common.stage2a.convert(z[g]+z[g-1],eff[g])}
        rows.append(dict(group=g,mismatches={n:metrics(d['parts'][g],p)['bit_mismatches'] for n,p in wrong.items()}))
    half=np.zeros((16,64),'<f2')
    for p in d['parts']:half=np.add(half,p,dtype=np.float16)
    tree=[p.astype('<f4') for p in d['parts']]
    while len(tree)>1:tree=[np.add(tree[i],tree[i+1],dtype=np.float32) if i+1<len(tree) else tree[i] for i in range(0,len(tree),2)]
    return dict(groups=rows,fp16_accumulator=metrics(d['out'],half)['bit_mismatches'],
        fp32_tree=metrics(d['out'],tree[0].astype('<f2'))['bit_mismatches'],
        reversed_order=metrics(d['out'],ordered(d['parts'][::-1])[1])['bit_mismatches'])


def check_controls(name,result):
    if name in ('synthetic_distinct','synthetic_rotated_scale') and any(v==0 for r in result['groups'] for v in r['mismatches'].values()):
        raise ValueError('synthetic stale input/Scale/L0C witness is not discriminating')
    if name=='synthetic_order_rounding' and any(result[k]==0 for k in ('fp16_accumulator','fp32_tree','reversed_order')):
        raise ValueError('ordered FP32 rounding witness is not discriminating')


def packed_real(proof):
    if len(proof['groups'])!=20 or [g['group_index'] for g in proof['groups']]!=list(range(20)):raise ValueError('real group order differs')
    names=('a_i8.bin','b_i8.bin','scale_f32.bin','quant_u64_doc_candidate.bin','scale_effective_f32.bin')
    chunks={n:[] for n in names};partials=[[],[]]
    for g in proof['groups']:
        c=json.loads(verify(g['case']).read_text());local.reference.verify_case(c)
        for n in names:chunks[n].append(verify(c['files'][n]).read_bytes())
        for repeat in (0,1):partials[repeat].append(verify(g['outputs'][repeat]).read_bytes())
    p=[b''.join(x) for x in partials]
    if p[0]!=p[1] or len(p[0])!=40960:raise ValueError('Stage3A partial repeat differs')
    result={n:b''.join(v) for n,v in chunks.items()};result['partials_fp16.bin']=p[0]
    verify_tree(proof['final'])
    result['sum_fp32.bin']=verify(proof['final']['fp32']).read_bytes()
    result['golden_fp16.bin']=verify(proof['final']['fp16']).read_bytes()
    if any(verify(x).read_bytes()!=result['golden_fp16.bin'] for x in proof['final']['npu_outputs']):raise ValueError('Stage3B1 NPU identity differs')
    return result


def prepare(root,proof):
    root.mkdir();cases=[];real=packed_real(proof)
    for name in CASE_NAMES:
        folder=root/name;folder.mkdir()
        if name=='real_20_groups':
            content=real
        else:
            a,b,s=synthetic(name);d=reference(a,b,s)
            content={'a_i8.bin':a.tobytes(),'b_i8.bin':b.tobytes(),'scale_f32.bin':s.tobytes(),
                'quant_u64_doc_candidate.bin':d['words'].tobytes(),'scale_effective_f32.bin':d['effective'].tobytes(),
                'partials_fp16.bin':d['parts'].tobytes(),'sum_fp32.bin':d['total'].tobytes(),'golden_fp16.bin':d['out'].tobytes()}
        for n,data in content.items():(folder/n).write_bytes(data)
        a=np.frombuffer(content['a_i8.bin'],'i1').reshape(20,16,128);b=np.frombuffer(content['b_i8.bin'],'i1').reshape(20,128,64)
        s=np.frombuffer(content['scale_f32.bin'],'<f4').reshape(20,64)
        witnesses=controls(a,b,s);check_controls(name,witnesses)
        case=dict(name=name,settings=SETTINGS,files={n:identity(folder/n) for n in SIZES},controls=witnesses,
                  real_source=dict(groups=proof['groups'],final=proof['final']) if name=='real_20_groups' else None,
                  status='CPU_PREPARED_NPU_NOT_RUN',numpy_version=np.__version__)
        write(folder/'case.json',case);verify_case(case);cases.append(identity(folder/'case.json'))
    return cases


def verify_case(case):
    if case.get('settings')!=SETTINGS or set(case['files'])!=set(SIZES) or case['name'] not in CASE_NAMES:raise ValueError('wrong twenty-group contract')
    for n,size in SIZES.items():
        verify(case['files'][n])
        if case['files'][n]['bytes']!=size:raise ValueError('case byte count mismatch')
    load=lambda n,t,shape:np.fromfile(case['files'][n]['path'],t).reshape(shape)
    a=load('a_i8.bin','i1',(20,16,128));b=load('b_i8.bin','i1',(20,128,64));s=load('scale_f32.bin','<f4',(20,64))
    d=reference(a,b,s)
    for n,k in [('quant_u64_doc_candidate.bin','words'),('scale_effective_f32.bin','effective'),('partials_fp16.bin','parts'),('sum_fp32.bin','total'),('golden_fp16.bin','out')]:
        if d[k].tobytes()!=verify(case['files'][n]).read_bytes():raise ValueError('strict CPU reference mismatch: '+n)
    if case['name']=='real_20_groups':
        expected=packed_real(case['real_source'])
        if any(verify(case['files'][n]).read_bytes()!=data for n,data in expected.items()):raise ValueError('real input or bound reduction Golden changed')
    else:
        if any(x.tobytes()!=y.tobytes() for x,y in zip((a,b,s),synthetic(case['name']))):raise ValueError('synthetic inputs changed')
    checks=controls(a,b,s,d);check_controls(case['name'],checks)
    if checks!=case['controls']:raise ValueError('mutation witness changed')
    return d


def compare(case,mode,outs):
    d=verify_case(case)
    if type(mode) is not int or mode not in range(21) or len(outs)!=2 or any(len(x)!=2048 for x in outs):raise ValueError('mode/two full repeats required')
    target=d['out'] if mode==0 else d['parts'][mode-1]
    checks=[metrics(target,np.frombuffer(x,'<f2').reshape(16,64)) for x in outs]
    ok=outs[0]==outs[1] and all(x['bitwise_equal'] and not x['actual_nonfinite'] for x in checks)
    if case['name']=='real_20_groups':
        binding=dict(kind='STAGE3B1_BOUND_FP16',identity=case['real_source']['final']['fp16']) if mode==0 else dict(
            kind='STAGE3A_BOUND_NPU_PARTIAL',group=mode-1,identities=case['real_source']['groups'][mode-1]['outputs'])
    else:
        binding=dict(kind='INDEPENDENT_SYNTHETIC_REFERENCE',identity=case['files']['golden_fp16.bin' if mode==0 else 'partials_fp16.bin'],
                     offset=0 if mode==0 else (mode-1)*2048,bytes=2048)
    return dict(status='BITWISE_MATCH' if ok else 'FAIL_NUMERICAL',mode=mode,comparisons=checks,repeat_equal=outs[0]==outs[1],reference_binding=binding)


def check_execution(ex,device,mode):
    if ex.get('op')!=NAME or ex.get('shape')!=[16,2560,64] or ex.get('group_count')!=20 or type(ex.get('output_mode')) is not int or ex.get('output_mode')!=mode or mode not in range(21) or ex.get('ub_guard_protocol')!='POISON_ON_GUARD_FAILURE':raise ValueError('twenty-group execution scope differs')
    old=dict(ex);old['op']=common.NAME;old['shape']=[16,128,64];common.check_execution(old,device)
