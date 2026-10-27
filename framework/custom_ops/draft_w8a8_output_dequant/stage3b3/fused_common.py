"""Stage3B3 provenance and independent ordered two-group references."""
import json
from pathlib import Path
import sys
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'stage3b2'))
import local_common as local
from local_common import common,reduction,identity,verify,write,verify_tree,fresh,metrics
NAME='DFlashW8A8TwoGroup';SNAKE='d_flash_w8_a8_two_group'
SETTINGS=dict(op=NAME,shape=[16,256,64],groups=2,group_k=128,base=[16,64,128],block_dim=1,
    A=[2,16,128],B=[2,128,64],scale=[2,64],output=[16,64],C='VECIN_FP16_NZ',dequant='TENSOR',
    user_ub=12288,ub_guard_bytes=2560,reserve=8192,scratch=239104,
    lifecycle='same object; rebind workspace/A/B/Scale; Iterate(false); GetTensorC; PIPE_ALL; End; PIPE_ALL per group',
    modes={'0':'ordered_sum','1':'group0_after_both','2':'group1_after_both'},
    arithmetic='FP32 +0; FP32(partial0); FP32(partial1); final FP16',performance='NOT_RUN')
SIZES={'a_i8.bin':4096,'b_i8.bin':16384,'scale_f32.bin':512,'quant_u64_doc_candidate.bin':1024,
       'scale_effective_f32.bin':512,'partials_fp16.bin':4096,'sum_fp32.bin':4096,'golden_fp16.bin':2048}


def sources():
    paths=[p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts]
    paths += [HERE.parent/'stage3b2/local_common.py',HERE.parent/'stage3b/s3b_common.py',
              HERE.parent/'stage2b/run_stage2b.py',common.FIX/'compile_raw.py']
    return [identity(p) for p in sorted(set(paths))]


def evidence(args):
    # Offline validation only: not one prior executable is launched.
    prior=local.evidence(args)
    rp=args.stage3b2_root/'smoke/report.json';r=json.loads(rp.read_text());verify_tree(r)
    if r.get('status')!='STAGE3B2_LOCAL_NZ_MATCH' or r.get('phase')!='complete' or r.get('device')!=args.device or r.get('settings')!=local.SETTINGS or r.get('sources')!=local.sources() or r.get('evidence')!=prior:
        raise ValueError('accepted unchanged Stage3B2 Local NZ evidence required')
    b=json.loads(verify(r['local_build_report']).read_text());verify_tree(b)
    if b.get('status')!='STAGE3B2_BUILT_NPU_NOT_RUN' or any(b.get(k)!=r[k] for k in ('evidence','sources','settings','device')):
        raise ValueError('Stage3B2 build binding differs')
    if len(r.get('cases',[]))!=6:raise ValueError('six Local NZ witnesses required')
    for row,old in zip(r['cases'],prior['cases']):
        if row.get('status')!='PASS' or row.get('prior')!=old:raise ValueError('Local case order/provenance differs')
        local.check_execution(json.loads(verify(row['execution']).read_text()),args.device)
        if local.compare(old,[verify(x).read_bytes() for x in row['outputs']])!=row['comparison'] or row['comparison']['status']!='CANDIDATE_BITWISE_MATCH':
            raise ValueError('Local numerical proof changed')
    reduce_report=json.loads(verify(prior['reduction_report']).read_text())
    a_id=reduce_report['stage3a_evidence']['report'];a_path=verify(a_id)
    if args.stage3a_root.resolve() not in a_path.parents:raise ValueError('bound Stage3A is outside selected root')
    _,_,_,a_proof,_=reduction.evidence(a_path,args.device,args.cann_root)
    if a_proof!=reduce_report['stage3a_evidence']:raise ValueError('Stage3A reduction binding differs')
    a=json.loads(a_path.read_text())
    return dict(local_report=identity(rp),local_proof=prior,stage3a=a_proof,
        groups=a['groups'][:2],preflight=prior['preflight'],earlier_kernels_executed=False)


def reference(a,b,scale):
    if a.dtype!=np.dtype('i1') or a.shape!=(2,16,128) or b.dtype!=np.dtype('i1') or b.shape!=(2,128,64) or scale.shape!=(2,64):
        raise ValueError('fixed two-group reference shapes required')
    encoded=[common.stage2a.encode_doc_candidate(scale[g]) for g in range(2)]
    words=np.stack([x[0] for x in encoded]);effective=np.stack([x[1] for x in encoded])
    z=np.stack([common.stage2a.dot(a[g],b[g]) for g in range(2)])
    parts=np.stack([common.stage2a.convert(z[g],effective[g]) for g in range(2)])
    if not np.isfinite(parts).all():raise ValueError('nonfinite partial reference')
    total=np.zeros((16,64),'<f4')
    for p in parts:total=np.add(total,p.astype('<f4'),dtype=np.float32)
    out=total.astype('<f2')
    if not np.isfinite(out).all():raise ValueError('nonfinite sum reference')
    return dict(words=words,effective=effective,z=z,parts=parts,total=total,out=out)


def synthetic(swapped=False):
    a=np.stack([((np.arange(2048)*(7+g*4)+g*9)%19-9).astype('i1').reshape(16,128) for g in range(2)])
    b=np.stack([((np.arange(8192)*(11+g*6)+g*3)%23-11).astype('i1').reshape(128,64) for g in range(2)])
    s=np.stack([np.array([2.**(-(n%4)) for n in range(64)],'<f4'),
                np.array([2.**(-4-(n//16)) for n in range(64)],'<f4')])
    return a,b,s[::-1].copy() if swapped else s


def controls(a,b,s):
    ref=reference(a,b,s);eff=ref['effective'];z=ref['z']
    tests={'stale_scale_g1':common.stage2a.convert(z[1],eff[0]),
           'stale_a_g1':common.stage2a.convert(common.stage2a.dot(a[0],b[1]),eff[1]),
           'stale_b_g1':common.stage2a.convert(common.stage2a.dot(a[1],b[0]),eff[1]),
           'carry_l0c_g1':common.stage2a.convert(z[0]+z[1],eff[1])}
    result={name:metrics(ref['parts'][1],value)['bit_mismatches'] for name,value in tests.items()}
    swapped=reference(a,b,s[::-1].copy())
    result['swapped_final']=metrics(ref['out'],swapped['out'])['bit_mismatches']
    return result


def prepare(root,proof):
    root.mkdir();values=[];real=[]
    for g,item in enumerate(proof['groups']):
        c=json.loads(verify(item['case']).read_text());local.reference.verify_case(c)
        load=lambda n,t,shape:np.fromfile(verify(c['files'][n]),t).reshape(shape)
        outs=[verify(x).read_bytes() for x in item['outputs']]
        if outs[0]!=outs[1] or outs[0]!=verify(c['files']['doc_candidate_fp16.bin']).read_bytes():raise ValueError('real Stage3A partial Golden differs')
        real.append((load('a_i8.bin','i1',(16,128)),load('b_i8.bin','i1',(128,64)),load('scale_f32.bin','<f4',(64,)),outs[0]))
    specs=[('real_g0_g1',np.stack([x[0] for x in real]),np.stack([x[1] for x in real]),np.stack([x[2] for x in real]))]
    specs += [(name,*synthetic(swap)) for name,swap in [('synthetic_distinct',False),('synthetic_swapped',True)]]
    for name,a,b,s in specs:
        d=reference(a,b,s);folder=root/name;folder.mkdir()
        if name=='real_g0_g1' and d['parts'].tobytes()!=b''.join(x[3] for x in real):raise ValueError('CPU candidate differs from accepted Stage3A partials')
        content={'a_i8.bin':a,'b_i8.bin':b,'scale_f32.bin':s,'quant_u64_doc_candidate.bin':d['words'],
            'scale_effective_f32.bin':d['effective'],'partials_fp16.bin':d['parts'],'sum_fp32.bin':d['total'],'golden_fp16.bin':d['out']}
        for n,v in content.items():v.tofile(folder/n)
        case=dict(name=name,settings=SETTINGS,files={n:identity(folder/n) for n in content},
            source_groups=proof['groups'] if name=='real_g0_g1' else [],controls=controls(a,b,s),
            status='CPU_ONLY_NPU_NOT_RUN',activation='already quantized from full X; only pack existing group bytes')
        if name!='real_g0_g1' and any(v==0 for v in case['controls'].values()):raise ValueError('synthetic witness cannot detect a stale state/Scale error')
        write(folder/'case.json',case);verify_case(case);values.append(identity(folder/'case.json'))
    return values


def verify_case(case):
    if case.get('settings')!=SETTINGS or set(case['files'])!=set(SIZES):raise ValueError('wrong two-group case contract')
    for n,size in SIZES.items():
        verify(case['files'][n])
        if case['files'][n]['bytes']!=size:raise ValueError('case bytes mismatch')
    load=lambda n,t,shape:np.fromfile(case['files'][n]['path'],t).reshape(shape)
    a=load('a_i8.bin','i1',(2,16,128));b=load('b_i8.bin','i1',(2,128,64));s=load('scale_f32.bin','<f4',(2,64))
    d=reference(a,b,s)
    for n,k in [('quant_u64_doc_candidate.bin','words'),('scale_effective_f32.bin','effective'),('partials_fp16.bin','parts'),('sum_fp32.bin','total'),('golden_fp16.bin','out')]:
        if d[k].tobytes()!=verify(case['files'][n]).read_bytes():raise ValueError('CPU reference differs: '+n)
    if case['name']=='real_g0_g1':
        if len(case['source_groups'])!=2 or [g['group_index'] for g in case['source_groups']]!=[0,1]:raise ValueError('real group order differs')
        for name in ('a_i8.bin','b_i8.bin','scale_f32.bin','quant_u64_doc_candidate.bin'):
            chunks=[verify(json.loads(verify(g['case']).read_text())['files'][name]).read_bytes() for g in case['source_groups']]
            if b''.join(chunks)!=verify(case['files'][name]).read_bytes():raise ValueError('packed real data differs from Stage3A')
        for repeat in (0,1):
            if b''.join(verify(g['outputs'][repeat]).read_bytes() for g in case['source_groups'])!=d['parts'].tobytes():raise ValueError('real partial Golden identity changed')
    elif case['name'] in ('synthetic_distinct','synthetic_swapped'):
        expected=synthetic(case['name']=='synthetic_swapped')
        if any(x.tobytes()!=y.tobytes() for x,y in zip((a,b,s),expected)):raise ValueError('synthetic fixture changed')
    else:raise ValueError('unknown case')
    if controls(a,b,s)!=case['controls']:raise ValueError('stale-state witness changed')
    return d


def compare(case,mode,outs):
    d=verify_case(case)
    if mode not in (0,1,2) or len(outs)!=2 or any(len(x)!=2048 for x in outs):raise ValueError('mode/two repeats required')
    target=d['out'] if mode==0 else d['parts'][mode-1]
    checks=[metrics(target,np.frombuffer(x,'<f2').reshape(16,64)) for x in outs]
    ok=outs[0]==outs[1] and all(x['bitwise_equal'] and x['actual_nonfinite']==0 for x in checks)
    return dict(status='BITWISE_MATCH' if ok else 'FAIL_NUMERICAL',mode=mode,comparisons=checks,repeat_equal=outs[0]==outs[1])


def check_execution(ex,device,mode):
    if ex.get('op')!=NAME or ex.get('shape')!=[16,256,64] or ex.get('group_count')!=2 or ex.get('output_mode')!=mode or ex.get('ub_guard_protocol')!='POISON_ON_GUARD_FAILURE':raise ValueError('two-group execution scope differs')
    old=dict(ex);old['op']=common.NAME;old['shape']=[16,128,64];common.check_execution(old,device)
