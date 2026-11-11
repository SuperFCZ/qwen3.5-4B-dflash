#!/usr/bin/env python3
"""Native eager W8A16/FP16 MLP and an explicitly mixed W8A8-gate/W8A16-down bridge.

This process never loads the B-NZ private OPP. Its output is consumed from the
separately checked ACLNN process. Complete W8A8 MLP timing is NOT_RUN.
"""
import argparse
import ctypes
import os
from pathlib import Path
import sys
import time
import numpy as np
import real_mlp_common as c
import run_real_gate as gate
from reference_r import check as fp16_check, POLICY
from real_mlp_dso_identity import bind_native_runtime, verify_native_binding, mapped_runtime, compare_loaded


def guard_inputs(torch, values, device):
    holders=[]; tensors=[]
    for value in values:
        data=torch.from_numpy(np.array(value,copy=True)).to(device)
        holder=torch.full((data.numel()+512,),91,dtype=data.dtype,device=device)
        view=holder[256:-256].view(data.shape);view.copy_(data)
        holders.append((holder,data,view));tensors.append(view)
    def check():
        for holder,expected,view in holders:
            if not torch.all(holder[:256]==91).item() or not torch.all(holder[-256:]==91).item():
                raise ValueError('native eager input surrounding guard changed')
            if not torch.equal(view,expected):raise ValueError('native eager input readonly failed')
    return tensors,check


def boundaries(torch, x, gate_fn, down_fn):
    g=gate_fn(x);a,b=g.split(9728,dim=-1)
    silu=torch.nn.functional.silu(a)  # FP16 boundary, same AirDFlashOps implementation
    product=silu*b
    down=down_fn(product)
    return dict(gate_up=g,silu=silu,multiply=product,down=down)


def snapshot(torch, values, root, tag, checks):
    torch.npu.synchronize();checks()
    arrays={};files={}
    for name,value in values.items():
        if value.device.type!='npu' or value.dtype!=torch.float16 or not torch.isfinite(value).all().item():
            raise ValueError('native boundary invalid '+name)
        # This checks explicit bridge storage, not undocumented native internal UB.
        holder=torch.full((value.numel()+512,),91,dtype=value.dtype,device=value.device)
        out=holder[256:-256].view(value.shape)
        out.fill_(float('nan'));torch.npu.synchronize()
        if not torch.isnan(out).all().item():raise ValueError('native output poison failed')
        out.copy_(value);torch.npu.synchronize()
        if not torch.isfinite(out).all().item() or not torch.all(holder[:256]==91).item() or not torch.all(holder[-256:]==91).item():
            raise ValueError('native boundary copy coverage/guard failed')
        array=out.detach().cpu().contiguous().numpy()
        arrays[name]=array;files[name]=c.save(root,tag+'-'+name+'.bin',array)
    return arrays,files


def native_case(torch, a, root, case, prep, gate_row, timed, r):
    from models.dflash_v1.weight_quant_matmul import weight_quant_linear
    g,d=prep['weights']['gate_up'],prep['weights']['down']
    x=c.array(case['files']['gate_up_x.bin'],'<f2',(16,2560))
    weights=[]
    for w,kind in ((g,'gate_up'),(d,'down')):
        _,k,n=c.SHAPES[kind]
        weights.extend([c.array(w['files']['q_group_nk.bin'],'i1',(n,k)),
                        c.array(w['files']['s_group_gn.bin'],'<f2',(k//128,n)).T.copy(),
                        c.array(w['files']['w_effective_nk.bin'],'<f2',(n,k))])
    device_gate=c.array(gate_row['outputs'][0],'<f2',(16,19456))
    tensors,check=guard_inputs(torch,[x,*weights,device_gate],f'npu:{a.device}')
    tx,gq,gs,gw,dq,ds,dw,bnz=tensors
    quant=lambda x,q,s:weight_quant_linear(x,q,s)
    functions={
        'W8A16':lambda:boundaries(torch,tx,lambda x:quant(x,gq,gs),lambda x:quant(x,dq,ds)),
        'FP16_EFFECTIVE':lambda:boundaries(torch,tx,lambda x:torch.nn.functional.linear(x,gw),lambda x:torch.nn.functional.linear(x,dw)),
        'MIXED_BNZ_GATE_W8A16_DOWN':lambda:boundaries(torch,tx,lambda _:bnz,lambda x:quant(x,dq,ds)),
    }
    result=dict(case=case['id'],context_rows=case['capture']['context_rows'],boundaries={},metrics={},timings={})
    r.setdefault('cases',[]).append(result)
    anchors={};cached={}
    for name,fn in functions.items():
        repeats=[];result['boundaries'][name]=[]
        for repeat in range(2):
            cached[name]=fn()
            arrays,files=snapshot(torch,cached[name],root,name+f'-{repeat}',check)
            repeats.append(arrays);result['boundaries'][name].append(files)
        if any(repeats[0][k].tobytes()!=repeats[1][k].tobytes() for k in repeats[0]):raise ValueError('native boundary Repeat drift '+name)
        anchors[name]=repeats[0]
    native=anchors['W8A16'];fp=anchors['FP16_EFFECTIVE']
    for key,file,shape in [('gate_up','gate_up_w8a16.bin',(16,19456)),('multiply','down_x.bin',(16,9728)),('down','down_w8a16.bin',(16,2560))]:
        if native[key].tobytes()!=c.array(case['files'][file],'<f2',shape).tobytes():raise ValueError('original eager Draft boundary changed '+key)
    gate_ref=c.array(case['files']['gate_up_reference_f64.bin'],'<f8',(16,19456))
    down_ref=c.fp64_mm(fp['multiply'],weights[5])
    result['independent_fp16_checks']=dict(gate_up=fp16_check(gate_ref,fp['gate_up']),down=fp16_check(down_ref,fp['down']),policy=POLICY)
    for name,value in anchors.items():
        result['metrics'][name]={key:dict(vs_native_w8a16=c.errors(native[key],value[key]),vs_same_source_fp16=c.errors(fp[key],value[key])) for key in value}
    result['mixed_scope']='Gate from separately safety-checked custom OPP; actual NPU SiLU/Multiply/W8A16 Down. NOT_ALL_W8A8; no combined resident call.'
    result['complete_w8a8_mlp']='NOT_RUN_DOWN_KERNEL_UNSUPPORTED'
    result['native_guard_scope']='surrounding input and boundary-copy GM guards; native internal allocations/UB guards NOT_EXPOSED; not custom-OP Guard equivalence'
    r['runtime_libraries']=mapped_runtime(Path(r['native_dso_binding']['cann_root']),r['native_dso_binding'])
    if not timed:return
    previous=c.read(a.smoke_report)
    compare_loaded(r['runtime_libraries'],previous['runtime_libraries'])
    prior_case=next(x for x in previous['cases'] if x['case']==case['id'])
    for name,values in anchors.items():
        for key,value in values.items():
            if value.tobytes()!=c.verify(prior_case['boundaries'][name][0][key]).read_bytes():
                raise ValueError('native output differs from separate Smoke '+name+'/'+key)
    # Complete MLP timing is only meaningful for the two native paths. Never sum
    # independent gate/process/bridge latencies and label that as integrated W8A8.
    jobs={}
    for name,fn in functions.items():
        if not name.startswith('MIXED'):jobs[name+'/mlp']=(lambda fn=fn:fn()['down'],anchors[name]['down'])
        y=cached[name]['gate_up'];z=cached[name]['multiply']
        jobs[name+'/swiglu']=(lambda y=y:torch.nn.functional.silu(y[:,:9728])*y[:,9728:],anchors[name]['multiply'])
        down_fn=(lambda z=z:torch.nn.functional.linear(z,dw)) if name=='FP16_EFFECTIVE' else (lambda z=z:quant(z,dq,ds))
        jobs[name+'/down']=(down_fn,anchors[name]['down'])
    jobs['W8A16/gate_up']=(lambda:quant(tx,gq,gs),native['gate_up'])
    jobs['FP16_EFFECTIVE/gate_up']=(lambda:torch.nn.functional.linear(tx,gw),fp['gate_up'])
    libc=ctypes.CDLL(None)
    for rnd in range(3):
        order=list(jobs) if rnd%2==0 else list(reversed(jobs))
        for name in order:
            fn,expected=jobs[name]
            def verify_output(out,tag):
                arrays,files=snapshot(torch,{'output':out},root,tag,check)
                if arrays['output'].tobytes()!=expected.tobytes():raise ValueError('native timed output drift '+name)
                return files
            record=dict(round=rnd,samples_ms=[],checks=[])
            result['timings'].setdefault(name,[]).append(record)
            tag=name.replace('/','_')+f'-round{rnd}'
            for i in range(2):record['checks'].append(verify_output(fn(),tag+f'-before{i}'))
            libc.fflush(None); before=(a.output_dir/'native.log').stat().st_size
            for _ in range(5):out=fn();torch.npu.synchronize()
            for _ in range(30):
                start=time.perf_counter_ns();out=fn();torch.npu.synchronize()
                record['samples_ms'].append((time.perf_counter_ns()-start)/1e6)
            libc.fflush(None)
            if (a.output_dir/'native.log').stat().st_size!=before:raise ValueError('native timed log changed')
            record['checks'].append(verify_output(out,tag+'-tail'))
            record['checks'].append(verify_output(fn(),tag+'-post'))
            record['statistics']=c.stats(record['samples_ms'],30)
    result['statistics']={name:c.stats([s for rnd in rows for s in rnd['samples_ms']],90) for name,rows in result['timings'].items()}
    result['timing_scope']='TORCH_EAGER_DISPATCH_EXECUTE_SYNC; allocations and dispatcher included; no H2D/D2H/checks/IO in interval; not ACLNN resident Runner metric'


def main(a):
    root=c.fresh(a.output_dir)
    r=dict(status='NOT_RUN',sources=c.sources(),prepare_report=c.identity(a.prepare_report),gate_smoke_report=c.identity(a.gate_smoke_report),
        npu='NOT_RUN',benchmark='NOT_RUN',quality='UNRESOLVED',production='NOT_APPROVED',full_w8a8_mlp='NOT_RUN_DOWN_KERNEL_UNSUPPORTED')
    try:
        prep=c.prepared(a.prepare_report);g=c.read(a.gate_smoke_report);c.verify_tree(g)
        if (g.get('status')!=gate.STATES['smoke'] or g.get('error') or g.get('integrity_error') or
            g['prepare_report']!=r['prepare_report'] or g['sources']!=r['sources'] or g['device']!=a.device):raise ValueError('Gate Smoke prerequisite differs')
        gate.pair.verify_build(g['runtime'],g['evidence'],Path(g['cann_root']))
        current={}
        bound=gate.history(argparse.Namespace(paired_report=c.verify(g['paired_report']),
            cann_root=Path(g['cann_root']),device=a.device),root,current)
        if bound!=g['evidence']:raise ValueError('native process historical SDK/runtime changed')
        c.pe.compare_machine(g['machine'],current['machine']);r['machine']=current['machine']
        if [x['case'] for x in g['runs']]!=[x['id'] for x in prep['cases']]:raise ValueError('real sample inventory drift')
        for case,row in zip(prep['cases'],g['runs']):
            if row['validation']!=gate.validate(row,case,g['evidence'],a.device):raise ValueError('Gate smoke changed')
        if a.phase=='benchmark':
            old=c.read(a.smoke_report);c.verify_tree(old)
            if old.get('status')!='REAL_MLP_BOUNDARIES_RECORDED_W8_DOWN_NOT_RUN' or old.get('error') or old.get('integrity_error') or any(old[k]!=r[k] for k in ('sources','prepare_report','gate_smoke_report')):raise ValueError('MLP smoke missing/changed')
            r['smoke_report']=c.identity(a.smoke_report)
        if os.environ.get('ASCEND310P_SIMULATION_ONLY')=='1' or os.environ.get('ASCEND_CUSTOM_OPP_PATH'):
            raise ValueError('real native-only process required; unset custom OPP path, no fallback')
        if a.phase=='benchmark':verify_native_binding(old['native_dso_binding'],Path(g['cann_root']))
        r['native_dso_binding']=bind_native_runtime(Path(g['cann_root']),g['evidence']['bnz']['build']['variants']['candidate']['build'])
        if a.phase=='benchmark' and old['native_dso_binding']!=r['native_dso_binding']:
            raise ValueError('native vendor/private DSO binding differs from Smoke')
        sys.path[:0]=[str(c.REPO),str(c.REPO/'framework/python')]
        import torch
        import torch_npu
        from models.dflash_v1.weight_quant_matmul import require_weight_quant_matmul
        require_weight_quant_matmul();torch.npu.set_device(a.device)
        if '310P' not in torch.npu.get_device_name(a.device).upper():raise ValueError('310P required')
        r['environment']=dict(device=a.device,name=torch.npu.get_device_name(a.device),torch=str(torch.__version__),torch_npu=str(torch_npu.__version__),
            torch_module=c.identity(torch.__file__),torch_npu_module=c.identity(torch_npu.__file__))
        if a.phase=='benchmark' and old['environment']!=r['environment']:
            raise ValueError('native Torch runtime differs from Smoke')
        r['runtime_libraries_after_import']=mapped_runtime(Path(g['cann_root']),r['native_dso_binding'])
        if a.phase=='benchmark':compare_loaded(r['runtime_libraries_after_import'],old['runtime_libraries'])
        # Record stdout/stderr; inspect log size outside each continuous interval.
        sys.stdout.flush();sys.stderr.flush()
        log=(root/'native.log').open('wb');os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
        r['npu']='ATTEMPTED'
        with torch.inference_mode():
            for case,row in zip(prep['cases'],g['runs']):
                dest=root/case['id'];dest.mkdir()
                native_case(torch,a,dest,case,prep,row,a.phase=='benchmark',r)
        r['runtime_libraries']=mapped_runtime(Path(g['cann_root']),r['native_dso_binding'])
        if a.phase=='benchmark':compare_loaded(r['runtime_libraries'],old['runtime_libraries'],complete=True)
        r['npu']='COMPLETED'
        r['status']='REAL_MLP_BOUNDARIES_RECORDED_W8_DOWN_NOT_RUN' if a.phase=='smoke' else 'REAL_NATIVE_MLP_TIMED_W8_DOWN_NOT_RUN'
        if a.phase=='benchmark':r['benchmark']='NATIVE_EAGER_AND_MIXED_SUFFIX_ONLY'
    except Exception as exc:r.update(status='BLOCKED',error=f'{type(exc).__name__}: {exc}')
    finally:
        try:
            c.verify_tree(r);c.prepared(a.prepare_report)
            if r['sources']!=c.sources():raise ValueError('source changed')
            if 'runtime_libraries' in r and r['runtime_libraries']!=mapped_runtime(Path(g['cann_root']),r['native_dso_binding']):
                raise ValueError('native library identity/alias changed')
        except Exception as exc:r.update(status='BLOCKED',integrity_error=str(exc))
        c.write(root/'report.json',r)
    return int(r['status']=='BLOCKED')


def parse(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('phase',choices=('smoke','benchmark'))
    for key in ('prepare-report','gate-smoke-report','output-dir'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--smoke-report',type=Path);p.add_argument('--device',type=int,default=0)
    a=p.parse_args(argv)
    if a.phase=='benchmark' and a.smoke_report is None:p.error('--smoke-report required')
    if a.device<0:p.error('negative device')
    for key,val in vars(a).items():
        if isinstance(val,Path):setattr(a,key,val.resolve())
    return a


if __name__=='__main__':raise SystemExit(main(parse()))
