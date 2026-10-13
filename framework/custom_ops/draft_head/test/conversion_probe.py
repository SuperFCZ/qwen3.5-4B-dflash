#!/usr/bin/env python3
"""Device evidence collector and fail-closed gate; CPU reference is NOT device evidence."""
import argparse
import json
import math
import re
import subprocess
from pathlib import Path
import numpy as np
import build_config
from common import K, N, WEIGHT_BYTES, record, sha256, write_json, verify_launch
from fixtures import set_nan_first_weights, check_nan_first_inputs


def sample_bits():
    positive=[0.,2**-25,2**-24,2**-14,1.+2**-11,1.+3*2**-11,
              65472.,65488.,65488.+2**-8,65503.,65504.,65519.,65520.,65536.,131008.]
    values=np.array(positive+[-v for v in positive],dtype='<f4').view('<u4').tolist()
    # FP32 subnormals, explicit quiet NaN payloads, both infinities; padding is tested too.
    values += [0x00000001,0x80000001,0x007fffff,0x807fffff,0x7fc00000,0xffc12345,0x7f800000,0xff800000]
    return np.array(values+[0]*(64-len(values)),dtype='<u4')


def vector_report(raw,stages,casts):
    original=np.frombuffer(raw,dtype='<u4').reshape(64)
    f32=np.frombuffer(stages,dtype='<u4').reshape(3,64)
    f16=np.frombuffer(casts,dtype='<u2').reshape(3,64)
    errors=[]; rows=[]
    for i,b in enumerate(original):
        value=original[i:i+1].view('<f4')[0]
        stage=f32[:,i].copy().view('<f4')
        rounded=f16[:,i].copy().view('<f2')
        row=dict(index=i,input_value=str(float(value)),input_fp32=f'0x{int(b):08x}',
                 stages_fp32=[f'0x{int(v):08x}' for v in f32[:,i]],
                 outputs_fp16=[f'0x{int(v):04x}' for v in f16[:,i]])
        rows.append(row)
        if f32[0,i]!=b: errors.append(dict(index=i,rule='raw input copy changed'))
        if math.isnan(value):
            # Do not assume scalar std::min/fmax semantics or accept NaN -> bound/zero.
            if not all(math.isnan(v) for v in stage) or f32[2,i]!=b:
                errors.append(dict(index=i,rule='Mins/Maxs must preserve original FP32 NaN bits'))
            if not all(math.isnan(v) for v in rounded) or f16[2,i]!=f16[0,i]:
                errors.append(dict(index=i,rule='candidate must preserve control Cast NaN bits'))
        else:
            upper=np.float32(min(float(value),65504.))
            final=np.float32(max(float(upper),-65504.))
            expected32=np.array([value,upper,final],dtype='<f4').view('<u4')
            # Signed zero is retained by the explicit first-operand min/max oracle.
            if not np.array_equal(f32[:,i],expected32):
                errors.append(dict(index=i,rule='FP32 limits/zero/subnormal preservation',expected=[f'0x{v:08x}' for v in expected32]))
            with np.errstate(over='ignore',under='ignore',invalid='ignore'):
                expected16=np.array([value,upper,final],dtype='<f2').view('<u2')
            # Control overflow is the unknown we are measuring: do not assume
            # whether CAST_NONE alone saturates or emits Inf. Only in-range
            # stages and the fully clamped candidate have a prescribed result.
            checked=[j for j,v in enumerate((value,upper,final)) if math.isfinite(v) and abs(v)<=65504]
            if any(f16[j,i]!=expected16[j] for j in checked):
                errors.append(dict(index=i,rule='RNE Cast/finite saturation',checked_stages=checked,
                                   expected=[f'0x{v:04x}' for v in expected16]))
    return dict(status='PASS' if not errors else 'FAIL',scope='measured vector operations against candidate requirements',
                stage_names=['CAST_NONE control','Mins(+65504) then Cast','Mins(+65504)/Maxs(-65504) then Cast'],rows=rows,errors=errors)


def precast_report(data,name):
    raw=np.frombuffer(data,dtype='<u4').reshape(2,16,64)
    rows=[]
    for row in range(15):
        for token in (63,64):
            b=raw[token//64,row,token%64]
            value=np.array([b],dtype='<u4').view('<f4')[0]
            rows.append(dict(row=row,token=token,fp32_bits=f'0x{int(b):08x}',value=str(float(value))))
    # Only overflow has a prescribed pre-cast finite sum. Other cases record facts,
    # and are subsequently checked against Native by the unchanged full comparator.
    expected=int(np.float32(131008).view(np.uint32))
    errors=[r for r in rows if name=='overflow_inf' and int(r['fp32_bits'],16)!=expected]
    return dict(status='PASS' if not errors else 'FAIL',rows=rows,errors=errors,
                finite_overflow_expected_fp32=f'0x{expected:08x}' if name=='overflow_inf' else None)


def environment(cann_root,device):
    root=Path(cann_root).resolve(); versions=[]
    candidates=[root/'version.cfg',root/'version.info',root/'compiler/version.info',root/'runtime/version.info']
    candidates+=list(root.glob('*/ascend_toolkit_install.info'))
    for path in candidates:
        if path.is_file(): versions.append(dict(path=str(path),text=path.read_text(errors='replace'),sha256=sha256(path)))
    errors=[]
    if not any(re.search(r'(?<!\d)9\.0\.0(?!\d)',v['text']) for v in versions):
        errors.append('cannot verify CANN 9.0.0 from toolkit version files')
    result=subprocess.run(['npu-smi','info'],text=True,capture_output=True,timeout=30)
    if result.returncode or '310P' not in result.stdout: errors.append('npu-smi must identify a 310P device')
    return dict(status='FAIL' if errors else 'PASS',errors=errors,cann_root=str(root),versions=versions,device_id=device,npu_smi=result.stdout,stderr=result.stderr,
                scope='current 310P/CANN9 experiment; not a general hardware rule')


def execute(args,folder,mode,weight):
    cmd=[str(args.runner.resolve()),mode,str(args.device_id),str(folder),str(weight)]
    with (folder/'runner.log').open('w') as log:
        subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=args.timeout)
    status=json.loads((folder/'execution.json').read_text())
    if status.get('status')!='PASS' or not status.get('guards_intact') or not status.get('input_readonly') or status.get('repeats')!=3:
        raise ValueError('diagnostic guards/readonly/repeat failure')
    names=['vector-fp32','vector-fp16']+(['precast','logits','ids'] if mode=='head' else [])
    files={}
    for name in names:
        paths=[folder/f'{name}-{i}.bin' for i in range(3)]
        if len({sha256(p) for p in paths})!=1: raise ValueError(f'{name}: repeat drift')
        files[name]=[record(p,folder) for p in paths]
    return dict(command=cmd,execution=status,files=files,values=record(folder/'values.bin',folder))


def run(args):
    root=args.output_dir.resolve(); root.mkdir(parents=True,exist_ok=False)
    report=dict(status='RUNNING',candidate_adoption='NOT_DECIDED',full_41_case_acceptance='NOT_RUN',
                full_draft_validation='NOT_RUN',probes=[],runner_sha256=sha256(args.runner))
    summary=root/'diagnostics.json'; write_json(summary,report)
    try:
        config=build_config.load(args.build_config)
        if not config['diagnostics']: raise ValueError('probe build identity required')
        report['build']=config; report['environment']=environment(args.cann_root,args.device_id)
        write_json(summary,report)
        if report['environment']['status']!='PASS': raise ValueError(report['environment']['errors'])
        raw=sample_bits().tobytes()
        folder=root/'cast'; folder.mkdir(); (folder/'values.bin').write_bytes(raw)
        result=execute(args,folder,'cast','-')
        result['vector']=vector_report(raw,(folder/'vector-fp32-0.bin').read_bytes(),(folder/'vector-fp16-0.bin').read_bytes())
        result['name']='standalone-cast'; report['probes'].append(result); write_json(summary,report)
        # Always collect the crucial raw Head result, even if standalone vector semantics fail.
        for name in ('overflow_inf','nan_first'):
            folder=root/name; folder.mkdir(); (folder/'values.bin').write_bytes(raw)
            x=np.zeros((15,K),dtype='<f2'); x[:,0]=2 if name=='overflow_inf' else 1
            x.tofile(folder/'hidden.bin'); weight=folder/'head.bin'
            with weight.open('wb') as f: f.truncate(WEIGHT_BYTES)
            w=np.memmap(weight,mode='r+',dtype='<f2',shape=(N,K))
            if name=='overflow_inf': w[63,0]=w[64,0]=65504
            else: set_nan_first_weights(w)
            w.flush(); del w
            if name=='nan_first': check_nan_first_inputs(weight,x.tobytes(),15)
            identity=sha256(weight)
            result=execute(args,folder,'head',weight)
            if identity!=sha256(weight): raise ValueError('diagnostic input file changed')
            result.update(name=name,weight=record(weight,folder),hidden=record(folder/'hidden.bin',folder),
                launch=verify_launch(folder/'runner.log',15,True,config['core_limit'],config['conversion']),
                precast=precast_report((folder/'precast-0.bin').read_bytes(),name),
                vector=vector_report(raw,(folder/'vector-fp32-0.bin').read_bytes(),(folder/'vector-fp16-0.bin').read_bytes()))
            # Independent Vector test must agree when embedded after the Head path.
            result['standalone_vector_equal']=all((folder/f'{n}-0.bin').read_bytes()==(root/'cast'/f'{n}-0.bin').read_bytes()
                                                   for n in ('vector-fp32','vector-fp16'))
            report['probes'].append(result); write_json(summary,report)
        if build_config.load(args.build_config)!=config or sha256(args.runner)!=report['runner_sha256']: raise ValueError('source/runner drift')
        passed=all(p['vector']['status']=='PASS' and p.get('precast',{}).get('status','PASS')=='PASS' and p.get('standalone_vector_equal',True)
                   for p in report['probes'])
        report['status']='PASS' if passed else 'FAIL'
        report['next_step']='sensitive subset then full 41-case gate' if passed else 'STOP: inspect raw bits; do not adopt candidate; investigate Compare/Select if special values changed'
    except Exception as error:
        report.update(status='FAIL',error=f'{type(error).__name__}: {error}')
    write_json(summary,report)
    print(f"{report['status']}: diagnostics only; full 41-case NOT_RUN; {summary}",flush=True)
    return 0 if report['status']=='PASS' else 1

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('runner','output-dir','cann-root','build-config'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device-id',type=int,default=0); p.add_argument('--timeout',type=int,default=1800)
    raise SystemExit(run(p.parse_args()))
