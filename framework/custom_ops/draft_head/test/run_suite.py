#!/usr/bin/env python3
"""Full-vocabulary strict head gates plus separately labelled native/custom timing."""
import argparse
import json
from pathlib import Path
import subprocess
from common import ABI,K,N,WEIGHT_BYTES,load_capture,checked_file,record,sha256,write_json,bits,argmax_half,verify_launch
import fixtures
import build_config


def validate_native(path,cap_hash):
    data=json.loads(path.read_text())
    if (data.get('abi')!=ABI or data.get('status')!='PASS' or data.get('capture_sha256')!=cap_hash or
        data.get('weight_binding')!='runtime-readonly-ND' or
        [(x.get('m'),x.get('kind')) for x in data.get('models',[])]!=[(m,k) for m in range(1,16) for k in ('logits','top1')]):
        raise ValueError('incomplete native Head/Top1 export')
    models={}
    for item in data['models']:
        if item.get('status')!='PASS': raise ValueError('native model compilation failed')
        audit=item.get('runtime_input_abi',{})
        if audit.get('status')!='PASS' or audit.get('logical_input_names')!=['hidden','weight']:
            raise ValueError('native public input normalization not audited')
        om=checked_file(path.parent,item['om']); deployment=checked_file(path.parent,item['deployment']); checked_file(path.parent,item['air'])
        d=json.loads(deployment.read_text())
        if d.get('status')!='PASS' or len(d.get('graphs',[]))!=1: raise ValueError('invalid native deployment')
        graph=d['graphs'][0]; m=item['m']; kind=item['kind']
        expected=dict(inputs=[dict(name='hidden',dtype='float16',shape=[m,K]),dict(name='weight',dtype='float16',shape=[N,K])],
                      outputs=[dict(name='y',dtype='float16' if kind=='logits' else 'int64',shape=[m,N] if kind=='logits' else [m])])
        if (graph.get('input_names')!=['hidden','weight'] or graph.get('output_names')!=['y'] or
            graph.get('metadata',{}).get('tensor_abi')!=expected or checked_file(deployment.parent,graph['om'])!=om):
            raise ValueError('native model must use the full Head ABI and exact OM binding')
        models[m,kind]=om
    return models


def validate_execution(x,mode,m,device,warmup,repetitions):
    expected=dict(status='PASS',mode=mode,runtime='AscendCL NPU',cpu_fallback=False,guards_intact=True,
                  input_readonly=True,repeat_equal=True,io_validated=True,m=m,k=K,n=N,device_id=device)
    if any(x.get(k)!=v for k,v in expected.items()): raise ValueError('runner identity/guard/readonly validation failed')
    if x.get('row_permutation_checked') is not (m>1): raise ValueError('missing changed-request check')
    t=x['timing']
    if mode=='audit':
        if t['repetitions']!=0 or t['warmup']!=0 or t['status']!='NOT_RUN' or t.get('protocol')!='checked-v1':
            raise ValueError('audit must not claim production timing')
    else:
        if (t.get('status')!='MEASURED' or t.get('protocol')!='continuous-v1' or t.get('warmup')!=warmup or t.get('repetitions')!=repetitions or
            t.get('per_timed_call_readback') is not False or t.get('per_timed_call_poison') is not False or
            t.get('correctness_before_calls')!=2 or t.get('correctness_after_calls')!=1 or t.get('timed_tail_checked') is not True):
            raise ValueError('timing protocol changed')
        import math
        values=t.get('execute_sync',{}).get('samples_ms',[])
        if len(values)!=repetitions or any(not math.isfinite(v) or v<=0 for v in values): raise ValueError('invalid timings')
        if mode=='fused':
            prepare=t.get('prepare',{}).get('samples_ms',[])
            if len(prepare)!=repetitions or any(not math.isfinite(v) or v<0 for v in prepare): raise ValueError('invalid workspace-query timing')


def compare_case(outputs,m,expected_ids=None,custom=True):
    import numpy as np
    logits=outputs['native-logits'][0]; native_ids=outputs['native-top1'][0]
    comparisons=dict(native_head_repeat=bits(logits,outputs['native-logits'][1],2),
                     native_top1_repeat=bits(native_ids,outputs['native-top1'][1],8),
                     native_references_consistent=bits(argmax_half(logits,m),native_ids,8))
    if expected_ids is not None:
        comparisons['synthetic_expected_ids']=bits(np.asarray(expected_ids,dtype='<i8').tobytes(),native_ids,8)
    if custom:
        comparisons.update(custom_ids=bits(native_ids,outputs['fused'][0],8),
                           custom_repeat=bits(outputs['fused'][0],outputs['fused'][1],8),
                           audit_ids=bits(native_ids,outputs['audit'][0],8),
                           audit_repeat=bits(outputs['audit'][0],outputs['audit'][1],8),
                           full_fp16_logits=bits(logits,outputs['audit-logits'][0],2),
                           audit_logits_repeat=bits(outputs['audit-logits'][0],outputs['audit-logits'][1],2),
                           audit_rounding_argmax=bits(argmax_half(outputs['audit-logits'][0],m),outputs['audit'][0],8))
    return dict(status='PASS' if all(c['equal'] for c in comparisons.values()) else 'FAIL',comparisons=comparisons)


def run(args):
    if not 3<=args.warmup<=100 or not 10<=args.repetitions<=1000: raise ValueError('need warmup>=3, repetitions>=10')
    if args.device_id<0 or not args.native_runner.is_file(): raise ValueError('need valid device/native runner')
    if not args.native_only and (not args.runner or not args.runner.is_file()): raise ValueError('custom runner missing')
    cap_path=args.capture.resolve(); cap=load_capture(cap_path); cap_hash=sha256(cap_path)
    models=validate_native(args.native_manifest.resolve(),cap_hash)
    root=args.output_dir.resolve(); root.mkdir(parents=True,exist_ok=False)
    config=build_config.load(args.build_config) if args.build_config else None
    if not args.native_only and (not config or config.get('abi')!='dflash-head-build-v1' or not args.runner):
        raise ValueError('custom validation needs build config and runner')
    runners={'native':args.native_runner.resolve()}
    if not args.native_only: runners['custom']=args.runner.resolve()
    identities={k:dict(path=str(v),sha256=sha256(v)) for k,v in runners.items()}
    report=dict(abi=ABI,status='RUNNING',scope='native-only' if args.native_only else 'full strict C operator',
                capture=str(cap_path),capture_sha256=cap_hash,native_manifest=str(args.native_manifest.resolve()),
                native_manifest_sha256=sha256(args.native_manifest),build=config,runners=identities,
                custom_correctness='NOT_RUN',native_weight_binding='runtime-readonly-ND',
                numerical_gate='all FP16 logit bits + exact INT64 IDs; no tolerance or NaN-payload relaxation',
                special_policy='FP16 numeric order; zeros tie; first NaN; minimum original ID; native OM gate required',
                production_const_head_parity='NOT_RUN',full_draft_validation='NOT_RUN',end_to_end='NOT_RUN',cases=[])
    summary=root/'suite.json'; write_json(summary,report)
    cases=[]
    w=checked_file(cap_path.parent,cap['weight']['file'],WEIGHT_BYTES)
    for gear in (16,64):
        hidden=checked_file(cap_path.parent,cap['gears'][f'c{gear}']['hidden']).read_bytes()
        for m in range(1,16):
            cases.append(dict(name=f'real-c{gear}-m{m:02d}',m=m,hidden=hidden[:m*K*2],weight=w,source=f'real-c{gear}-prefix'))
    for case in fixtures.prepare(root/'synthetic'):
        cases.append(dict(name='synthetic-'+case['name'],m=case['m'],source=case['source'],expected_ids=case['expected_ids'],
                          hidden=checked_file(root/'synthetic',case['hidden']).read_bytes(),
                          weight=checked_file(root/'synthetic',case['weight'],WEIGHT_BYTES)))
    for case in cases:
        m=case['m']; directory=root/case['name']; directory.mkdir()
        (directory/'hidden.bin').write_bytes(case['hidden'])
        row=dict(name=case['name'],m=m,source=case['source'],status='RUNNING',phase='native-logits',executions={},outputs={},launches={})
        row['inputs']=dict(hidden=record(directory/'hidden.bin',root),weight=dict(path=str(case['weight']),sha256=sha256(case['weight'])))
        report['cases'].append(row); write_json(summary,report)
        try:
            raw={}
            modes=['native-logits','native-top1']+([] if args.native_only else ['audit','fused'])
            for mode in modes:
                row['phase']=mode; dest=directory/mode; dest.mkdir(); (dest/'hidden.bin').symlink_to(directory/'hidden.bin')
                runner=runners['native' if mode.startswith('native') else 'custom']
                om=models[m,mode.removeprefix('native-')] if mode.startswith('native') else '-'
                counts=(0,0) if mode=='audit' else (args.warmup,args.repetitions)
                command=[str(runner),mode,str(args.device_id),str(dest),str(case['weight']),str(m),*map(str,counts),str(om)]
                write_json(dest/'command.json',command)
                with (dest/'runner.log').open('w') as log: subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=args.timeout)
                execution=json.loads((dest/'execution.json').read_text()); validate_execution(execution,mode,m,args.device_id,*counts)
                row['executions'][mode]=execution
                raw[mode]=[(dest/f'actual-{i}.bin').read_bytes() for i in range(2)]
                files=[dest/f'actual-{i}.bin' for i in range(2)]
                if mode=='audit':
                    raw['audit-logits']=[(dest/f'logits-actual-{i}.bin').read_bytes() for i in range(2)]
                    files += [dest/f'logits-actual-{i}.bin' for i in range(2)]
                else:
                    for name in ('benchmark-last.bin','postcheck.bin'):
                        if (dest/name).read_bytes()!=raw[mode][0]: raise ValueError('timed tail/postcheck drift')
                        files.append(dest/name)
                if m>1:
                    width=N*2 if mode=='native-logits' else 8
                    original=raw[mode][0]
                    permuted=original[width:]+original[:width]
                    if (dest/'permuted.bin').read_bytes()!=permuted or (dest/'restored.bin').read_bytes()!=original:
                        raise ValueError('changed/restored request evidence differs')
                    files += [dest/'permuted.bin',dest/'restored.bin']
                    if mode=='audit':
                        golden=raw['audit-logits'][0]; width=N*2
                        if (dest/'logits-permuted.bin').read_bytes()!=golden[width:]+golden[:width] or (dest/'logits-restored.bin').read_bytes()!=golden:
                            raise ValueError('changed/restored audit logits differ')
                        files += [dest/'logits-permuted.bin',dest/'logits-restored.bin']
                if mode in ('fused','audit'):
                    launch=verify_launch(dest/'runner.log',m,mode=='audit',config['core_limit'])
                    if execution['workspace_bytes']<launch['system_workspace_bytes']+launch['user_workspace_bytes']:
                        raise ValueError('ACLNN workspace smaller than declared requirement')
                    row['launches'][mode]=launch
                row['outputs'][mode]=[record(file,root) for file in files]
            comparison=compare_case(raw,m,case.get('expected_ids'),not args.native_only)
            write_json(directory/'comparison.json',comparison)
            row.update(status=comparison['status'],phase='complete',comparison=record(directory/'comparison.json',root))
            if sha256(case['weight'])!=row['inputs']['weight']['sha256']: raise ValueError('weight changed during benchmark')
            checked_file(root,row['inputs']['hidden'],m*K*2)
            print(f"{case['name']}: {row['status']} {comparison['comparisons']}",flush=True)
        except Exception as error:
            row.update(status='FAIL',error=f'{type(error).__name__}: {error}')
            print(f"{case['name']}: FAIL during {row['phase']}: {row['error']}",flush=True)
        write_json(summary,report)
    unchanged=sha256(cap_path)==cap_hash and sha256(args.native_manifest)==report['native_manifest_sha256'] and all(sha256(v['path'])==v['sha256'] for v in identities.values())
    try:
        load_capture(cap_path)
        validate_native(args.native_manifest.resolve(),cap_hash)
    except (ValueError,OSError): unchanged=False
    if config is not None:
        try: unchanged=unchanged and build_config.load(args.build_config)==config
        except ValueError: unchanged=False
    passed=unchanged and len(report['cases'])==41 and all(c['status']=='PASS' for c in report['cases'])
    report.update(status='PASS' if passed else 'FAIL',sources_unchanged=unchanged,
                  custom_correctness='NOT_RUN' if args.native_only else ('PASS' if passed else 'FAIL'))
    report['timing_summary']=[dict(name=c['name'],m=c['m'],source=c['source'],
        modes={mode:execution['timing']['execute_sync'] for mode,execution in c['executions'].items() if mode!='audit'})
        for c in report['cases'] if c['status']=='PASS']
    write_json(summary,report)
    print(f"{report['status']}: {report['scope']}; custom={report['custom_correctness']}; {summary}; full Draft/Decode NOT_RUN",flush=True)
    return 0 if passed else 1

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('capture','native-manifest','native-runner','output-dir'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--runner',type=Path); p.add_argument('--build-config',type=Path); p.add_argument('--native-only',action='store_true')
    p.add_argument('--device-id',type=int,default=0); p.add_argument('--warmup',type=int,default=5)
    p.add_argument('--repetitions',type=int,default=30); p.add_argument('--timeout',type=int,default=3600)
    return run(p.parse_args())
if __name__=='__main__': raise SystemExit(main())
