#!/usr/bin/env python3
"""Isolated builds, observable profiling-only controls, strict paired Head gates."""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from common import HERE,K,WEIGHT_BYTES,checked_file,load_capture,sha256,record,write_json,verify_launch
import build_config
from conversion_probe import environment
from run_suite import validate_execution,validate_native

CASES=[f'real-c{g}-m{m:02d}' for g in (16,64) for m in (1,8,15)]


def call(command,log,env,timeout=None):
    write_json(log.with_suffix('.command.json'),command)
    with log.open('w') as out: subprocess.run(command,env=env,stdout=out,stderr=subprocess.STDOUT,check=True,timeout=timeout)


def build(root,label,conversion='compare_select',top1='scalar',profile='none'):
    state=root/f'{label}-build.json'
    env=os.environ.copy(); env.update(HEAD_CONVERSION=conversion,HEAD_TOP1=top1,HEAD_PROFILE_STAGE=profile,
        HEAD_NATIVE_ONLY='0',HEAD_DIAGNOSTICS='0',HEAD_TOP1_PROBE='1' if top1=='vector' else '0',HEAD_BUILD_ONLY='1',HEAD_BUILD_STATE=str(state))
    call(['bash',str(HERE/'run_server.sh')],root/f'{label}-build.log',env)
    result=json.loads(state.read_text()); result['environment']={**os.environ,**result['environment']}
    config=build_config.load(result['config'])
    if (config['conversion'],config['top1'],config['profile'])!=(conversion,top1,profile): raise ValueError('wrong build')
    return result


def strict(args,root,label,state,subset=False):
    out=root/label
    cmd=[sys.executable,str(HERE/'test/run_suite.py'),'--capture',str(args.capture),'--native-manifest',str(args.native_manifest),
         '--runner',state['runner'],'--native-runner',state['native_runner'],'--build-config',state['config'],
         '--output-dir',str(out),'--device-id',str(args.device_id),'--warmup','5','--repetitions','30','--timeout',str(args.timeout)]
    if subset: cmd+=['--cases',','.join(CASES)]
    call(cmd,root/f'{label}.log',state['environment'])
    data=json.loads((out/'suite.json').read_text())
    if data['status']!='PASS' or (not subset and data.get('full_41_case_acceptance')!='PASS'): raise ValueError('strict gate failed')
    return out,data


def observers(args,root,label,state):
    config=build_config.load(state['config']); cap=load_capture(args.capture)
    weight=checked_file(args.capture.parent,cap['weight']['file'],WEIGHT_BYTES)
    report=dict(status='RUNNING',scope='profiling-only observable Matmul; NOT a Top1 correctness test',
                correctness='NOT_APPLICABLE',full_41_case_acceptance='NOT_RUN',cases=[],build=config,
                capture_sha256=sha256(args.capture),runner_sha256=sha256(state['runner']))
    out=root/label; out.mkdir()
    for gear in (16,64):
        hidden=checked_file(args.capture.parent,cap['gears'][f'c{gear}']['hidden']).read_bytes()
        for m in (1,8,15):
            folder=out/f'real-c{gear}-m{m:02d}'; folder.mkdir(); (folder/'hidden.bin').write_bytes(hidden[:m*K*2])
            cmd=[state['runner'],'fused',str(args.device_id),str(folder),str(weight),str(m),'5','30','-']
            call(cmd,folder/'runner.log',state['environment'],args.timeout)
            execution=json.loads((folder/'execution.json').read_text())
            validate_execution(execution,'fused',m,args.device_id,5,30,profiling=True)
            launch=verify_launch(folder/'runner.log',m,False,config['core_limit'],config['conversion'],config['top1'],config['profile'])
            if execution['workspace_bytes']<launch['user_workspace_bytes']+launch['system_workspace_bytes']: raise ValueError('workspace undersized')
            first=(folder/'actual-0.bin').read_bytes()
            import struct
            if len(first)!=m*8 or any(v>=0 for v in struct.unpack('<'+'q'*m,first)): raise ValueError('observer did not emit a fingerprint')
            for name in ('actual-1.bin','benchmark-last.bin','postcheck.bin'):
                if (folder/name).read_bytes()!=first: raise ValueError('observer repeat/timed tail changed')
            report['cases'].append(dict(name=folder.name,launch=launch,execution=execution,command=cmd,
                input=record(folder/'hidden.bin',out),fingerprint=record(folder/'actual-0.bin',out)))
            write_json(out/'suite.json',report)
    if build_config.load(state['config'])!=config or sha256(state['runner'])!=report['runner_sha256']: raise ValueError('source drift')
    report['status']='MEASURED_NOT_CORRECTNESS'; write_json(out/'suite.json',report)
    return out,report


def profile(args,root,label,state,case_dir):
    # Separate invocations; these samples are NEVER used as uninstrumented latency.
    dest=root/f'profile-{label}'; dest.mkdir(); (dest/'hidden.bin').write_bytes((case_dir/'hidden.bin').read_bytes())
    cap=load_capture(args.capture); weight=checked_file(args.capture.parent,cap['weight']['file'],WEIGHT_BYTES)
    runner_cmd=[state['runner'],'fused',str(args.device_id),str(dest),str(weight),'15','5','30','-']
    profiler=shutil.which('msprof',path=state['environment'].get('PATH'))
    if not profiler: raise ValueError('msprof unavailable; explicitly use --skip-msprof to mark profiling NOT_RUN')
    cmd=[profiler,'--ai-core=on','--aic-mode=task-based','--aic-metrics=PipeUtilization',
         '--output='+str(dest/'profiler'),'--application='+shlex.join(runner_cmd)]
    call(cmd,dest/'profile.log',state['environment'],args.timeout)
    execution=json.loads((dest/'execution.json').read_text())
    config=build_config.load(state['config'])
    validate_execution(execution,'fused',15,args.device_id,5,30,profiling=config['profile']!='none')
    write_json(dest/'profile.json',dict(status='COMMAND_COMPLETED',correctness='NOT_APPLICABLE',
               cube_activity='REQUIRES_REVIEW_OF_RAW_MSPROF',command=cmd,execution=execution))
    return str(dest)


def paired_report(original,baseline,vector):
    for data in (original,baseline,vector):
        if data.get('status')!='PASS' or data['build'].get('profile')!='none': raise ValueError('non-correctness result in paired benchmark')
        if set(c['name'] for c in data['cases'])!=set(CASES): raise ValueError('paired cases incomplete')
    if (original['build']['conversion'],baseline['build']['conversion'],vector['build']['conversion'])!=('cast','compare_select','compare_select'):
        raise ValueError('conversion identities differ')
    if (original['build']['top1'],baseline['build']['top1'],vector['build']['top1'])!=('scalar','scalar','vector'):
        raise ValueError('Top1 identities differ')
    for field in ('capture_sha256','native_manifest_sha256'):
        if len({data[field] for data in (original,baseline,vector)})!=1: raise ValueError('paired golden identity differs')
    out=[]
    for name in CASES:
        rows=[next(c for c in data['cases'] if c['name']==name) for data in (original,baseline,vector)]
        identity=lambda c:(c['inputs']['hidden']['sha256'],c['inputs']['weight']['sha256'])
        if len({identity(c) for c in rows})!=1: raise ValueError('paired inputs differ')
        for field in ('base_k','single_k','partitions','matmul_ub_bytes','user_ub_bytes','user_workspace_bytes'):
            if len({row['launches']['fused'][field] for row in rows})!=1: raise ValueError('paired tiling/workspace changed')
        times={}
        for label,row in zip(('original_cast','baseline','vector'),rows):
            for mode in ('native-logits','native-top1','fused'):
                e=row['executions'][mode]; validate_execution(e,mode,row['m'],e['device_id'],5,30)
                times[f'{label}/{mode}']=e['timing']['execute_sync']
        out.append(dict(name=name,timing=times,
            baseline_to_vector_median=times['baseline/fused']['median_ms']/times['vector/fused']['median_ms'],
            native_to_vector_median=times['vector/native-top1']['median_ms']/times['vector/fused']['median_ms']))
    return out


def run(args):
    root=args.output_dir.resolve(); root.mkdir(parents=True,exist_ok=False)
    # Existing immutable capture/OM only. Never fall back to capture/export.
    os.environ.update(HEAD_CAPTURE=str(args.capture),HEAD_NATIVE_MANIFEST=str(args.native_manifest),DEVICE_ID=str(args.device_id),MODEL_PYTHON=sys.executable)
    report=dict(status='RUNNING',full_41_case_acceptance='NOT_RUN',performance='NOT_RUN',candidate_adoption='NOT_DECIDED',
                full_draft_validation='NOT_RUN',profiling={},full_gates={},builds={},evidence={})
    try:
        load_capture(args.capture); validate_native(args.native_manifest,sha256(args.capture))
        report['environment']=environment(os.environ.get('CANN_ROOT',os.environ.get('ASCEND_HOME_PATH','/usr/local/Ascend/ascend-toolkit/latest')),args.device_id)
        write_json(root/'suite.json',report)
        if report['environment']['status']!='PASS': raise ValueError(report['environment']['errors'])
        states={}; profile_results={}
        states['vector']=build(root,'vector',conversion='compare_select',top1='vector')
        report['builds']['vector']=str(root/'vector-build.json')
        report['evidence']['top1_probe']=str(root/'top1-probe/suite.json')
        write_json(root/'suite.json',report)
        call([sys.executable,str(HERE/'test/vector_top1_probe.py'),'--runner',states['vector']['top1_runner'],
              '--output-dir',str(root/'top1-probe'),'--device-id',str(args.device_id)],root/'top1-probe.log',states['vector']['environment'])
        primitive=json.loads((root/'top1-probe/suite.json').read_text())
        if primitive['status']!='PASS' or primitive['rows']!=2191: raise ValueError('complete primitive gate required')
        if args.primitive_only:
            report.update(status='PRIMITIVE_PASS',scope='2191-row primitive only; paired/full correctness NOT_RUN')
            write_json(root/'suite.json',report); print(root/'suite.json',flush=True); return 0
        for label,stage in (('matmul','matmul'),('convert','convert')):
            states[label]=build(root,label,profile=stage)
            out,data=observers(args,root,label,states[label]); profile_results[label]=data; report['evidence'][label]=str(out/'suite.json')
            report['builds'][label]=str(root/f'{label}-build.json')
            report['profiling'][label]='NOT_RUN' if args.skip_msprof else profile(args,root,label,states[label],out/'real-c16-m15')
            write_json(root/'suite.json',report)
        for label,conv,top in (('original','cast','scalar'),('baseline','compare_select','scalar')):
            states[label]=build(root,label,conversion=conv,top1=top)
            report['builds'][label]=str(root/f'{label}-build.json')
        # Representative paired gates before full acceptance.
        results={}
        for label in ('original','baseline','vector'):
            out,data=strict(args,root,label+'-paired',states[label],subset=True); results[label]=data
            report['evidence'][label+'_paired']=str(out/'suite.json')
        report['paired']=paired_report(results['original'],results['baseline'],results['vector'])
        m15=[p for p in report['paired'] if p['name'].endswith('m15')]
        report['m15_faster_than_baseline']=all(p['baseline_to_vector_median']>1 for p in m15)
        report['m15_at_or_below_native']=all(p['native_to_vector_median']>=1 for p in m15)
        report['diagnostic_deltas']=[]
        for pair in report['paired']:
            measured={label:next(c['execution']['timing']['execute_sync'] for c in data['cases'] if c['name']==pair['name'])
                      for label,data in profile_results.items()}
            report['diagnostic_deltas'].append(dict(name=pair['name'],observers=measured,
                convert_minus_matmul_median_ms=measured['convert']['median_ms']-measured['matmul']['median_ms'],
                baseline_minus_convert_median_ms=pair['timing']['baseline/fused']['median_ms']-measured['convert']['median_ms'],
                scope='diagnostic differences include observer/merge overhead; not additive hardware pipe times'))
        report['performance']='MEASURED'; write_json(root/'suite.json',report)
        for label in ('baseline','vector'):
            report['full_gates'][label]='RUNNING'; write_json(root/'suite.json',report)
            out,data=strict(args,root,label+'-full',states[label]); report['evidence'][label+'_full']=str(out/'suite.json')
            report['full_gates'][label]='PASS'; write_json(root/'suite.json',report)
            report['profiling'][label]='NOT_RUN' if args.skip_msprof else profile(args,root,label,states[label],out/'real-c16-m15')
        report.update(status='PASS',full_41_case_acceptance='PASS',performance='MEASURED')
    except Exception as e:
        report.update(status='FAIL',error=f'{type(e).__name__}: {e}')
        for key,value in report['full_gates'].items():
            if value=='RUNNING': report['full_gates'][key]='FAIL'
        if 'FAIL' in report['full_gates'].values(): report['full_41_case_acceptance']='FAIL'
    write_json(root/'suite.json',report)
    print(f"{report['status']}: {root/'suite.json'}; adoption remains NOT_DECIDED",flush=True)
    return 0 if report['status']=='PASS' else 1

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('capture','native-manifest','output-dir'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device-id',type=int,default=0); p.add_argument('--timeout',type=int,default=1800)
    p.add_argument('--skip-msprof',action='store_true')
    p.add_argument('--primitive-only',action='store_true',help='stop after all 2191 diagnostic rows; full correctness NOT_RUN')
    a=p.parse_args(); a.capture=a.capture.resolve(); a.native_manifest=a.native_manifest.resolve(); raise SystemExit(run(a))
