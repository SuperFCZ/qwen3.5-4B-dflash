#!/usr/bin/env python3
"""Same effective-weight native FP16 vs accepted W8 Off/Raw, Gate only."""
import argparse
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
from fp16_gate_common import compare,HERE,SHAPE,NAME,MODES,ORDERS,OUTPUTS,identity,verify,write_json,record,generate_weights,check_execution,numeric,stats,pipe_report
from fp16_gate_evidence import accepted,FROZEN
from gate_copy_sdk import inspect as inspect_sdk
from run_gate_copy import launch_plan,copy_evidence


def command(argv,log,env,timeout=1800):
    write_json(log.with_suffix('.command.json'),dict(argv=list(map(str,argv)),environment={k:env.get(k) for k in ('CANN_ROOT','ASCEND_OPP_PATH','ASCEND_CUSTOM_OPP_PATH','LD_LIBRARY_PATH')},timeout=timeout))
    with log.open('w') as stream:subprocess.run(list(map(str,argv)),env=env,cwd=log.parent,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=timeout)


def child_environment(cann,vendor=None):
    env=dict(os.environ,CANN_ROOT=str(cann));env.pop('LD_PRELOAD',None);env.pop('ASCEND_CUSTOM_OPP_PATH',None)
    base=[p for p in env.get('LD_LIBRARY_PATH','').split(':') if p and '/vendors/' not in p]
    paths=[str(cann/'lib64'),str(cann/'aarch64-linux/lib64')]+base
    if vendor:env['ASCEND_CUSTOM_OPP_PATH']=str(vendor);paths.insert(0,str(vendor/'op_api/lib'))
    env['LD_LIBRARY_PATH']=':'.join(paths)
    return env


def build_native(args,root,env):
    if platform.machine() not in ('aarch64','arm64') or sys.platform!='linux':raise ValueError('real Linux aarch64 CANN receiver required')
    sdk=inspect_sdk(args.cann_root,'prefetch');write_json(root/'sdk.json',sdk)
    public=args.cann_root/'include/aclnnop/aclnn_mm.h'
    if public.is_file():header=public.resolve()
    else:
        headers={p.resolve() for p in args.cann_root.rglob('aclnn_mm.h') if p.is_file()}
        if len(headers)!=1:raise ValueError('public aclnn_mm.h missing and fallback discovery ambiguous: '+str(sorted(map(str,headers))))
        header=next(iter(headers))
    text=header.read_text()
    if any(s not in text for s in ('aclnnMmGetWorkspaceSize','aclnnMm','int8_t')):raise ValueError('unsupported native Mm header')
    command(['cmake','-S',HERE,'-B',root/'build','-DCANN_ROOT='+str(args.cann_root),'-DNATIVE_MM_HEADER='+str(header),'-DCOMMON_INCLUDE='+str(HERE.parent/'test')],root/'cmake.log',env)
    command(['cmake','--build',root/'build','--parallel'],root/'build.log',env)
    runner=root/'build/fp16_direct_mm';command(['ldd',runner],root/'ldd.log',env)
    libs={}
    for line in (root/'build/CMakeCache.txt').read_text().splitlines():
        for name in ('ascendcl','opapi','nnopbase'):
            if line.startswith('LIB_'+name+':FILEPATH='):libs[name]=identity(line.split('=',1)[1])
    if set(libs)!={'ascendcl','opapi','nnopbase'}:raise ValueError('native linked library identities missing')
    return runner,dict(runner=identity(runner),native_header=identity(header),native_libraries=libs,sdk=record(root/'sdk.json',root),
        sources=[identity(HERE/n) for n in ('native_mm.cpp','CMakeLists.txt')]+[identity(HERE.parent/'test/runner_common.h')],
        api='aclnnMm',cube_math_type=0,weight_binding='runtime resident ND_KN FP16',
        internal_accumulation_and_tiling='CANN native; not asserted identical to custom N64/K256',simulator='NOT_RUN')


def cpu_sample(files,weights,actual):
    m,k,n=SHAPE;rows=[0,7,15];cols=[0,63,64,9727,9728,n-1]
    x=np.memmap(files['x.bin']['path'],mode='r',dtype='<f2',shape=(m,k))
    w=np.memmap(weights['files']['w_fp16_nk.bin']['path'],mode='r',dtype='<f2',shape=(n,k))
    expected=(x[rows].astype(np.float64)@w[cols].astype(np.float64).T).astype('<f2')
    y=np.frombuffer(actual,dtype='<f2').reshape(m,n)[np.ix_(rows,cols)]
    return dict(status='CPU_DIAGNOSTIC_ONLY_NOT_GOLDEN',rows=rows,columns=cols,
        arithmetic='FP64 full K from the identical pre-rounded FP16 weights',comparison=compare(expected.tobytes(),y.tobytes(),SimpleNamespace(m=len(rows),n=len(cols))))


def invoke(mode,folder,source,weights,runner,env,device,timed,profiler=None):
    folder.mkdir(parents=True,exist_ok=False)
    case=folder/'case';case.mkdir()
    files={'x.bin':source['files']['x.bin']}
    if mode=='fp16':files['w_fp16_kn.bin']=weights['files']['w_fp16_kn.bin']
    else:files.update({n:source['files'][n] for n in ('w_nz.bin','s_gn.bin')})
    for name,item in files.items():(case/name).symlink_to(verify(item))
    argv=[str(runner),str(device),str(case)]
    if mode!='fp16':argv += list(map(str,SHAPE))
    if timed:argv += ['5','30','--continuous']
    write_json(case/'command.json',argv)
    if profiler:
        launcher=folder/'replay.sh';launcher.write_text('#!/usr/bin/env bash\nset -eo pipefail\nexec '+shlex.join(argv)+' > '+shlex.quote(str(case/'runner.log'))+' 2>&1\n')
        prof=[str(profiler),'--ai-core=on','--aic-mode=task-based','--aic-metrics=PipeUtilization',f'--output={folder / "profiler"}','--application='+shlex.join(['bash',str(launcher)])]
        command(prof,folder/'profile.log',env,3600)
    else:command(argv,case/'runner.log',env)
    # The profiler launcher redirects application output explicitly, so the
    # original W8 Host launch/config audit runs for profiled execution too.
    ex=json.loads((case/'execution.json').read_text());check_execution(ex,mode,device,timed)
    names=OUTPUTS if timed else OUTPUTS[:2]
    outputs=[(case/name).read_bytes() for name in names]
    if any(x!=outputs[0] for x in outputs):raise ValueError('repeat/timed-tail/postcheck drift')
    diff=numeric(verify(source['golden']).read_bytes(),outputs[0],mode)
    launch=None
    if mode!='fp16':
        cfg=source['modes'][mode]['config'];launch=launch_plan(case/'runner.log',cfg,ex['workspace_bytes'],'prefetch')
        copy_evidence(case/'runner.log','loop')
    for item in files.values():verify(item)
    result=dict(mode=mode,profiled=bool(profiler),formal_latency=timed and not profiler,execution=ex,
        execution_file=identity(case/'execution.json'),outputs={name:identity(case/name) for name in names},
        golden_comparison=diff,launch=launch,input_hashes={name:item['sha256'] for name,item in files.items()})
    if profiler:
        result['pipe_counters']=pipe_report(folder/'profiler')
        if not result['pipe_counters']['files']:raise ValueError('msprof completed without CSV artifacts')
    if timed and not profiler:result['timing']=stats(ex['timing']['execute_sync']['samples_ms'])
    write_json(folder/'result.json',result)
    return result,outputs[0]



def attach_w8_instructions(path,source):
    sys.path.insert(0,str(HERE.parent/'simulator'))
    from bottleneck import Evidence,load_mode
    evidence=Evidence(path.resolve().parent);report=json.loads(evidence.read(path).read_text())
    if report['status']!='BOUNDED_SIMULATOR_DIAGNOSTIC_ONLY' or report['phase']!='complete' or report['paired_input_identity']!='MATCHED':raise ValueError('incomplete historical bounded evidence')
    modes={}
    for mode in ('off','raw'):
        spec=load_mode(evidence,report,mode)
        if spec['config']['source_sha256']['op_kernel/d_flash_group_quant_linear.cpp']!=source['modes'][mode]['config']['source_sha256']['op_kernel/d_flash_group_quant_linear.cpp']:raise ValueError('bounded/full hardware Kernel origin differs')
        for name in ('x.bin','w_nz.bin','s_gn.bin'):
            if spec['inputs'][name]!=source['files'][name]['sha256']:raise ValueError('bounded instruction inputs differ')
        if spec['inputs']['expected.bin']!=source['golden']['sha256']:raise ValueError('bounded Golden differs')
        modes[mode]={name:identity(spec[name]) for name in ('trace','instr','code','binary','source')}
    evidence.verify()
    return dict(status='REUSED_HISTORICAL_BOUNDED_EVIDENCE',report=identity(path),n_tile=report['n_tile'],modes=modes,
                scope='One N64 tile, full K2560; not full Gate instruction counts; no FP16 instruction comparison inferred'),evidence.files


def experiment(args):
    root=args.output_dir.absolute()
    forbidden=[HERE.parent.resolve(),args.prefetch_evidence.resolve().parent,args.a51_evidence.resolve().parent]
    if root.resolve()!=root or any(root==p or p in root.parents for p in forbidden):raise ValueError('use a new standalone output outside old evidence/repository')
    root.mkdir(parents=True,exist_ok=False);root.chmod(0o700)
    report=dict(status='RUNNING',phase='accepted_evidence',shape=SHAPE,projection=NAME,
        prechecks={},rounds=[],profiles={},performance='NOT_RUN',w8_native_bitwise='NOT_RUN',
        fp16_native_golden_bitwise='NOT_RUN',full_draft_token_kv='NOT_RUN',end_to_end='NOT_RUN',
        simulator=dict(fp16='NOT_RUN',w8='NOT_RUN',reason='Native aclnnMm has no validated single-N-tile bounded execution interface. W8 Simulator/Runner is not reused as FP16 support.'),
        attribution=dict(dequant_only_cost='UNRESOLVED',actual_weight_traffic_and_bandwidth='UNRESOLVED',cube_and_sync_instruction_cost='UNRESOLVED',
            reason='Native FP16 and custom W8 differ in layout handling, tiling, Cube scheduling and dequantization. Pipe counters may overlap.'))
    summary=root/'comparison.json';write_json(summary,report);watched={}
    try:
        source=accepted(args.a51_evidence,args.prefetch_evidence);watched.update(source['watched'])
        if getattr(args,'w8_bounded_report',None):
            old,files=attach_w8_instructions(args.w8_bounded_report,source);watched.update(files)
            report['w8_instruction_evidence']=old;write_json(root/'w8-bounded-evidence.json',old)
        for path in HERE.glob('*'):
            if path.is_file():item=identity(path);watched[item['path']]=item
        report['source_identity']=source['identity'];report['w8_builds']=source['modes'];report['phase']='offline_weights';write_json(summary,report)
        weights=generate_weights(source['files'],root/'weights');report['weights']=weights
        for item in weights['files'].values():watched[item['path']]=item
        if args.prepare_only:report.update(status='PREPARED_NOT_RUN',phase='complete')
        else:
            env=child_environment(args.cann_root)
            report['phase']='native_build';write_json(summary,report)
            fp16_runner,built=build_native(args,root,env);report['native_build']=built
            for item in [built['runner'],built['native_header'],*built['sources'],*built['native_libraries'].values()]:watched[item['path']]=item
            profiler=shutil.which(args.msprof)
            if not profiler:raise ValueError('msprof unavailable; no fabricated PipeUtilization')
            item=identity(profiler);watched[item['path']]=item;report['profiler']=item
            command(['uname','-a'],root/'host.log',env,30)
            if shutil.which('npu-smi'):command(['npu-smi','info'],root/'npu-smi.log',env,60)
            report['device']=dict(id=args.device_id,required_soc='Ascend310P3',same_process_configuration=True)
            runners=dict(fp16=fp16_runner,**{mode:Path(source['modes'][mode]['runner']) for mode in ('off','raw')})
            envs=dict(fp16=env,**{mode:child_environment(args.cann_root,Path(source['modes'][mode]['opp'])/'vendors/customize') for mode in ('off','raw')})
            first={};report['phase']='all_untimed_prechecks';write_json(summary,report)
            for mode in MODES:
                result,first[mode]=invoke(mode,root/'prechecks'/mode,source,weights,runners[mode],envs[mode],args.device_id,False)
                result['cpu_sample']=cpu_sample(source['files'],weights,first[mode]);report['prechecks'][mode]=result;write_json(summary,report)
            report.update(w8_native_bitwise='PASS',fp16_native_golden_bitwise=report['prechecks']['fp16']['golden_comparison']['status'],phase='unprofiled_timing');write_json(summary,report)
            for index,order in enumerate(ORDERS):
                record_round=dict(index=index,order=order,results={});report['rounds'].append(record_round)
                for mode in order:
                    result,output=invoke(mode,root/'timing'/str(index)/mode,source,weights,runners[mode],envs[mode],args.device_id,True)
                    if output!=first[mode]:raise ValueError('cross-process/permutation output drift')
                    if result.get('launch')!=report['prechecks'][mode].get('launch'):raise ValueError('timed Host launch plan changed')
                    record_round['results'][mode]=result;write_json(summary,report)
            report['timing_summary']={mode:stats([s for r in report['rounds'] for s in r['results'][mode]['timing']['samples_ms']]) for mode in MODES}
            report['latency_ratios']={mode:report['timing_summary'][mode]['median_ms']/report['timing_summary']['fp16']['median_ms'] for mode in ('off','raw')}
            report['latency_interpretation']='Observed native-runtime-ND FP16 versus custom-runtime-NZ W8 implementations, 6 crossed sessions x 30 samples each; not isolated dequantization cost or Draft speedup. Ratios do not assert FP16 bitwise parity.'
            report.update(performance='MEASURED_UNPROFILED',phase='separate_msprof');write_json(summary,report)
            for mode in MODES:
                result,output=invoke(mode,root/'profiling'/mode,source,weights,runners[mode],envs[mode],args.device_id,True,profiler)
                if output!=first[mode]:raise ValueError('profiled output differs from unprofiled precheck')
                if result.get('launch')!=report['prechecks'][mode].get('launch'):raise ValueError('profiled Host launch plan changed')
                report['profiles'][mode]=result;write_json(summary,report)
            report.update(status='EXPERIMENT_COMPLETED_DIAGNOSTIC_ONLY',phase='complete')
    except Exception as error:report.update(status='BLOCKED',error=f'{type(error).__name__}: {error}')
    finally:
        try:
            for item in watched.values():verify(item)
        except Exception as error:report.update(status='BLOCKED',integrity_error=str(error))
        report['watched']=watched;write_json(summary,report)
        lines=['# Same effective-weight Gate FP16 / W8 experiment','', 'Status: '+report['status']+'; phase: '+report['phase'],
               'M16 K2560 N19456; layer-0 only; no historical FP16 checkpoint.',
               'W8 Native bitwise: '+report['w8_native_bitwise']+'; FP16 vs W8 Golden: '+report['fp16_native_golden_bitwise'],'',
               '| Implementation | Unprofiled median ms | p95 ms | samples |','| --- | ---: | ---: | ---: |']
        for mode,t in report.get('timing_summary',{}).items():lines.append(f"| {mode} | {t['median_ms']:.6f} | {t['p95_ms']:.6f} | {len(t['samples_ms'])} |")
        lines += ['', 'Each session: continuous-v1 warmup=5 / reps=30. All six permutations; model weight preparation, allocation, transfers, correctness checks and profiler are outside formal execute+sync timing.',
            'Native FP16 ND may perform additional internal layout work. Its bitwise differences require review; no tolerance replaces W8 validation.',
            'Profiler Task Duration and raw MTE2/VECTOR/SCALAR/CUBE columns are separate. No sum of overlapping pipe counters.',
            'Dequant-only overhead, measured bandwidth, instruction/synchronization causality: UNRESOLVED.',
            'Native FP16 bounded Simulator: NOT_RUN / no validated interface; no W8 runner substitution. Full Gate Simulator and Draft/Decode: NOT_RUN.',
            report.get('error','')]
        (root/'comparison.md').write_text('\n'.join(lines)+'\n')
    print(report['status']+': '+str(summary));return int(report['status']=='BLOCKED')


def parse():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--a51-evidence',type=Path,default=Path('/home/w00949577/qwen35-dflash-run/a51-fallback-20261012-063342/suite.json'))
    p.add_argument('--prefetch-evidence',type=Path,default=Path('/home/w00949577/qwen35-dflash-run/gate-raw-prefetch-reverse-20261020-025853/suite.json'))
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--cann-root',type=Path,default=Path(os.environ.get('CANN_ROOT','/usr/local/Ascend/cann-9.0.0')))
    p.add_argument('--w8-bounded-report',type=Path,help='Optional existing completed bounded report; read-only input/source identity validation')
    p.add_argument('--device-id',type=int,default=0);p.add_argument('--msprof',default='msprof');p.add_argument('--prepare-only',action='store_true')
    args=p.parse_args()
    if args.device_id<0:p.error('negative device')
    args.cann_root=args.cann_root.resolve();return args

if __name__=='__main__':raise SystemExit(experiment(parse()))
