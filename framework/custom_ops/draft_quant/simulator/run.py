#!/usr/bin/env python3
"""Isolated ACLNN/camodel preparation and simulation; never hardware timing."""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE/'test'))
from a2_common import checked_file,load_bundle,record,sha256,write_json,validate_layout
from a3_launch import load_build_config,configure_build,launch_evidence
from opp_preflight import check_install,check_root
from run_gate_copy_pair import source_evidence
from gate_copy_sdk import inspect as inspect_sdk
from simulator_prepare import runner_sources,debug_cmake,validate_help,profile_command,smoke_inputs,elf_machine,artifacts,focus_lines

FROZEN=('op_kernel/d_flash_group_quant_linear.cpp','op_host/d_flash_group_quant_linear.cpp',
        'op_host/d_flash_group_quant_linear_contract.h','op_host/d_flash_group_quant_linear_tiling.h',
        'op_host/d_flash_group_quant_linear_build_config.h','op_kernel/d_flash_group_quant_linear_build_config.h')
BORROWED_RUNNER=('test/main.cpp','test/runner_common.h')


def ident(p):
    p=Path(p).resolve();return dict(path=str(p),bytes=p.stat().st_size,sha256=sha256(p))


def command(argv,log,env,cwd,timeout=1800):
    write_json(log.with_suffix('.command.json'),dict(argv=list(map(str,argv)),cwd=str(cwd),
               environment={k:env.get(k) for k in ('CANN_ROOT','ASCEND_CUSTOM_OPP_PATH','LD_LIBRARY_PATH','LD_PRELOAD','DFLASH_SIM_RUNTIME','DFLASH_SIM_DRIVER','DFLASH_PROFILER_ROOT')}))
    with log.open('w') as out:
        proc=subprocess.Popen(list(map(str,argv)),cwd=cwd,env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            proc.wait();raise RuntimeError('Simulator/build timeout; terminated this command process group, no fallback')
        if code:raise subprocess.CalledProcessError(code,list(map(str,argv)))
    return log.read_text(errors='replace')


def inspect_elf(path,log,env,cwd,host=False,debug=False):
    if host and elf_machine(path)!=183:raise ValueError('ELF is not AArch64: '+str(path))
    text=command(['readelf','-hSWd',path],log,env,cwd)
    if debug and ('.debug_info' not in text or '.debug_line' not in text):raise ValueError('Kernel lacks -g DWARF line info: '+str(path))
    return dict(**ident(path),readelf=record(log,cwd),debug_info='.debug_info' in text,debug_line='.debug_line' in text)


def accepted(args,root,env):
    pair=json.loads(args.prefetch_evidence.read_text())
    if pair.get('status')!='PASS' or pair.get('experiment')!='prefetch' or set(pair.get('runs',{}))!={'off','raw'}:raise ValueError('need accepted Off/Raw pair')
    comparison=checked_file(args.prefetch_evidence.parent,pair['comparison'])
    result=json.loads(comparison.read_text())
    if result.get('status')!='PASS' or not any(c.get('name')=='layer-0-gate_up' and c.get('status')=='PASS' for c in result.get('cases',[])):raise ValueError('accepted Gate bitwise evidence missing')
    source=source_evidence(args.a51_evidence)
    configs={};evidence={}
    for mode in ('off','raw'):
        item=pair['runs'][mode];path=Path(item['path'])
        if sha256(path)!=item['sha256']:raise ValueError('accepted run changed')
        run=json.loads(path.read_text());cfg=load_build_config(Path(run['build_config_path']))
        if run.get('status')!='PASS' or cfg!=run['build'] or cfg.get('gate_prefetch_mode')!=mode or cfg.get('gate_copy_mode')!='loop' or cfg['k_tile']!=256 or cfg['scale_mode']!='broadcast':raise ValueError('accepted source configuration differs')
        if sha256(run['build_config_path'])!=run['build_config_sha256']:raise ValueError('build config changed')
        for name in (*FROZEN,*BORROWED_RUNNER):
            if cfg['source_sha256'].get(name)!=sha256(HERE/name):raise ValueError('frozen operator source changed: '+name)
        timing=checked_file(path.parent,run['timing']);timed=json.loads(timing.read_text())
        if timed.get('status')!='PASS' or timed['bundle_sha256']!=sha256(source['bundle']) or timed['native_om_manifest_sha256']!=sha256(source['native']):raise ValueError('different accepted Capture/Native identity')
        build_root=Path(cfg['header']).parents[2]
        install=check_install(build_root/'opp')
        audit=[]
        for i,k in enumerate(install['kernels']):
            audit.append(inspect_elf(Path(k['binary']),root/f'{mode}-accepted-{i}-elf.log',env,root))
        cmakes={}
        for rel in ('CMakeLists.txt','op_kernel/CMakeLists.txt','op_host/CMakeLists.txt'):
            p=build_root/'CustomOp'/rel
            if p.is_file():cmakes[rel]=dict(**ident(p),text=p.read_text())
        old_runner=Path(timed['runners']['custom']['path'])
        if sha256(old_runner)!=timed['runners']['custom']['sha256']:raise ValueError('accepted runner changed')
        runner_audit=inspect_elf(old_runner,root/f'{mode}-accepted-runner.log',env,root,host=True)
        configs[mode]=cfg;evidence[mode]=dict(run=ident(path),build_config=ident(run['build_config_path']),
            headers=[ident(cfg['header']),ident(cfg['kernel_header'])],timing=ident(timing),runner=runner_audit,
            build=cfg,installed=install,installed_files=[ident(p) for p in sorted((build_root/'opp').rglob('*')) if p.is_file()],kernels=audit,generated_cmake=cmakes)
    for key in ('core_limit','pipeline_mode','dequant_mode','kv_m80_mode','scale_mode','k_tile','source_sha256'):
        if configs['off'][key]!=configs['raw'][key]:raise ValueError('accepted pair differs beyond Gate raw prefetch: '+key)
    return source,configs,dict(pair=ident(args.prefetch_evidence),comparison=ident(comparison),modes=evidence)


def preflight(args,root):
    if platform.machine() not in ('aarch64','arm64') or sys.platform!='linux':raise ValueError('requires Linux aarch64 receiver; local CPU tests are not simulator execution')
    sim=args.sim_lib.resolve();cann=args.cann_root.resolve();tool=args.msopprof.resolve()
    if 'dav_2002' not in sim.parts or not sim.is_dir():raise ValueError('Ascend310P3 library must resolve to dav_2002/lib')
    env=dict(os.environ,CANN_ROOT=str(cann),ASCEND_HOME_PATH=str(cann),ASCEND_INSTALL_PATH=str(cann))
    # No inherited preload/old custom OPP in configure/build or tool preflight.
    env.pop('LD_PRELOAD',None);env.pop('ASCEND_CUSTOM_OPP_PATH',None)
    for binary in ('readelf','cmake','msopgen'):
        if not shutil.which(binary,path=env.get('PATH')):raise ValueError('required build tool missing: '+binary)
    command(['cmake','--version'],root/'cmake-version.log',env,root,60)
    help_text=command([tool,'simulator','--help'],root/'simulator-help.log',env,root,60);validate_help(help_text)
    libs={name:sim/name for name in ('libruntime_camodel.so','libnpu_drv_camodel.so')}
    lib_records={name:inspect_elf(p,root/(name+'-elf.log'),env,root,host=True) for name,p in libs.items()}
    sdk=inspect_sdk(cann,'prefetch');write_json(root/'sdk.json',sdk)
    source,configs,old=accepted(args,root,env)
    return env,source,configs,dict(status='PREPARED_NOT_EXECUTED',target='Ascend310P3',arch='dav_2002',
        tool=ident(tool),simulator_directory=str(sim),libraries=lib_records,sdk=record(root/'sdk.json',root),accepted=old,
        profiler_version='26.0.0 user-reported; simulator --help bound by hash',help=ident(root/'simulator-help.log'))


def build(mode,cfg,root,args,env,kernel_adapter=None,command_runner=None):
    # Optional adapters are confined to this new Simulator project. The normal
    # entry has neither adapter and still builds the byte-identical algorithm.
    build_command=command_runner or command
    directory=root/mode;directory.mkdir();project=directory/'CustomOp';opp=directory/'opp';check_root(directory)
    build_command(['msopgen','gen','-i',HERE/'DFlashGroupQuantLinear.json','-c','ai_core-Ascend310P3','-lan','cpp','-out',project],directory/'msopgen.log',env,directory)
    for name in ('d_flash_group_quant_linear.cpp','d_flash_group_quant_linear_contract.h','d_flash_group_quant_linear_tiling.h'):
        shutil.copyfile(HERE/'op_host'/name,project/'op_host'/name)
    shutil.copyfile(HERE/'op_kernel/d_flash_group_quant_linear.cpp',project/'op_kernel/d_flash_group_quant_linear.cpp')
    configure_build(project/'op_host/d_flash_group_quant_linear_build_config.h',directory/'build-config.json',cfg['core_limit'],cfg['dequant_mode'],
        project/'op_kernel/d_flash_group_quant_linear_build_config.h',cfg['pipeline_mode'],cfg['kv_m80_mode'],cfg['scale_mode'],cfg['k_tile'],'loop',mode)
    for name in FROZEN:
        # The kernel-facing config is generated intentionally; algorithm source is copied byte-for-byte.
        if name.endswith('build_config.h'):continue
        if sha256(project/name)!=sha256(HERE/name):raise ValueError('isolated source copy differs')
    adaptation=None
    if kernel_adapter is not None:
        adaptation=kernel_adapter(project/'op_kernel/d_flash_group_quant_linear.cpp')
        write_json(directory/'bounded-source.json',adaptation)
        (directory/'bounded-source.diff').write_text(adaptation['diff'])
    debug_cmake(project/'op_kernel/CMakeLists.txt')
    write_json(directory/'source-map.json',dict(kernel=ident(project/'op_kernel/d_flash_group_quant_linear.cpp'),
               focus_lines=focus_lines(project/'op_kernel/d_flash_group_quant_linear.cpp'),tile_mapping='UNRESOLVED_TILE_MAPPING'))
    build_command(['bash','build.sh'],directory/'build.log',env,project,3600)
    packages=list((project/'build_out').glob('custom_opp_*.run'))
    if len(packages)!=1:raise ValueError('not exactly one generated OPP package')
    build_command(['bash',packages[0],'--install-path='+str(opp)],directory/'install.log',env,directory,600)
    installed=check_install(opp);write_json(directory/'opp-preflight.json',installed)
    binary_audit=[]
    for i,k in enumerate(installed['kernels']):
        binary_audit.append(inspect_elf(Path(k['binary']),directory/f'debug-kernel-{i}.log',env,directory,debug=True))
    generated=runner_sources(HERE,directory/'runner-source')
    build_command(['cmake','-S',Path(__file__).parent,'-B',directory/'runner-build','-DCANN_ROOT='+str(args.cann_root.resolve()),
             '-DSIM_LIB='+str(args.sim_lib.resolve()),'-DOPP_ROOT='+str(opp),'-DRUNNER_SOURCE='+str(generated)],directory/'cmake.log',env,directory)
    build_command(['cmake','--build',directory/'runner-build','--parallel'],directory/'runner-build.log',env,directory)
    runner=directory/'runner-build/dflash_gate_simulator'
    inspect_elf(runner,directory/'runner-elf.log',env,directory,host=True)
    vendor=opp/'vendors/customize';sim=args.sim_lib.resolve()
    app_env=dict(env,ASCEND_CUSTOM_OPP_PATH=str(vendor),LD_LIBRARY_PATH=':'.join([str(sim),str(vendor/'op_api/lib'),str(args.cann_root.resolve()/'lib64'),str(args.cann_root.resolve()/'aarch64-linux/lib64'),env.get('LD_LIBRARY_PATH','')]),
                 LD_PRELOAD=str(sim/'libruntime_camodel.so')+':'+str(sim/'libnpu_drv_camodel.so'),
                 DFLASH_PROFILER_ROOT=str(args.msopprof.resolve().parent.parent),DFLASH_SIM_RUNTIME=str(sim/'libruntime_camodel.so'),DFLASH_SIM_DRIVER=str(sim/'libnpu_drv_camodel.so'))
    build_command(['ldd',runner],directory/'ldd.log',app_env,directory)
    build_command([runner,'--link-check',directory],directory/'link-check.log',app_env,directory)
    audit=dict(status='BUILT_NOT_SIMULATED',package=ident(packages[0]),kernels=binary_audit,runner=ident(runner),
               config=ident(directory/'build-config.json'),source_map=record(directory/'source-map.json',directory),
               simulator_source_adaptation=adaptation,
               runner_sources=[ident(generated),ident(generated.parent/'runner_common.h'),ident(Path(__file__).parent/'runtime_guard.h'),ident(Path(__file__).parent/'CMakeLists.txt')])
    write_json(directory/'build-report.json',audit)
    return runner,app_env,audit


def simulate(mode,stage,runner,env,root,args,source):
    directory=root/mode/stage;directory.mkdir();case=directory/'input'
    sources={}
    if stage=='smoke':
        smoke_inputs(case);shape=(16,256,64);expected=case/'expected.bin'
    else:
        case.mkdir();shape=(16,2560,19456)
        manifest=Path(source['bundle']);bundle=load_bundle(manifest,'a4')
        projection=next(c for c in bundle['cases'] if c['name']=='layer-0-gate_up');validate_layout(manifest.parent,projection,'a4')
        sources[str(manifest)]=ident(manifest)
        for name in ('x.bin','w_nz.bin','s_gn.bin'):
            src=checked_file(manifest.parent,projection['files'][name]);sources[str(src)]=ident(src)
            shutil.copyfile(src,case/name)
            if sha256(case/name)!=projection['files'][name]['sha256']:raise ValueError('Gate input copy differs')
        golden=Path(source['golden']);original=json.loads(golden.read_text());row=next(c for c in original['cases'] if c['name']=='layer-0-gate_up')
        sources[str(golden)]=ident(golden)
        expected=checked_file(golden.parent,row['outputs']['native_om'][0]);sources[str(expected)]=ident(expected)
        shutil.copyfile(expected,case/'expected.bin')
        if sha256(case/'expected.bin')!=sources[str(expected)]['sha256']:raise ValueError('Golden copy differs')
        expected=case/'expected.bin'
    config=load_build_config(root/mode/'build-config.json')
    inputs={p.name:ident(p) for p in case.iterdir() if p.is_file()}
    write_json(directory/'inputs.json',dict(shape=shape,files=inputs,original_sources=sources,origin='new toolchain basis fixture' if stage=='smoke' else 'bit-identical frozen C16 layer-0 bytes',raw_prefetch_selected=mode=='raw' and stage=='gate'))
    out=directory/'profile'
    command(profile_command(args.msopprof,runner,out,case,shape,args.timeout_minutes),directory/'simulator.log',env,directory,args.timeout_minutes*60+120)
    result=json.loads((case/'execution.json').read_text())
    if result.get('status')!='SIMULATOR_COMPLETED' or result.get('runtime')!='AscendCL camodel' or result.get('repetitions')!=1 or result.get('guards_intact') is not True or result.get('input_readonly') is not True or result.get('timing',{}).get('status')!='NOT_RUN':raise ValueError('simulator runner/guards/call count failed')
    if [result.get(k) for k in ('m','k','n')]!=list(shape):raise ValueError('simulator shape differs')
    launch=launch_evidence(case/'application.log',shape,config['core_limit'],result['workspace_bytes'],config['dequant_mode'],config['pipeline_mode'],5,config['kv_m80_mode'],config['scale_mode'],256,config['k_tile_policy'],mode)
    for item in inputs.values():
        if ident(item['path'])!=item:raise ValueError('simulator input/expected changed')
    for item in sources.values():
        if ident(item['path'])!=item:raise ValueError('original Capture/Golden changed')
    same=(case/'actual-0.bin').read_bytes()==expected.read_bytes()
    collected=artifacts(out,record)
    built=json.loads((root/mode/'build-report.json').read_text())
    expected_hashes={k['sha256'] for k in built['kernels']}
    profile_binaries=[ident(p) for p in out.rglob('*.o') if p.is_file()]
    binding=dict(status='MATCHED' if any(p['sha256'] in expected_hashes for p in profile_binaries) else 'UNVERIFIED',binaries=profile_binaries,expected_sha256=sorted(expected_hashes))
    report=dict(status='SIMULATOR_DIAGNOSTIC_ONLY',launch=launch,simulator_output_bitwise_match=same,artifacts=collected,profile_kernel_identity=binding,
                actual=ident(case/'actual-0.bin'),expected=ident(expected),npu_correctness='NOT_RUN',hardware_performance='NOT_RUN',
                full_draft_token_kv='NOT_RUN',repeat_test='NOT_RUN_SINGLE_CALL',core_selection='parse core 0; full launch and all tiles execute')
    write_json(directory/'result.json',report)
    if not same or collected['status']!='COLLECTED' or binding['status']!='MATCHED':raise ValueError('simulator output mismatch, artifact incomplete, or profiled Kernel binary identity unverified')
    return report


def run(args):
    root=args.output_dir.absolute()
    if root.resolve()!=root:raise ValueError('use a non-symlink simulator output directory')
    root.mkdir(parents=True,exist_ok=False);root.chmod(0o700)
    report=dict(status='PREPARING',stage=args.stage,phase='preflight',runs={},npu_correctness='NOT_RUN',hardware_performance='NOT_RUN',full_draft_token_kv='NOT_RUN')
    original={str(HERE/n):sha256(HERE/n) for n in (*FROZEN,*BORROWED_RUNNER)}
    summary=root/'simulator-report.json';write_json(summary,report)
    try:
        for p in (args.a51_evidence,args.prefetch_evidence):original[str(p)]=sha256(p)
        env,source,configs,prepared=preflight(args,root);report['preflight']=prepared
        def watch_records(value):
            if isinstance(value,dict):
                if 'path' in value and 'sha256' in value and Path(value['path']).is_absolute():original[value['path']]=value['sha256']
                for child in value.values():watch_records(child)
            elif isinstance(value,list):
                for child in value:watch_records(child)
        watch_records(prepared)
        # Accepted installed objects/metadata are read-only and rechecked at exit.
        for mode in prepared['accepted']['modes'].values():
            for item in mode['installed']['kernels']:
                original[item['binary']]=item['binary_sha256'];original[item['json']]=item['json_sha256']
        if args.stage=='prepare':report['status']='PREPARED_NOT_EXECUTED'
        else:
            modes=('off',) if args.stage=='smoke' else ('off','raw')
            for mode in modes:
                report['phase']='build-'+mode;write_json(summary,report)
                runner,app_env,built=build(mode,configs[mode],root,args,env)
                report['runs'][mode]=dict(build=built,smoke='NOT_RUN',gate='NOT_RUN')
                report['phase']='smoke-'+mode;write_json(summary,report)
                report['runs'][mode]['smoke']=simulate(mode,'smoke',runner,app_env,root,args,source)
                if args.stage=='gate':
                    report['phase']='gate-'+mode;write_json(summary,report)
                    report['runs'][mode]['gate']=simulate(mode,'gate',runner,app_env,root,args,source)
            report.update(status='SIMULATOR_DIAGNOSTIC_ONLY',phase='complete')
    except Exception as error:report.update(status='BLOCKED',error=f'{type(error).__name__}: {error}')
    finally:
        if any(not Path(p).is_file() or sha256(p)!=h for p,h in original.items()):report.update(status='BLOCKED',integrity_error='original source/evidence changed')
        write_json(summary,report)
        lines=['# Gate Simulator diagnostic','', 'Status: '+report['status']+'; phase: '+report['phase'],
               'NPU correctness / hardware performance / full Draft: NOT_RUN.',
               'Simulator cycles are not hardware wall-time. core-id=0 limits parsing, not executed tiles.',
               'Exact N/K tile attribution remains UNRESOLVED_TILE_MAPPING; source-map.json is a source-line focus list only.','']
        if report.get('error'):lines+=['Error: '+report['error'],'']
        for mode,r in report['runs'].items():
            for stage in ('smoke','gate'):
                value=r.get(stage,'NOT_RUN')
                lines += [mode+'/'+stage+': '+(value['status'] if isinstance(value,dict) else value)]
                if isinstance(value,dict):
                    lines += ['Artifacts: '+json.dumps(value['artifacts']['files'],ensure_ascii=False)]
        lines+=['','Raw logs, command/env records, ELF/CMake audits, SHA256 and output checks are in simulator-report.json and each phase directory.']
        (root/'simulator-analysis.md').write_text('\n'.join(lines)+'\n')
    print(report['status']+': '+str(summary)+'; NPU/hardware performance NOT_RUN',flush=True)
    return int(report['status']=='BLOCKED')


def parse(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--msopprof',type=Path,default=Path('/home/w00949577/tools/msopprof-26-isolated/tools/msopprof/bin/msopprof'))
    p.add_argument('--sim-lib',type=Path,default=Path('/usr/local/Ascend/cann-9.0.0/tools/simulator/Ascend310P3/lib'))
    p.add_argument('--cann-root',type=Path,default=Path(os.environ.get('CANN_ROOT','/usr/local/Ascend/cann-9.0.0')))
    p.add_argument('--a51-evidence',type=Path,default=Path('/home/w00949577/qwen35-dflash-run/a51-fallback-20261012-063342/suite.json'))
    p.add_argument('--prefetch-evidence',type=Path,default=Path('/home/w00949577/qwen35-dflash-run/gate-raw-prefetch-reverse-20261020-025853/suite.json'))
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--stage',choices=('prepare','smoke','gate'),default='smoke')
    p.add_argument('--timeout-minutes',type=int,default=5)
    args=p.parse_args(argv)
    if not 1<=args.timeout_minutes<=2880:p.error('timeout must be 1..2880 minutes')
    return args

if __name__=='__main__':raise SystemExit(run(parse()))
