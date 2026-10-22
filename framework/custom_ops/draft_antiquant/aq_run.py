#!/usr/bin/env python3
"""Separate audit -> source-reviewed build -> untimed K128; no larger shapes."""
import argparse
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from aq_common import HERE, identity, verify, write, fresh, experiment_sources
from aq_sdk import collect, reviewed

QUANT = HERE.parent / 'draft_quant'


def command(argv, log, env, timeout=1800):
    argv=[str(a) for a in argv]
    write(str(log)+'.command.json', dict(argv=argv, cwd=str(log.parent),
        environment={k:env.get(k) for k in ('ASCEND_HOME_PATH','ASCEND_CUSTOM_OPP_PATH','LD_LIBRARY_PATH')}))
    with log.open('w') as f:
        result=subprocess.run(argv,cwd=log.parent,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
    if result.returncode:
        raise RuntimeError(f'command exit {result.returncode}; see {log}')


def environment(cann, vendor=None):
    # Reuse established isolation policy, not a cached NPU/Python process.
    sys.path.insert(0,str(QUANT/'fp16_compare'))
    from experiment import child_environment
    env=child_environment(cann,vendor)
    env.update(ASCEND_HOME_PATH=str(cann),ASCEND_INSTALL_PATH=str(cann),ASCEND_OPP_PATH=str(cann/'opp'))
    return env


def installed(root):
    sys.path.insert(0,str(QUANT/'test'))
    from opp_preflight import check_root, check_binary_path
    check_root(root)
    directory=root/'vendors/customize/op_impl/ai_core/tbe/kernel/ascend310p/d_flash_anti_quant_k128'
    binaries=sorted(directory.glob('*.o'))
    if not binaries:raise ValueError('new AntiQuant kernel absent from installed OPP')
    entries=[]
    for binary in binaries:
        meta=check_binary_path(binary)
        if not binary.stat().st_size or not isinstance(json.loads(meta.read_text()),dict):
            raise ValueError('empty binary or malformed paired JSON')
        entries.append(dict(binary=identity(binary),metadata=identity(meta)))
    header=root/'vendors/customize/op_api/include/aclnn_d_flash_anti_quant_k128.h'
    if not header.is_file() or not header.stat().st_size:raise ValueError('new ACLNN header missing')
    return dict(kernels=entries,header=identity(header))


def source_no_explicit_dequant(path):
    text=Path(path).read_text()
    text=re.sub(r'/\*.*?\*/|//[^\n]*','',text,flags=re.S)
    if re.search(r'\b(?:Cast|Brcb|Mul|Muls)\s*(?:<[^;]+?>)?\s*\(',text):
        raise ValueError('explicit UB dequant forbidden in this candidate')
    if text.count('mm.SetAntiQuantVector(')!=1 or 'CubeFormat::NZ, int8_t, true' not in text:
        raise ValueError('candidate AntiQuant/GM INT8 NZ contract changed')
    return dict(status='SOURCE_CHECKED',source=identity(path),
        scope='No explicit user Cast/Brcb/Mul/Muls calls; internal SDK antiquant instructions NOT yet inspected')


def build(args, root, report):
    audit,review=reviewed(args.audit,args.review)
    if audit['cann_root']!=str(args.cann_root):raise ValueError('build SDK differs from audited SDK')
    if sys.platform!='linux' or platform.machine() not in ('aarch64','arm64'):
        raise ValueError('real Linux aarch64 CANN receiver required')
    for name in ('msopgen',):
        if name not in audit['tools'] or not shutil.which(name) or identity(shutil.which(name))!=audit['tools'][name]['file']:
            raise ValueError('tool differs from SDK audit: '+name)
    report.update(audit=identity(args.audit),review=identity(args.review),
                  source_review=review,phase='build',cann_compile='RUNNING',
                  explicit_dequant=source_no_explicit_dequant(HERE/'op_kernel/d_flash_anti_quant_k128.cpp'))
    write(root/'report.json',report)
    env=environment(args.cann_root);project=root/'CustomOp';opp=root/'opp'
    command(['msopgen','gen','-i',HERE/'DFlashAntiQuantK128.json','-c','ai_core-Ascend310P3',
             '-lan','cpp','-out',project,'-op','DFlashAntiQuantK128'],root/'msopgen.log',env)
    for sub in ('op_host','op_kernel'):
        for p in (HERE/sub).iterdir():shutil.copyfile(p,project/sub/p.name)
    command(['bash','-c','cd "$1" && bash build.sh','build',project],root/'build.log',env,3600)
    report['cann_compile']='COMPILE_COMPLETED_NUMERICS_NOT_RUN'
    packages=list((project/'build_out').glob('custom_opp_*.run'))
    if len(packages)!=1:raise ValueError('expected exactly one isolated OPP package')
    command(['bash',packages[0],'--install-path='+str(opp)],root/'install.log',env)
    report['installed']=installed(opp)
    preferred=args.cann_root/'include/aclnnop/aclnn_mm.h'
    headers={p.resolve() for p in args.cann_root.rglob('aclnn_mm.h')}
    if preferred.is_file():header=preferred.resolve()
    elif len(headers)==1:header=next(iter(headers))
    else:raise ValueError('native aclnnMm header missing/ambiguous')
    if args.cann_root not in header.parents:raise ValueError('native header escapes SDK')
    command(['cmake','-S',HERE/'test','-B',root/'runner-build',
             '-DCANN_ROOT='+str(args.cann_root),'-DOPP_ROOT='+str(opp),'-DNATIVE_MM_HEADER='+str(header),
             '-DCOMMON_INCLUDE='+str(QUANT/'test'),'-DCMAKE_EXPORT_COMPILE_COMMANDS=ON'],root/'cmake.log',env)
    command(['cmake','--build',root/'runner-build','--parallel'],root/'runner-build.log',env)
    runners={mode:root/'runner-build'/('antiquant_'+mode) for mode in ('native','custom')}
    for mode,p in runners.items():
        command(['ldd',p],root/(mode+'-ldd.log'),environment(args.cann_root,opp/'vendors/customize' if mode=='custom' else None))
        if 'not found' in (root/(mode+'-ldd.log')).read_text():raise ValueError('unresolved runtime dependency: '+mode)
    files=[identity(p) for p in opp.rglob('*') if p.is_file()]+[identity(p) for p in runners.values()]+[
        identity(packages[0]),identity(header),identity(QUANT/'test/runner_common.h')]
    configs=[p for p in project.rglob('*') if p.is_file() and
             (p.name in ('CMakeCache.txt','compile_commands.json','CMakePresets.json') or p.suffix=='.cmake')]
    files += [identity(p) for p in configs]
    linked={}
    for line in (root/'runner-build/CMakeCache.txt').read_text().splitlines():
        if line.startswith('LIB_') and ':FILEPATH=' in line:
            name,path=line.split(':FILEPATH=',1);item=identity(path)
            allowed=opp if name=='LIB_custom' else args.cann_root
            if allowed not in Path(item['path']).parents:raise ValueError('linked library outside selected root: '+path)
            linked[name]=item;files.append(item)
    if set(linked)!={'LIB_custom','LIB_ascendcl','LIB_opapi','LIB_nnopbase','LIB_acl_op_compiler'}:
        raise ValueError('incomplete linked library identities')
    for sub in ('op_host','op_kernel'):
        files += [identity(p) for p in (project/sub).iterdir() if p.is_file()]
    source_no_explicit_dequant(project/'op_kernel/d_flash_anti_quant_k128.cpp')
    reviewed(args.audit,args.review)  # SDK/source immutable across compilation
    report.update(status='K128_BUILT_NPU_NOT_RUN',phase='complete',files=files,
        runners={m:identity(p) for m,p in runners.items()},opp=str(opp),linked_libraries=linked,
        build_config=dict(soc='Ascend310P3',op='DFlashAntiQuantK128',shape=[16,128,64],block_dim=1,
            a='GM ND FP16',b='GM NZ INT8 transpose=true',c='GM ND FP16',base=[16,64,128],
            parameter_buffers='2 x VECIN FP16[64], 128 bytes each, TPipe aligned',
            lifetime='GM copy -> PIPE_ALL -> SetAntiQuantVector -> IterateAll -> PIPE_ALL -> End',
            sdk_buffer='UB - 256 - 8192, rounded down to 32; actual tiling printed at GetWorkspaceSize',
            compiler_configuration_files=[identity(p) for p in configs]))


def check_execution(ex,mode,device):
    expected=dict(status='EXECUTION_COMPLETED_NUMERICS_UNCHECKED',
        op='aclnnMm' if mode=='native' else 'DFlashAntiQuantK128',soc='Ascend310P3',shape=[16,128,64],
        guards=True,readonly=True,repeat_equal=True,repetitions=2,device=device,cpu_fallback=False)
    for k,v in expected.items():
        if type(ex.get(k))!=type(v) or ex.get(k)!=v:raise ValueError('invalid execution '+k)
    timing=ex.get('timing',{})
    if timing.get('status')!='NOT_RUN' or timing.get('warmup')!=0 or timing.get('repetitions')!=0:
        raise ValueError('K128 stage must be untimed')


def numerical_report(payload, outputs):
    from types import SimpleNamespace
    sys.path.insert(0,str(QUANT/'test'))
    from reference import compare
    if set(outputs)!={'native','custom'} or any(len(v)!=2 for v in outputs.values()):
        raise ValueError('both implementations and two repeats required')
    if any(len(v)!=2048 for vs in outputs.values() for v in vs):
        raise ValueError('K128 FP16 output size differs')
    if any(vs[0]!=vs[1] for vs in outputs.values()):
        raise ValueError('repeat stability gate failed')
    result={}
    for mode,values in outputs.items():
        result[mode]=[dict(cpu_fp32=compare(payload['cpu_fp32.bin'],v,SimpleNamespace(m=16,n=64)),
                           cpu_fp64=compare(payload['cpu_fp64.bin'],v,SimpleNamespace(m=16,n=64)),
                           native=compare(outputs['native'][0],v,SimpleNamespace(m=16,n=64))) for v in values]
    matched=all(c['finite'] and c['bitwise_equal'] and c['max_ulp']==0
        for rows in result.values() for row in rows for c in row.values())
    return dict(status='K128_BITWISE_MATCH' if matched else 'FAIL_NUMERICAL',comparisons=result,
        rule='Exact bits; report CPU reduction-order disagreements without tolerance or Golden changes',
        group_refresh='NOT_RUN',full_gate_golden='NOT_APPLICABLE_TO_PARTIAL_K')


def k128(args,root,report):
    audit,review=reviewed(args.audit,args.review)
    if audit['cann_root']!=str(args.cann_root):raise ValueError('execution SDK differs from audited SDK')
    built=json.loads(args.build_report.read_text())
    if built.get('status')!='K128_BUILT_NPU_NOT_RUN' or built.get('audit')!=identity(args.audit) or built.get('review')!=identity(args.review):
        raise ValueError('requires source-reviewed independent K128 build')
    for item in built['files']:verify(item)
    if installed(Path(built['opp']))!=built['installed']:raise ValueError('installed OPP identity differs')
    report.update(audit=identity(args.audit),review=identity(args.review),build=identity(args.build_report),
                  phase='frozen_slice',cann_compile=built['cann_compile'])
    write(root/'report.json',report)
    from aq_prepare import prepare
    sliced=prepare(args.a51_evidence,args.prefetch_evidence,root/'data')
    report['slice']=sliced;report['cpu_preparation']='CPU_SLICE_VERIFIED';report['phase']='untimed_k128'
    write(root/'report.json',report)
    outputs={};report['executions']={}
    for mode in ('native','custom'):
        folder=root/mode;folder.mkdir()
        for name,item in sliced['files'].items():(folder/name).symlink_to(verify(item))
        runner=verify(built['runners'][mode])
        env=environment(args.cann_root,Path(built['opp'])/'vendors/customize' if mode=='custom' else None)
        report['npu']='RUNNING';write(root/'report.json',report)
        command([runner,args.device_id,folder],folder/'runner.log',env,300)
        ex=json.loads((folder/'execution.json').read_text());check_execution(ex,mode,args.device_id)
        if mode=='custom' and 'ANTIQUANT_K128 blockDim=1 single=16/64/128 base=16/64/128' not in (folder/'runner.log').read_text():
            raise ValueError('actual AntiQuant tiling evidence missing')
        paths=[folder/f'actual-{i}.bin' for i in range(2)];outputs[mode]=[p.read_bytes() for p in paths]
        if any(len(v)!=2048 for v in outputs[mode]) or outputs[mode][0]!=outputs[mode][1]:
            raise ValueError('output size/repeat differs')
        report['executions'][mode]=dict(execution=ex,report=identity(folder/'execution.json'),
            log=identity(folder/'runner.log'),outputs=[identity(p) for p in paths])
        write(root/'report.json',report)
    payload={name:verify(item).read_bytes() for name,item in sliced['files'].items()}
    numbers=numerical_report(payload,outputs);write(root/'numerical.json',numbers)
    report.update(numerical=numbers,npu=numbers['status'],phase='complete',
        status='K128_DIAGNOSTIC_BITWISE_PASS' if numbers['status']=='K128_BITWISE_MATCH' else 'FAIL_NUMERICAL')
    for item in sliced['watched'].values():verify(item)
    for item in built['files']:verify(item)
    reviewed(args.audit,args.review)


def parse():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('audit','build','k128'))
    p.add_argument('--cann-root',type=Path,default=Path(os.environ.get('CANN_ROOT','/usr/local/Ascend/cann-9.0.0')))
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--audit',type=Path);p.add_argument('--review',type=Path);p.add_argument('--build-report',type=Path)
    p.add_argument('--device-id',type=int,default=0)
    p.add_argument('--a51-evidence',type=Path,default=Path('/home/w00949577/qwen35-dflash-run/a51-fallback-20261012-063342/suite.json'))
    p.add_argument('--prefetch-evidence',type=Path,default=Path('/home/w00949577/qwen35-dflash-run/gate-raw-prefetch-reverse-20261020-025853/suite.json'))
    a=p.parse_args();a.cann_root=a.cann_root.resolve()
    if a.device_id<0:p.error('negative device id')
    if a.stage!='audit' and (not a.audit or not a.review):p.error('source-bound --audit and --review required before compilation')
    if a.stage=='k128' and not a.build_report:p.error('--build-report required')
    return a


def main(args):
    if args.stage=='audit':
        report,path=collect(args.cann_root,args.output_dir);print(report['status']+': '+str(path))
        return int(report['status'].startswith('BLOCKED'))
    protected=[args.cann_root,args.a51_evidence.parent,args.prefetch_evidence.parent]
    protected += [p.parent for p in (args.audit,args.review,args.build_report) if p]
    root=fresh(args.output_dir,protected)
    report=dict(status='RUNNING',phase='source_review',cann_compile='NOT_RUN',npu='NOT_RUN',
        cpu_preparation='NOT_RUN',k256='NOT_RUN',k2560='NOT_RUN',performance='NOT_RUN',
        simulator='NOT_RUN',full_gate='NOT_RUN',full_draft='NOT_RUN',sources=experiment_sources())
    try:
        if args.stage=='build':build(args,root,report)
        else:k128(args,root,report)
    except Exception as e:
        report.update(status='BLOCKED_UNSUPPORTED' if str(e).startswith('BLOCKED_UNSUPPORTED') else 'BLOCKED',
                      error=f'{type(e).__name__}: {e}')
        if report['cann_compile']=='RUNNING':report['cann_compile']='FAILED_OR_INCOMPLETE'
        if report['npu']=='RUNNING':report['npu']='FAILED_OR_INCOMPLETE'
    finally:
        try:
            for item in report['sources']:verify(item)
            for item in report.get('slice',{}).get('watched',{}).values():verify(item)
        except Exception as e:report.update(status='BLOCKED',integrity_error=str(e))
        write(root/'report.json',report)
    print(report['status']+': '+str(root/'report.json'))
    return int(report['status'].startswith(('BLOCKED','FAIL')))


if __name__=='__main__':raise SystemExit(main(parse()))
