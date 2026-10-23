#!/usr/bin/env python3
"""Independent W8A8 ND -> optional NZ -> frozen K128, all stages fail closed."""
import argparse
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from w8_common import HERE,QUANT,OPS,identity,verify,write,fresh,sources
from w8_sdk import preflight,reviewed


def env(cann,opp=None):
    e=dict(os.environ);e.pop('LD_PRELOAD',None);e.pop('ASCEND_CUSTOM_OPP_PATH',None)
    e.update(CANN_ROOT=str(cann),ASCEND_HOME_PATH=str(cann),ASCEND_INSTALL_PATH=str(cann),ASCEND_OPP_PATH=str(cann/'opp'))
    libs=[str(cann/'lib64'),str(cann/'aarch64-linux/lib64')]+[p for p in e.get('LD_LIBRARY_PATH','').split(':') if p and '/vendors/' not in p]
    if opp:
        vendor=opp/'vendors/customize';e['ASCEND_CUSTOM_OPP_PATH']=str(vendor);libs.insert(0,str(vendor/'op_api/lib'))
    e['LD_LIBRARY_PATH']=':'.join(libs);return e


def command(argv,log,environment,timeout=1800):
    argv=list(map(str,argv));write(str(log)+'.command.json',dict(argv=argv,environment={k:environment.get(k) for k in ('ASCEND_HOME_PATH','ASCEND_OPP_PATH','ASCEND_CUSTOM_OPP_PATH','LD_LIBRARY_PATH')}))
    with log.open('w') as f:r=subprocess.run(argv,cwd=log.parent,env=environment,stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
    if r.returncode:raise RuntimeError(f'command exit {r.returncode}: {log}')


def verify_tree(value):
    if isinstance(value,dict):
        if set(value)=={'path','bytes','sha256'}:verify(value)
        else:
            for v in value.values():verify_tree(v)
    elif isinstance(value,list):
        for v in value:verify_tree(v)


def require_proof(path,status,settings,device=None,preflight_identity=None,build_identity=None):
    if path is None:raise ValueError('required prior NPU stage evidence missing: '+status)
    r=json.loads(path.read_text())
    if r.get('status')!=status or r.get('phase')!='complete' or r.get('settings')!=settings:
        raise ValueError('prior stage not accepted with identical reference settings: '+status)
    if device is not None and r.get('device')!=device:raise ValueError('prior stage device differs')
    if preflight_identity is not None and r.get('preflight')!=preflight_identity:raise ValueError('prior SDK preflight differs')
    if build_identity is not None and r.get('build_report')!=build_identity:raise ValueError('prior OPP build differs')
    verify_tree(r)
    names={'scale_one','scale_half','channel_powers','zero','signed_boundaries','precision_probe','overflow_probe','subnormal_probe'}
    if r.get('dataset')!='synthetic' or len(r.get('cases',[]))!=8 or {c.get('name') for c in r['cases']}!=names:
        raise ValueError('all unique synthetic cases required')
    if status!='SYNTHETIC_'+r.get('layout','').upper()+'_PASS':raise ValueError('prior layout/status mismatch')
    from w8_reference import compare_case
    for entry in r['cases']:
        case=json.loads(verify(entry['case_file']).read_text())
        result=compare_case(case,[verify(f).read_bytes() for f in entry['raw']['outputs']],
                            [verify(f).read_bytes() for f in entry['quant']['outputs']])
        if result!=entry['comparison'] or result['status']!='W8A8_REFERENCE_MATCH':raise ValueError('prior numerical evidence inconsistent')
        for mode in ('raw','quant'):
            ex=json.loads(verify(entry[mode]['execution_file']).read_text());check_execution(ex,mode,r['layout'],r['device'])
    return identity(path)


def installed(opp):
    sys.path.insert(0,str(QUANT/'test'))
    from opp_preflight import check_root,check_binary_path
    check_root(opp);result={}
    for mode,(_,snake) in OPS.items():
        folder=opp/'vendors/customize/op_impl/ai_core/tbe/kernel/ascend310p'/snake
        binaries=sorted(folder.glob('*.o'))
        if not binaries:raise ValueError('missing independent installed kernel: '+snake)
        items=[]
        for p in binaries:
            q=check_binary_path(p)
            if not p.stat().st_size or not isinstance(json.loads(q.read_text()),dict):raise ValueError('invalid OPP object/JSON')
            items.append(dict(binary=identity(p),metadata=identity(q)))
        header=opp/'vendors/customize/op_api/include'/('aclnn_'+snake+'.h')
        if not header.is_file() or not header.stat().st_size:raise ValueError('missing independent ACLNN header')
        result[mode]=dict(kernels=items,header=identity(header))
    return result


def build(args,root,r,p,review):
    if sys.platform!='linux' or platform.machine() not in ('aarch64','arm64'):raise ValueError('Linux aarch64 CANN receiver required')
    if args.layout=='nz':r['nd_proof']=require_proof(args.nd_proof,'SYNTHETIC_ND_PASS',r['settings'],args.device,r['preflight'])
    if 'msopgen' not in p['tools'] or not shutil.which('msopgen') or identity(shutil.which('msopgen'))!=p['tools']['msopgen']['file']:
        raise ValueError('msopgen missing or differs from preflight')
    text=(HERE/'op_kernel/kernel_common.h').read_text()
    if re.search(r'\b(?:Cast|Brcb|Mul|Muls|AscendAntiQuant|SetAntiQuantVector)\s*[<(]',text):raise ValueError('weight dequant/fallback forbidden')
    r.update(phase='cann_build',cann_compile='RUNNING');write(root/'report.json',r)
    definition=json.loads((HERE/'operators.json').read_text())
    if args.layout=='nz':
        for op in definition:op['input_desc'][1]['format']=['FRACTAL_NZ']
    write(root/'operators.json',definition);project=root/'CustomOp';e=env(args.cann_root)
    for index,(name,_) in enumerate(OPS.values()):
        command(['msopgen','gen','-i',root/'operators.json','-c','ai_core-Ascend310P3','-lan','cpp','-out',project,'-op',name]+(['-m','1'] if index else []),root/f'msopgen-{index}.log',e)
    config='#pragma once\n#define W8A8_B_NZ '+str(int(args.layout=='nz'))+'\n#define W8A8_SCALED_C_DTYPE matmul_tiling::DataType::'+r['settings']['host_scaled_c_dtype']+'\n'
    for sub in ('op_host','op_kernel'):
        for source in (HERE/sub).iterdir():shutil.copyfile(source,project/sub/source.name)
        (project/sub/'experiment_config.h').write_text(config)
    command(['bash','-c','cd "$1" && bash build.sh','build',project],root/'build.log',e,3600)
    r['cann_compile']='BUILD_COMPLETED_NPU_NOT_RUN'
    packages=list((project/'build_out').glob('custom_opp_*.run'))
    if len(packages)!=1:raise ValueError('isolated package missing/ambiguous')
    opp=root/'opp';command(['bash',packages[0],'--install-path='+str(opp)],root/'install.log',e)
    r['installed']=installed(opp)
    command(['cmake','-S',HERE/'test','-B',root/'runner-build','-DCANN_ROOT='+str(args.cann_root),
        '-DOPP_ROOT='+str(opp),'-DCOMMON_INCLUDE='+str(QUANT/'test'),'-DCMAKE_EXPORT_COMPILE_COMMANDS=ON'],root/'cmake.log',e)
    command(['cmake','--build',root/'runner-build','--parallel'],root/'runner-build.log',e)
    runner=root/'runner-build/w8a8_runner';command(['ldd',runner],root/'ldd.log',env(args.cann_root,opp))
    if 'not found' in (root/'ldd.log').read_text():raise ValueError('unresolved runtime library')
    linked={}
    for line in (root/'runner-build/CMakeCache.txt').read_text().splitlines():
        if line.startswith('LIB_') and ':FILEPATH=' in line:
            key,value=line.split(':FILEPATH=',1);item=identity(value)
            if (opp if key=='LIB_custom' else args.cann_root) not in Path(item['path']).parents:raise ValueError('library escaped selected install')
            linked[key]=item
    if set(linked)!={'LIB_custom','LIB_ascendcl','LIB_nnopbase','LIB_acl_op_compiler'}:raise ValueError('missing linked library identity')
    artifacts=[identity(p) for p in opp.rglob('*') if p.is_file()]+[identity(packages[0]),identity(runner),identity(QUANT/'test/runner_common.h')]
    artifacts += [identity(p) for sub in ('op_host','op_kernel') for p in (project/sub).iterdir() if p.is_file()]
    artifacts += [identity(p) for p in project.rglob('*') if p.is_file() and p.name in ('CMakeCache.txt','compile_commands.json','CMakePresets.json')]
    r.update(status='BUILT_NOT_NPU_VERIFIED',phase='complete',runner=identity(runner),opp=str(opp),artifacts=artifacts,linked=linked,
        requested_config=dict(shape=[16,128,64],base=[16,64,128],block_dim=1,B=args.layout,
            A='INT8 ND',C='INT32 raw / FP16 quant',quant_mode='TENSOR / SetQuantVector GM uint64[64]',explicit_weight_dequant=False))


def check_execution(ex,mode,layout,device):
    expected=dict(status='EXECUTION_COMPLETED_NUMERICS_UNCHECKED',op=OPS[mode][0],soc='Ascend310P3',shape=[16,128,64],
        guards=True,readonly=True,repeat_equal=True,repetitions=2,device=device,cpu_fallback=False,layout=layout,raw_accumulator=mode=='raw')
    if any(type(ex.get(k)) is not type(v) or ex.get(k)!=v for k,v in expected.items()):raise ValueError('execution/guard/readonly/repeat ABI mismatch')
    t=ex.get('timing',{})
    if t.get('status')!='NOT_RUN' or t.get('warmup')!=0 or t.get('repetitions')!=0:raise ValueError('no performance stage in this PoC')


def invoke(case,mode,root,built,args):
    root.mkdir(parents=True)
    for name,item in case['files'].items():(root/name).symlink_to(verify(item))
    command([verify(built['runner']),args.device,root,args.layout,mode],root/'runner.log',env(args.cann_root,Path(built['opp'])),300)
    ex=json.loads((root/'execution.json').read_text());check_execution(ex,mode,args.layout,args.device)
    if f'W8A8_PLAN raw={int(mode=="raw")} nz={int(args.layout=="nz")}' not in (root/'runner.log').read_text():raise ValueError('actual Host plan missing')
    outputs=[identity(root/f'actual-{i}.bin') for i in range(2)]
    return dict(execution_file=identity(root/'execution.json'),log=identity(root/'runner.log'),outputs=outputs)


def smoke(args,root,r):
    from w8_reference import synthetic,frozen,compare_case
    import numpy as np
    if args.build_report is None:raise ValueError('--build-report required')
    built=json.loads(args.build_report.read_text())
    if built.get('status')!='BUILT_NOT_NPU_VERIFIED' or built['layout']!=args.layout or built['settings']!=r['settings'] or built['preflight']!=r['preflight'] or built['review']!=r['review']:
        raise ValueError('build/review/layout identity mismatch')
    verify_tree(built)
    if installed(Path(built['opp']))!=built['installed']:raise ValueError('OPP changed')
    r['build_report']=identity(args.build_report);r['cann_compile']=built['cann_compile']
    if args.layout=='nz':r['nd_proof']=require_proof(args.nd_proof,'SYNTHETIC_ND_PASS',r['settings'],args.device,r['preflight'])
    if args.dataset=='frozen':r['synthetic_proof']=require_proof(args.synthetic_proof,'SYNTHETIC_'+args.layout.upper()+'_PASS',r['settings'],args.device,r['preflight'],identity(args.build_report))
    r['phase']='cpu_prepare';write(root/'report.json',r)
    cases=synthetic(root/'data',r['settings']['codec']) if args.dataset=='synthetic' else frozen(root/'data',r['settings']['codec'],args.a51_evidence,args.prefetch_evidence)
    r['cases']=[];r['phase']='npu_correctness';r['cpu_reference']='READY';write(root/'report.json',r)
    for case in cases:
        entry=dict(name=case['name'],case_file=identity(Path(case['files']['a.bin']['path']).parent/'case.json'));r['cases'].append(entry)
        r['npu']='RUNNING';write(root/'report.json',r)
        entry['raw']=invoke(case,'raw',root/'execute'/case['name']/'raw',built,args)
        raw=[verify(x).read_bytes() for x in entry['raw']['outputs']]
        oracle=np.fromfile(case['files']['acc_int64.bin']['path'],'<i8')
        if any(len(x)!=4096 or not np.array_equal(np.frombuffer(x,'<i4').astype(np.int64),oracle) for x in raw):
            entry['raw_gate']='FAIL_INT32_ORACLE'
            entry['raw_comparison']=[dict(bytes=len(x),mismatches=int(np.sum(np.frombuffer(x,'<i4').astype(np.int64)!=oracle)),
                max_abs_integer=int(np.max(np.abs(np.frombuffer(x,'<i4').astype(np.int64)-oracle)))) if len(x)==4096 else dict(bytes=len(x),error='OUTPUT_SIZE') for x in raw]
            write(root/'report.json',r)
            raise ValueError('INT32 accumulator mismatch; quant stage not executed')
        entry['quant']=invoke(case,'quant',root/'execute'/case['name']/'quant',built,args)
        result=compare_case(case,raw,[verify(x).read_bytes() for x in entry['quant']['outputs']]);entry['comparison']=result
        write(root/'report.json',r)
        if result['status']!='W8A8_REFERENCE_MATCH':raise ValueError('encoded W8A8 reference mismatch; no tolerance/adaptive codec')
        for item in case.get('watched',{}).values():verify(item)
    verify_tree(built)
    r.update(status=('SYNTHETIC_' if args.dataset=='synthetic' else 'FROZEN_K128_')+args.layout.upper()+'_PASS',
        phase='complete',npu='REQUESTED_W8A8_CHECKS_PASSED',w8a16_acceptance='NOT_APPLICABLE_DIFFERENT_NUMERICAL_SCHEME')


def main(args):
    protected=[args.cann_root,args.a51_evidence.parent,args.prefetch_evidence.parent]
    protected += [p.parent for p in (args.preflight,args.build_report,args.nd_proof,args.synthetic_proof) if p]
    if args.review:protected.append(args.review)
    root=fresh(args.output_dir,protected)
    r=dict(status='UNRESOLVED',phase='preflight',layout=args.layout,dataset=args.dataset,device=args.device,
        cann_compile='NOT_RUN',npu='NOT_RUN',performance='NOT_RUN',k256='NOT_RUN',full_gate='NOT_RUN',draft='NOT_RUN',simulator='NOT_RUN',sources=sources())
    try:
        if args.stage=='preflight':
            r['preflight_result']=preflight(args.cann_root,root,args.sdk_file);r['status']=r['preflight_result']['status'];r['phase']='complete'
        else:
            if args.preflight is None or args.review is None:raise ValueError('--preflight and source-cited --review required')
            p,review=reviewed(args.preflight,args.review,args.layout)
            if p['cann_root']!=str(args.cann_root):raise ValueError('selected SDK differs from reviewed root')
            r.update(settings=review['settings'],preflight=identity(args.preflight),review=identity(args.review))
            if args.stage=='build':build(args,root,r,p,review)
            else:smoke(args,root,r)
            reviewed(args.preflight,args.review,args.layout)
    except Exception as e:
        text=str(e);status='BLOCKED_UNSUPPORTED' if text.startswith('BLOCKED_UNSUPPORTED') else 'UNRESOLVED' if text.startswith('UNRESOLVED') else 'BLOCKED'
        r.update(status=status,error=f'{type(e).__name__}: {e}')
        if r['cann_compile']=='RUNNING':r['cann_compile']='FAILED_OR_INCOMPLETE'
        if r['npu']=='RUNNING':r['npu']='FAILED_OR_INCOMPLETE'
    finally:
        try:
            for item in r['sources']:verify(item)
            for entry in r.get('cases',[]):
                case=json.loads(verify(entry['case_file']).read_text());verify_tree(case)
        except Exception as e:r.update(status='BLOCKED',integrity_error=str(e))
        write(root/'report.json',r)
    print(r['status']+': '+str(root/'report.json'))
    return int(r['status'].startswith('BLOCKED') or args.stage!='preflight' and r['status']=='UNRESOLVED')


def parse():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('preflight','build','smoke'))
    p.add_argument('--cann-root',type=Path,default=Path(os.environ.get('CANN_ROOT','/usr/local/Ascend/cann-9.0.0')))
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--layout',choices=('nd','nz'),default='nd')
    p.add_argument('--dataset',choices=('synthetic','frozen'),default='synthetic');p.add_argument('--device',type=int,default=0)
    for key in ('preflight','review','build-report','nd-proof','synthetic-proof'):p.add_argument('--'+key,type=Path)
    p.add_argument('--sdk-file',type=Path,action='append',default=[])
    p.add_argument('--a51-evidence',type=Path,default=Path('/home/w00949577/qwen35-dflash-run/a51-fallback-20261012-063342/suite.json'))
    p.add_argument('--prefetch-evidence',type=Path,default=Path('/home/w00949577/qwen35-dflash-run/gate-raw-prefetch-reverse-20261020-025853/suite.json'))
    a=p.parse_args();a.cann_root=a.cann_root.resolve()
    if a.device<0:p.error('negative device')
    if any(p.is_absolute() or '..' in p.parts for p in a.sdk_file):p.error('--sdk-file must be explicit SDK-relative path')
    return a

if __name__=='__main__':raise SystemExit(main(parse()))
