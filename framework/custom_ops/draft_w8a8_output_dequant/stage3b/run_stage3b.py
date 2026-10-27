#!/usr/bin/env python3
"""Stage3B-1 reduction only: prepare/build/smoke and fixed-file read-only VDEQ inspection."""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import sys
from s3b_common import HERE,s3a,common,e2c,identity,verify,write,verify_tree,fresh,sources,resolve_report,evidence,prepare,verify_case,check_execution,compare,NAME,SNAKE,SETTINGS,BUDGET
from vdeq_inspect import inspect
command=e2c.stage2b.command


def installed(opp):
    sys.path.insert(0,str(common.QUANT/'test'))
    from opp_preflight import check_root,check_binary_path
    from compile_raw import elf_info
    check_root(opp);directory=opp/'vendors/customize/op_impl/ai_core/tbe/kernel/ascend310p'/SNAKE
    paths=sorted(directory.glob('*.o'))
    if not paths:raise ValueError('missing isolated reduction object')
    entries=[]
    for p in paths:
        q=check_binary_path(p)
        if opp.resolve() not in p.resolve().parents or opp.resolve() not in q.resolve().parents:raise ValueError('OPP kernel escapes installation')
        meta=json.loads(q.read_text())
        if not isinstance(meta,dict):raise ValueError('kernel JSON is not an object')
        obj,elf=elf_info(p,False);entries.append(dict(binary=obj,metadata=identity(q),elf=elf))
    api=opp/'vendors/customize/op_api';header=api/'include'/('aclnn_'+SNAKE+'.h');library=api/'lib/libcust_opapi.so'
    host=opp/'vendors/customize/op_impl/ai_core/tbe/op_tiling/liboptiling.so'
    for p in (header,library,host):
        if not p.is_file() or not p.stat().st_size or opp.resolve() not in p.resolve().parents:raise ValueError('missing/escaped installed reduction API/Host')
    h,info=elf_info(host,True)
    return dict(kernels=entries,header=identity(header),api=identity(library),host=h,host_elf=info,
                all_files=[identity(p) for p in sorted(opp.rglob('*')) if p.is_file()])


def build(args,root,r,pf):
    if sys.platform!='linux' or platform.machine() not in ('aarch64','arm64'):raise ValueError('requires Linux aarch64 CANN receiver')
    for name in ('msopgen','cmake','c++'):
        path=shutil.which(name)
        if not path or name not in pf['tools'] or identity(path)!=pf['tools'][name]['file']:raise ValueError('compiler/tool differs from accepted environment: '+name)
    sys.path.insert(0,str(common.QUANT/'test'))
    from opp_preflight import check_root
    check_root(root)
    r['vdeq_static']=inspect(args.cann_root,root/'vdeq-static',pf)
    e=common.env(args.cann_root);project=root/'CustomOp'
    shutil.copyfile(HERE/'operator.json',root/'operator.json');r.update(phase='msopgen',cann_compile='RUNNING');write(root/'report.json',r)
    command(['msopgen','gen','-i',root/'operator.json','-c','ai_core-Ascend310P3','-lan','cpp','-out',project,'-op',NAME],root/'msopgen.log',e,r,'msopgen')
    bound=[]
    for sub in ('op_host','op_kernel'):
        wanted=set()
        for p in (HERE/sub).iterdir():
            if p.is_file():shutil.copyfile(p,project/sub/p.name);wanted.add(p.name);bound.append(identity(project/sub/p.name))
        actual={p.name for p in (project/sub).iterdir() if p.suffix in ('.h','.hpp','.cpp','.cc')}
        if actual!=wanted:raise ValueError('unexpected generated reduction source inventory')
    r['generated_sources']=bound;r['phase']='cann_compile';write(root/'report.json',r)
    command(['bash','-c','cd "$1" && bash build.sh','stage3b',project],root/'build.log',e,r,'build',3600)
    for item in bound:verify(item)
    r['cann_compile']='BUILD_RETURNED_ZERO';packages=list((project/'build_out').glob('custom_opp_*.run'))
    if len(packages)!=1:raise ValueError('missing/ambiguous reduction package')
    r['package']=identity(packages[0]);opp=root/'opp';r.update(phase='isolated_install',installation='RUNNING');write(root/'report.json',r)
    command(['bash',packages[0],'--install-path='+str(opp)],root/'install.log',e,r,'install')
    r['phase']='installed_opp_preflight';r['installed']=installed(opp);r['installation']='ISOLATED_OPP_VERIFIED'
    r['phase']='runner_build';write(root/'report.json',r)
    command(['cmake','-S',HERE/'test','-B',root/'runner-build','-DCANN_ROOT='+str(args.cann_root),'-DOPP_ROOT='+str(opp),
             '-DCOMMON_INCLUDE='+str(common.QUANT/'test'),'-DCMAKE_EXPORT_COMPILE_COMMANDS=ON'],root/'cmake.log',e,r,'runner_configure')
    command(['cmake','--build',root/'runner-build','--parallel'],root/'runner-build.log',e,r,'runner_build')
    runner=root/'runner-build/stage3b_runner';command(['ldd',runner],root/'ldd.log',common.env(args.cann_root,opp),r,'ldd')
    if 'not found' in (root/'ldd.log').read_text():raise ValueError('unresolved runner library')
    links={}
    for line in (root/'runner-build/CMakeCache.txt').read_text().splitlines():
        if line.startswith('LIB_') and ':FILEPATH=' in line:
            name,path=line.split(':FILEPATH=',1);item=identity(path)
            if (opp if name=='LIB_custom' else args.cann_root).resolve() not in Path(item['path']).parents:raise ValueError('wrong linked library')
            links[name]=item
    if set(links)!={'LIB_custom','LIB_ascendcl','LIB_nnopbase','LIB_acl_op_compiler'}:raise ValueError('missing linked identity')
    r.update(status='STAGE3B1_BUILT_NPU_NOT_RUN',phase='complete',cann_compile='COMPLETED',runner=identity(runner),
        opp=str(opp),linked_libraries=links,compile_commands=identity(root/'runner-build/compile_commands.json'))


def smoke(args,root,r,case):
    if args.build_report is None:raise ValueError('--build-report required')
    b=json.loads(args.build_report.read_text());verify_tree(b)
    if b.get('status')!='STAGE3B1_BUILT_NPU_NOT_RUN' or b.get('settings')!=SETTINGS or b.get('sources')!=r['sources'] or b.get('stage3a_evidence')!=r['stage3a_evidence']:raise ValueError('reduction build/source/input binding differs')
    if installed(Path(b['opp']))!=b['installed']:raise ValueError('isolated reduction OPP changed')
    r.update(build_report=identity(args.build_report),cann_compile=b['cann_compile'],installation=b['installation'])
    folder=root/'execute';folder.mkdir();(folder/'partials_fp16.bin').symlink_to(verify(case['files']['partials_fp16.bin']))
    r.update(phase='reduction_npu',npu='RUNNING');write(root/'report.json',r)
    command([verify(b['runner']),args.device,folder],folder/'runner.log',common.env(args.cann_root,Path(b['opp'])),r,'reduction',300)
    r['execution']=identity(folder/'execution.json');check_execution(json.loads(verify(r['execution']).read_text()),args.device)
    if 'STAGE3B_REDUCE_PLAN groups=20 elements=1024 blockDim=1 explicitUB=12288 reserve=8192' not in (folder/'runner.log').read_text():raise ValueError('reduction resource plan missing')
    r['outputs']=[identity(folder/f'actual-{i}.bin') for i in range(2)]
    r['comparison']=compare(case,[verify(item).read_bytes() for item in r['outputs']]);write(root/'report.json',r)
    if r['comparison']['status']!='REDUCTION_BITWISE_MATCH':raise ValueError('reduction differs from Stage3A partial Golden; no tolerance or conversion fallback')
    verify_tree(b);verify_case(case)
    if installed(Path(b['opp']))!=b['installed']:raise ValueError('OPP changed during reduction')
    r.update(status='STAGE3B1_REDUCTION_MATCH',phase='complete',npu='MATCHED_REAL_STAGE3A_PARTIALS_ONLY')


def main(args):
    report_path=resolve_report(args.stage3a_root,args.stage3a_report)
    protected=[args.stage3a_root,report_path.parent,args.cann_root]
    if args.build_report:protected.append(args.build_report.parent)
    root=fresh(args.output_dir,protected)
    r=dict(stage='3B-1',status='NOT_RUN',phase='stage3a_evidence',device=args.device,settings=SETTINGS,sources=sources(),
        cann_compile='NOT_RUN',installation='NOT_RUN',npu='NOT_RUN',performance='NOT_RUN',stage3b2='DESIGN_ONLY',
        previous_stages_rerun=False,native_semantics_dependency='NONE',full_n19456='NOT_RUN')
    try:
        stack,total,half,r['stage3a_evidence'],pf=evidence(report_path,args.device,args.cann_root)
        write(root/'report.json',r)
        if args.stage=='inspect-vdeq':
            r['vdeq_static']=inspect(args.cann_root,root/'vdeq-static',pf);r.update(status='READ_ONLY_VDEQ_EVIDENCE',phase='complete')
        elif args.stage=='build':build(args,root,r,pf)
        else:
            case=prepare(root/'data',stack,total,half,r['stage3a_evidence']);r['case']=identity(root/'data/case.json');verify_case(case)
            if args.stage=='prepare':r.update(status='STAGE3B1_CPU_PREPARED_NPU_NOT_RUN',phase='complete')
            else:smoke(args,root,r,case)
    except Exception as error:
        r.update(status='FAILED',error=f'{type(error).__name__}: {error}')
        for key in ('cann_compile','installation','npu'):
            if r[key]=='RUNNING':r[key]='FAILED_OR_INCOMPLETE'
    finally:
        try:
            for item in r['sources']:verify(item)
            if 'stage3a_evidence' in r:verify_tree(r['stage3a_evidence'])
            if 'case' in r:verify_tree(json.loads(verify(r['case']).read_text()))
        except Exception as error:r.update(status='FAILED',integrity_error=str(error))
        write(root/'report.json',r)
    print(r['status']+': '+str(root/'report.json'));return int(r['status']=='FAILED')


def parse():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('prepare','build','smoke','inspect-vdeq'))
    p.add_argument('--stage3a-root',type=Path,default=Path('/home/w00949577/qwen35-dflash-run/w8a8-stage3a-20261027-011523'))
    p.add_argument('--stage3a-report',type=Path);p.add_argument('--build-report',type=Path)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--device',type=int,default=0)
    p.add_argument('--cann-root',type=Path,default=Path(os.environ.get('CANN_ROOT','/usr/local/Ascend/cann-9.0.0')))
    args=p.parse_args();args.cann_root=args.cann_root.resolve()
    if args.device<0:p.error('negative device')
    if args.build_report and args.stage!='smoke':p.error('--build-report only applies to smoke')
    return args

if __name__=='__main__':raise SystemExit(main(parse()))
