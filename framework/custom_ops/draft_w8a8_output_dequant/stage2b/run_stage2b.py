#!/usr/bin/env python3
"""Experimental Stage2B only. Build once; smoke basic, zero, nonpower separately."""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from s2b_common import HERE,FIX,QUANT,NAME,SNAKE,CODEC,GROUPS,identity,verify,write,fresh,verify_tree,env,sources,upstream,check_execution
from s2b_reference import prepare,compare,verify_origin


def command(argv,log,environment,report,label,timeout=1800):
    argv=list(map(str,argv));entry=dict(argv=argv,returncode=None,status='RUNNING',log=str(log))
    report.setdefault('commands',{})[label]=entry
    write(str(log)+'.command.json',dict(argv=argv,environment={k:environment.get(k) for k in ('ASCEND_CUSTOM_OPP_PATH','ASCEND_OPP_PATH','LD_LIBRARY_PATH')}))
    try:
        with log.open('w') as f:done=subprocess.run(argv,cwd=log.parent,env=environment,stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
        entry.update(returncode=done.returncode,status='COMPLETED' if done.returncode==0 else 'FAILED')
        if done.returncode:raise RuntimeError(f'{label} returned {done.returncode}: {log}')
    except subprocess.TimeoutExpired:entry['status']='TIMED_OUT';raise
    finally:
        if log.is_file():entry['log_identity']=identity(log)


def installed(opp):
    sys.path.insert(0,str(QUANT/'test'))
    from opp_preflight import check_root,check_binary_path
    from compile_raw import elf_info
    check_root(opp)
    directory=opp/'vendors/customize/op_impl/ai_core/tbe/kernel/ascend310p'/SNAKE
    paths=sorted(directory.glob('*.o'))
    if not paths:raise ValueError('missing Stage2B installed kernel')
    records=[]
    for p in paths:
        meta=check_binary_path(p)
        if opp.resolve() not in p.resolve().parents or opp.resolve() not in meta.resolve().parents:raise ValueError('installed kernel escaped private OPP')
        binary,elf=elf_info(p,False)
        if not isinstance(json.loads(meta.read_text()),dict):raise ValueError('invalid installed kernel JSON')
        records.append(dict(binary=binary,metadata=identity(meta),elf=elf))
    header=opp/'vendors/customize/op_api/include'/('aclnn_'+SNAKE+'.h')
    library=opp/'vendors/customize/op_api/lib/libcust_opapi.so'
    alias=opp/'vendors/customize/op_impl/ai_core/tbe/op_tiling/liboptiling.so'
    for p in (header,library,alias):
        if not p.is_file() or not p.stat().st_size or opp.resolve() not in p.resolve().parents:raise ValueError('missing/escaped installed API or runtime Host library: '+str(p))
    host,info=elf_info(alias,True)
    # Binding the actual installed package, not equating it to an uninstalled build .so.
    return dict(kernels=records,header=identity(header),api=identity(library),host=host,host_elf=info,host_alias=str(alias),
                all_files=[identity(p) for p in sorted(opp.rglob('*')) if p.is_file()])


def build(args,root,r):
    if sys.platform!='linux' or platform.machine() not in ('aarch64','arm64'):raise ValueError('CANN Linux aarch64 receiver required')
    sys.path.insert(0,str(QUANT/'test'))
    from opp_preflight import check_root
    check_root(root)
    pf=json.loads(verify(r['upstream']['stage1']['preflight']).read_text())
    for name in ('msopgen','cmake','c++'):
        tool=shutil.which(name)
        if name not in pf['tools'] or not tool or identity(tool)!=pf['tools'][name]['file']:raise ValueError('tool differs from Stage1 environment: '+name)
    r['tools']=pf['tools'];e=env(args.cann_root);project=root/'CustomOp'
    r.update(phase='msopgen',cann_compile='RUNNING');write(root/'report.json',r)
    definition=root/'operator.json';shutil.copyfile(HERE/'operator.json',definition)
    command(['msopgen','gen','-i',definition,'-c','ai_core-Ascend310P3','-lan','cpp','-out',project,'-op',NAME],root/'msopgen.log',e,r,'msopgen')
    bound=[]
    for sub in ('op_host','op_kernel'):
        wanted=set()
        for p in (HERE/sub).iterdir():
            if p.is_file():shutil.copyfile(p,project/sub/p.name);wanted.add(p.name);bound.append(identity(project/sub/p.name))
        actual={p.name for p in (project/sub).iterdir() if p.suffix in ('.cpp','.h','.hpp','.cc')}
        if actual!=wanted:raise ValueError('unexpected generated source inventory')
    r['generated_sources']=bound;r['phase']='cann_compile';write(root/'report.json',r)
    command(['bash','-c','cd "$1" && bash build.sh','stage2b',project],root/'build.log',e,r,'build',3600)
    r['cann_compile']='BUILD_RETURNED_ZERO'
    for f in bound:verify(f)
    packages=list((project/'build_out').glob('custom_opp_*.run'))
    if len(packages)!=1:raise ValueError('missing/ambiguous Stage2B package')
    r['package']=identity(packages[0]);opp=root/'opp';r['phase']='isolated_install';r['installation']='RUNNING';write(root/'report.json',r)
    command(['bash',packages[0],'--install-path='+str(opp)],root/'install.log',e,r,'install')
    r['phase']='installed_opp_preflight';r['installed']=installed(opp);r['installation']='ISOLATED_OPP_VERIFIED'
    r['phase']='runner_build';write(root/'report.json',r)
    command(['cmake','-S',HERE/'test','-B',root/'runner-build','-DCANN_ROOT='+str(args.cann_root),
        '-DOPP_ROOT='+str(opp),'-DCOMMON_INCLUDE='+str(QUANT/'test'),'-DCMAKE_EXPORT_COMPILE_COMMANDS=ON'],root/'cmake.log',e,r,'runner_configure')
    command(['cmake','--build',root/'runner-build','--parallel'],root/'runner-build.log',e,r,'runner_build')
    runner=root/'runner-build/stage2b_runner'
    command(['ldd',runner],root/'ldd.log',env(args.cann_root,opp),r,'ldd')
    if 'not found' in (root/'ldd.log').read_text():raise ValueError('unresolved runtime library')
    links={}
    for line in (root/'runner-build/CMakeCache.txt').read_text().splitlines():
        if line.startswith('LIB_') and ':FILEPATH=' in line:
            key,value=line.split(':FILEPATH=',1);item=identity(value)
            expected=opp if key=='LIB_custom' else args.cann_root
            if expected.resolve() not in Path(item['path']).parents:raise ValueError('Runner linked wrong library: '+key)
            links[key]=item
    if set(links)!={'LIB_custom','LIB_ascendcl','LIB_nnopbase','LIB_acl_op_compiler'}:raise ValueError('missing Runner library identity')
    r.update(status='STAGE2B_BUILT_NPU_NOT_RUN',phase='complete',cann_compile='COMPLETED',opp=str(opp),
             runner=identity(runner),linked_libraries=links,compile_commands=identity(root/'runner-build/compile_commands.json'))


def check_basic(path,built_id,upstream_ids,device):
    if path is None:raise ValueError('advanced Scale groups require --basic-report')
    r=json.loads(path.read_text());verify_tree(r)
    if r.get('status')!='STAGE2B_BASIC_MATCH' or r.get('group')!='basic' or r.get('phase')!='complete' or r.get('build_report')!=built_id or r.get('upstream')!=upstream_ids or r.get('device')!=device or r.get('codec_status')!=CODEC:
        raise ValueError('basic proof must match this build, device, codec and evidence')
    if len(r.get('cases',[]))!=len(GROUPS['basic']) or {c['name'] for c in r['cases']}!=set(GROUPS['basic']):raise ValueError('complete basic suite required')
    for c in r['cases']:
        case=json.loads(verify(c['case']).read_text());verify_origin(case,upstream_ids)
        if case['name']!=c['name']:raise ValueError('basic case name binding differs')
        ex=json.loads(verify(c['execution']).read_text());check_execution(ex,device)
        compared=compare(case,[verify(f).read_bytes() for f in c['outputs']])
        if compared!=c['comparison'] or compared['status']!='CANDIDATE_BITWISE_MATCH':raise ValueError('basic numeric evidence invalid')
    return identity(path)


def smoke(args,root,r):
    if args.build_report is None:raise ValueError('--build-report required')
    b=json.loads(args.build_report.read_text());verify_tree(b)
    if b.get('status')!='STAGE2B_BUILT_NPU_NOT_RUN' or b.get('settings')!=r['settings'] or b.get('upstream')!=r['upstream'] or b.get('sources')!=r['sources']:
        raise ValueError('Stage2B build/config/source/evidence mismatch')
    if installed(Path(b['opp']))!=b['installed']:raise ValueError('Stage2B installed OPP changed')
    r.update(build_report=identity(args.build_report),cann_compile=b['cann_compile'],installation=b['installation'])
    if args.group!='basic':r['basic_proof']=check_basic(args.basic_report,r['build_report'],r['upstream'],args.device)
    r['phase']='cpu_golden';cases=prepare(root/'data',r['upstream'],args.group);r['cases']=[]
    for case in cases:
        folder=root/'execute'/case['name'];folder.mkdir(parents=True)
        entry=dict(name=case['name'],case=identity(Path(case['files']['a_i8.bin']['path']).parent/'case.json'),directory=str(folder));r['cases'].append(entry)
        for name,item in case['files'].items():(folder/name).symlink_to(verify(item))
        r.update(phase='npu_'+case['name'],npu='RUNNING');write(root/'report.json',r)
        command([verify(b['runner']),args.device,folder],folder/'runner.log',env(args.cann_root,Path(b['opp'])),r,'case_'+case['name'],300)
        entry['execution']=identity(folder/'execution.json');ex=json.loads(verify(entry['execution']).read_text());check_execution(ex,args.device)
        log=(folder/'runner.log').read_text()
        if 'STAGE2B_PLAN arch=2002 M=16 K=128 N=64 blockDim=1 HostC=DT_FLOAT16 dequant=TENSOR' not in log or 'single=16/64/128 base=16/64/128' not in log:
            raise ValueError('actual Stage2B Host plan missing')
        entry['outputs']=[identity(folder/f'actual-{i}.bin') for i in range(2)]
        entry['comparison']=compare(case,[verify(f).read_bytes() for f in entry['outputs']])
        write(root/'report.json',r)
        if entry['comparison']['status']!='CANDIDATE_BITWISE_MATCH':raise ValueError('numerical mismatch; candidate codec/rounding NOT_CONFIRMED; stop without fallback')
    verify_tree(b)
    r.update(status='STAGE2B_'+args.group.upper()+'_MATCH',phase='complete',npu='MATCHED_SELECTED_CASES_ONLY',
        codec_validation='OBSERVED_MATCH_FOR_'+args.group.upper()+'_CASES_ONLY',
        host_ctype_validation='HOST_DT_FLOAT16_ACCEPTED_FOR_THIS_RUN')


def main(args):
    protected=[args.cann_root,args.stage1.parent,args.stage2a_index.parent,args.stage2a_cases.parent]
    protected += [p.parent for p in (args.build_report,args.basic_report) if p]
    root=fresh(args.output_dir,protected)
    r=dict(status='NOT_RUN',phase='evidence',stage='2B',group=args.group,device=args.device,sources=sources(),
       codec_status=CODEC,codec_validation='NOT_RUN',cann_compile='NOT_RUN',installation='NOT_RUN',npu='NOT_RUN',performance='NOT_RUN',
       settings=dict(op=NAME,shape=[16,128,64],A='INT8_ND',B='INT8_ND_NON_TRANSPOSED',O='FP16_ND',scale='UINT64_ND_64',
        HostC='DT_FLOAT16',dequant='TENSOR',block_dim=1,base=[16,64,128],codec='doc_mask13_bit46',rounding_model='FP32_product_then_FP16_RNE'),
       raw_stage1_rerun=False,k2560='NOT_RUN',full_gate='NOT_RUN',draft='NOT_RUN')
    try:
        r['upstream']=upstream(args);write(root/'report.json',r)
        if args.stage=='build':build(args,root,r)
        else:smoke(args,root,r)
        if upstream(args)!=r['upstream']:raise ValueError('upstream evidence changed during run')
    except Exception as e:
        r.update(status='FAILED',error=f'{type(e).__name__}: {e}')
        if r['cann_compile']=='RUNNING':r['cann_compile']='FAILED_OR_INCOMPLETE'
        if r['installation']=='RUNNING':r['installation']='FAILED_OR_INCOMPLETE'
        if r['npu']=='RUNNING':r['npu']='FAILED_OR_INCOMPLETE'
    finally:
        try:
            for f in r['sources']:verify(f)
            if 'upstream' in r:verify_tree(r['upstream'])
            for c in r.get('cases',[]):verify_tree(json.loads(verify(c['case']).read_text()))
        except Exception as e:r.update(status='FAILED',integrity_error=str(e))
        write(root/'report.json',r)
    print(r['status']+': '+str(root/'report.json'))
    return int(r['status']=='FAILED')


def parse():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('build','smoke'))
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--device',type=int,default=0)
    p.add_argument('--group',choices=tuple(GROUPS),default='basic');p.add_argument('--build-report',type=Path);p.add_argument('--basic-report',type=Path)
    p.add_argument('--cann-root',type=Path,default=Path(os.environ.get('CANN_ROOT','/usr/local/Ascend/cann-9.0.0')))
    p.add_argument('--stage1',type=Path,default=Path('/home/w00949577/qwen35-dflash-run/w8a8-raw-reviewed-20261026-060329/smoke-20261026-064554/report.json'))
    p.add_argument('--stage2a-index',type=Path,default=Path('/home/w00949577/qwen35-dflash-run/w8a8-stage2a-20261026-083731/sdk-index.json'))
    p.add_argument('--stage2a-cases',type=Path,default=Path('/home/w00949577/qwen35-dflash-run/w8a8-stage2a-20261026-083731/cpu-cases/manifest.json'))
    a=p.parse_args();a.cann_root=a.cann_root.resolve()
    if a.device<0:p.error('negative device')
    if a.stage=='build' and (a.group!='basic' or a.basic_report or a.build_report):p.error('build has no case/proof selection')
    return a

if __name__=='__main__':raise SystemExit(main(parse()))
