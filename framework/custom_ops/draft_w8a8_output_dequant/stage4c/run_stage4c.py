#!/usr/bin/env python3
"""Runner-only build, guarded Smoke, unprofiled baseline, then independent msprof."""
import argparse,json,os,platform,shlex,shutil
from pathlib import Path
from bench_common import *
from profile_metrics import collect
from run_stage2b import command


def environment(proof,cann):
    env=common.env(cann,Path(proof['opp']))
    env['DFLASH_STAGE4B_EXPECTED_CORE_POLICY']=','.join(str(proof['policy'][k]) for k in ('aic','aiv','all','selected','arch'))
    return env


def machine(root,proof,cann,r):
    result=dict(device=r['device'],required_soc='Ascend310P3',os=platform.uname()._asdict(),cann_root=str(cann),
                cann_identity=proof['preflight'],version_files=[],driver='UNAVAILABLE',npu_smi='UNAVAILABLE')
    for path in (Path('/usr/local/Ascend/driver/version.info'),cann/'version.cfg',cann/'version.info',cann/'aarch64-linux/ascend_toolkit_install.info'):
        if path.is_file():
            item=identity(path);result['version_files'].append(dict(file=item,text=path.read_text(errors='replace')[:16384]))
            if path==Path('/usr/local/Ascend/driver/version.info'):result['driver']='VERSION_FILE_RECORDED'
    smi=shutil.which('npu-smi')
    if smi:
        result['npu_smi_tool']=identity(smi)
        try:
            command([smi,'info'],root/'npu-smi.log',environment(proof,cann),r,'npu_smi',60)
            result['npu_smi']=identity(root/'npu-smi.log')
        except Exception as error:result['npu_smi_error']=str(error)
    return result


def build(args,root,r):
    if platform.system()!='Linux' or platform.machine() not in ('aarch64','arm64'):raise ValueError('Linux aarch64 CANN receiver required')
    proof=r['evidence'];pf=json.loads(verify(proof['preflight']).read_text())
    for name in ('cmake','c++'):
        tool=shutil.which(name)
        if not tool or identity(tool)!=pf['tools'][name]['file']:raise ValueError('compiler differs from accepted environment')
    env=environment(proof,args.cann_root);r['phase']='runner_build'
    command(['cmake','-S',HERE/'test','-B',root/'runner-build','-DCANN_ROOT='+str(args.cann_root),'-DOPP_ROOT='+proof['opp'],
        '-DCOMMON_INCLUDE='+str(common.QUANT/'test'),'-DCMAKE_EXPORT_COMPILE_COMMANDS=ON'],root/'cmake.log',env,r,'configure')
    command(['cmake','--build',root/'runner-build','--parallel'],root/'runner-build.log',env,r,'runner_build')
    runner=root/'runner-build/stage4c_runner';command(['ldd',runner],root/'ldd.log',env,r,'ldd')
    if 'not found' in (root/'ldd.log').read_text():raise ValueError('missing Runner dependency')
    links={}
    for line in (root/'runner-build/CMakeCache.txt').read_text().splitlines():
        if line.startswith('LIB_') and ':FILEPATH=' in line:
            key,value=line.split(':FILEPATH=',1);item=identity(value)
            expected=Path(proof['opp']) if key=='LIB_custom' else args.cann_root
            if expected.resolve() not in Path(item['path']).parents:raise ValueError('wrong linked library')
            links[key]=item
    if set(links)!={'LIB_custom','LIB_ascendcl','LIB_nnopbase','LIB_acl_op_compiler'}:raise ValueError('missing link identities')
    r.update(status='STAGE4C_BENCHMARK_PREPARED',phase='complete',runner_compile='COMPLETED',runner=identity(runner),
        linked_libraries=links,compile_commands=identity(root/'runner-build/compile_commands.json'),
        kernel_build='NOT_RUN_REUSED_STAGE4B',opp_installation='NOT_RUN_REUSED_STAGE4B')


def accept_build(args,r):
    if args.runner_build_report is None:raise ValueError('--runner-build-report required')
    b=json.loads(args.runner_build_report.read_text());verify_tree(b)
    if b.get('status')!='STAGE4C_BENCHMARK_PREPARED' or any(b.get(k)!=r[k] for k in ('sources','evidence','device')):raise ValueError('benchmark build binding differs')
    r['runner_build_report']=identity(args.runner_build_report)
    return b


def invoke(args,folder,r,b,limit,timed,profile=False):
    folder.mkdir(parents=True);proof=r['evidence']
    for name in ('a_i8.bin','b_i8.bin','quant_u64_doc_candidate.bin','golden_fp16.bin'):(folder/name).symlink_to(verify(proof['files'][name]))
    argv=[verify(b['runner']),args.device,folder,limit]+(['--benchmark'] if timed else [])
    env=environment(proof,args.cann_root)
    if profile:
        profiler=shutil.which(args.msprof)
        if not profiler:raise ValueError('msprof unavailable')
        launcher=folder.parent/'launch.sh';launcher.write_text('#!/usr/bin/env bash\nset -euo pipefail\nexec '+shlex.join(list(map(str,argv)))+' > '+shlex.quote(str(folder/'runner.log'))+' 2>&1\n')
        profile_argv=[profiler,'--ai-core=on','--aic-mode=task-based','--aic-metrics=PipeUtilization',
                      '--output='+str(folder.parent/'profiler'),'--application='+shlex.join(['bash',str(launcher)])]
        r['profiler']=identity(profiler);r['profile_command']=profile_argv
        command(profile_argv,folder.parent/'profile.log',env,r,'msprof',1800)
    else:command(argv,folder/'runner.log',env,r,str(folder.relative_to(Path(r['output_dir']))),600)
    ex_id=identity(folder/'execution.json');ex=json.loads(verify(ex_id).read_text());stats=check_execution(ex,args.device,limit,timed)
    outputs=verify_outputs(proof,folder,timed)
    plan=full.plan_from_log((folder/'runner.log').read_text(),0,limit)
    if plan['core_policy']!=proof['policy']:raise ValueError('Core policy changed')
    admission=identity(folder/'resource-admission.json')
    if json.loads(verify(admission).read_text())!=stage4b.admission_expected(ex,plan,False):raise ValueError('resource admission differs')
    return dict(status='FROZEN_GOLDEN_MATCH',scheme='W8A8_FROZEN_STAGE4B',core_limit=limit,plan=plan,execution=ex_id,outputs=outputs,
                admission=admission,workspace_bytes=ex['workspace_bytes'],timing=stats,profiled=profile,
                formal_latency_evidence=timed and not profile)


def prior_report(path,status,r):
    if path is None:raise ValueError('required completed stage report missing: '+status)
    p=json.loads(path.read_text());verify_tree(p)
    if p.get('status')!=status or p.get('phase')!='complete' or any(p.get(k)!=r[k] for k in ('sources','evidence','device','runner_build_report')):raise ValueError('prior stage scope/build/evidence differs')
    expected=[1,2,0] if status=='STAGE4C_SMOKE_MATCH' else [x for row in ORDER for x in row]
    if [e['core_limit'] for e in p.get('runs',[])]!=expected:raise ValueError('prior run coverage incomplete')
    for e in p['runs']:
        if e.get('status')!='FROZEN_GOLDEN_MATCH' or e.get('profiled') is not False:raise ValueError('prior correctness/profile scope invalid')
        timed=status!='STAGE4C_SMOKE_MATCH'
        ex=json.loads(verify(e['execution']).read_text());stats=check_execution(ex,r['device'],e['core_limit'],timed)
        folder=Path(e['execution']['path']).parent
        if stats!=e['timing'] or e.get('formal_latency_evidence') is not timed or verify_outputs(r['evidence'],folder,timed)!=e['outputs']:raise ValueError('prior samples/output bindings differ')
        plan=full.plan_from_log((folder/'runner.log').read_text(),0,e['core_limit'])
        if plan!=e['plan'] or plan['core_policy']!=r['evidence']['policy'] or json.loads(verify(e['admission']).read_text())!=stage4b.admission_expected(ex,plan,False):raise ValueError('prior resource proof differs')
    if status=='STAGE4C_BASELINE_MEASURED' and [(e.get('round'),e.get('position')) for e in p['runs']]!=[(i,j) for i in range(3) for j in range(3)]:raise ValueError('round/order identities differ')
    return p


def main(args):
    if args.stage=='profile' and args.benchmark_report and args.runner_build_report is None:
        args.runner_build_report=verify(json.loads(args.benchmark_report.read_text())['runner_build_report'])
    protected=[args.cann_root,args.stage4b_root,args.stage4a_root]
    protected += [p.parent for p in (args.runner_build_report,args.smoke_report,args.benchmark_report,args.a31_suite) if p]
    root=fresh(args.output_dir,protected)
    r=dict(stage='4C',status='NOT_RUN',phase='evidence',device=args.device,sources=sources(),output_dir=str(root),
        npu='NOT_RUN',performance='NOT_RUN',end_to_end='NOT_RUN',online_preprocessing='ONLINE_PREPROCESSING_NOT_RUN',
        kernel_modified=False,opp_modified=False,reference_policy='unchanged Stage4B Golden')
    comparison=dict(status='NOT_RUN');prof=dict(status='NOT_RUN',formal_latency_evidence=False);inventory=dict(status='ONLINE_PREPROCESSING_NOT_RUN')
    try:
        r['evidence']=evidence(args);inventory=preprocessing(r['evidence'])
        comparison=a31_readiness(args.a31_suite,r['evidence']);r['machine']=machine(root,r['evidence'],args.cann_root,r)
        if args.stage=='prepare':r.update(status='STAGE4C_EVIDENCE_PREPARED',phase='complete')
        elif args.stage=='build':build(args,root,r)
        else:
            b=accept_build(args,r);r['runs']=[]
            if args.stage=='smoke':
                for limit in LIMITS:
                    r.update(phase=f'smoke_limit{limit}',npu='RUNNING');write(root/'benchmark-report.json',r)
                    r['runs'].append(invoke(args,root/f'limit-{limit}',r,b,limit,False))
                r.update(status='STAGE4C_SMOKE_MATCH',phase='complete',npu='MATCHED_UNTIMED_CHECKS_ONLY')
            elif args.stage=='benchmark':
                prior_report(args.smoke_report,'STAGE4C_SMOKE_MATCH',r);r['smoke_report']=identity(args.smoke_report)
                for round_id,order in enumerate(ORDER):
                    for position,limit in enumerate(order):
                        r.update(phase=f'round{round_id}_limit{limit}',npu='RUNNING');write(root/'benchmark-report.json',r)
                        item=invoke(args,root/f'round-{round_id}'/f'limit-{limit}',r,b,limit,True)
                        item.update(round=round_id,position=position);r['runs'].append(item)
                r['summary']={str(limit):{kind:statistics_ms([v for e in r['runs'] if e['core_limit']==limit for v in e['timing'][kind]['samples_ms']],90)
                    for kind in ('execute_sync','prepare')} for limit in LIMITS}
                for limit in LIMITS:
                    blocks={e['plan']['blocks'] for e in r['runs'] if e['core_limit']==limit}
                    if len(blocks)!=1:raise ValueError('actual launch count drift across rounds')
                    r['summary'][str(limit)].update(actual_blocks=blocks.pop(),rounds=3,samples_per_round=30,
                        scope='preprocessed resident inputs; frozen in-kernel guards included; not end-to-end')
                r.update(status='STAGE4C_BASELINE_MEASURED',phase='complete',npu='FROZEN_GOLDEN_MATCH',performance='UNPROFILED_W8A8_RESIDENT_INPUTS_ONLY')
            else:
                baseline=prior_report(args.benchmark_report,'STAGE4C_BASELINE_MEASURED',r)
                r['benchmark_report']=identity(args.benchmark_report);r.update(phase='profile_auto',npu='RUNNING')
                result=invoke(args,root/'profile'/'case',r,b,0,True,True);raw=collect(root/'profile'/'profiler')
                prof=dict(status='STAGE4C_PROFILE_COMPLETED' if raw['status']=='RAW_OPERATOR_METRICS_FOUND' else 'PROFILE_COLLECTED_METRICS_UNAVAILABLE',
                    benchmark_report=r['benchmark_report'],execution=result,raw=raw,profiler=r['profiler'],command=r['profile_command'],
                    static_tile_distribution=result['plan']['owners'],formal_latency_evidence=False,
                    bottleneck='UNRESOLVED until actual counter/timeline interpretation; pipe activity is not additive wall time')
                r.update(status=prof['status'],phase='complete',npu='PROFILE_GOLDEN_MATCH',performance='PROFILE_EXCLUDED_FROM_FORMAL_LATENCY')
    except Exception as error:
        r.update(status='FAILED',error=f'{type(error).__name__}: {error}')
        if r['npu']=='RUNNING':r['npu']='FAILED_OR_INCOMPLETE'
        if args.stage=='profile':prof.update(status='FAILED',error=r['error'])
    finally:
        try:
            for item in r['sources']:verify(item)
            if 'evidence' in r:
                verify_tree(r['evidence'])
                if installed(Path(r['evidence']['opp']))!=r['evidence']['installed']:raise ValueError('frozen OPP changed')
            for item in r.get('machine',{}).get('version_files',[]):verify(item['file'])
        except Exception as error:r.update(status='FAILED',integrity_error=str(error))
        if r['status']=='FAILED' and args.stage=='profile':prof.update(status='FAILED',error=r.get('integrity_error',r.get('error','profile validation failed')))
        r['artifacts']=[identity(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name in ('execution.json','runner.log','actual-0.bin','actual-1.bin','benchmark-last.bin','postcheck.bin')]
        write(root/'benchmark-report.json',r);write(root/'comparison-report.json',comparison)
        write(root/'profile-report.json',prof);write(root/'preprocessing-inventory.json',inventory)
    print(r['status']+': '+str(root/'benchmark-report.json'));return int(r['status']=='FAILED')


def parse():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('prepare','build','smoke','benchmark','profile'))
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--runner-build-report',type=Path);p.add_argument('--smoke-report',type=Path);p.add_argument('--benchmark-report',type=Path)
    p.add_argument('--msprof',default='msprof');p.add_argument('--a31-suite',type=Path,default=Path('/home/w00949577/z50058744/qwen3.5-4B-dflash/framework/custom_ops/draft_quant/.runs/a31-KEg03LMS/data/suite.json'))
    for name,default in [('stage4b','w8a8-full-gate-20261027-093617'),('stage4a','w8a8-core-policy-20261027-083223'),('stage3b4','w8a8-twenty-group-20261027-055348'),('stage3b3','w8a8-two-group-20261027-042044'),('stage3b2','w8a8-local-nz-20261027-031856'),('stage3b1','w8a8-stage3b-20261027-024844'),('stage3a','w8a8-stage3a-20261027-011523'),('stage2b','w8a8-stage2b-20261026-100533')]:
        p.add_argument('--'+name+'-root',type=Path,default=Path('/home/w00949577/qwen35-dflash-run')/default)
    for name in ('build','basic','zero','nonpower'):p.add_argument('--'+name+'-report',type=Path)
    p.add_argument('--stage2c-report',type=Path,default=Path('/home/w00949577/qwen35-dflash-run/w8a8-stage2c-20261026-104943/report.json'))
    p.add_argument('--device',type=int,default=0);p.add_argument('--cann-root',type=Path,default=Path(os.environ.get('CANN_ROOT','/usr/local/Ascend/cann-9.0.0')))
    args=p.parse_args();args.cann_root=args.cann_root.resolve()
    for name in ('build','basic','zero','nonpower'):
        if getattr(args,name+'_report') is None:setattr(args,name+'_report',args.stage2b_root/name/'report.json')
    if args.device<0:p.error('negative device')
    return args

if __name__=='__main__':raise SystemExit(main(parse()))
