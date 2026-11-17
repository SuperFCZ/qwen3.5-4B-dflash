#!/usr/bin/env python3
"""One-shape A/B Host-only scheduler research. No Kernel Execute entrypoint."""
import argparse,json,sys,traceback
from pathlib import Path
from scheduler_common import c,HERE,sources,borrowed_sources
from scheduler_derive import verify
from scheduler_evidence import baseline,verify_prefix_review
from scheduler_contracts import read_query,compare_pair
import platform_runtime_env as envs
import dso_manifest as dsos
from platform_contracts import validate_runtime,loaded_libraries,verify_loaded

STATES=dict(source='CORE_SCHEDULER_SOURCE_REVIEWED',build='CORE_SCHEDULER_HOST_OPPS_BUILT',atc='CORE_SCHEDULER_ATC_HOST_OBSERVED',aclnn='CORE_SCHEDULER_ACLNN_HOST_OBSERVED',compare='CORE_SCHEDULER_HOST_COMPARISON_RECORDED')


def stage(path,phase):
    r=c.read(path);c.verify_tree(r)
    if phase=='source':verify_prefix_review(r['binding']['prefix_review'])
    if r.get('status')!=STATES[phase] or r.get('error') or r.get('integrity_error') or r['sources']!=sources() or r['borrowed_sources']!=borrowed_sources():raise ValueError('scheduler predecessor/source mismatch')
    return r


def base_platform(query,expected,variant):
    p=query['platform'];e=expected['observation']
    for key,value in e.items():
        if key in ('host_dso','blockDim'):continue
        if p[key]!=value:raise ValueError('base PlatformInfo changed: '+key)
    if variant=='A' and p['blockDim']!=e['blockDim']:raise ValueError('control launch policy differs from baseline')


def verify_environment(env,binding):
    extra=env.get('DFLASH_SCHEDULER_EVIDENCE_PATH')
    if not extra or not Path(extra).is_absolute():raise ValueError('missing independent scheduler evidence path')
    envs.assert_invocation({k:v for k,v in env.items() if k!='DFLASH_SCHEDULER_EVIDENCE_PATH'},binding)


def main(a):
    protected=[p for p in (a.baseline_root,) if p]+[p.parent for p in (a.source_report,a.build_report,a.atc_report,a.aclnn_report) if p]
    root=c.fresh(a.output_dir,protected);r=dict(status='BLOCKED',phase=a.phase,sources=sources(),borrowed_sources=borrowed_sources(),npu='NOT_RUN',execute_calls=0,benchmark='NOT_RUN',full_draft='NOT_RUN')
    try:
        c.clean();verify()
        if a.phase=='source':r['binding']=baseline(a.baseline_root)
        else:
            src=stage(a.source_report,'source');r['source_report']=c.identity(a.source_report);binding=src['binding'];prior=binding['prior']
            for item in binding['references'].values():c.verify(item)
            standard=envs.standard_identity(binding['runtime_environment']);sm=c.read(c.verify(binding['standard_manifest']));dsos.verify_manifest(sm)
            if a.phase=='build':
                r['variants']={}
                for variant in ('A','B'):
                    out=root/variant;out.mkdir();package=out/'package';env,env_binding=envs.for_prior(prior,package/'opp',ready=False)
                    if envs.standard_identity(env_binding)!=standard:raise ValueError('standard environment changed')
                    request=out/'request.json';c.write(request,dict(output=str(package),prior=prior,variant=variant,sources=sources(),borrowed_sources=borrowed_sources()))
                    c.command([sys.executable,'-B',HERE/'scheduler_build_worker.py',request],out/'build.log',env)
                    built=c.read(package/'report.json')
                    if built.get('status')!='SCHEDULER_HOST_DSOS_BUILT_NPU_NOT_RUN' or built.get('error') or built.get('integrity_error'):raise ValueError('Host OPP build failed: '+variant)
                    env,env_binding=envs.for_prior(prior,built['opp']['install'])
                    c.command(['cmake','-S',HERE/'runner','-B',out/'runner-build','-DCANN_ROOT='+prior['current']['cann_root'],'-DCOMMON_INCLUDE='+str(c.QUANT/'test'),'-DPRIVATE_OPP='+built['opp']['install']],out/'configure.log',env)
                    c.command(['cmake','--build',out/'runner-build','--parallel','2'],out/'compile.log',env)
                    runner=c.identity(out/'runner-build/platform_diag_runner');deps=c.dependencies(runner,out/'ldd.log',env)
                    manifest=dsos.with_private(sm,built['opp'],binding['standard_manifest']);c.write(out/'dso-manifest.json',manifest)
                    r['variants'][variant]=dict(package=built,package_report=c.identity(package/'report.json'),runner=runner,dependencies=deps,runtime_environment=env_binding,dso_manifest=c.identity(out/'dso-manifest.json'))
                for x in r['variants'].values():
                    if not x['package']['opp_comparison']['all_kernel_bytes_equal']:raise ValueError('frozen Kernel changed')
            else:
                built=stage(a.build_report,'build');r['build_report']=c.identity(a.build_report)
                if built['source_report']!=r['source_report']:raise ValueError('mixed Source/Build')
                if a.phase in ('atc','aclnn'):
                    r['variants']={}
                    for variant in ('A','B'):
                        b=built['variants'][variant];opp=b['package']['opp'];c.verify_opp(opp)
                        env,env_binding=envs.for_prior(prior,opp['install']);envs.verify_binding(b['runtime_environment'])
                        if env_binding['values']!=b['runtime_environment']['values'] or envs.standard_identity(env_binding)!=standard:raise ValueError('runtime environment drift')
                        manifest=c.read(c.verify(b['dso_manifest']));dsos.verify_manifest(manifest)
                        out=root/variant;out.mkdir();item=dict(runtime_environment=env_binding,opp=opp,dso_manifest=b['dso_manifest']);r['variants'][variant]=item
                        if a.phase=='atc':
                            dest=out/'fc';env.update(AI_RUN_DIR=str(out),ASCEND_WORK_PATH=str(dest))
                            request=out/'request.json';c.write(request,dict(output=str(dest),frozen=binding['frozen'],evidence=prior['current']['evidence'],opp=opp,environment=env,runtime_environment=env_binding,sources=sources(),borrowed_sources=borrowed_sources()))
                            try:c.command([sys.executable,'-B',HERE/'scheduler_atc_worker.py',request],out/'worker.log',env)
                            finally:
                                if (dest/'report.json').is_file():item['worker_report']=c.identity(dest/'report.json');item['captures']=c.read(dest/'report.json').get('captures',[])
                            child=c.read(dest/'report.json')
                            if child.get('status')!='SCHEDULER_SINGLE_FC_ATC_OBSERVED' or child.get('error') or child.get('integrity_error'):raise ValueError('ATC Host query failed: '+variant)
                            item.update(child_environment=child['child_environment'],elf_environment=child['elf_environment'])
                        else:
                            dest=out/'fc';dest.mkdir();platform=dest/'platform.jsonl';platform.touch(exist_ok=False);sched=dest/'scheduling.jsonl';sched.touch(exist_ok=False)
                            for name,file in binding['case']['inputs'].items():(dest/name).symlink_to(c.verify(file))
                            env.update(DFLASH_PLATFORM_DIAG_PATH=str(platform),DFLASH_PLATFORM_DIAG_ROUTE='ACLNN',DFLASH_SCHEDULER_EVIDENCE_PATH=str(sched));verify_environment(env,env_binding)
                            item['child_environment']=dict(env)
                            if c.dependencies(b['runner'],out/'ldd.log',env)!=b['dependencies']:raise ValueError('Runner dependency drift')
                            try:c.command([c.verify(b['runner']),0,dest],dest/'runner.log',env)
                            finally:item['retained']=[c.identity(dest/n) for n in ('platform.jsonl','scheduling.jsonl','execution.json','runtime-events.jsonl','maps.txt','runner.log') if (dest/n).is_file()]
                            result=c.read(dest/'execution.json');journal=[json.loads(x) for x in (dest/'runtime-events.jsonl').read_text().splitlines()];validate_runtime(result,journal)
                            item.update(execution=c.identity(dest/'execution.json'),journal=c.identity(dest/'runtime-events.jsonl'),maps=c.identity(dest/'maps.txt'),loaded_libraries=loaded_libraries(dest/'maps.txt',opp,Path(prior['current']['cann_root']),manifest))
                        item['query']=read_query(dest/'platform.jsonl',dest/'scheduling.jsonl',opp,a.phase.upper(),variant)
                        if a.phase=='aclnn' and (result['workspace']!=binding['aclnn_workspace'] or result['workspace']!=item['query']['schedule']['resources']['workspace']):raise ValueError('actual ACLNN Workspace changed')
                        base_platform(item['query'],binding['observed'][a.phase],variant);verify_environment(item['child_environment'],env_binding);c.verify_opp(opp)
                    r['pair']=compare_pair(r['variants']['A']['query'],r['variants']['B']['query'])
                else:
                    atc=stage(a.atc_report,'atc');runtime=stage(a.aclnn_report,'aclnn');r['atc_report']=c.identity(a.atc_report);r['aclnn_report']=c.identity(a.aclnn_report)
                    for x in (atc,runtime):
                        if x['source_report']!=r['source_report'] or x['build_report']!=r['build_report']:raise ValueError('mixed query reports')
                    r['routes']={}
                    for route,report in (('ATC',atc),('ACLNN',runtime)):
                        for variant,item in report['variants'].items():
                            q=item['query'];actual=read_query(c.verify(q['platform_file']),c.verify(q['scheduling_file']),item['opp'],route,variant)
                            if actual!=q:raise ValueError('query/report disagreement')
                            verify_environment(item['child_environment'],item['runtime_environment']);dsos.verify_manifest(c.read(c.verify(item['dso_manifest'])))
                            if route=='ACLNN':
                                result=c.read(c.verify(item['execution']));validate_runtime(result,[json.loads(x) for x in c.verify(item['journal']).read_text().splitlines()])
                                if result['workspace']!=binding['aclnn_workspace'] or result['workspace']!=q['schedule']['resources']['workspace']:raise ValueError('actual ACLNN Workspace differs from sealed baseline')
                                if loaded_libraries(c.verify(item['maps']),item['opp'],Path(prior['current']['cann_root']),c.read(c.verify(item['dso_manifest'])))!=item['loaded_libraries']:raise ValueError('mapped identity drift')
                        r['routes'][route]=compare_pair(report['variants']['A']['query'],report['variants']['B']['query'])
                    r['cross_route']={}
                    for variant in ('A','B'):
                        x=atc['variants'][variant];y=runtime['variants'][variant]
                        if x['query']['host_dso']!=y['query']['host_dso']:raise ValueError('route Host DSO mismatch')
                        if x['query']['schedule']['resources']!=y['query']['schedule']['resources'] or x['query']['tiling_sha256']!=y['query']['tiling_sha256']:raise ValueError('ATC/ACLNN Tiling or resource mismatch')
                        ee=envs.compare_invocations({k:v for k,v in x['child_environment'].items() if k!='DFLASH_SCHEDULER_EVIDENCE_PATH'},{k:v for k,v in y['child_environment'].items() if k!='DFLASH_SCHEDULER_EVIDENCE_PATH'})
                        r['cross_route'][variant]=dict(ATC=x['query']['schedule']['selected_block_dim'],ACLNN=y['query']['schedule']['selected_block_dim'],environment=ee,resource_map_differences={k:dict(ATC=x['query']['platform']['resource_map'].get(k),ACLNN=y['query']['platform']['resource_map'].get(k)) for k in set(x['query']['platform']['resource_map'])|set(y['query']['platform']['resource_map']) if x['query']['platform']['resource_map'].get(k)!=y['query']['platform']['resource_map'].get(k)})
                    r['decision']='HOST_ONLY_CANDIDATE_RECORDED_DEVICE_SMOKE_REQUIRES_SEPARATE_APPROVAL'
                    r['remaining']='Runtime PlatFormInfos generic Core getter producer/selection and entitlement are not established by CalcTschBlockDim arithmetic; no performance inference'
        r['status']=STATES[a.phase]
    except Exception as e:r['error']=f'{type(e).__name__}: {e}';(root/'traceback.txt').write_text(traceback.format_exc())
    finally:
        try:
            c.verify_tree(r)
            if 'binding' in r:verify_prefix_review(r['binding']['prefix_review'])
            if r['sources']!=sources() or r['borrowed_sources']!=borrowed_sources():raise ValueError('source changed')
            for item in r.get('variants',{}).values():
                if 'loaded_libraries' in item:verify_loaded(item['loaded_libraries'])
            c.clean()
        except Exception as e:r.update(status='BLOCKED',integrity_error=str(e))
        c.write(root/'report.json',r)
    small={k:r.get(k) for k in ('status','phase','error','integrity_error','pair','routes','cross_route','decision','npu','execute_calls')}
    print(json.dumps(small,indent=2));print('REPORT:',root/'report.json');return int(r['status']=='BLOCKED')


def parse(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False);p.add_argument('phase',choices=STATES);p.add_argument('--output-dir',type=Path,required=True)
    for key in ('baseline-root','source-report','build-report','atc-report','aclnn-report'):p.add_argument('--'+key,type=Path)
    a=p.parse_args(argv);needs={'source':['baseline_root'],'build':['source_report'],'atc':['source_report','build_report'],'aclnn':['source_report','build_report'],'compare':['source_report','build_report','atc_report','aclnn_report']}[a.phase]
    for key in needs:
        if getattr(a,key) is None:p.error('--'+key.replace('_','-')+' required')
    for key,value in vars(a).items():
        if isinstance(value,Path):setattr(a,key,value.resolve())
    return a


if __name__=='__main__':raise SystemExit(main(parse()))
