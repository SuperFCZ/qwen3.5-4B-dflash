#!/usr/bin/env python3
"""One FC shape, read-only PlatformInfo diagnostics. Never executes Matmul."""
import argparse,json,sys,traceback
from pathlib import Path
from platform_common import c,ec,HERE,sources
from run_atc import load_stage,bound_admission
from atc_evidence import payloads,option
from derive import verify
from platform_contracts import read_platform,validate_runtime,compare,loaded_libraries,verify_loaded,environment

STATES=dict(source='PLATFORM_DIAG_SOURCE_BOUND',build='PLATFORM_DIAG_OPP_RUNNER_BUILT',atc='PLATFORM_DIAG_ATC_OBSERVED',aclnn='PLATFORM_DIAG_ACLNN_HOST_OBSERVED',compare='PLATFORM_DIAG_COMPARISON_RECORDED')


def source(a):
    admitted=load_stage(a.admission_report,'admit');prior=bound_admission(admitted);e=prior['current']['evidence'];root=Path(prior['current']['cann_root']).resolve(strict=True)
    sdk=[]
    for row in c.read(HERE/'sdk_evidence.json')['files']:
        item=c.identity(root/row['relative'])
        if item['sha256']!=row['sha256'] or root not in Path(item['path']).parents:raise ValueError('SDK header identity differs: '+row['relative'])
        sdk.append(dict(file=item,lines=row['lines'],status='CURRENT_INSTALL_MATCHED_REVIEWED_LOCAL_SOURCE'))
    hroot=c.verify(e['H']['reports']['worker-input.json']).parent
    aggregate=hroot/'a-probes.json';summary=c.read(aggregate)
    cases=[x for x in summary['cases'] if x['name']=='c16-fc' and x['status']=='PASS']
    if summary['status']!='PASS' or len(cases)!=1:raise ValueError('historical single FC graph missing')
    folder=hroot/'a-c16-fc';row=cases[0]
    ap=c.linked(folder,row['artifacts']['air']);dp=c.linked(folder,row['artifacts']['deployment']);air=c.read(ap);dep=c.read(dp)
    files=payloads(ap,air);g=air['graphs'][0];control=g['metadata']['draft_ac']
    if control['scope']!='single-op-A' or (control['a_count'],control['c_count'],control.get('audit_count',0))!=(1,0,0):raise ValueError('not one A projection')
    abi=g['metadata']['tensor_abi']
    if abi!=dict(inputs=[dict(name='x',dtype='float16',shape=[16,12800])],outputs=[dict(name='y',dtype='float16',shape=[16,2560])]):raise ValueError('single FC ABI differs')
    if c.linked(dp.parent,dep['air_manifest'])!=ap or len(dep['graphs'])!=1:raise ValueError('AIR/Deployment binding')
    compiled=dep['graphs'][0]
    for key in ('metadata','runtime_input_abi','custom_op_audit'):
        if compiled[key]!=g[key]:raise ValueError('single graph audit differs')
    command=compiled['atc_command'];atc=e['tools']['atc']
    if c.identity(command[0])!=atc or option(command,'--model')!=str(c.linked(ap.parent,g['air'])):raise ValueError('single graph ATC identity')
    case=next(x for x in e['projections']['16'] if x['name']=='fc')
    bundle_path=Path(e['H']['state']['args']['a16_bundle']);bundle=c.read(bundle_path)
    fc=next(x for x in bundle['cases'] if x['name']=='fc')
    if control['projections']!=[dict(n=2560,k=12800,weight_sha256=fc['files']['q_nk.bin']['sha256'],scale_gn_sha256=fc['files']['s_gn.bin']['sha256'])]:raise ValueError('AIR weights/scales do not bind frozen FC inputs')
    for name,item in case['inputs'].items():
        if c.identity(c.linked(bundle_path.parent,fc['files'][name]))!=item:raise ValueError('FC tensor identity differs')
    om=c.linked(dp.parent,compiled['om'])
    if om!=c.linked(folder,row['artifacts']['om']):raise ValueError('historical single FC OM binding differs')
    if case['shape']!=[16,12800,2560]:raise ValueError('frozen FC shape differs')
    return dict(prior=prior,sdk=sdk,historical_fc=c.identity(aggregate),case=case,
        frozen=dict(om=c.identity(om),air=c.identity(ap),deployment=c.identity(dp),payloads=files,command=command,atc_binary=atc,compiler_identity_text=dep['compiler']['identity']))


def stage(path,name):
    r=c.read(path);c.verify_tree(r)
    verify_loaded(r.get('loaded_libraries',[]))
    if r.get('status')!=STATES[name] or r.get('error') or r.get('integrity_error') or r['sources']!=sources():raise ValueError('diagnostic predecessor/source mismatch')
    return r


def main(a):
    root=c.fresh(a.output_dir,[p.parent for p in (a.admission_report,a.source_report,a.build_report,a.atc_report,a.aclnn_report) if p])
    r=dict(status='BLOCKED',phase=a.phase,sources=sources(),npu='NOT_RUN',kernel_execute_calls=0,benchmark='NOT_RUN',full_draft='NOT_RUN')
    try:
        c.clean();verify()
        if a.phase=='source':
            r['binding']=source(a);r['admission_report']=c.identity(a.admission_report)
            b=r['binding'];env=c.selected_env(b['prior']['current']['evidence'],b['prior']['reports']['build']['variants']['B']['opp']['install'])
            r['atc_dependencies']=c.dependencies(b['frozen']['atc_binary'],root/'atc-ldd.log',env)
        else:
            src=stage(a.source_report,'source');r['source_report']=c.identity(a.source_report);binding=src['binding'];prior=binding['prior']
            if ec.previous(prior['root'])['references']!=prior['references']:raise ValueError('old predecessor drift')
            if a.phase=='build':
                req=root/'request.json';dest=root/'package'
                c.write(req,dict(output=str(dest),prior=prior,variant='B',sources=sources()))
                c.command([sys.executable,'-B',HERE/'build_worker.py',req],root/'build.log',c.selected_env(prior['current']['evidence'],dest/'opp'))
                built=c.read(dest/'report.json')
                if built.get('error') or built.get('integrity_error') or built.get('status')!='PLATFORM_DIAG_HOST_DSOS_BUILT_NPU_NOT_RUN':raise ValueError('diagnostic Host build failed')
                r['package']=built;r['build_worker']=c.identity(dest/'report.json')
                env=c.selected_env(prior['current']['evidence'],built['opp']['install'])
                c.command(['cmake','-S',HERE/'runner','-B',root/'runner-build','-DCANN_ROOT='+prior['current']['cann_root'],'-DCOMMON_INCLUDE='+str(c.QUANT/'test'),'-DPRIVATE_OPP='+built['opp']['install']],root/'runner-configure.log',env)
                c.command(['cmake','--build',root/'runner-build','--parallel','2'],root/'runner-build.log',env)
                r['runner']=c.identity(root/'runner-build/platform_diag_runner');r['dependencies']=c.dependencies(r['runner'],root/'runner-ldd.log',env);r['environment_contract']=environment(env)
            else:
                built=stage(a.build_report,'build');r['build_report']=c.identity(a.build_report)
                if built['source_report']!=r['source_report']:raise ValueError('mixed source/build')
                opp=built['package']['opp'];c.verify_opp(opp);env=c.selected_env(prior['current']['evidence'],opp['install'])
                env.pop('DFLASH_GROUP_QUANT_EVIDENCE_PATH',None)
                env.pop('DFLASH_PLATFORM_DIAG_PATH',None);env.pop('DFLASH_PLATFORM_DIAG_ROUTE',None)
                if environment(env)!=built['environment_contract']:raise ValueError('ATC/ACLNN selection environment differs from diagnostic build')
                r['environment_contract']=environment(env)
                if a.phase=='atc':
                    req=root/'request.json';dest=root/'fc'
                    c.write(req,dict(output=str(dest),frozen=binding['frozen'],evidence=prior['current']['evidence'],opp=opp,sources=sources()))
                    env.update(AI_RUN_DIR=str(root),ASCEND_WORK_PATH=str(dest))
                    try:c.command([sys.executable,'-B',HERE/'atc_worker.py',req],root/'atc-worker.log',env)
                    finally:
                        if (dest/'report.json').is_file():
                            r['worker']=c.identity(dest/'report.json');r['captures']=c.read(dest/'report.json').get('captures',[])
                    child=c.read(dest/'report.json')
                    if child.get('status')!='SINGLE_FC_ATC_COMPILED_NOT_EXECUTED' or child.get('integrity_error'):raise ValueError('single FC ATC failed')
                    r['worker']=c.identity(dest/'report.json');r['captures']=child['captures'];r['platform']=read_platform(c.verify(child['platform']),'ATC',opp)
                elif a.phase=='aclnn':
                    out=root/'fc';out.mkdir();diag=out/'platform.jsonl';diag.touch(exist_ok=False)
                    for name,item in binding['case']['inputs'].items():(out/name).symlink_to(c.verify(item))
                    env.update(DFLASH_PLATFORM_DIAG_PATH=str(diag),DFLASH_PLATFORM_DIAG_ROUTE='ACLNN')
                    if c.dependencies(built['runner'],root/'runner-ldd.log',env)!=built['dependencies']:raise ValueError('Runner library identity drift')
                    try:c.command([c.verify(built['runner']),0,out],out/'runner.log',env)
                    finally:r['retained_runtime']=[c.identity(out/name) for name in ('runner.log','runtime-events.jsonl','execution.json','maps.txt','platform.jsonl') if (out/name).is_file()]
                    result=c.read(out/'execution.json')
                    journal=[json.loads(x) for x in (out/'runtime-events.jsonl').read_text().splitlines()]
                    validate_runtime(result,journal)
                    r['loaded_libraries']=loaded_libraries(out/'maps.txt',opp,Path(prior['current']['cann_root']))
                    r.update(maps=c.identity(out/'maps.txt'),execution=c.identity(out/'execution.json'),runtime_events=c.identity(out/'runtime-events.jsonl'),platform=read_platform(diag,'ACLNN',opp))
                    if result['workspace']<r['platform']['observation']['workspace']:raise ValueError('ACLNN workspace below Host request')
                elif a.phase=='compare':
                    atc=stage(a.atc_report,'atc');runtime=stage(a.aclnn_report,'aclnn')
                    if any(x['build_report']!=r['build_report'] or x['source_report']!=r['source_report'] for x in (atc,runtime)):raise ValueError('route comparison identity differs')
                    for route,observed in (('ATC',atc),('ACLNN',runtime)):
                        if read_platform(c.verify(observed['platform']['source']),route,opp)!=observed['platform']:raise ValueError('saved PlatformInfo is not the bound raw observation')
                    journal=[json.loads(x) for x in c.verify(runtime['runtime_events']).read_text().splitlines()]
                    validate_runtime(c.read(c.verify(runtime['execution'])),journal)
                    if loaded_libraries(c.verify(runtime['maps']),opp,Path(prior['current']['cann_root']))!=runtime['loaded_libraries']:raise ValueError('saved loaded libraries differ from raw maps')
                    r['atc_report']=c.identity(a.atc_report);r['aclnn_report']=c.identity(a.aclnn_report)
                    r['comparison']=compare(atc['platform'],runtime['platform'])
                c.verify_opp(opp)
        r['status']=STATES[a.phase]
    except Exception as e:r['error']=f'{type(e).__name__}: {e}';(root/'traceback.txt').write_text(traceback.format_exc())
    finally:
        try:
            c.verify_tree(r)
            verify_loaded(r.get('loaded_libraries',[]))
            if r['sources']!=sources():raise ValueError('diagnostic source changed')
            c.clean()
        except Exception as e:r.update(status='BLOCKED',integrity_error=str(e))
        c.write(root/'report.json',r)
    fields={k:r.get(k) for k in ('status','phase','error','integrity_error','npu','kernel_execute_calls','comparison')}
    if 'platform' in r:fields['platform']=r['platform']['observation']
    print(json.dumps(fields,indent=2,ensure_ascii=False));print('REPORT:',root/'report.json');return int(r['status']=='BLOCKED')


def parse(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('phase',choices=STATES);p.add_argument('--output-dir',type=Path,required=True)
    for key in ('admission-report','source-report','build-report','atc-report','aclnn-report'):p.add_argument('--'+key,type=Path)
    a=p.parse_args(argv)
    needed={'source':['admission_report'],'build':['source_report'],'atc':['source_report','build_report'],'aclnn':['source_report','build_report'],'compare':['source_report','build_report','atc_report','aclnn_report']}[a.phase]
    for key in needed:
        if getattr(a,key) is None:p.error('--'+key.replace('_','-')+' required')
    for k,v in vars(a).items():
        if isinstance(v,Path):setattr(a,k,v.resolve())
    return a


if __name__=='__main__':raise SystemExit(main(parse()))
