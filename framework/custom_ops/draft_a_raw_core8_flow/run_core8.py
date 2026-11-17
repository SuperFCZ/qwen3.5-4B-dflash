#!/usr/bin/env python3
"""Explicit independent eight-core stages; never auto-run NPU or performance."""
import argparse
import sys
import traceback
from pathlib import Path
from c8_common import c,ec,HERE,SIDECAR,legacy,sources
import assets8
import probes8
from resource8 import compare
from derive_export import verify as verify_worker
from raw_protocol import verify_run,order,summarize
from run_experiment import run_om
from run_environment import check_workers

STATES=dict(admit='A_RAW_CORE8_ADMITTED_BC16_REUSED',build='A_RAW_CORE8_PACKAGES_RUNNERS_READY',
    probes='A_RAW_CORE8_104_PROBES_MATCH',export='A_RAW_CORE8_FOUR_OMS_COMPILED',
    smoke='A_RAW_CORE8_TOKEN_KV_MATCH',benchmark='A_RAW_CORE8_PAIRED_MEASURED')


def load(path,stage):
    r=c.read(path);c.verify_tree(r)
    if r.get('status')!=STATES[stage] or r.get('error') or r.get('integrity_error'):raise ValueError('failed Core8 predecessor: '+stage)
    if r['sources8']!=sources():raise ValueError('Core8 source drift')
    return r


def bound(r):
    _,prior=legacy(c.verify(r['assets']['old_admission']))
    if prior['references']!=r['assets']['references']:raise ValueError('input provenance changed')
    c.verify_tree(r['assets']);c.verify_opp(r['assets']['B8_package']['opp'])
    assets8.reverify(r['assets']['B8_c16'],prior,r['assets']['B8_package']['opp'],'B',16)
    return prior


def verify_build(r,prior):
    c.verify_tree(r)
    for v in ('B','R'):
        p=r['packages'][v];c.verify_opp(p['opp'])
        if not p['opp_comparison']['all_kernel_bytes_equal']:raise ValueError('device Kernel bytes changed')
        header=p['project_after']['op_host/d_flash_group_quant_linear_build_config.h']['file']
        if header['sha256']!=prior['current']['evidence'][v]['configuration']['header_sha256']:raise ValueError('actual configured Header differs')
    from raw_evidence import configs,header_pair
    configs(prior['current']['evidence']['B']['configuration'],prior['current']['evidence']['R']['configuration'])
    b=r['packages']['B'];a=r['packages']['R']
    header_pair(c.verify(b['project_after']['op_host/d_flash_group_quant_linear_build_config.h']['file']),c.verify(a['project_after']['op_host/d_flash_group_quant_linear_build_config.h']['file']))
    for file in ('op_host/d_flash_group_quant_linear.cpp','op_host/host_evidence.h'):
        if b['project_after'][file]['file']['sha256']!=a['project_after'][file]['file']['sha256']:raise ValueError('B8/R8 Host source differs')


def check_models(r,prior,build,probes):
    if set(r['models'])!={'B','R'}:raise ValueError('B8/R8 models missing')
    for v in ('B','R'):
        if set(r['models'][v])!={'16','64'}:raise ValueError('C16/C64 model missing')
        for gear,entry in r['models'][v].items():
            assets8.reverify(entry,prior,build['packages'][v]['opp'],v,int(gear))
            if entry['host_resources']['plans']!=probes['plans'][v][gear]:raise ValueError('OM/probe actual 8-core resources differ')
    for gear in ('16','64'):
        b=r['models']['B'][gear];a=r['models']['R'][gear]
        if any(b['worker'][k]!=a['worker'][k] for k in ('abi','counts','weight_bindings','head_sha256','logical_ir_sha256')):raise ValueError('B8/R8 graph or weight change')
        compare(b['host_resources']['plans'],a['host_resources']['plans'])


def execution_view(r,prior,build,export):
    return dict(evidence=prior['current']['evidence'],build=dict(runner=build['om_runner'],runner_dependencies=build['om_runner_dependencies'],
        variants={v:{'opp':build['packages'][v]['opp']} for v in ('B','R')}),
        export={'models':{v:{g:e['worker'] for g,e in rows.items()} for v,rows in export['models'].items()}},runs=r.setdefault('runs',[]))


def main(a):
    protected=[a.sidecar_root]+[p.parent for p in (getattr(a,k+'_report') for k in list(STATES)[:-1]) if p]
    root=c.fresh(a.output_dir,protected)
    r=dict(status='BLOCKED',phase=a.phase,sources8=sources(),core_policy='EXACT_8_AVAILABLE_8_BLOCK_AUTO_0',
        labels={'B':'B8_OFF','R':'R8_RAW'},npu='NOT_RUN',atc='NOT_RUN',benchmark='NOT_RUN',profile='NOT_RUN',production='NOT_MODIFIED')
    prior=None;admitted=None;build=None;probe=None;export=None;smoke=None
    try:
        c.clean();verify_worker()
        if a.phase=='admit':
            r['assets'],prior=assets8.admit(a.sidecar_root,root)
            r['source_root']=prior['root']
        else:
            chain={};stages={}
            for name in list(STATES)[:list(STATES).index(a.phase)]:
                p=getattr(a,name+'_report');value=load(p,name)
                if any(value.get(k+'_report')!=ref for k,ref in chain.items()):raise ValueError('mixed Core8 stage chain')
                chain[name]=c.identity(p);r[name+'_report']=chain[name];stages[name]=value
            admitted=stages['admit'];prior=bound(admitted);r['source_root']=prior['root']
            build=stages.get('build');probe=stages.get('probes');export=stages.get('export');smoke=stages.get('smoke')
            if build:verify_build(build,prior)
            if probe:
                plans=probes8.validate_all(probe,prior,build['packages'],build['projection_runners'])
                if plans!=probe['plans']:raise ValueError('8-core probe plans changed')
            if export:check_models(export,prior,build,probe)
            if a.phase=='build':
                # B8 is reused as-is; only missing R8 Host sidecar package is built.
                r['packages']={'B':admitted['assets']['B8_package']}
                out=root/'R8';req=root/'R8-request.json'
                c.write(req,dict(output=str(out),prior=prior,variant='R',core8_sources=sources()))
                c.command([sys.executable,'-B',SIDECAR/'build_worker.py',req],root/'R8-build.log',c.selected_env(prior['current']['evidence'],out/'opp'))
                r['packages']['R']=c.read(out/'report.json');r['R8_build_report']=c.identity(out/'report.json')
                if r['packages']['R'].get('status')!='A_RAW_SIDECAR_HOST_OPP_BUILT_NPU_NOT_RUN':raise ValueError('R8 Host build failed')
                r['projection_runners']={};r['runner_dependencies']={}
                for v in ('B','R'):
                    dest=root/(v+'8-runner');env=c.selected_env(prior['current']['evidence'],r['packages'][v]['opp']['install'])
                    c.command(['cmake','-S',c.QUANT/'test','-B',dest,'-DCANN_ROOT='+prior['current']['cann_root'],'-DASCEND_OPP_PATH='+r['packages'][v]['opp']['install'],'-DDFLASH_K_TILE=256'],root/(v+'-configure.log'),env)
                    c.command(['cmake','--build',dest,'--target','dflash_group_quant_linear_test','--parallel','2'],root/(v+'-build.log'),env)
                    r['projection_runners'][v]=c.identity(dest/'dflash_group_quant_linear_test')
                    r['runner_dependencies'][v]=c.dependencies(r['projection_runners'][v],root/(v+'-ldd.log'),env)
                original=prior['reports']['build'];r.update(om_runner=original['runner'],om_runner_dependencies=original['runner_dependencies'])
                verify_build(r,prior)
            elif a.phase=='probes':
                r['npu']='ATTEMPTED';r['runs']=[]
                for v in ('B','R'):
                    env=c.selected_env(prior['current']['evidence'],build['packages'][v]['opp']['install'])
                    if c.dependencies(build['projection_runners'][v],root/(v+'-ldd.log'),env)!=build['runner_dependencies'][v]:raise ValueError('probe Runner dependency drift')
                    for g in (16,64):
                        for case in prior['current']['evidence']['projections'][str(g)]:
                            probes8.run_one(root/(v+'8')/f'c{g}'/case['name'],prior,build['packages'][v],build['projection_runners'][v],v,g,case,prior['current']['device'],r['runs'])
                            c.write(root/'report.json',r)
                r['plans']=probes8.validate_all(r,prior,build['packages'],build['projection_runners'])
                if r['plans']['B']['16']!=admitted['assets']['B8_c16']['host_resources']['plans']:raise ValueError('reused B8-C16 OM plan differs from new NPU probes')
                r['npu']='COMPLETED_104_NEW_PROBES'
            elif a.phase=='export':
                selector={'current':prior['current'],'reports':{'build':{'variants':{v:{'opp':build['packages'][v]['opp']} for v in ('B','R')}}}}
                checks=root/'import-check';checks.mkdir();checked={}
                try:check_workers(selector,checks,checked)
                finally:c.write(checks/'report.json',checked);r['import_check']=c.identity(checks/'report.json')
                r['models']={'B':{'16':admitted['assets']['B8_c16']},'R':{}};r['B8_C16_action']='REUSED_NO_RECOMPILE';r['atc']='ATTEMPTED'
                old= c.read(c.verify(admitted['assets']['old_admission']))
                for v,g in (('B',64),('R',16),('R',64)):
                    out=root/(v+'8')/f'c{g}';req=root/f'{v}8-c{g}-request.json'
                    c.write(req,dict(output=str(out),variant=v,gear=g,device=prior['current']['device'],evidence=prior['current']['evidence'],sources=c.sources(),
                        continuation_sources=sources(),private_opp=build['packages'][v]['opp'],atc_binary=prior['current']['evidence']['tools']['atc'],compiler_identity_text=old['frozen']['compiler_identity_text']))
                    env=c.selected_env(prior['current']['evidence'],build['packages'][v]['opp']['install']);env.update(AI_RUN_DIR=str(root),ASCEND_WORK_PATH=str(out))
                    c.command([sys.executable,'-B',HERE/'export_worker.py',req],root/f'{v}8-c{g}-export.log',env)
                    child=c.read(out/'report.json');entry=assets8.model(child,prior,build['packages'][v]['opp'],v,g,out/'resource8')
                    entry['worker_report']=c.identity(out/'report.json');r['models'][v][str(g)]=entry
                    c.write(root/'report.json',r)
                check_models(r,prior,build,probe);r.update(atc='COMPLETED',npu='EXPORT_TENSOR_PREPARATION_ONLY_DRAFT_NOT_RUN')
            elif a.phase in ('smoke','benchmark'):
                view=execution_view(r,prior,build,export)
                args=argparse.Namespace(device=prior['current']['device'])
                if a.phase=='benchmark':
                    if [(x['gear'],x['variant']) for x in smoke['runs']]!=[(g,v) for g in (16,64) for v in ('B','R')]:raise ValueError('full B8/R8 Smoke missing')
                    for row in smoke['runs']:
                        if row['validation']!=verify_run(row,prior['current']['evidence']['H']['gears'][str(row['gear'])],args.device,False):raise ValueError('Smoke drift')
                    view['smoke']=smoke
                r['npu']='ATTEMPTED'
                if a.phase=='smoke':
                    for g in (16,64):
                        group=[run_om(args,root/f'c{g}'/(v+'8'),view,v,g) for v in ('B','R')]
                        if group[0]['validation']['workspace_bytes']!=group[1]['validation']['workspace_bytes']:raise ValueError('B8/R8 model Workspace differs')
                else:
                    r['benchmark']='ATTEMPTED'
                    for i in range(3):
                        for g in (16,64):
                            for j,v in enumerate(order(i,g)):
                                row=run_om(args,root/f'round-{i}'/f'c{g}'/(v+'8'),view,v,g,True);row.update(round=i,position=j)
                                c.write(root/'report.json',r)
                    r['comparison']=summarize(r['runs']);r['comparison']['experiment']='B8_R8_FULL_DRAFT_ONLY'
                    r['comparison']['seven_core_samples_included']=False;c.write(root/'comparison.json',r['comparison']);r['benchmark']='MEASURED'
                r['npu']='COMPLETED'
        r['status']=STATES[a.phase]
    except Exception as e:
        r['error']=f'{type(e).__name__}: {e}';(root/'traceback.txt').write_text(traceback.format_exc())
    finally:
        try:
            c.verify_tree(r)
            if r['sources8']!=sources():raise ValueError('Core8 source changed during stage')
            if prior:
                after=ec.previous(prior['root'])
                if after['references']!=prior['references']:raise ValueError('historical evidence changed')
            for value in (build,r if a.phase=='build' and 'packages' in r else None):
                if value:
                    for package in value['packages'].values():c.verify_opp(package['opp'])
            c.clean()
        except Exception as e:r.update(status='BLOCKED',integrity_error=str(e))
        c.write(root/'report.json',r)
    print(__import__('json').dumps({k:r.get(k) for k in ('status','phase','error','integrity_error','npu','atc','benchmark','B8_C16_action')},indent=2));print('REPORT:',root/'report.json')
    return int(r['status']=='BLOCKED')


def parse(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False);p.add_argument('phase',choices=STATES)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--sidecar-root',type=Path,default=Path('/home/w00949577/qwen35-dflash-run/a-only-raw-sidecar-kV7ZFn'))
    for name in list(STATES)[:-1]:p.add_argument('--'+name+'-report',type=Path)
    a=p.parse_args(argv)
    for name in list(STATES)[:list(STATES).index(a.phase)]:
        if getattr(a,name+'_report') is None:p.error('--'+name+'-report required')
    for k,v in vars(a).items():
        if isinstance(v,Path):setattr(a,k,v.resolve())
    return a


if __name__=='__main__':raise SystemExit(main(parse()))
