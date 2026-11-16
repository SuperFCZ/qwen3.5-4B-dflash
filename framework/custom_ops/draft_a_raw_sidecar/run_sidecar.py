#!/usr/bin/env python3
"""B-C16 Host evidence only: source admission -> new Host OPP -> ATC resource gate."""
import argparse
from pathlib import Path
import sys
import traceback
from sidecar_common import c,ec,HERE,sources,audit_host
from run_atc import load_stage,bound_admission
from run_environment import check_workers
from atc_evidence import compare_om
from split_atc import one_capture
from sidecar_atc import collect_host_evidence,verify_host_evidence
from derive_worker import verify as verify_worker_source

STATES=dict(admit='A_RAW_SIDECAR_SOURCE_ADMITTED',build='A_RAW_SIDECAR_HOST_OPP_BUILT_NPU_NOT_RUN',bc16='A_RAW_BC16_HOST_EVIDENCE_SIDECAR_MATCH')
PREVIOUS_OM_SHA='f4ca093e62230bba0bb5e6c3a5865189d32c783e72abfbd8ad8184d0257b9587'
PREVIOUS_OM_BYTES=3090431312


def stage(path,name):
    r=c.read(path);c.verify_tree(r)
    if r.get('status')!=STATES[name] or r.get('error') or r.get('integrity_error'):raise ValueError('failed sidecar predecessor: '+name)
    if r['sidecar_sources']!=sources():raise ValueError('sidecar source drift')
    return r


def legacy(admission_path):
    old=load_stage(admission_path,'admit');prior=bound_admission(old)
    return old,prior


def verify_previous(path,prior):
    old=c.read(path);c.verify_tree(old)
    if old.get('status')!='A_ONLY_ATC_RESUMED_NOT_EXECUTED' or old.get('error') or old.get('integrity_error'):
        raise ValueError('latest B-C16 ATC worker did not complete')
    if (old.get('variant'),old.get('gear'))!=('B',16) or old['private_opp']!=prior['reports']['build']['variants']['B']['opp']:
        raise ValueError('latest worker is not the sealed normal B package')
    c.verify_opp(old['private_opp']);one_capture(old['atc_captures'],old['atc_command'])
    if old['om']['sha256']!=PREVIOUS_OM_SHA or old['om']['bytes']!=PREVIOUS_OM_BYTES:raise ValueError('latest OM differs from user-frozen identity')
    if c.verify(old['compiler_stdout_log']).read_bytes()!=c.verify(old['atc_log']).read_bytes():raise ValueError('old stdout copies differ')
    return dict(report=c.identity(path),om=old['om'],air=old['artifacts']['air'],deployment=old['artifacts']['deployment'],stdout=old['atc_log'],stderr=old['atc_stderr'])


def main(a):
    protected=[p.parent for p in (a.admission_report,a.previous_worker,a.source_report,a.build_report) if p]
    root=c.fresh(a.output_dir,protected)
    r=dict(status='BLOCKED',phase=a.phase,sidecar_sources=sources(),atc='NOT_RUN',npu='NOT_RUN',draft_execution='NOT_RUN',smoke='NOT_RUN',benchmark='NOT_RUN',profile='NOT_RUN')
    prior=None;old=None
    try:
        c.clean();r['source_audit']=audit_host();r['worker_base']=verify_worker_source()
        if a.phase=='admit':
            old,prior=legacy(a.admission_report)
            r.update(admission_report=c.identity(a.admission_report),predecessors=prior['references'],previous_worker=verify_previous(a.previous_worker,prior))
        else:
            admitted=stage(a.source_report,'admit');old,prior=legacy(c.verify(admitted['admission_report']))
            if prior['references']!=admitted['predecessors']:raise ValueError('historical predecessor drift')
            c.verify_tree(admitted['previous_worker'])
            r.update(source_report=c.identity(a.source_report),admission_report=admitted['admission_report'],predecessors=prior['references'])
            if a.phase=='build':
                req=root/'build-request.json';out=root/'B'
                c.write(req,dict(output=str(out),prior=prior,variant='B',sidecar_sources=sources()))
                env=c.selected_env(prior['current']['evidence'],out/'opp')
                c.command([sys.executable,'-B',HERE/'build_worker.py',req],root/'build-worker.log',env)
                built=c.read(out/'report.json')
                if built.get('status')!=STATES['build'] or built.get('error') or built.get('integrity_error'):raise ValueError('new Host package failed')
                r['B']=built;r['build_worker_report']=c.identity(out/'report.json')
            else:
                built=stage(a.build_report,'build')
                if built['source_report']!=r['source_report']:raise ValueError('mixed sidecar source/build')
                package=built['B'];c.verify_opp(package['opp'])
                if not package['opp_comparison']['all_kernel_bytes_equal']:raise ValueError('device Kernel bytes changed')
                r['build_report']=c.identity(a.build_report);r['private_opp']=package['opp']
                # Environment selection only. No old NPU acceptance is assigned
                # to this new Host package, and no Projection is executed here.
                selector={'current':prior['current'],'reports':{'build':{'variants':{'B':{'opp':package['opp']},'R':prior['reports']['build']['variants']['R']}}}}
                check=root/'import-check';check.mkdir();import_report={}
                try:check_workers(selector,check,import_report)
                finally:c.write(check/'report.json',import_report);r['import_check']=c.identity(check/'report.json')
                out=root/'B/c16';req=root/'B-c16-request.json'
                c.write(req,dict(output=str(out),evidence=prior['current']['evidence'],sources=c.sources(),continuation_sources=sources(),
                    private_opp=package['opp'],frozen=old['frozen']))
                env=c.selected_env(prior['current']['evidence'],package['opp']['install']);env.update(AI_RUN_DIR=str(root),ASCEND_WORK_PATH=str(out))
                r['atc']='ATTEMPTED'
                try:c.command([sys.executable,'-B',HERE/'bc16_worker.py',req],root/'B-c16-worker.log',env)
                finally:
                    if (out/'report.json').is_file():
                        child=c.read(out/'report.json');r['worker_report']=c.identity(out/'report.json');r['atc_captures']=child.get('atc_captures',[])
                if child.get('status')!='A_ONLY_SIDECAR_ATC_COMPILED_NOT_EXECUTED' or child.get('error') or child.get('integrity_error'):raise ValueError('BC16 sidecar worker failed')
                capture=one_capture(child['atc_captures'],child['atc_command'])
                if child['host_evidence']!=capture['host_evidence']:raise ValueError('worker/capture sidecar identity differs')
                plan=collect_host_evidence(child['host_evidence'],root/'host-resources',16,prior['current']['evidence']['B']['configuration'])
                from raw_resources import accepted_gate
                accepted_gate(plan['plans']['16,2560,19456'],prior['current']['evidence'],'B')
                # All six shapes must match the old B ACLNN plans, not defaults.
                old_plans={}
                for x in prior['reports']['probes']['runs']:
                    if x['variant']!='B' or x['gear']!=16:continue
                    key=','.join(map(str,x['shape']));value=x['validation']['plan']
                    if key in old_plans and old_plans[key]!=value:raise ValueError('old same-shape B plans conflict')
                    old_plans[key]=value
                if plan['plans']!=old_plans:raise ValueError('new Host Tiling resources differ from accepted B C16 probes')
                verify_host_evidence(plan,16,prior['current']['evidence']['B']['configuration'])
                r.update(model=child,host_resources=plan,om_comparison=compare_om(admitted['previous_worker']['om'],child['om']),
                    atc='COMPLETED',new_opp_npu_acceptance='NOT_RUN_OLD_104_PROBES_NOT_TRANSFERRED')
        r['status']=STATES[a.phase]
    except Exception as e:
        r['error']=f'{type(e).__name__}: {e}';(root/'traceback.txt').write_text(traceback.format_exc())
    finally:
        try:
            c.verify_tree(r)
            if r['sidecar_sources']!=sources():raise ValueError('sidecar source changed during run')
            if prior:
                after=ec.previous(prior['root'])
                if after['references']!=prior['references']:raise ValueError('old evidence changed')
            if 'private_opp' in r:c.verify_opp(r['private_opp'])
            c.clean()
        except Exception as e:r.update(status='BLOCKED',integrity_error=str(e))
        c.write(root/'report.json',r)
    small={k:r.get(k) for k in ('status','phase','error','integrity_error','atc','npu','draft_execution','new_opp_npu_acceptance')}
    if 'model' in r:small['files']={k:r['model'][k] for k in ('om','atc_log','atc_stderr','host_evidence')}
    if 'host_resources' in r:small['sidecar_schema']=r['host_resources']['sidecar_schema']
    print(__import__('json').dumps(small,indent=2,ensure_ascii=False));print('REPORT:',root/'report.json')
    return int(r['status']=='BLOCKED')


def parse(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('phase',choices=STATES);p.add_argument('--output-dir',type=Path,required=True)
    for k in ('admission-report','previous-worker','source-report','build-report'):p.add_argument('--'+k,type=Path)
    a=p.parse_args(argv)
    required={'admit':('admission_report','previous_worker'),'build':('source_report',),'bc16':('source_report','build_report')}[a.phase]
    for k in required:
        if getattr(a,k) is None:p.error('--'+k.replace('_','-')+' required')
    for k,v in vars(a).items():
        if isinstance(v,Path):setattr(a,k,v.resolve())
    return a


if __name__=='__main__':raise SystemExit(main(parse()))
