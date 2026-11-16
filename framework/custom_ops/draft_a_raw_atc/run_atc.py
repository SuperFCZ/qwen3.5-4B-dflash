#!/usr/bin/env python3
"""Explicit admit -> B-C16 ATC-only resume -> remaining three exports. No NPU replay."""
import argparse
from pathlib import Path
import sys
import traceback
from split_atc import c,ec,HERE,sources,collect_stdout,legacy
from atc_evidence import find_export,old_worker,verify_model
from worker_derivation import verify_derivation
from run_environment import check_workers

STATES=dict(admit='A_RAW_ATC_RESUME_ADMITTED',resume='A_RAW_BC16_STDOUT_RESOURCES_MATCH',complete='A_RAW_FOUR_STDOUT_OMS_COMPILED_DRAFT_NOT_RUN')


def load_stage(path,stage):
    r=c.read(path);c.verify_tree(r)
    if r.get('status')!=STATES[stage] or r.get('error') or r.get('integrity_error'):raise ValueError('failed continuation predecessor: '+stage)
    if r['continuation_sources']!=sources() or r['environment_sources']!=ec.sources():raise ValueError('continuation source inventory changed')
    return r


def bound_admission(r):
    prior=ec.previous(r['source_root'])
    if prior['references']!=r['predecessors']:raise ValueError('original predecessor identity drift')
    frozen=old_worker(c.verify(r['frozen']['failed_export']),prior)
    if frozen!=r['frozen']:raise ValueError('frozen B-C16 assets changed after admission')
    return prior


def child_request(root,prior,admission,variant,gear,resume=False):
    dest=root/variant/f'c{gear}';request=root/f'{variant}-c{gear}-request.json'
    req=dict(evidence=prior['current']['evidence'],sources=c.sources(),continuation_sources=sources(),
        variant=variant,gear=gear,device=prior['current']['device'],output=str(dest),
        private_opp=prior['reports']['build']['variants'][variant]['opp'],
        atc_binary=admission['frozen']['atc_binary'],compiler_identity_text=admission['frozen']['compiler_identity_text'])
    if resume:req['frozen']=admission['frozen']
    c.write(request,req)
    env=c.selected_env(req['evidence'],prior['reports']['build']['variants'][variant]['opp']['install'])
    env.update(AI_RUN_DIR=str(root),ASCEND_WORK_PATH=str(dest),PYTHONDONTWRITEBYTECODE='1')
    return dest,request,env


def collect_worker(dest,prior,variant,gear):
    model=c.read(dest/'report.json')
    if model.get('error') or model.get('integrity_error') or model['status']=='BLOCKED':raise ValueError('worker failed: '+str(dest))
    model['host_resources']=collect_stdout(c.verify(model['atc_log']),dest/'host-plans',gear,prior['current']['evidence'][variant]['configuration'])
    model['worker_report']=c.identity(dest/'report.json');verify_model(model,prior)
    return model


def launch_worker(report,dest,request,env,variant,gear,resume=False):
    label=f'{variant}-c{gear}'
    try:c.command([sys.executable,'-B',HERE/('resume_worker.py' if resume else 'worker.py'),request],request.parent/(label+'-worker.log'),env)
    finally:
        row={};path=dest/'report.json'
        if path.is_file():
            row['report']=c.identity(path);row['atc_captures']=c.read(path).get('atc_captures',[])
        # Retain available raw streams even if a process dies before its report.
        for name in ('stdout','stderr'):
            log=dest/'atc-streams/call-0'/(name+'.log')
            if log.is_file():row[name]=c.identity(log)
        report.setdefault('worker_evidence',{})[label]=row


def main(a):
    protected=[p.parent for p in (a.failed_export_report,a.admission_report,a.resume_report) if p]
    if a.failed_export_root:protected.append(a.failed_export_root)
    root=c.fresh(a.output_dir,protected)
    r=dict(status='BLOCKED',phase=a.phase,continuation_sources=sources(),environment_sources=ec.sources(),
        npu='NOT_RUN',draft_execution='NOT_RUN',smoke='NOT_RUN',benchmark='NOT_RUN',profile='NOT_RUN',atc='NOT_RUN')
    prior=None;admission=None
    try:
        c.clean();r['derivation']=verify_derivation()
        if a.phase=='admit':
            failed=a.failed_export_report or find_export(a.failed_export_root)
            first=c.read(failed);source_root=c.verify(first['precheck_report']).parent.parent
            prior=ec.previous(source_root)
            r.update(source_root=str(source_root),predecessors=prior['references'],frozen=old_worker(failed,prior))
        else:
            admission=load_stage(a.admission_report,'admit');prior=bound_admission(admission)
            r.update(source_root=admission['source_root'],predecessors=prior['references'],admission_report=c.identity(a.admission_report))
            checks=root/'import-check';checks.mkdir();env_result={}
            try:check_workers(prior,checks,env_result)
            finally:
                c.write(checks/'report.json',env_result);r['import_check']=c.identity(checks/'report.json')
            r['atc']='ATTEMPTED'
            if a.phase=='resume':
                dest,req,env=child_request(root,prior,admission,'B',16,True)
                launch_worker(r,dest,req,env,'B',16,True)
                r['model']=collect_worker(dest,prior,'B',16)
            else:
                resumed=load_stage(a.resume_report,'resume')
                if resumed['admission_report']!=r['admission_report']:raise ValueError('mixed ATC resume/admission')
                verify_model(resumed['model'],prior)
                r['resume_report']=c.identity(a.resume_report);r['models']={'B':{'16':resumed['model']},'R':{}}
                r['npu']='EXPORT_TENSOR_PREPARATION_ONLY_DRAFT_NOT_RUN'
                for variant,gear in (('B',64),('R',16),('R',64)):
                    dest,req,env=child_request(root,prior,admission,variant,gear)
                    launch_worker(r,dest,req,env,variant,gear)
                    r['models'][variant][str(gear)]=collect_worker(dest,prior,variant,gear)
                    c.write(root/'report.json',r)
                for gear in ('16','64'):
                    b=r['models']['B'][gear];v=r['models']['R'][gear]
                    if any(b[k]!=v[k] for k in ('abi','counts','weight_bindings','head_sha256','logical_ir_sha256')):raise ValueError('B/R non-Raw graph/weight difference')
                    legacy.compare(b['host_resources'],v['host_resources'])
                r['resource_comparison']='B_R_GATE_AND_NON_GATE_CONTRACTS_MATCH'
            r['atc']='COMPLETED'
        r['status']=STATES[a.phase]
    except Exception as error:
        r['error']=f'{type(error).__name__}: {error}';(root/'traceback.txt').write_text(traceback.format_exc())
    finally:
        try:
            c.verify_tree(r)
            if r['continuation_sources']!=sources() or r['environment_sources']!=ec.sources():raise ValueError('source changed during continuation')
            if prior:
                end=ec.previous(prior['root'])
                if end['references']!=prior['references']:raise ValueError('original evidence changed during continuation')
                if admission:bound_admission(admission)
            c.clean()
        except Exception as error:r.update(status='BLOCKED',integrity_error=str(error))
        c.write(root/'report.json',r)
    small={k:r.get(k) for k in ('status','phase','error','integrity_error','atc','npu','draft_execution','smoke','benchmark')}
    if 'frozen' in r:small['frozen']={k:r['frozen'][k] for k in ('worker_report','air','deployment','atc_binary')}
    if 'model' in r:
        small['logs']={k:r['model'][k] for k in ('atc_log','atc_stderr','om_comparison')}
        small['host_stdout_schema']=r['model']['host_resources']['stdout_schema']
    if 'models' in r:small['models']={v:{g:{k:m[k] for k in ('om','atc_log','atc_stderr')} for g,m in group.items()} for v,group in r['models'].items()}
    print(__import__('json').dumps(small,indent=2,ensure_ascii=False));print('REPORT:',root/'report.json')
    return int(r['status']=='BLOCKED')


def parse(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('phase',choices=STATES);p.add_argument('--output-dir',type=Path,required=True)
    group=p.add_mutually_exclusive_group();group.add_argument('--failed-export-root',type=Path);group.add_argument('--failed-export-report',type=Path)
    p.add_argument('--admission-report',type=Path);p.add_argument('--resume-report',type=Path)
    a=p.parse_args(argv)
    if a.phase=='admit' and not (a.failed_export_root or a.failed_export_report):p.error('failed Export root/report required')
    if a.phase!='admit' and not a.admission_report:p.error('--admission-report required')
    if a.phase=='complete' and not a.resume_report:p.error('--resume-report required')
    if a.phase!='admit' and (a.failed_export_root or a.failed_export_report):p.error('use the admitted frozen assets')
    for k,v in vars(a).items():
        if isinstance(v,Path):setattr(a,k,v.resolve())
    return a


if __name__=='__main__':raise SystemExit(main(parse()))
