#!/usr/bin/env python3
"""Check the historical export environment; optionally resume Export only."""
import argparse
from pathlib import Path
import sys
import raw_env_contract as ec
c=ec.c


def check_workers(prior,root,report):
    report['variants']={}
    for variant in ('B','R'):
        output=root/variant;request=root/(variant+'-request.json')
        req=dict(output=str(output),evidence=prior['current']['evidence'],sources=c.sources(),environment_sources=ec.sources())
        c.write(request,req)
        env=c.selected_env(req['evidence'],prior['reports']['build']['variants'][variant]['opp']['install'])
        try:c.command([sys.executable,'-B',ec.HERE/'raw_env_worker.py',request],root/(variant+'-imports.log'),env)
        finally:
            if (output/'report.json').is_file():
                report['variants'][variant]=c.read(output/'report.json')
                report.setdefault('child_reports',{})[variant]=c.identity(output/'report.json')
        ec.validate_child(report['variants'][variant])
    if ec.snapshot(report['variants']['B'])!=ec.snapshot(report['variants']['R']):raise ValueError('B/R Python dependency environment differs')


def checked_environment(path,prior):
    r=c.read(path);c.verify_tree(r)
    if r.get('status')!='A_RAW_EXPORT_ENVIRONMENT_IMPORTABLE' or r.get('error') or r.get('integrity_error'):
        raise ValueError('prior environment check not PASS')
    if r['environment_sources']!=ec.sources():raise ValueError('environment gate source drift')
    if r['references']!=prior['references']:raise ValueError('environment check belongs to other predecessor evidence')
    if set(r['variants'])!={'B','R'}:raise ValueError('environment variants missing')
    for v,child in r['variants'].items():
        if c.read(c.verify(r['child_reports'][v]))!=child:raise ValueError('environment child report drift')
        ec.validate_child(child)
    return r


def main(args):
    protected=[args.source_root]
    if args.environment_report:protected.append(args.environment_report.parent)
    root=c.fresh(args.output_dir,protected)
    report=dict(status='BLOCKED',phase=args.phase,npu='NOT_RUN',export='NOT_RUN',
        environment_sources=ec.sources(),reuse='NOT_CHECKED')
    try:
        c.clean();prior=ec.previous(args.source_root)
        report.update(references=prior['references'],reuse=prior['status'])
        old=None
        if args.phase=='export':old=checked_environment(args.environment_report,prior)
        check_workers(prior,root,report)
        if args.phase=='check':report['status']='A_RAW_EXPORT_ENVIRONMENT_IMPORTABLE'
        else:
            for variant in ('B','R'):
                if ec.snapshot(old['variants'][variant])!=ec.snapshot(report['variants'][variant]):raise ValueError('import environment drift since check')
            report['environment_report']=c.identity(args.environment_report)
            dest=root/'export';argv=[sys.executable,'-B',ec.RAW/'run_experiment.py','export','--output-dir',dest,
                '--device',prior['current']['device'],'--cann-root',prior['current']['cann_root']]
            for stage,ref in prior['references'].items():argv+=['--'+stage+'-report',c.verify(ref)]
            report['export']='ATTEMPTED';report['npu']='EXPORT_DEVICE_TENSOR_PREPARATION_POSSIBLE_DRAFT_NOT_RUN'
            try:c.command(argv,root/'export-driver.log')
            finally:
                if (dest/'report.json').is_file():report['export_report']=c.identity(dest/'report.json')
            child=c.read(c.verify(report['export_report']))
            if child.get('status')!=ec.experiment.STATES['export'] or child.get('error') or child.get('integrity_error'):raise ValueError('original Export gate failed')
            ec.experiment.verify_export(child)
            report.update(status='A_RAW_EXPORT_RESUMED_WITH_IMPORT_GATE',export='COMPLETED_DRAFT_NOT_RUN')
    except Exception as error:
        import traceback
        report['error']=f'{type(error).__name__}: {error}'
        (root/'traceback.txt').write_text(traceback.format_exc())
    finally:
        try:
            c.verify_tree(report)
            for child in report.get('variants',{}).values():ec.verify_loaded_files(child.get('loaded_files',{}))
            if report['environment_sources']!=ec.sources():raise ValueError('environment gate source changed during run')
            if 'references' in report:
                end=ec.previous(args.source_root)
                if end['references']!=report['references']:raise ValueError('predecessor changed during environment gate')
            c.clean()
        except Exception as error:report.update(status='BLOCKED',integrity_error=str(error))
        c.write(root/'report.json',report)
    ec.print_summary(report);print('REPORT:',root/'report.json')
    return int(report['status']=='BLOCKED')


def parse(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('phase',choices=('check','export'))
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--environment-report',type=Path)
    a=p.parse_args(argv)
    if a.phase=='export' and not a.environment_report:p.error('--environment-report required for export')
    if a.phase=='check' and a.environment_report:p.error('--environment-report is only for export')
    for k,v in vars(a).items():
        if isinstance(v,Path):setattr(a,k,v.resolve())
    return a


if __name__=='__main__':raise SystemExit(main(parse()))
