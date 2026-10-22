#!/usr/bin/env python3
"""Reparse historical repository snapshots only; never read the live SDK tree."""
import argparse
import copy
import json
from pathlib import Path
import re
import audit
from aq_common import identity,verify,write,fresh
from tiling_metadata import MAX_METADATA_BYTES,is_repository,inspect_to_file


def read_report(path):
    if path.stat().st_size>64*1024*1024:raise ValueError('historical audit report exceeds 64 MiB')
    def pairs(items):
        result={}
        for key,value in items:
            if key in result:raise ValueError('duplicate audit report key')
            result[key]=value
        return result
    def constant(value):raise ValueError('nonstandard audit report number: '+value)
    with path.open('rb') as f:raw=f.read(64*1024*1024+1)
    if len(raw)>64*1024*1024:raise ValueError('historical audit report grew beyond limit')
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=constant)


def validate_history(path,old,watched):
    """Bind snapshots to old SDK identities without stat/open on SDK source paths."""
    root=path.resolve().parent
    def artifact(item):
        p=Path(item['path'])
        if not p.is_absolute() or root not in p.resolve(strict=True).parents:
            raise ValueError('historical artifact escapes audit directory: '+str(p))
        result=verify(item);watched[item['path']]=item
        return result
    if old.get('soc')!='Ascend310P3' or not isinstance(old.get('files'),list) or not old['files']:
        raise ValueError('requires populated Ascend310P3 audit snapshots')
    cann=Path(old['cann_root'])
    if not cann.is_absolute() or '..' in cann.parts:raise ValueError('invalid historical SDK root')
    seen=set();versions=[]
    for f in old['files']:
        source=f['source'];p=Path(source['path'])
        if not p.is_absolute() or '..' in p.parts or cann not in p.parents:
            raise ValueError('historical source identity outside selected SDK')
        if source['path'] in seen:raise ValueError('duplicate source in historical audit')
        seen.add(source['path'])
        snapshot=artifact(f['snapshot'])
        if any(source[k]!=f['snapshot'][k] for k in ('bytes','sha256')):
            raise ValueError('snapshot does not match historical SDK identity')
        if f['role']=='version':versions.append(dict(source=source,snapshot=f['snapshot'],text=snapshot.read_text(errors='replace')))
        for key in ('metadata_report','document_records'):
            if key in f:artifact(f[key])
    for key in ('scan_file','priority_evidence','input_path_evidence'):
        if key in old:artifact(old[key])
    if old.get('scan_file') and read_report(Path(old['scan_file']['path']))!=old['scan']:
        raise ValueError('historical embedded scan differs from scan artifact')
    return dict(status='SNAPSHOT_IDENTITIES_VERIFIED',count=len(old['files']),
        sdk_version='CANN_9_0_0_RECORDED' if any(re.search(r'(?<![\d.])9\.0\.0(?![\d.])',v['text']) for v in versions)
                    else 'UNRESOLVED_IN_REUSED_SNAPSHOTS',version_evidence=versions,
        live_sdk='NOT_READ_OR_REVALIDATED',
        historical_exporter_sources='Recorded in old audit; not revalidated against intentionally updated checkout',
        sdk_scan='NOT_RESUMED; an incomplete historical scan remains incomplete')


def run(args):
    prior=args.report.resolve(strict=True)
    old_identity=identity(prior);old=read_report(prior)
    # Do not resolve/stat the historical SDK root: it need not be mounted.
    sdk=Path(old['cann_root']);output=args.output_dir.absolute()
    if output==sdk or sdk in output.parents or output in sdk.parents:
        raise ValueError('output overlaps historical SDK path')
    root=fresh(args.output_dir,(prior.parent,))
    report=dict(status='RUNNING',scope='METADATA_ONLY_OFFLINE',historical_report=old_identity,
        soc='Ascend310P3',cann_root=old['cann_root'],cann_compile='NOT_RUN',npu='NOT_RUN',simulator='NOT_RUN',
        performance='NOT_RUN',full_gate='NOT_RUN',sdk_scan='NOT_RESUMED',files=[],
        scan_provenance='NO_NEW_SCAN; original state retained in historical_scan',
        historical_scan=copy.deepcopy(old.get('scan',{'status':'UNKNOWN'})),
        scan=dict(status='NOT_RESUMED',reused_snapshot_count=0,
                  unreviewed_repositories=copy.deepcopy(old.get('scan',{}).get('unreviewed_repositories',[]))),
        arithmetic=old.get('arithmetic',audit.counterexamples()),traffic=old.get('traffic',audit.traffic()),
        paths=audit.conclusions({'status':'HISTORICAL_SNAPSHOTS_ONLY'}),
        sources=[identity(p) for p in sorted(audit.HERE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts])
    watched={old_identity['path']:old_identity};results=[]
    try:
        report['snapshot_verification']=validate_history(prior,old,watched)
        report['files']=copy.deepcopy(old['files'])
        report['scan']['reused_snapshot_count']=len(report['files'])
        targets=[(i,f) for i,f in enumerate(report['files']) if f['role']=='tiling_metadata']
        if not targets:raise ValueError('historical audit contains no repository metadata snapshot')
        for index,f in targets:
            if not is_repository(Path(f['source']['path']).relative_to(Path(old['cann_root']))):
                raise ValueError('metadata target is not a 310P Matmul repository')
            snapshot=verify(f['snapshot'])
            if f['source']['bytes']>MAX_METADATA_BYTES:raise ValueError('repository snapshot exceeds 32 MiB')
            with snapshot.open('rb') as stream:raw=stream.read(MAX_METADATA_BYTES+1)
            records=root/f'repository-documents-{index}.jsonl'
            result=inspect_to_file(raw,f['source'],records)
            result.update(snapshot=f['snapshot'],document_records=identity(records),historical_metadata_report=f.get('metadata_report'))
            target=root/f'repository-metadata-{index}.json';write(target,result)
            f.update(metadata_report=identity(target),document_records=identity(records),metadata_status=result['status'],
                     content_classification=result['classification'])
            results.append(dict(source_index=index,source=f['source'],report=identity(target),
                status=result['status'],format=result['format'],validated_documents=result['validated_document_count']))
        complete=all(r['status']=='STRUCTURAL_REVIEW_COMPLETE' for r in results)
        report['status']='METADATA_REPARSED_SDK_REVIEW_INCOMPLETE' if complete else 'INCOMPLETE_METADATA_REVIEW'
    except Exception as e:report.update(status='BLOCKED_OFFLINE_REPARSE',error=f'{type(e).__name__}: {e}')
    finally:
        report['metadata_results']=results
        try:
            for item in watched.values():verify(item)
            for item in report['sources']:verify(item)
            for f in report['files']:
                verify(f['snapshot'])
                for key in ('metadata_report','document_records'):
                    if key in f:verify(f[key])
        except Exception as e:report.update(status='BLOCKED_OFFLINE_REPARSE',integrity_error=str(e))
        priority=audit.priority_index(report)
        priority['historical_scan']=dict(status=report['historical_scan']['status'],resumed=False,source_report=old_identity)
        if report['historical_scan']['status']!='COMPLETE_WITHIN_DECLARED_SCOPE':
            priority['unreviewed_evidence'].append(dict(reason='HISTORICAL_SDK_SCAN_TAIL_NOT_ENUMERATED_OR_RESUMED'))
        write(root/'priority-evidence.json',priority);report['priority_evidence']=identity(root/'priority-evidence.json')
        report['read_only_artifacts']=list(watched.values())
        write(root/'group128-capability.json',report)
        (root/'group128-capability.md').write_text('Metadata-only offline reparse. Historical SDK scan was NOT resumed.\n\n'+audit.markdown(report))
    print(report['status']+': '+str(root/'group128-capability.json'))
    return int(report['status']!='METADATA_REPARSED_SDK_REVIEW_INCOMPLETE')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    raise SystemExit(run(parser.parse_args()))
