#!/usr/bin/env python3
"""Fresh interpreter entry; install verified imports BEFORE loading task modules."""
import json
from pathlib import Path
import runpy
import sys
import traceback
from native_base import activate,digest,bound_native


def main():
    if len(sys.argv)!=3 or sys.argv[1] not in ('preflight','device','single'):
        raise ValueError('usage: native_worker.py preflight|device|single REQUEST_JSON')
    stage=sys.argv[1]; request=Path(sys.argv[2]).resolve(); state=json.loads(request.read_text())
    root=Path(state['args']['output_dir']).resolve()
    if request!=root/('preflight-input.json' if stage=='preflight' else 'worker-input.json'):
        raise ValueError('worker request outside evidence root')
    base,_,_=bound_native(state)
    if root.is_relative_to(base): raise ValueError('evidence output must be outside read-only Native Base')
    output=root/f'native-base-{stage}.json'; report=dict(status='FAIL',stage=stage)
    code=1
    try:
        expected=None
        if stage in ('device','single'):
            link=state['native_boundary']; path=(root/link['path']).resolve()
            if not path.is_relative_to(root) or digest(path)!=link['sha256']:
                raise ValueError('accepted Native boundary report changed')
            expected=json.loads(path.read_text())
        boundary,report=activate(state,Path(__file__).resolve().parents[4],expected=expected)
        report['request_sha256']=digest(request)
        output.write_text(json.dumps(report,indent=2)+'\n')
        entry=Path(__file__).with_name('preflight_worker.py' if stage=='preflight' else 'single_graph.py' if stage=='single' else 'run.py')
        sys.argv=[str(entry)]+([str(request)] if stage=='preflight' else ['--device-worker',str(request)])
        try: runpy.run_path(str(entry),run_name='__main__')
        except SystemExit as stop: code=stop.code if isinstance(stop.code,int) else 1
        if stage=='preflight' and code==0:
            report['accepted_args']=json.loads((root/'preflight-result.json').read_text())['args']
        report['modules']=boundary.verify_loaded()
        report['receiver']=boundary.receiver
        from source_lock import audit_source_lock
        end=audit_source_lock(boundary.base,boundary.identity)
        if end['status']!='PASS': raise RuntimeError('Native Base changed during worker')
        report['status']='PASS' if code==0 else 'FAIL'
    except BaseException as error:
        traceback.print_exc();report.update(status='FAIL',error=f'{type(error).__name__}: {error}');code=1
    output.write_text(json.dumps(report,indent=2)+'\n')
    if report['status']!='PASS':
        failed=root/('preflight-result.json' if stage=='preflight' else 'suite.json')
        if failed.is_file():
            result=json.loads(failed.read_text()); result.update(status='FAIL',native_boundary_error=report.get('error','worker failed'))
            failed.write_text(json.dumps(result,indent=2)+'\n')
    return code

if __name__=='__main__': raise SystemExit(main())
