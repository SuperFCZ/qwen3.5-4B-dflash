#!/usr/bin/env python3
"""Disposable evidence/NZ process. It never builds OPP or starts device tests."""
import json
import os
from pathlib import Path
import sys
import traceback
from common import record,sha256,write_json
from process_boundary import decode_args,encode_args,trace_npu_imports


def run(path):
    path=Path(path).resolve(); request=json.loads(path.read_text()); args=decode_args(request)
    root=args.output_dir.resolve()
    if path!=root/'preflight-input.json': raise ValueError('preflight request outside its evidence directory')
    os.environ['AI_RUN_DIR']=str(root)
    result=dict(status='FAIL',pid=os.getpid(),request_sha256=sha256(path))
    try:
        with trace_npu_imports(root/'preflight-imports.json'):
            from resolve import resolve
            resolve(args,root) # All existing checks, including real NZ inverse, unchanged.
            if getattr(args,'draft_only',False):
                from draft_only import require_draft_interfaces
                require_draft_interfaces(args,root/'draft-interface-preflight.json')
            if getattr(args,'full_draft',False):
                from full_interface import require_full_interfaces
                require_full_interfaces(args,root/'full-interface-preflight.json')
        result.update(status='PASS',**encode_args(args),preflight=record(root/'preflight.json',root),
                      imports=record(root/'preflight-imports.json',root))
    except Exception as error:
        traceback.print_exc()
        result['error']=f'{type(error).__name__}: {error}'
    write_json(root/'preflight-result.json',result)
    return 0 if result['status']=='PASS' else 1

if __name__=='__main__':
    if len(sys.argv)!=2: raise SystemExit('usage: preflight_worker.py PREFLIGHT_INPUT_JSON')
    raise SystemExit(run(sys.argv[1]))
