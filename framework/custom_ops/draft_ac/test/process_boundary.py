"""Stdlib-only process boundary and import evidence; never unload device modules."""
from contextlib import contextmanager
import os
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace
from common import checked_file,sha256,write_json


def encode_args(args):
    return dict(args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                path_args=[k for k,v in vars(args).items() if isinstance(v,Path)])


def decode_args(data):
    args=SimpleNamespace(**data['args'])
    for key in data['path_args']: setattr(args,key,Path(getattr(args,key)))
    return args


def npu_modules():
    return sorted(n for n in sys.modules if n=='torch_npu' or n.startswith('torch_npu.'))


def require_clean_npu(phase):
    loaded=npu_modules()
    if loaded: raise RuntimeError(f'torch_npu loaded before joint OPP selection ({phase}): {loaded[:8]}; start a fresh process')


@contextmanager
def trace_npu_imports(path):
    """Observe actual find_spec stacks in the preflight child; never supply a spec."""
    report=dict(pid=os.getpid(),ppid=os.getppid(),scope='preflight imports only; NPU execution NOT_RUN',
                autoload=os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD'),before=npu_modules(),first_lookup=None,first_import=None)
    class Observer:
        def find_spec(self,fullname,path=None,target=None):
            if (fullname=='torch_npu' or fullname.startswith('torch_npu.')) and report['first_import'] is None:
                frames=[dict(file=f.filename,line=f.lineno,function=f.name,source=f.line)
                        for f in traceback.extract_stack()[:-1]]
                event=dict(module=fullname,stack=frames)
                if report['first_lookup'] is None: report['first_lookup']=event
                # find_spec() availability queries do not import a module.
                # CPython's actual import loader calls _find_spec from here.
                if len(frames)>1 and frames[-2]['function']=='_find_and_load_unlocked':
                    report['first_import']=event
            return None
    observer=Observer(); sys.meta_path.insert(0,observer)
    try:
        yield report
    finally:
        sys.meta_path.remove(observer)
        report['after']=npu_modules()
        report['loaded']=bool(report['after'])
        report['origins']={}
        files={f['file'] for f in (report['first_import'] or {}).get('stack',[])}
        for name in ('torch','torch_npu','transformers','qwen35_dflash.ascend310p'):
            module=sys.modules.get(name); source=getattr(module,'__file__',None)
            if source: report['origins'][name]=source; files.add(source)
        report['source_sha256']={str(p):sha256(p) for f in files if (p:=Path(f)).is_file() and p.suffix=='.py'}
        write_json(path,report)


def validate_preflight_result(request,result,root):
    if (result.get('status')!='PASS' or result.get('request_sha256')!=sha256(root/'preflight-input.json')
            or type(result.get('pid')) is not int or result['pid']<=0 or result['pid']==os.getpid()):
        raise ValueError('preflight child status/request/process identity differs')
    allowed={'a_build_config','a16_bundle','a64_bundle','a16_native','a64_native','factory_config'}
    for key,value in request['args'].items():
        if key not in allowed and result['args'].get(key)!=value: raise ValueError('preflight changed request: '+key)
    import json
    provenance=json.loads(checked_file(root,result['preflight']).read_text())
    imports=json.loads(checked_file(root,result['imports']).read_text())
    if provenance.get('status')!='PASS' or imports.get('pid')!=result['pid']:
        raise ValueError('missing passing preflight/import evidence')
    for item in provenance['records'].values():
        if sha256(item['path'])!=item['sha256']: raise ValueError('preflight evidence changed across process boundary')
    return decode_args(result),provenance
