"""Fresh interpreter; imports only, no model construction/forward or ACL calls."""
import importlib
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
import raw_env_contract as ec
c=ec.c


def main(path):
    req=c.read(path);out=Path(req['output']);out.mkdir(parents=True,exist_ok=False)
    r=dict(status='BLOCKED',output_root=str(out),npu='NOT_RUN',interfaces='NOT_PREPARED',export='NOT_RUN');boundary=None
    try:
        c.clean();c.verify_tree(req);e=req['evidence'];h=e['H']
        r['python']=ec.python_identity()
        r['historical_environment']=ec.historical_environment(e)
        r['direct_imports']=ec.direct_imports(h['native_base'])
        r['current_distribution']=ec.distribution_info('transformers')
        expected=r['historical_environment']['modules']
        # Check origin/version before importing the heavy Native dependency tree.
        transformers=importlib.import_module('transformers')
        r['packages']={'transformers':ec.check_package(transformers,r['current_distribution'],expected['transformers'])}
        hint=Path(r['historical_environment']['interpreter_hint'])
        if Path(sys.executable).parent.resolve()!=hint.parent.resolve() or Path(sys.prefix).resolve()!=hint.parent.parent.resolve():
            raise ValueError('use the historical venv executable alias; do not splice PYTHONPATH or resolve to base Python')
        r['packages']['transformers']['installed_record']=ec.installed_record('transformers','transformers/__init__.py')
        init=importlib.import_module('transformers.initialization')
        if not callable(getattr(init,'copy_',None)):raise ValueError('Transformers initialization.copy_ unavailable')
        if Path(transformers.__file__).resolve().parent not in Path(init.__file__).resolve().parents:raise ValueError('initialization outside selected Transformers package')
        r['packages']['transformers.initialization']=ec.installed_record('transformers','transformers/initialization.py')
        if r['packages']['transformers.initialization']['file']!=c.identity(init.__file__):raise ValueError('initialization module is shadowed')
        for name in ('torch','torch_npu','torchair'):
            module=importlib.import_module(name);file=c.identity(module.__file__)
            if file!=expected[name]['file'] or str(getattr(module,'__version__','unknown'))!=expected[name]['version']:
                raise ValueError('historical runtime module identity differs: '+name)
            r['packages'][name]=dict(file=file,version=str(getattr(module,'__version__','unknown')))
        from native_base import activate,check_extension_compatibility
        args=dict(h['state']['args'],output_dir=str(out),draft_only=True,full_draft=False,preflight_only=False)
        boundary,bound=activate(dict(args=args),c.REPO);r['native_boundary']=bound
        from draft_only import native_history,draft_operators,DRAFT_OPS
        a=SimpleNamespace(**{k:Path(v) if k in h['state']['path_args'] else v for k,v in args.items()})
        config,native,_=native_history(a)
        from source_lock import require_source_lock
        require_source_lock(native,out/'source-lock.json',native_root=Path(h['native_base']))
        r['compatibility']=check_extension_compatibility(config,native)
        specs=draft_operators(config)
        if len(specs)!=3 or {s.torch_op for s in specs}!=DRAFT_OPS:raise ValueError('Draft operator declarations differ')
        from qwen35_dflash.ascend310p.custom_op_export import prepare_custom_op_export
        if not callable(prepare_custom_op_export):raise ValueError('operator preparation entry missing')
        r['operators']=[dict(torch_op=s.torch_op,ge_op_type=s.ge_op_type) for s in specs]
        # No prepare_custom_op_export / prepare_draft_interfaces call here.
        r['status']='DRAFT_IMPORTS_READY_INTERFACES_NOT_PREPARED'
    except Exception as error:
        r['error']=f'{type(error).__name__}: {error}'
        r['traceback']=traceback.format_exc();(out/'traceback.txt').write_text(r['traceback'])
    finally:
        try:
            if boundary:r['native_boundary']['modules']=boundary.verify_loaded()
            r['loaded_files'],r['process_modules']=ec.lifetime.capture(ec.loaded_files(),out)
            ec.verify_loaded_files(r['loaded_files'])
            ec.lifetime.verify(r['process_modules'],r['loaded_files'],out);c.verify_tree(req)
        except Exception as error:r.update(status='BLOCKED',integrity_error=str(error))
        c.write(out/'report.json',r)
    return int(r['status']=='BLOCKED')


if __name__=='__main__':raise SystemExit(main(Path(sys.argv[1])))
