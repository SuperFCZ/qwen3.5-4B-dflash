"""Pinned Native Base import boundary, installed only in fresh diagnostic workers.

No module replacement, SOURCE_LOCK rewrite, alternate checkout or bytecode cache.
Only the explicitly listed new A/C module names may come from the extension tree.
"""
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

NATIVE_SHA = '7872c0f687aca12421786ed9508b8416a987ec38d5d8bc116dee39861c119395'
NATIVE_REVISION = '319f51d0c51352c7f27f739e845ba93101f1ce8f'
EXTENSION_NAMES = tuple('qwen35_dflash.ascend310p.'+n for n in (
    'draft_ac_export', 'draft_ac_constants', 'draft_ac_payload', 'draft_ac_export_pipeline',
    'draft_ac_runtime_input_export', 'draft_ac_weight_prepack'))
_ACTIVE = None


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def managed(name):
    return name == 'models' or name.startswith('models.') or name == 'qwen35_dflash' or name.startswith('qwen35_dflash.')


def extension_inventory(root):
    root = Path(root).resolve()
    paths = [root/'framework/python/qwen35_dflash/ascend310p'/(n.rsplit('.',1)[1]+'.py') for n in EXTENSION_NAMES]
    for folder in ('draft_ac', 'draft_quant', 'draft_head'):
        paths.extend((root/f'framework/custom_ops/{folder}/test').glob('*.py'))
    paths.extend((root/'framework/custom_ops/draft_ac').glob('*.sh'))
    return {str(p.relative_to(root)): digest(p) for p in sorted(set(paths))}


def native_inventory(root, folders=('models','framework/python')):
    paths=set()
    for folder in folders:
        paths.update((root/folder).rglob('*.py'))
    if not paths: raise ValueError('Native Python source inventory is empty')
    result={}
    for p in sorted(paths):
        if p.is_symlink() or not p.resolve().is_relative_to(root):
            raise ValueError('Native module inventory escapes root: '+str(p))
        result[str(p.relative_to(root))]=digest(p)
    return result


def revision(root):
    def git(*args):
        return subprocess.check_output(['git','-C',str(root),*args],stderr=subprocess.DEVNULL,text=True).strip()
    try:
        if Path(git('rev-parse','--show-toplevel')).resolve() != root: return None
        return git('rev-parse','HEAD')
    except (OSError,subprocess.CalledProcessError): return None


def bound_native(state):
    args = state['args']
    if not args.get('draft_only') or not args.get('native_base'):
        raise ValueError('historical worker requires explicit --draft-only --native-base')
    deployment_path = Path(args['native_deployment']).resolve()
    deployment = json.loads(deployment_path.read_text())
    link = deployment['air_manifest']; path = Path(link['path'])
    path = (deployment_path.parent/path).resolve()
    if not path.is_file() or digest(path) != link['sha256']:
        raise ValueError('Native AIR hash/path differs')
    air = json.loads(path.read_text())
    drafts = [g for g in air['graphs'] if g['name']=='draft']
    deployed = [g for g in deployment['graphs'] if g['name']=='draft']
    if air.get('status')!='PASS' or deployment.get('status')!='PASS' or len(drafts)!=1 or len(deployed)!=1:
        raise ValueError('Native AIR/Deployment incomplete')
    identity = drafts[0]['metadata']['quant_source_lock']
    if identity != deployed[0]['metadata']['quant_source_lock']:
        raise ValueError('Native AIR/Deployment source identity differs')
    base = Path(args['native_base']).resolve()
    expected=dict(path=str(base/'SOURCE_LOCK.json'),sha256=NATIVE_SHA,verified_file_count=205)
    if identity != expected:
        raise ValueError(f'Native identity differs from explicit root / 319f51d profile: expected={expected}; actual={identity}')
    return base, identity, dict(path=str(path),sha256=digest(path))


class VerifiedLoader(importlib.machinery.SourceFileLoader):
    def __init__(self, name, path, boundary, kind):
        super().__init__(name,str(path)); self.boundary=boundary; self.kind=kind
    def exec_module(self, module):
        super().exec_module(module)
        self.boundary.loaded[self.name]['module_id']=id(module)
    def get_code(self, fullname):
        path=Path(self.path).resolve(); data=path.read_bytes()
        actual=hashlib.sha256(data).hexdigest()
        expected=self.boundary.expected(path,self.kind)
        if actual!=expected: raise ImportError('module source changed: '+str(path))
        self.boundary.loaded[fullname]=dict(path=str(path),sha256=actual,kind=self.kind)
        # Compile source bytes directly: never trust or write historical __pycache__.
        return self.source_to_code(data,str(path))


class Boundary(importlib.abc.MetaPathFinder):
    def __init__(self, base, extension, identity, extensions, source_report, native_sources):
        self.base=base; self.extension=extension; self.identity=identity
        self.extensions=extensions; self.loaded={}
        self.receiver=None; self.receiver_root=None; self.expected_receiver=None
        self.native_hashes={str((base/e['path']).resolve()):e['actual_sha256'] for e in source_report['files']}
        self.native_sources=native_sources
        for relative,value in native_sources.items():
            path=str((base/relative).resolve())
            if path in self.native_hashes and self.native_hashes[path]!=value:
                raise ValueError('Native source changed between lock audit and inventory: '+path)
            self.native_hashes[path]=value
    def expected(self,path,kind):
        if kind=='extension': return self.extensions[str(path.relative_to(self.extension))]
        if kind=='receiver': return self.receiver['python_sources'][str(path.relative_to(self.receiver_root))]
        if str(path) not in self.native_hashes: raise ImportError('Native module was not in the verified source inventory: '+str(path))
        return self.native_hashes[str(path)]
    def find_spec(self,fullname,path=None,target=None):
        if not managed(fullname): return None
        if fullname in EXTENSION_NAMES:
            origin=self.extension/'framework/python/qwen35_dflash/ascend310p'/(fullname.rsplit('.',1)[1]+'.py')
            return importlib.util.spec_from_file_location(fullname,origin,loader=VerifiedLoader(fullname,origin,self,'extension'))
        if fullname=='qwen35_dflash': search=[str(self.base/'framework/python')]
        elif fullname=='models': search=[str(self.base)]
        else:
            search=list(path or ())
            if not search or any(not Path(p).resolve().is_relative_to(self.base) and
                    not (fullname.startswith('models.') and self.receiver_root and Path(p).resolve().is_relative_to(self.receiver_root)) for p in search):
                raise ImportError('mixed Native package search path: '+fullname)
        native_search=[p for p in search if Path(p).resolve().is_relative_to(self.base)]
        # Only models' original receiver fallback may search outside Native Base.
        # It cannot override an existing historical module, regardless of path order.
        spec=importlib.machinery.PathFinder.find_spec(fullname,native_search)
        kind='native'
        if spec is None and fullname.startswith('models.') and self.receiver_root:
            receiver_search=[p for p in search if Path(p).resolve().is_relative_to(self.receiver_root)]
            spec=importlib.machinery.PathFinder.find_spec(fullname,receiver_search)
            kind='receiver'
        allowed_root=self.base if kind=='native' else self.receiver_root
        if spec is None or not spec.origin or not Path(spec.origin).resolve().is_relative_to(allowed_root):
            raise ImportError('module is not in verified Native Base: '+fullname)
        origin=Path(spec.origin).resolve()
        if fullname=='models.export_model_wrapper_qwen3_5' and self.receiver_root and origin!=self.receiver_root/'export_model_wrapper_qwen3_5.py':
            raise ImportError('receiver wrapper is shadowed outside its Native manifest binding')
        if origin.suffix!='.py': raise ImportError('Native source module must be Python source: '+fullname)
        return importlib.util.spec_from_file_location(fullname,origin,loader=VerifiedLoader(fullname,origin,self,kind),
            submodule_search_locations=list(spec.submodule_search_locations) if spec.submodule_search_locations is not None else None)
    def verify_loaded(self):
        if extension_inventory(self.extension)!=self.extensions:
            raise RuntimeError('Diagnostic Extension files changed after boundary selection')
        if native_inventory(self.base)!=self.native_sources:
            raise RuntimeError('Native Python source inventory changed after boundary selection')
        if self.receiver is not None:
            if native_inventory(self.receiver_root,('.',))!=self.receiver['python_sources'] or digest(self.receiver['input_manifest']['path'])!=self.receiver['input_manifest']['sha256']:
                raise RuntimeError('hash-bound receiver source changed')
        records={}
        for name,module in tuple(sys.modules.items()):
            if not managed(name): continue
            observed=self.loaded.get(name)
            actual=getattr(module,'__file__',None)
            if not observed or observed.get('module_id')!=id(module) or not actual or str(Path(actual).resolve())!=observed['path']:
                raise RuntimeError(f'untracked or mixed module origin: {name}; expected={observed}; actual={actual}')
            if digest(actual)!=observed['sha256']: raise RuntimeError('loaded module file changed: '+name)
            package_paths=list(getattr(module,'__path__',()))
            for entry in package_paths:
                resolved=Path(entry).resolve()
                if not resolved.is_relative_to(self.base) and not (name=='models' or name.startswith('models.')):
                    raise RuntimeError('mixed framework package path: '+name)
                if not resolved.is_relative_to(self.base) and not (self.receiver_root and resolved.is_relative_to(self.receiver_root)):
                    raise RuntimeError('unbound receiver package path: '+name)
            records[name]=dict(observed,__file__=actual,package_paths=package_paths)
        return records
    def bind_receiver(self,config,native):
        manifest=Path(config['input_manifest']).expanduser()
        metadata=native['metadata']
        if manifest.is_symlink() or digest(manifest)!=metadata['quant_input_manifest_sha256']:
            raise ValueError('receiver input manifest differs from frozen Native AIR')
        manifest=manifest.resolve()
        payload=json.loads(manifest.read_text())
        from qwen35_dflash.ascend310p.quant_factory import QUANT_BASE_REVISION
        if payload.get('schema_version')!=1 or payload.get('artifact_kind')!='qwen35-quant-air-om-input-manifest' or payload.get('status')!='PASS' or payload.get('quant_branch_base_revision')!=QUANT_BASE_REVISION:
            raise ValueError('invalid Native quant input manifest')
        from qwen35_dflash.ascend310p.input_manifest import _verify_file_group
        group=payload['groups']['receiver_wrapper']
        if group['sha256']!=metadata['target_input_identity']['receiver_wrapper']:
            raise ValueError('Native receiver wrapper group identity differs')
        paths=_verify_file_group('receiver_wrapper',group)
        root=Path(config['receiver_models_dir']).expanduser()
        if root.is_symlink(): raise ValueError('receiver root must not be a symlink')
        root=root.resolve()
        if str(root)!=metadata['receiver_models_dir'] or paths!=(root/'export_model_wrapper_qwen3_5.py',):
            raise ValueError('receiver module root differs from Native config/manifest')
        record=dict(root=str(root),input_manifest=dict(path=str(manifest),sha256=digest(manifest)),
                    wrapper_group=group,python_sources=native_inventory(root,('.',)))
        if self.expected_receiver is not None and self.expected_receiver!=record:
            raise ValueError('receiver sources differ from accepted Preflight')
        if self.receiver is not None and self.receiver!=record:
            raise ValueError('receiver binding changed during worker')
        self.receiver=record; self.receiver_root=root
        protect_writes(root)
        return record
    def before_factory(self,native,*,require_receiver=False):
        if require_receiver and self.receiver is None: raise RuntimeError('Native receiver binding was not preflighted')
        self.verify_loaded()
        from qwen35_dflash.ascend310p import quant_factory
        self.verify_loaded()
        expected=self.base/'framework/python/qwen35_dflash/ascend310p/quant_factory.py'
        verifier=quant_factory._verify_quant_source_lock
        if any(Path(fn.__code__.co_filename).resolve()!=expected for fn in
               (verifier,quant_factory.create_quant_incremental_graphs,quant_factory.create_quant_recompute_graph)):
            raise RuntimeError('Factory verifier code is not historical')
        actual=verifier()  # The original function executes; no replacement result.
        if actual!=self.identity or actual!=native['metadata']['quant_source_lock']:
            raise RuntimeError('historical Factory returned a different source identity')
        return actual


def protect_writes(base):
    def inside(path):
        if not isinstance(path,(str,bytes,os.PathLike)): return False
        return Path(os.fsdecode(path)).resolve().is_relative_to(base)
    def audit(event,args):
        paths=[]
        if event=='open':
            path,mode,flags=args
            if (mode and any(m in mode for m in 'wax+')) or flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND): paths=[path]
        elif event in ('os.remove','os.rmdir','os.mkdir','os.chmod','os.chown','os.truncate','os.utime'): paths=args[:1]
        elif event in ('os.rename','os.link'): paths=args[:2]
        elif event=='os.symlink': paths=args[1:2]
        if any(inside(p) for p in paths): raise PermissionError('Native Base is read-only: '+event)
    sys.addaudithook(audit)


def activate(state, extension, *, expected=None):
    global _ACTIVE
    preloaded=sorted(n for n in sys.modules if managed(n))
    if _ACTIVE is not None or preloaded:
        raise RuntimeError('Native boundary requires a fresh process; refusing preloaded modules: '+str(preloaded))
    base,identity,air=bound_native(state); extension=Path(extension).resolve()
    if base==extension or base.is_relative_to(extension) or extension.is_relative_to(base):
        raise ValueError('Native Base and Diagnostic Extension roots must be separate')
    if any((base/'framework/python/qwen35_dflash/ascend310p'/(n.rsplit('.',1)[1]+'.py')).exists() for n in EXTENSION_NAMES):
        raise ValueError('Diagnostic Extension module name collides with Native Base')
    output=Path(state['args']['output_dir']).resolve()
    if output.is_relative_to(base): raise ValueError('evidence must be outside read-only Native Base')
    from source_lock import audit_source_lock
    audit=audit_source_lock(base,identity)
    if audit['status']!='PASS': raise ValueError('Native Base complete source lock audit failed: '+json.dumps(audit))
    extensions=extension_inventory(extension)
    native_sources=native_inventory(base)
    if expected is not None:
        if (expected.get('status')!='PASS' or expected.get('native_base')!=str(base)
                or expected.get('extension_root')!=str(extension) or expected.get('identity')!=identity
                or expected.get('extensions')!=extensions or expected.get('native_python_sources')!=native_sources
                or expected.get('native_air')!=air or expected.get('accepted_args')!=state['args']
                or expected.get('extension_revision')!=revision(extension)
                or expected.get('native_checkout_revision')!=revision(base)):
            raise ValueError('Native/Extension boundary differs from accepted Preflight')
    boundary=Boundary(base,extension,identity,extensions,audit,native_sources)
    if expected is not None: boundary.expected_receiver=expected.get('receiver')
    sys.dont_write_bytecode=True
    protect_writes(base)
    sys.meta_path.insert(0,boundary); _ACTIVE=boundary
    return boundary,dict(status='RUNNING',pid=os.getpid(),native_base=str(base),extension_root=str(extension),
        identity=identity,native_air=air,source_audit=audit,extensions=extensions,native_python_sources=native_sources,
        accepted_args=state['args'],
        native_profile_revision=NATIVE_REVISION,native_checkout_revision=revision(base),
        extension_revision=revision(extension),extension_revision_scope='Git HEAD; file hashes bind actual working tree content',
        export_settings=dict(mode='draft-only',gears=[16,64],variants=['off','a','c','ac'],
            atc=state['args'].get('atc','atc'),precision_mode='must_keep_origin_dtype',deterministic=0,
            timing_protocol='continuous-v1',warmup=5,repetitions=30,native_golden='REUSE_ONLY',
            target_execution='NOT_RUN',end_to_end='NOT_RUN'),
        scope='import/source boundary only; Draft GE/ATC/NPU NOT_RUN')


def active():
    return _ACTIVE


def validate_preflight_boundary(root, request, *, accepted_args):
    """Coordinator-side, stdlib-only recheck before any OPP build."""
    root=Path(root).resolve(); path=root/'native-base-preflight.json'
    report=json.loads(path.read_text())
    base,identity,_=bound_native(request)
    extension=Path(__file__).resolve().parents[4]
    if (report.get('status')!='PASS' or report.get('accepted_args')!=accepted_args or report.get('request_sha256')!=digest(root/'preflight-input.json')
            or report.get('native_base')!=str(base) or report.get('identity')!=identity
            or report.get('extension_root')!=str(extension)
            or report.get('extensions')!=extension_inventory(extension)
            or report.get('native_python_sources')!=native_inventory(base)
            or report.get('extension_revision')!=revision(extension)
            or report.get('native_checkout_revision')!=revision(base)):
        raise ValueError('Native boundary Preflight missing, failed or changed')
    receiver=report.get('receiver')
    if receiver is not None:
        if (native_inventory(Path(receiver['root']),('.',))!=receiver['python_sources'] or
                digest(receiver['input_manifest']['path'])!=receiver['input_manifest']['sha256']):
            raise ValueError('receiver source changed after Preflight')
    from source_lock import audit_source_lock
    if audit_source_lock(base,identity)['status']!='PASS':
        raise ValueError('historical source changed across process boundary')
    return dict(path=path.name,bytes=path.stat().st_size,sha256=digest(path))


def check_extension_compatibility(config, native=None):
    if _ACTIVE is None: raise RuntimeError('missing isolated Native Base worker')
    if config.get('draft_weight_prepack') is not None or not config.get('draft_weight_prepack_manifest'):
        raise ValueError('319f51d diagnostic requires the existing immutable NZ Manifest; no automatic prepack')
    import importlib
    import inspect
    for name in ('quant_factory','incremental','draft_constants','exporter','runtime_input_export',
                 'weight_prepack','compiler','common_reuse','custom_op_export'):
        importlib.import_module('qwen35_dflash.ascend310p.'+name)
    for name in EXTENSION_NAMES: importlib.import_module(name)
    if native is not None:
        _ACTIVE.bind_receiver(config,native)
        from qwen35_dflash.ascend310p.quant_factory import _append_receiver_models
        _append_receiver_models(_ACTIVE.receiver_root)
        importlib.import_module('models.export_model_wrapper_qwen3_5')
    from qwen35_dflash.ascend310p.weight_prepack import prepack_weight_quant_constants
    signature=inspect.signature(prepack_weight_quant_constants)
    signature.bind(object(),object(),object(),object())
    return dict(status='PASS',native_prepack_signature=str(signature),receiver=_ACTIVE.receiver,modules=_ACTIVE.verify_loaded())
