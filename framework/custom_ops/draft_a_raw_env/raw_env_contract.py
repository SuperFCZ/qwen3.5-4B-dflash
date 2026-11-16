"""Import-only environment gate outside the frozen draft_a_raw source inventory."""
import ast
import base64
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
RAW=HERE.parent/'draft_a_raw'
sys.path.insert(0,str(RAW))
import raw_common as c
import run_experiment as experiment


def sources():
    return [c.identity(p) for p in sorted(HERE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts]


def previous(root):
    """Use the existing exact Source Hash, artifact and 104-projection verifiers."""
    root=Path(root).resolve(strict=True);first=c.read(root/'precheck/report.json')
    current=dict(sources=c.sources(),device=first['device'],cann_root=first['cann_root'])
    links={};reports={}
    for stage in ('precheck','build','probes'):
        path=root/stage/'report.json';r=experiment.prior(path,stage,current)
        if any(r.get(k+'_report')!=v for k,v in links.items()):raise ValueError('mixed predecessor chain')
        links[stage]=c.identity(path);reports[stage]=r
        if stage=='precheck':current['evidence']=r['evidence']
        elif r['evidence']!=current['evidence']:raise ValueError('predecessor evidence drift')
    if any(len(current['evidence']['projections'][g])!=26 for g in ('16','64')) or len(reports['probes']['runs'])!=104:
        raise ValueError('reuse requires all 104 projection checks')
    return dict(root=str(root),references=links,reports=reports,current=current,
        status='PRECHECK_BUILD_104_PROBES_IDENTITIES_REVERIFIED_NO_REPLAY')


def literal_version(path):
    tree=ast.parse(Path(path).read_text())
    for n in tree.body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='__version__' for t in n.targets):
            if isinstance(n.value,ast.Constant) and isinstance(n.value.value,str):return n.value.value
    raise ValueError('no literal package version in verified historical init: '+str(path))


def historical_environment(evidence):
    h=evidence['H'];state=h['state']
    imports=c.read(c.verify(state['report']['preflight_imports']))
    interface=c.read(c.verify(state['report']['draft_interface']))
    modules={}
    for name in ('torch','transformers'):
        alias=imports['origins'][name];expected=imports['source_sha256'][alias]
        file=c.identity(alias)
        if file['sha256']!=expected:raise ValueError('historical package source changed: '+alias)
        modules[name]=dict(alias=alias,file=file)
    for name in ('torch_npu','torchair'):
        item=interface[name];file=c.identity(item['path'])
        if file['sha256']!=item['sha256']:raise ValueError('historical interface module changed: '+name)
        modules[name]=dict(alias=item['path'],file=file,version=item['version'])
    modules['torch']['version']=h['environment']['torch']
    modules['transformers']['version']=literal_version(c.verify(modules['transformers']['file']))
    command_path=c.verify(h['reports']['worker-input.json']).parent/'preflight-command.json'
    command=c.read(command_path)
    if not isinstance(command,list) or not command or not Path(command[0]).is_absolute():raise ValueError('invalid historical interpreter hint')
    return dict(modules=modules,command=c.identity(command_path),interpreter_hint=command[0],
        hint_scope='Observed historical command; never automatically executed or substituted',
        status='HISTORICAL_PACKAGE_INIT_IDENTITIES_MATCHED')


def python_identity():
    # Preserve the invoked venv alias: executing its resolved base binary would
    # silently switch environments even though the ELF SHA is identical.
    result=dict(executable_alias=sys.executable,binary=c.identity(sys.executable),version=sys.version,
        prefix=sys.prefix,base_prefix=sys.base_prefix,sys_path=list(sys.path),
        environment={k:os.environ.get(k) for k in ('CONDA_PREFIX','VIRTUAL_ENV','PYTHONPATH','PYTHONHOME','TORCH_DEVICE_BACKEND_AUTOLOAD')})
    cfg=Path(sys.prefix)/'pyvenv.cfg'
    result['venv_config']=c.identity(cfg) if cfg.is_file() else None
    return result


def distribution_info(name):
    dist=metadata.distribution(name)
    init=Path(dist.locate_file(name+'/__init__.py')).resolve(strict=True)
    return dict(name=dist.metadata['Name'],version=dist.version,package_init=c.identity(init))


def installed_record(name,relative):
    dist=metadata.distribution(name)
    rows=[f for f in dist.files or [] if str(f)==relative]
    if len(rows)!=1 or rows[0].hash is None or rows[0].hash.mode!='sha256':
        raise ValueError('installed distribution lacks SHA256 RECORD: '+relative)
    row=rows[0];file=c.identity(dist.locate_file(row))
    digest=base64.urlsafe_b64encode(bytes.fromhex(file['sha256'])).decode().rstrip('=')
    if row.hash.value!=digest or row.size is not None and row.size!=file['bytes']:
        raise ValueError('installed distribution RECORD mismatch: '+relative)
    return dict(file=file,record_sha256_base64=row.hash.value,record_bytes=row.size)


def check_package(module,dist,expected):
    actual=c.identity(module.__file__)
    if actual!=dist['package_init']:raise ValueError('package shadowing / distribution origin mismatch')
    if str(module.__version__)!=dist['version']:raise ValueError('distribution/module version mismatch')
    if actual!=expected['file'] or str(module.__version__)!=expected['version']:
        raise ValueError('package does not match historical Native export environment')
    if [str(Path(p).resolve()) for p in module.__path__]!=[str(Path(actual['path']).parent)]:
        raise ValueError('mixed Transformers package search path')
    return dict(file=actual,version=str(module.__version__),distribution=dist)


def direct_imports(base):
    records=[]
    for rel in ('models/modeling_qwen3_5_hiai_nd.py','models/modeling_qwen3_5_hiai_nd_dflash_rollback.py','models/dflash_v1/modeling_qwen3_5_dflash.py'):
        p=Path(base)/rel;tree=ast.parse(p.read_text());lines=[]
        for n in ast.walk(tree):
            if isinstance(n,ast.ImportFrom) and (n.module=='transformers.initialization' or n.module=='transformers' and any(x.name=='initialization' for x in n.names)):
                lines.append(dict(line=n.lineno,module=n.module,names=[x.name for x in n.names]))
        records.append(dict(file=c.identity(p),imports=lines))
    return records


def loaded_files():
    files={}
    for name,module in tuple(sys.modules.items()):
        path=getattr(module,'__file__',None)
        if path and Path(path).is_file():files[name]=dict(alias=str(Path(path).absolute()),file=c.identity(path))
    return files


def verify_loaded_files(records):
    for name,item in records.items():
        if str(Path(item['alias']).resolve(strict=True))!=item['file']['path']:raise ValueError('module alias retargeted: '+name)
        c.verify(item['file'])


def snapshot(child):
    return {k:child[k] for k in ('python','packages','loaded_files','operators')}


def validate_child(child):
    if child.get('status')!='DRAFT_IMPORTS_READY_INTERFACES_NOT_PREPARED' or child.get('error') or child.get('integrity_error'):
        raise ValueError('environment import gate BLOCKED: '+str(child.get('error') or child.get('integrity_error')))
    if child.get('npu')!='NOT_RUN' or child.get('interfaces')!='NOT_PREPARED':raise ValueError('environment gate executed outside import-only scope')
    c.verify_tree(child);verify_loaded_files(child['loaded_files'])


def print_summary(report):
    out={k:report.get(k) for k in ('status','error','integrity_error','npu','export','reuse')}
    out['variants']={k:{field:v.get(field) for field in ('status','error','integrity_error','current_distribution','historical_environment','python','operators')} for k,v in report.get('variants',{}).items()}
    print(json.dumps(out,ensure_ascii=False,indent=2))
