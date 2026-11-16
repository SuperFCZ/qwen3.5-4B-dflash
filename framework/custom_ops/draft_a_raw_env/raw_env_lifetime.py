"""One source-bound PyTorch child-lifetime module; no /tmp-wide exception."""
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import raw_common as c

NAME='_remote_module_non_scriptable'
GENERATOR='torch.distributed.nn.jit.instantiator'
TEMPLATE='torch.distributed.nn.jit.templates.remote_module_template'
BYTES=2355
SHA256='8205b16956fb264841ecd8644784a0d157f87df79b17c16825dc1163433ce5d8'
LIFETIME='PYTORCH_CHILD_TEMPORARY_MODULE_V1'


def _payload(path):
    data=Path(path).read_bytes()
    if len(data)!=BYTES or hashlib.sha256(data).hexdigest()!=SHA256:
        raise ValueError('unreviewed generated PyTorch module content')
    return data


def capture(records,output):
    """Check the live module and owner before the interpreter's atexit cleanup."""
    persistent=dict(records);generated={}
    if NAME not in sys.modules:
        if NAME in records:raise ValueError('generated module record without live module')
        return persistent,generated
    if NAME not in records:raise ValueError('live generated module has no readable source identity')
    module=sys.modules[NAME];owner=sys.modules.get(GENERATOR);template=sys.modules.get(TEMPLATE)
    torch=sys.modules.get('torch')
    if not owner or not template or not torch or str(torch.__version__)!='2.8.0+cpu':
        raise ValueError('unreviewed generated module producer/version')
    td=getattr(owner,'_TEMP_DIR',None)
    if not isinstance(td,tempfile.TemporaryDirectory):raise ValueError('generated module owner is not TemporaryDirectory')
    temp=Path(td.name).resolve(strict=True)
    if (Path(owner.INSTANTIATED_TEMPLATE_DIR_PATH).resolve(strict=True)!=temp or
            not any(Path(p).is_absolute() and Path(p).resolve()==temp for p in sys.path)):
        raise ValueError('generated module owner/sys.path mismatch')
    source=temp/(NAME+'.py');item=persistent[NAME]
    if source.is_symlink() or Path(module.__file__).resolve(strict=True)!=source or Path(module.__spec__.origin).resolve(strict=True)!=source:
        raise ValueError('generated module origin differs from live owner')
    if Path(item['alias']).resolve(strict=True)!=source or item['file']!=c.identity(source):raise ValueError('generated module identity differs')
    package=Path(torch.__file__).resolve(strict=True).parent
    producer_files={GENERATOR:package/'distributed/nn/jit/instantiator.py',
        TEMPLATE:package/'distributed/nn/jit/templates/remote_module_template.py'}
    for name,path in producer_files.items():
        value=sys.modules[name]
        if Path(value.__file__).resolve(strict=True)!=path or persistent.get(name,{}).get('file')!=c.identity(path):
            raise ValueError('generated module producer source mismatch: '+name)
    data=_payload(source)
    directory=Path(output).resolve(strict=True)/'module-evidence';directory.mkdir(exist_ok=False)
    archive=directory/(NAME+'.snapshot')
    with archive.open('xb') as stream:stream.write(data)
    if c.identity(source)!=item['file'] or _payload(archive)!=data:raise ValueError('generated source changed while sealing')
    generated[NAME]=dict(lifetime=LIFETIME,original_alias=item['alias'],original_realpath=item['file']['path'],
        observed_bytes=BYTES,observed_sha256=SHA256,owner_temp_root=str(temp),owner_temp_alias=str(Path(td.name).absolute()),capture_pid=os.getpid(),
        live_check='OWNER_ORIGIN_AND_EXACT_CONTENT_VERIFIED',snapshot=c.identity(archive),
        producer={name:persistent[name] for name in producer_files})
    del persistent[NAME]
    return persistent,generated


def verify(generated,persistent,output):
    if set(generated)-{NAME}:raise ValueError('unknown process-scoped module')
    if NAME in persistent:raise ValueError('generated module must have a lifecycle binding')
    for name,item in generated.items():
        expected={'lifetime','original_alias','original_realpath','observed_bytes','observed_sha256','owner_temp_root','owner_temp_alias',
            'capture_pid','live_check','snapshot','producer'}
        if set(item)!=expected or item['lifetime']!=LIFETIME or item['live_check']!='OWNER_ORIGIN_AND_EXACT_CONTENT_VERIFIED':
            raise ValueError('generated module lifecycle evidence incomplete')
        if item['observed_bytes']!=BYTES or item['observed_sha256']!=SHA256 or type(item['capture_pid']) is not int or item['capture_pid']<=0:
            raise ValueError('generated module observed identity differs')
        origin=Path(item['owner_temp_root'])/(name+'.py')
        alias=Path(item['owner_temp_alias'])/(name+'.py')
        if not origin.is_absolute() or not alias.is_absolute() or item['original_alias']!=str(alias) or item['original_realpath']!=str(origin):
            raise ValueError('generated module observed owner/origin differs')
        snapshot=c.verify(item['snapshot'])
        expected_path=Path(output).resolve(strict=True)/'module-evidence'/(name+'.snapshot')
        if snapshot!=expected_path:raise ValueError('generated snapshot outside child evidence directory')
        _payload(snapshot)
        if set(item['producer'])!={GENERATOR,TEMPLATE}:raise ValueError('generated module producer inventory differs')
        for key,value in item['producer'].items():
            if persistent.get(key)!=value:raise ValueError('generated producer is not the verified loaded module: '+key)
            if str(Path(value['alias']).resolve(strict=True))!=value['file']['path']:raise ValueError('producer alias changed')
            c.verify(value['file'])


def comparison(generated):
    """Only random child origins/archive locations are excluded, never content."""
    return {name:{k:item[k] for k in ('lifetime','observed_bytes','observed_sha256','live_check','producer')} for name,item in generated.items()}
