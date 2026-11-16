"""Unique independent experiment utilities. Import performs no device operations."""
import hashlib,json,os,re,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[2]
AC=HERE.parent/'draft_ac';QUANT=HERE.parent/'draft_quant'
sys.path[:0]=[str(AC/'test'),str(QUANT/'test')]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def identity(path):
    p=Path(path).resolve(strict=True)
    return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))


def verify(item):
    if not Path(item['path']).is_absolute():raise ValueError('identity needs an explicitly bound absolute path: '+item['path'])
    if identity(item['path'])!=item:raise ValueError('file identity changed: '+item['path'])
    return Path(item['path'])


def read(path):return json.loads(Path(path).read_text())


def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def linked(base,item):
    p=Path(item['path']);p=p if p.is_absolute() else Path(base)/p
    if sha(p)!=item['sha256'] or 'bytes' in item and p.stat().st_size!=item['bytes']:raise ValueError('linked file changed: '+str(p))
    return p.resolve(strict=True)


HEAD_ALIAS_KIND='historical-head-library-alias-v1'


def alias_target(path):
    """Observe every link component, without creating or rewriting a link."""
    path=Path(path)
    if not path.is_absolute():raise ValueError('library alias must be absolute: '+str(path))
    pending=list(path.parts[1:]);current=Path(path.anchor);links=[]
    while pending:
        part=pending.pop(0)
        if part=='..':current=current.parent;continue
        current=current/part
        if current.is_symlink():
            target=os.readlink(current);links.append(dict(path=str(current),target=target))
            if len(links)>40:raise ValueError('library symlink loop or excessive depth: '+str(path))
            destination=Path(target)
            if destination.is_absolute():current=Path(destination.anchor);parts=destination.parts[1:]
            else:current=current.parent;parts=destination.parts
            pending=list(parts)+pending
    resolved=path.resolve(strict=True)
    if current!=resolved:raise ValueError('library alias resolution changed: '+str(path))
    return resolved,links


def verify_head_alias(binding):
    if (set(binding)!={'kind','install_root','resolved_install_root','alias','links','file'} or
            binding['kind']!=HEAD_ALIAS_KIND):raise ValueError('historical Head alias schema differs')
    root=Path(binding['install_root']);alias=Path(binding['alias'])
    if not root.is_absolute() or not alias.is_absolute() or root not in alias.parents:
        raise ValueError('historical Head library outside declared OPP: '+str(alias))
    real_root=root.resolve(strict=True)
    if str(real_root)!=binding['resolved_install_root']:raise ValueError('historical Head OPP root retargeted')
    target,links=alias_target(alias)
    if real_root not in target.parents:raise ValueError('historical Head alias escapes OPP: '+str(alias))
    if str(target)!=binding['file']['path'] or links!=binding['links']:
        raise ValueError('historical Head alias target/chain changed: '+str(alias))
    verify(binding['file'])  # Original expected bytes/SHA; no acceptance of new content.
    if alias_target(alias)!=(target,links):raise ValueError('historical Head alias changed during verification')


def verify_tree(v):
    if isinstance(v,dict):
        if v.get('kind')==HEAD_ALIAS_KIND:verify_head_alias(v)
        elif set(v)=={'path','bytes','sha256'}:verify(v)
        else:
            for x in v.values():verify_tree(x)
    elif isinstance(v,list):
        for x in v:verify_tree(x)


def sources():
    files=[p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    # Reused diagnostic code is bound separately from immutable Native Source Lock.
    files+=list((AC/'test').glob('*.py'))+list((QUANT/'test').glob('*.py'))
    files+=list((REPO/'framework/python/qwen35_dflash/ascend310p').glob('draft_ac_*.py'))
    for folder in (QUANT,HERE.parent/'draft_head'):
        for sub in ('op_host','op_kernel'):files += [p for p in (folder/sub).iterdir() if p.is_file()]
    files += list((QUANT/'test').glob('*.cpp'))+list((QUANT/'test').glob('*.h'))+list(QUANT.glob('run*.sh'))
    files += [QUANT/'DFlashGroupQuantLinear.json',QUANT/'test/CMakeLists.txt',HERE.parent/'draft_head/DFlashDraftLmHeadTop1.json']
    return [identity(p) for p in sorted(set(files))]


def fresh(path,protected=()):
    p=Path(path).resolve()
    for root in (REPO,*protected):
        root=Path(root).resolve()
        if p==root or root in p.parents:raise ValueError('output inside protected evidence/source: '+str(root))
    p.mkdir(parents=True,exist_ok=False);return p


def clean():
    if any(n in ('torch','models') or n.startswith(('torch.','torch_npu','qwen35_dflash','models.')) for n in sys.modules):
        raise ValueError('coordinator/evidence process must not load model/device modules')


def verify_opp(opp):
    root=Path(opp['install']).resolve();pf=opp['preflight'];vendor=root/'vendors/customize'
    if Path(pf['vendor']).resolve()!=vendor:raise ValueError('private vendor root differs')
    expected=set()
    for row in pf['files']:
        p=Path(row['path']).resolve(strict=True)
        if vendor not in p.parents or sha(p)!=row['sha256']:raise ValueError('OPP library/header identity: '+str(p))
        expected.add(str(p))
    actual={str(p.resolve()) for p in vendor.rglob('*') if p.is_file() and (p.suffix in ('.h','.so','.json') or '.so.' in p.name)}
    if expected!=actual:raise ValueError('OPP file inventory differs')
    binaries=set()
    for row in pf['kernels']:
        for key,hash_key in [('binary','sha256'),('json','json_sha256')]:
            p=Path(row[key]).resolve(strict=True)
            if vendor not in p.parents or sha(p)!=row[hash_key]:raise ValueError('OPP Kernel identity: '+str(p))
        binaries.add(str(Path(row['binary']).resolve()))
    if not binaries or binaries!={str(p.resolve()) for p in (vendor/'op_impl/ai_core/tbe/kernel/ascend310p').rglob('*.o')}:raise ValueError('OPP Kernel inventory differs')


def command(argv,log,env=None):
    args=list(map(str,argv));write(log.with_suffix('.command.json'),args)
    with log.open('wb') as f:rc=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,env=env).returncode
    if rc:raise RuntimeError(f'command returned {rc}: {log}')


def dependencies(runner,log,env=None):
    command(['ldd',verify(runner)],log,env)
    text=log.read_text()
    if 'not found' in text:raise ValueError('Runner unresolved libraries: '+str(log))
    paths=set(re.findall(r'(?:=>\s+|^\s*)(/\S+)\s+\(',text,re.M))
    if not paths:raise ValueError('empty Runner dependency evidence')
    return [dict(alias=p,file=identity(p)) for p in sorted(paths)]


def selected_env(evidence,opp):
    env=os.environ.copy();old=evidence['historical_environment'];hroot=Path(evidence['H']['opp_root']).resolve()
    for key in ('ASCEND_CUSTOM_OPP_PATH','LD_LIBRARY_PATH'):
        kept=[]
        for entry in old[key].split(os.pathsep):
            if not entry:continue
            p=Path(entry).resolve()
            if p==hroot or hroot in p.parents:continue
            vendor=p.parent.parent if p.name=='lib' and p.parent.name=='op_api' else p
            if any((vendor/'op_impl/ai_core/tbe/kernel/ascend310p'/name).exists() for name in ('d_flash_group_quant_linear','d_flash_draft_lm_head_top1')):
                raise ValueError('ambiguous inherited A/C vendor path: '+entry)
            kept.append(entry)
        prefix=str(Path(opp)/'vendors/customize'/('op_api/lib' if key=='LD_LIBRARY_PATH' else ''))
        env[key]=os.pathsep.join([prefix]+kept)
    if evidence.get('H',{}).get('environment',{}).get('cann_root'):
        cann=Path(evidence['H']['environment']['cann_root']).resolve()
        if env.get('ASCEND_OPP_PATH') and Path(env['ASCEND_OPP_PATH']).resolve()!=(cann/'opp').resolve():raise ValueError('default OPP outside selected CANN')
        env['ASCEND_OPP_PATH']=str(cann/'opp')
    env.update(TORCH_DEVICE_BACKEND_AUTOLOAD='0',PYTHONDONTWRITEBYTECODE='1')
    return env
