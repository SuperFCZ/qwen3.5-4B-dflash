"""Static, fail-closed ATC entrypoint -> ELF dependency audit. Never run ATC."""
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
from platform_common import c

# Exact server-confirmed wrapper, also read from the local CANN 9.0.0 install.
# No SDK script body is copied into this repository. Unknown scripts are rejected.
WRAPPER_SHA='90aa69b5e18a19e65d1f934f1cb68089f9d8092a4dc00e2fb1152ca2b7c15f35'
REVIEWED_WRAPPERS={WRAPPER_SHA:dict(
    name='CANN9_ATC_BASH_WRAPPER_1233',bytes=1233,mode=0o555,shebang='#!/bin/bash',interpreter='/bin/bash',
    target_bytes=670800,target_sha256='a32df12914369e4f41a60deff26e8b641c877c070f02ecebfacb121aa5703170',target_mode=0o555)}
ENV_KEYS=('PATH','LD_LIBRARY_PATH','PYTHONPATH')
FORBIDDEN_ENV=('BASH_ENV','ENV','SHELLOPTS','BASHOPTS','LD_PRELOAD','LD_AUDIT','LD_TRACE_LOADED_OBJECTS','CDPATH','GLOBIGNORE')
SCOPE='STATIC_WRAPPER_COMMANDS_AND_ELF_LDD; not ATC execution, Python imports or runtime dlopen coverage'


def snapshot(alias,executable=False):
    alias=Path(alias)
    if not alias.is_absolute():raise ValueError('dependency alias must be absolute: '+str(alias))
    target,links=c.alias_target(alias)
    if not target.is_file():raise ValueError('dependency is not a regular file: '+str(alias))
    before=target.stat();item=c.identity(target);after=target.stat()
    if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_mode)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_mode):raise ValueError('dependency changed while hashing: '+str(alias))
    mode=stat.S_IMODE(after.st_mode);can_execute=bool(mode&0o111) and os.access(alias,os.X_OK)
    if executable and not can_execute:raise ValueError('dependency is not executable: '+str(alias))
    return dict(alias=str(alias),links=links,file=item,mode=mode,mode_octal=oct(mode),executable=can_execute)


def verify_snapshot(row):
    if snapshot(row['alias'],row['executable'])!=row:raise ValueError('dependency identity/alias/permissions changed: '+row['alias'])


def is_elf(path):
    with Path(path).open('rb') as f:return f.read(4)==b'\x7fELF'


def path_tool(name,env):
    path=env.get('PATH','')
    if not path or any(not x or not Path(x).is_absolute() for x in path.split(os.pathsep)):
        raise ValueError('static dependency audit needs explicit absolute PATH entries')
    found=shutil.which(name,path=path)
    if not found:raise ValueError('dependency tool not found in selected PATH: '+name)
    return snapshot(found,True)


def check_environment(env):
    bad=[k for k in FORBIDDEN_ENV if env.get(k)]
    bad += [k for k in env if k.startswith('BASH_FUNC_')]
    if bad:raise ValueError('static dependency audit incomplete: shell/loader overrides: '+','.join(sorted(bad)))
    # Empty/relative loader entries depend on a later ATC cwd and cannot be bound here.
    if env.get('LD_LIBRARY_PATH') and any(not x or not Path(x).is_absolute() for x in env['LD_LIBRARY_PATH'].split(':')):
        raise ValueError('static dependency audit incomplete: cwd-dependent LD_LIBRARY_PATH')


def plan(expected,alias,cann_root,env):
    """Only an exact reviewed script can select a shell call-chain profile."""
    c.verify(expected);entry=snapshot(alias,True);root=Path(cann_root).resolve(strict=True)
    if entry['file']!=expected:raise ValueError('ATC entrypoint differs from historical identity')
    real=Path(expected['path'])
    if root not in real.parents:raise ValueError('ATC entrypoint outside selected CANN')
    check_environment(env)
    base={k:env.get(k) for k in ENV_KEYS};commands=[]
    if is_elf(real):
        kind='DIRECT_ELF';profile=None;exports={}
        commands.append(dict(role='atc',executable=entry,environment=base))
    else:
        profile=REVIEWED_WRAPPERS.get(expected['sha256'])
        if not profile:raise ValueError('dependency audit incomplete: unreviewed ATC wrapper '+str(real))
        if expected['bytes']!=profile['bytes'] or entry['mode']!=profile['mode'] or real.read_bytes().splitlines()[0].decode()!=profile['shebang']:
            raise ValueError('reviewed ATC wrapper size/mode/shebang changed')
        # Wrapper: realpath($0); dirname(realpath); cd/pwd -> PKG_PATH; then bin/atc.bin.
        # cd, pwd and export are Bash builtins. realpath and dirname use inherited PATH.
        if real.parent.name!='bin' or re.search(r'[\s*?\[\]]',str(real)):
            raise ValueError('wrapper path does not satisfy reviewed unquoted expansion contract')
        package=real.parent.parent
        target=snapshot(package/'bin/atc.bin',True)
        if root not in Path(target['file']['path']).parents:raise ValueError('wrapper target escapes selected CANN')
        if (target['file']['bytes'],target['file']['sha256'],target['mode'])!=(profile['target_bytes'],profile['target_sha256'],profile['target_mode']):
            raise ValueError('ATC wrapper target differs from server-confirmed ELF identity: '+str(package/'bin/atc.bin')+'; observed '+str((target['file']['bytes'],target['file']['sha256'],target['mode_octal'])))
        commands.append(dict(role='bash_interpreter',executable=snapshot(profile['interpreter'],True),environment=base))
        for name in ('realpath','dirname'):commands.append(dict(role=name,executable=path_tool(name,env),environment=base))
        exports=dict(LD_LIBRARY_PATH=str(package/'lib64')+':'+env.get('LD_LIBRARY_PATH',''),
                     PYTHONPATH=str(package/'python/site-packages')+':'+env.get('PYTHONPATH',''))
        effective=dict(env,**exports);check_environment(effective)
        commands.append(dict(role='atc.bin',executable=target,environment={k:effective.get(k) for k in ENV_KEYS}))
        kind='REVIEWED_SHELL_WRAPPER'
    for row in commands:
        if not is_elf(row['executable']['file']['path']):raise ValueError('dependency audit incomplete: non-ELF command '+row['role'])
    return dict(entry=entry,kind=kind,profile=profile,wrapper_exports=exports,commands=commands,
                inherited_environment=base,cann_root=str(root),scope=SCOPE)


def parse_ldd(text):
    """Require every non-vDSO row to name a resolved absolute file."""
    if re.search(r'\bnot found\b',text,re.I):raise ValueError('ELF dynamic dependency not found')
    paths=set()
    for raw in text.splitlines():
        line=raw.strip()
        if not line:continue
        if re.fullmatch(r'linux-(?:vdso|gate)(?:64)?\.so\.\d+\s+\(0x[0-9a-fA-F]+\)',line):continue
        m=re.fullmatch(r'(?:\S+\s+=>\s+)?(/.+?)\s+\(0x[0-9a-fA-F]+\)',line)
        if not m:raise ValueError('unresolved/unknown ldd record: '+line)
        paths.add(m.group(1))
    if not paths:raise ValueError('ELF dynamic dependency list is empty')
    return [snapshot(p) for p in sorted(paths)]


def audit(expected,alias,cann_root,env,output):
    """Write failure evidence too. Only ldd is launched; no shell/ATC command is run."""
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    r=dict(status='BLOCKED',dependency_audit='INCOMPLETE',scope=SCOPE,atc='NOT_RUN',aclnn='NOT_RUN',npu='NOT_RUN',executables=[])
    try:
        r['expected_entry']=expected;r['entry_observation']=snapshot(alias,True)
        r['chain']=plan(expected,alias,cann_root,env);r['ldd_tool']=path_tool('ldd',env)
        for i,item in enumerate(r['chain']['commands']):
            row=dict(item,returncode=None,dependencies=[]);r['executables'].append(row)
            verify_snapshot(item['executable']);verify_snapshot(r['ldd_tool'])
            child_env=dict(env)
            for key,value in item['environment'].items():
                if value is None:child_env.pop(key,None)
                else:child_env[key]=value
            # Stable ldd diagnostics, without changing loader/PATH selection.
            child_env['LC_ALL']='C'
            out=root/(str(i)+'-'+item['role']+'.stdout.log');err=root/(str(i)+'-'+item['role']+'.stderr.log')
            row['argv']=[r['ldd_tool']['alias'],item['executable']['alias']]
            try:
                with out.open('xb') as f,err.open('xb') as g:
                    row['returncode']=subprocess.run(row['argv'],stdout=f,stderr=g,env=child_env,check=False).returncode
            finally:
                for key,path in (('stdout',out),('stderr',err)):
                    if path.is_file():row[key]=c.identity(path)
            if row['returncode']!=0:raise ValueError('ldd returned '+str(row['returncode'])+' for '+item['role'])
            if err.read_bytes():raise ValueError('ldd stderr requires review: '+item['role'])
            row['dependencies']=parse_ldd(out.read_text())
            verify_snapshot(item['executable'])
        r.update(status='ATC_STATIC_ELF_DEPENDENCIES_BOUND',dependency_audit='COMPLETE_STATIC_WRAPPER_CHAIN')
        verify(r)
    except Exception as e:r.update(status='BLOCKED',dependency_audit='INCOMPLETE',error=f'{type(e).__name__}: {e}')
    c.write(root/'report.json',r)
    if r['status']=='BLOCKED':raise ValueError('ATC dependency audit incomplete: '+r['error']+'; report: '+str(root/'report.json'))
    return r


def verify(r):
    if not isinstance(r,dict) or r.get('status')!='ATC_STATIC_ELF_DEPENDENCIES_BOUND' or r.get('dependency_audit')!='COMPLETE_STATIC_WRAPPER_CHAIN':raise ValueError('ATC dependency audit was not completed')
    c.verify_tree(r);verify_snapshot(r['chain']['entry']);verify_snapshot(r['ldd_tool'])
    chain=r['chain']
    if r['expected_entry']!=chain['entry']['file'] or r['entry_observation']!=chain['entry']:raise ValueError('ATC entrypoint historical/observed identity differs')
    env={k:v for k,v in chain['inherited_environment'].items() if v is not None}
    if plan(chain['entry']['file'],chain['entry']['alias'],chain['cann_root'],env)!=chain:raise ValueError('recorded ATC static chain differs from reviewed wrapper')
    rows=r['executables'];commands=chain['commands']
    if len(rows)!=len(commands) or not rows:raise ValueError('incomplete ELF call-chain inventory')
    for row,command in zip(rows,commands):
        if any(row.get(k)!=v for k,v in command.items()) or row['returncode']!=0:raise ValueError('ELF dependency call-chain differs')
        verify_snapshot(row['executable'])
        if not is_elf(row['executable']['file']['path']):raise ValueError('bound ELF type changed')
        if row['argv']!=[r['ldd_tool']['alias'],row['executable']['alias']]:raise ValueError('ldd command identity differs')
        if c.verify(row['stderr']).read_bytes():raise ValueError('bound ldd stderr is not empty')
        actual=parse_ldd(c.verify(row['stdout']).read_text())
        if actual!=row['dependencies']:raise ValueError('ELF library identity/alias/permissions changed')
