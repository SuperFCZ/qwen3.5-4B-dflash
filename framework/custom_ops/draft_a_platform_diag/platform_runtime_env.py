"""Child-only Diagnostic OPP environment; never append historical user paths."""
import os
import shutil
from pathlib import Path
import sys
from platform_common import c

POLICY='PLATFORM_DIAGNOSTIC_PRIVATE_OPP_ONLY_V1'
DRIVER_ROOT=Path('/usr/local/Ascend/driver')
CANN_LD=('lib64','aarch64-linux/lib64','lib64/plugin/opskernel','lib64/plugin/nnengine',
         'opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64','tools/aml/lib64','tools/aml/lib64/plugin')
DRIVER_LD=('lib64','lib64/common','lib64/driver')
CANN_LIBS=('libascendcl.so','libnnopbase.so','libacl_op_compiler.so','libplatform.so','libregister.so')
DEVICE_KEYS=('ASCEND_VISIBLE_DEVICES','ASCEND_RT_VISIBLE_DEVICES','ASCEND_DEVICE_ID','ASCEND_VNPU_SPECS','ASCEND_RUNTIME_OPTIONS')
KEEP=('HOME','USER','LOGNAME','TMPDIR','TMP','TEMP','LANG','LC_ALL','LC_CTYPE','TZ','VIRTUAL_ENV','CONDA_PREFIX',*DEVICE_KEYS)
OUTPUT_KEYS={'AI_RUN_DIR','ASCEND_WORK_PATH','DFLASH_PLATFORM_DIAG_PATH','DFLASH_PLATFORM_DIAG_ROUTE'}


def directory(path,owner,scope,required=False):
    path=Path(path);owner=Path(owner).resolve()
    if not path.is_dir():
        if required:raise ValueError('required runtime directory missing: '+str(path))
        return None
    target,links=c.alias_target(path)
    if target!=owner and owner not in target.parents:raise ValueError('runtime directory escapes selected '+scope+': '+str(path))
    return dict(alias=str(path),realpath=str(target),links=links,scope=scope)


def library(path,owner,scope):
    path=Path(path);target,links=c.alias_target(path);owner=Path(owner).resolve()
    if owner not in target.parents:raise ValueError('runtime library escapes selected '+scope+': '+str(path))
    return dict(alias=str(path),links=links,file=c.identity(target),scope=scope)


def make_environment(cann_root,opp_root=None,*,tools=None,base=None,ready=True,driver_root=DRIVER_ROOT):
    """Allowlist construction, not a best-effort filter of an inherited LD path."""
    base=dict(os.environ if base is None else base);cann=Path(cann_root).resolve(strict=True);driver=Path(driver_root).resolve(strict=True)
    env={k:base[k] for k in KEEP if k in base};env.setdefault('LANG','C.UTF-8')
    directories=[];ld=[];seen=set()
    def add(path,owner,scope,required=False):
        row=directory(path,owner,scope,required)
        if row and row['realpath'] not in seen:
            seen.add(row['realpath']);directories.append(row);ld.append(row['alias'])
    opp=Path(opp_root).resolve() if opp_root is not None else None
    if opp:
        private=opp/'vendors/customize';api=private/'op_api/lib'
        if ready:add(api,opp,'PRIVATE_DIAGNOSTIC_OPP',True)
        else:ld.append(str(api))  # Only the new output path; not an inherited OPP.
        env['ASCEND_CUSTOM_OPP_PATH']=str(private)
    else:env['ASCEND_CUSTOM_OPP_PATH']=''
    for i,rel in enumerate(CANN_LD):add(cann/rel,cann,'CANN',i==0)
    for rel in DRIVER_LD:add(driver/rel,driver,'DRIVER')
    directory(cann/'opp',cann,'CANN',True)
    env.update(CANN_ROOT=str(cann),ASCEND_HOME_PATH=str(cann),ASCEND_TOOLKIT_HOME=str(cann),
               ASCEND_AICPU_PATH=str(cann),ASCEND_OPP_PATH=str(cann/'opp'),LD_LIBRARY_PATH=':'.join(ld),
               TORCH_DEVICE_BACKEND_AUTOLOAD='0',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1')
    python_dirs=[]
    for rel in ('python/site-packages','opp/built-in/op_impl/ai_core/tbe'):
        row=directory(cann/rel,cann,'CANN',True);python_dirs.append(row['alias'])
    env['PYTHONPATH']=':'.join(python_dirs)
    # Preserve the invoked venv alias; a resolved base Python would select another environment.
    paths=[cann/'bin',cann/'tools/ccec_compiler/bin',Path(sys.executable).absolute().parent,Path(sys.base_prefix)/'bin']
    for item in (tools or {}).values():paths.append(c.verify(item).parent)
    paths += [Path(p) for p in ('/usr/local/sbin','/usr/local/bin','/usr/sbin','/usr/bin','/sbin','/bin')]
    env['PATH']=':'.join(dict.fromkeys(str(p) for p in paths if p.is_dir()))
    for name,item in (tools or {}).items():
        selected=shutil.which(name,path=env['PATH'])
        if not selected or c.identity(selected)!=item:raise ValueError('minimal PATH selects a different bound tool: '+name)
    libs=[library(cann/'lib64'/name,cann,'CANN') for name in CANN_LIBS]
    libs.append(library(driver/'lib64/driver/libascend_hal.so',driver,'DRIVER'))
    if opp and ready:libs.append(library(opp/'vendors/customize/op_api/lib/libcust_opapi.so',opp,'PRIVATE_DIAGNOSTIC_OPP'))
    binding=dict(policy=POLICY,cann_root=str(cann),driver_root=str(driver),opp_root=str(opp) if opp else None,
                 ready=ready,values=dict(env),directories=directories,libraries=libs,
                 replaced_parent_paths={k:base.get(k) for k in ('ASCEND_CUSTOM_OPP_PATH','ASCEND_OPP_PATH','LD_LIBRARY_PATH','PYTHONPATH','PATH')},
                 inheritance='No historical custom-OPP/LD/Python path append; system libraries use the system loader defaults')
    verify_binding(binding)
    return env,binding


def for_prior(prior,opp=None,*,ready=True):
    return make_environment(prior['current']['cann_root'],opp,tools=prior['current']['evidence']['tools'],ready=ready)


def standard_identity(binding):
    return dict(policy=binding['policy'],cann_root=binding['cann_root'],driver_root=binding['driver_root'],
                directories=[x for x in binding['directories'] if x['scope']!='PRIVATE_DIAGNOSTIC_OPP'],
                libraries=[x for x in binding['libraries'] if x['scope']!='PRIVATE_DIAGNOSTIC_OPP'],
                process={k:v for k,v in binding['values'].items() if k not in ('ASCEND_CUSTOM_OPP_PATH','LD_LIBRARY_PATH')})


def verify_binding(binding):
    if binding.get('policy')!=POLICY:raise ValueError('runtime environment policy changed')
    c.verify_tree(binding)
    cann=Path(binding['cann_root']);driver=Path(binding['driver_root']);opp=Path(binding['opp_root']) if binding['opp_root'] else None
    allowed={str(cann/x):('CANN',cann) for x in CANN_LD}
    allowed.update({str(driver/x):('DRIVER',driver) for x in DRIVER_LD})
    if opp:allowed[str(opp/'vendors/customize/op_api/lib')]=('PRIVATE_DIAGNOSTIC_OPP',opp)
    for row in binding['directories']:
        if row['alias'] not in allowed or row['scope']!=allowed[row['alias']][0]:raise ValueError('unapproved runtime directory: '+row['alias'])
        target,links=c.alias_target(row['alias'])
        if allowed[row['alias']][1].resolve() not in target.parents:raise ValueError('runtime directory escapes scope')
        if str(target)!=row['realpath'] or links!=row['links']:raise ValueError('runtime directory retargeted: '+row['alias'])
    for row in binding['libraries']:
        target,links=c.alias_target(row['alias'])
        if str(target)!=row['file']['path'] or links!=row['links']:raise ValueError('runtime library retargeted: '+row['alias'])
    values=binding['values'];opp=binding['opp_root'];cann=Path(binding['cann_root'])
    for key in ('CANN_ROOT','ASCEND_HOME_PATH','ASCEND_TOOLKIT_HOME','ASCEND_AICPU_PATH'):
        if values.get(key)!=str(cann):raise ValueError('CANN environment root changed: '+key)
    expected=str(Path(opp)/'vendors/customize') if opp else ''
    if values['ASCEND_CUSTOM_OPP_PATH']!=expected or values['ASCEND_OPP_PATH']!=str(cann/'opp'):
        raise ValueError('non-diagnostic OPP in child environment')
    expected_ld=[x['alias'] for x in binding['directories']]
    if opp and not binding['ready']:expected_ld.insert(0,str(Path(opp)/'vendors/customize/op_api/lib'))
    if values['LD_LIBRARY_PATH']!=':'.join(expected_ld):raise ValueError('unbound library path in runtime environment')
    if any(k.startswith('LD_') and k!='LD_LIBRARY_PATH' or k.startswith('BASH_FUNC_') or k in ('BASH_ENV','ENV','PYTHONHOME') for k in values):
        raise ValueError('unexpected loader/startup override')


def assert_invocation(env,binding):
    verify_binding(binding)
    core={k:v for k,v in env.items() if k not in OUTPUT_KEYS}
    if core!=binding['values']:raise ValueError('child environment differs from sealed minimal environment')


def compare_invocations(atc,aclnn):
    diff={k:dict(ATC=atc.get(k),ACLNN=aclnn.get(k)) for k in sorted(set(atc)|set(aclnn)) if atc.get(k)!=aclnn.get(k)}
    if set(diff)-OUTPUT_KEYS:raise ValueError('ATC/ACLNN runtime environment differs beyond diagnostic output paths/route')
    return dict(policy=POLICY,launch_differences=diff,stable_environment_equal=True,
                atc_wrapper_note='The reviewed ATC wrapper additionally prepends its package lib64 and python/site-packages; recorded separately as elf_environment')
