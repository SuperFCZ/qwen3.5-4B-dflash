from pathlib import Path
from scheduler_common import c,HERE
import run_diag as old
import dso_manifest as dsos

BASE_HOST_SHA='1ef6d60cd1fddb51e1e6aa7e1181d13e40ec659f24434aa748da3cbe826e10ac'


def directory_view(path):
    p=Path(path)
    if not p.exists():return dict(alias=str(p),exists=False)
    real,links=c.alias_target(p)
    return dict(alias=str(p),exists=True,realpath=str(real),links=links,is_directory=real.is_dir())


def prefix_review(cann,atc,runtime,manifest):
    cann=Path(cann);dirs={x:directory_view(cann/x) for x in ('lib64','aarch64-linux/lib64','python/site-packages','aarch64-linux/python/site-packages')}
    def select(name,env):
        for path in env['LD_LIBRARY_PATH'].split(':'):
            p=Path(path)/name
            if p.is_file():return c.identity(p)
        return None
    names=sorted({Path(x['file']['path']).name for x in manifest['records']});resolution={}
    for name in names:
        a=select(name,atc);b=select(name,runtime)
        resolution[name]=dict(ATC_ELF=a,ACLNN=b,equal=a==b,scope='LD_PATH_CANDIDATES_NOT_ACTUAL_ATC_FULL_MAPS')
    return dict(directories=dirs,library_resolution=resolution,actual_atc_full_maps='NOT_CAPTURED',
                conclusion='Equal real directories/known candidates narrow the loader hypothesis; PlatformInfo producer and Python module origins remain unproven')


def verify_prefix_review(review):
    for row in review['directories'].values():
        if directory_view(row['alias'])!=row:raise ValueError('ATC prefix directory/alias changed after Source')
    c.verify_tree(review)


def baseline(root):
    root=Path(root).resolve(strict=True);reports={k:old.stage(root/k/'report.json',k) for k in ('source','build','atc','aclnn','compare')}
    ids={k:c.identity(root/k/'report.json') for k in reports}
    for k in ('build','atc','aclnn','compare'):
        if reports[k]['source_report']!=ids['source']:raise ValueError('mixed baseline Source')
    for k in ('atc','aclnn','compare'):
        if reports[k]['build_report']!=ids['build']:raise ValueError('mixed baseline Build')
    if reports['compare']['atc_report']!=ids['atc'] or reports['compare']['aclnn_report']!=ids['aclnn']:raise ValueError('mixed baseline Compare')
    for k in ('atc','aclnn'):
        if reports[k]['platform']['host_dso']['sha256']!=BASE_HOST_SHA:raise ValueError('baseline Host DSO differs from user-confirmed identity')
    if reports['atc']['platform']['host_dso']!=reports['aclnn']['platform']['host_dso']:raise ValueError('baseline actual Host libraries differ')
    package=reports['build']['package'];c.verify_opp(package['opp'])
    metadata=[]
    for item in package['opp']['preflight']['kernels']:
        p=Path(item['json']);raw=c.read(p)
        fields={}
        def walk(v,loc='$'):
            if isinstance(v,dict):
                for key,value in v.items():
                    if key.lower() in ('coretype','kerneltype','kernel_type','socversion','blockdim'):fields[loc+'.'+key]=value
                    walk(value,loc+'.'+key)
            elif isinstance(v,list):
                for i,x in enumerate(v):walk(x,loc+'.'+str(i))
        walk(raw);metadata.append(dict(file=c.identity(p),fields=fields))
    source=reports['source'];cann=Path(source['binding']['prior']['current']['cann_root']);sdk=[]
    for row in c.read(HERE/'sdk-review.json')['installed_headers']:
        item=c.identity(cann/row['relative'])
        if item['sha256']!=row['sha256'] or cann.resolve() not in Path(item['path']).parents:raise ValueError('SDK header identity differs: '+row['relative'])
        sdk.append(dict(file=item,lines=row['lines']))
    libraries=[]
    for rel in ('lib64/libtiling_api.a','lib64/libplatform.so'):
        p=cann/rel;item=c.identity(p)
        if cann.resolve() not in Path(item['path']).parents:raise ValueError('SDK library outside selected CANN')
        libraries.append(dict(alias=str(p),file=item,scope='NEW_SERVER_SOURCE_SEAL_NOT_LOCAL_BINARY_EQUIVALENCE'))
    manifest=c.read(c.verify(reports['build']['dso_manifest']));dsos.verify_manifest(manifest)
    return dict(references=ids,prior=source['binding']['prior'],frozen=source['binding']['frozen'],case=source['binding']['case'],
                runtime_environment=source['runtime_environment'],standard_manifest=source['dso_manifest'],
                observed={k:reports[k]['platform'] for k in ('atc','aclnn')},aclnn_workspace=c.read(c.verify(reports['aclnn']['execution']))['workspace'],sdk=sdk,sdk_libraries=libraries,kernel_metadata=metadata,
                prefix_review=prefix_review(cann,reports['atc']['atc_elf_environment'],reports['aclnn']['child_environment'],manifest),
                kernel_execution='NOT_RUN_IN_THIS_EXPERIMENT')
