"""Read-only review of J3GWjX; old BLOCKED remains BLOCKED."""
import json
from pathlib import Path
from platform_common import c
from platform_contracts import read_platform,validate_runtime
from dso_manifest import POLICY,maps_rows


def audit(root):
    root=Path(root).resolve(strict=True);p=root/'aclnn/report.json';r=c.read(p);policy=c.read(POLICY)
    if r.get('status')!='BLOCKED' or r.get('integrity_error'):raise ValueError('old ACLNN is not an intact BLOCKED report')
    src=c.read(c.verify(r['source_report']));build=c.read(c.verify(r['build_report']))
    if src.get('status')!='PLATFORM_DIAG_SOURCE_BOUND' or build.get('status')!='PLATFORM_DIAG_OPP_RUNNER_BUILT' or build['source_report']!=r['source_report']:raise ValueError('old Source/Build chain differs')
    cann=Path(src['binding']['prior']['current']['cann_root']).resolve();standard=str(cann/policy['standard_opp'][0]['relative'])
    if standard not in r.get('error',''):raise ValueError('different old rejection')
    files={}
    for name,pin in policy['reference_files'].items():
        path=(root/'aclnn/fc'/name).resolve(strict=True)
        rows=[x for x in r.get('retained_runtime',[]) if x['path']==str(path)]
        if len(rows)!=1 or any(rows[0][k]!=pin[k] for k in ('bytes','sha256')):raise ValueError('old artifact pin differs: '+name)
        c.verify(rows[0]);files[name]=rows[0]
    execution=c.read(c.verify(files['execution.json']));events=[json.loads(x) for x in c.verify(files['runtime-events.jsonl']).read_text().splitlines()]
    validate_runtime(execution,events);opp=build['package']['opp'];c.verify_opp(opp)
    platform=read_platform(c.verify(files['platform.jsonl']),'ACLNN',opp)
    aliases=maps_rows(c.verify(files['maps.txt']).read_text())
    if standard not in aliases:raise ValueError('reviewed CANN OPP missing from old mappings')
    private=Path(opp['install']).resolve()
    for alias in aliases:
        if 'libcust_op' in Path(alias).name and alias!=standard and private not in Path(alias).parents:raise ValueError('foreign private OPP remains in old maps: '+alias)
    atcp=root/'atc/report.json';atc=c.read(atcp)
    if atc.get('status')!='PLATFORM_DIAG_ATC_OBSERVED' or atc['source_report']!=r['source_report'] or atc['build_report']!=r['build_report']:raise ValueError('old ATC links differ')
    observed=read_platform(c.verify(atc['platform']['source']),'ATC',opp)
    if observed!=atc['platform']:raise ValueError('old ATC observation changed')
    return dict(status='STANDARD_OPP_BLOCKED_REFERENCE_REVERIFIED_NOT_PROMOTED',report=c.identity(p),files=files,private_opp=opp,
        source_report=r['source_report'],build_report=r['build_report'],atc_report=c.identity(atcp),platform=platform,atc_platform=observed,
        execution=execution,old_error=r['error'],old_status='BLOCKED',scope='REFERENCE_ONLY_NO_RETROACTIVE_PASS',
        reuse='NOT_ADMITTED_NEW_DSO_MANIFEST_AND_SOURCE_IDENTITY_REQUIRED')
