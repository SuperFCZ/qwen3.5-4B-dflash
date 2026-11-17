"""Reverify contaminated historical artifacts without promoting their old status."""
from pathlib import Path
import json
from platform_common import c,sources
from platform_contracts import read_platform,validate_runtime

# User's read-only terminal verification, 2026-11-17. No historical report is rewritten.
PINS={
 'platform.jsonl':(28416,'3e83ab14b5ebb0333532fe531dcdde0e2f1a3cedf62e2e6ac9fec2fbddce0570'),
 'execution.json':(293,'16f3891af770ec7a0569f054de584ef5fec124fe3cd18972637f9518c10b347c'),
 'runtime-events.jsonl':(590,'f4770627e241e854b20579577d52d30d2647aa128899a816d5bb65e5b4f2abac'),
 'maps.txt':(38854,'b6b696460ab79375853ce8502e11ed15ed0b530dd9e7bef81553f87665ff9737'),
 'runner.log':(5822,'ac29c3e75d8c8ec1ff1667c6057073e6daae1c623653a63cca984ff5ea51699d')}
FOREIGN='/home/l00886453/custom_ops_llm/vendors/customize_dynamic/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so'


def audit(root):
    root=Path(root).resolve(strict=True);path=root/'aclnn/report.json';r=c.read(path)
    if r.get('status')!='BLOCKED' or r.get('integrity_error') or FOREIGN not in r.get('error',''):raise ValueError('not the identified contaminated ACLNN report')
    records=r.get('retained_runtime',[]);files={}
    for name,(size,sha) in PINS.items():
        p=(root/'aclnn/fc'/name).resolve(strict=True)
        matches=[x for x in records if x['path']==str(p)]
        if len(matches)!=1 or matches[0]['bytes']!=size or matches[0]['sha256']!=sha:raise ValueError('old retained identity/pin differs: '+name)
        c.verify(matches[0]);files[name]=matches[0]
    build_path=c.verify(r['build_report']);build=c.read(build_path)
    source_path=c.verify(r['source_report']);source=c.read(source_path)
    if build.get('status')!='PLATFORM_DIAG_OPP_RUNNER_BUILT' or build['source_report']!=r['source_report']:raise ValueError('old Build/Source chain differs')
    if source.get('status')!='PLATFORM_DIAG_SOURCE_BOUND':raise ValueError('old Source status differs')
    opp=build['package']['opp'];c.verify_opp(opp)
    execution=c.read(c.verify(files['execution.json']));events=[json.loads(x) for x in c.verify(files['runtime-events.jsonl']).read_text().splitlines()]
    validate_runtime(execution,events)
    platform=read_platform(c.verify(files['platform.jsonl']),'ACLNN',opp)
    # The pinned maps and original rejection establish the old contamination.
    # Do not retroactively invent hashes for foreign libraries that were not sealed.
    mappings=[line.split(None,5)[5] for line in c.verify(files['maps.txt']).read_text().splitlines() if len(line.split(None,5))==6]
    if FOREIGN not in mappings:raise ValueError('old pinned maps do not contain the reported foreign OPP')
    rejection=r['error']
    atc_path=root/'atc/report.json';atc=c.read(atc_path)
    if atc.get('status')!='PLATFORM_DIAG_ATC_OBSERVED' or atc['build_report']!=r['build_report'] or atc['source_report']!=r['source_report']:raise ValueError('old ATC chain differs')
    atc_platform=read_platform(c.verify(atc['platform']['source']),'ATC',opp)
    if atc_platform!=atc['platform']:raise ValueError('old ATC saved platform differs from raw bytes')
    c.verify_opp(opp)
    return dict(status='CONTAMINATED_RUNTIME_REFERENCE_REVERIFIED_NOT_ADMITTED',scope='REFERENCE_ONLY_NOT_FORMAL_PASS',
        old_acl_report=c.identity(path),old_atc_report=c.identity(atc_path),old_build_report=r['build_report'],old_source_report=r['source_report'],
        files=files,execution=execution,platform=platform,atc_platform=atc_platform,loaded_library_rejection=rejection,foreign_library_identity='NOT_RECORDED_NO_RETROACTIVE_HASH',
        old_environment=r.get('environment_contract'),old_atc_environment=atc.get('environment_contract'),
        current_source_identity_equal=source.get('sources')==sources(),
        atc_reuse='NOT_ADMITTED_CHANGED_SOURCE_BUILD_AND_ENVIRONMENT_CONTRACT',
        next_atc='Recompile only the frozen single-FC AIR after new Build; no full Draft export',
        kernel_execute_calls=0,npu='NOT_RUN_NEW_EXPERIMENT')
