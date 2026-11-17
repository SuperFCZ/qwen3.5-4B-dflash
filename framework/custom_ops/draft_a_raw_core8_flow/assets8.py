"""Verify existing assets into a NEW 8-core identity, without editing 7-core reports."""
import hashlib
from pathlib import Path
from c8_common import c,sidecar_stage,legacy
from resource8 import resources,verify_resource
from split_atc import one_capture
from atc_evidence import COUNTS,payloads

OM={'bytes':3090431253,'sha256':'f67c41cda787fc552a6c35c5ed7dfde26a98f49e8b8cc8d40b52f383f03e19af'}
SIDECAR={'bytes':7810,'sha256':'2e021c01a46db3c41864cef5acc3b06893695ae81d6abefa4836a925c36a66e4'}


def model(record,prior,opp,variant,gear,out=None):
    c.verify_tree(record);c.verify_opp(opp)
    if (record.get('status') not in ('A_ONLY_SIDECAR_ATC_COMPILED_NOT_EXECUTED','A_ONLY_OM_COMPILED_NOT_EXECUTED') or
        record.get('error') or record.get('integrity_error') or record.get('private_opp')!=opp or
        (record.get('variant'),record.get('gear'))!=(variant,gear) or record['import_boundary']['status']!='PASS'):
        raise ValueError('OM worker identity/status differs')
    abi=prior['current']['evidence']['H']['gears'][str(gear)]['abi']
    if record['abi']!=abi or record['counts']!=COUNTS:raise ValueError('full Draft ABI/counts differ')
    ap=c.verify(record['artifacts']['air']);air=c.read(ap);files=payloads(ap,air)
    dp=c.verify(record['artifacts']['deployment']);dep=c.read(dp)
    if dep['status']!='PASS' or len(dep['graphs'])!=1 or c.linked(dp.parent,dep['air_manifest'])!=ap:raise ValueError('AIR/Deployment mismatch')
    g=air['graphs'][0];compiled=dep['graphs'][0]
    if g['metadata']['draft_ac']['mode']!='a' or g['metadata']['draft_ac']['gear']!=gear or g['metadata']['tensor_abi']!=abi:raise ValueError('AIR gear/mode/ABI differs')
    if g['runtime_input_abi']['status']!='PASS' or g['runtime_input_abi']['draft_ac'].get('status')!='PASS' or g['runtime_input_abi']['draft_ac']['counts']!=COUNTS:raise ValueError('AIR audit missing')
    observed={r['ge_op_type']:r['ge_node_occurrences'] for r in g['custom_op_audit']}
    if any(observed.get(k,0)!=v for k,v in COUNTS.items()) or observed.get('WeightQuantBatchMatmulV2',0):raise ValueError('26 A / 0 C / 0 residual WeightQuant required')
    for k in ('metadata','runtime_input_abi','custom_op_audit'):
        if compiled[k]!=g[k]:raise ValueError('Deployment audit differs')
    if c.linked(dp.parent,compiled['om'])!=c.verify(record['om']) or record['artifacts']['om']!=record['om']:raise ValueError('OM identity differs')
    if compiled['atc_command']!=record['atc_command']:raise ValueError('ATC command differs')
    capture=one_capture(record['atc_captures'],record['atc_command'])
    if any(record[k]!=capture[v] for k,v in (('atc_log','stdout'),('atc_stderr','stderr'),('host_evidence','host_evidence'))):raise ValueError('three log identities differ')
    if c.identity(record['atc_command'][0])!=prior['current']['evidence']['tools']['atc']:raise ValueError('ATC binary changed')
    root=record.get('logical_ir_source_root',record['output_root'])
    if hashlib.sha256(c.verify(record['graph']).read_bytes().replace(root.encode(),b'<VARIANT_ROOT>')).hexdigest()!=record['logical_ir_sha256']:raise ValueError('logical IR changed')
    cfg=prior['current']['evidence'][variant]['configuration']
    if out is not None:
        resource=resources(record['host_evidence'],out,gear,cfg,variant)
        return dict(worker=record,host_resources=resource,payload_files=files)


def reverify(entry,prior,opp,variant,gear):
    raw=c.read(c.verify(entry['worker_report']))
    if raw!=entry['worker']:raise ValueError('Worker report drift')
    model(raw,prior,opp,variant,gear)
    verify_resource(entry['host_resources'],gear,prior['current']['evidence'][variant]['configuration'],variant)


def admit(root,out):
    root=Path(root)
    source=sidecar_stage(root/'source/report.json','admit');build=sidecar_stage(root/'build/report.json','build')
    if build['source_report']!=c.identity(root/'source/report.json'):raise ValueError('sidecar Source/Build chain differs')
    admission,prior=legacy(c.verify(source['admission_report']))
    if prior['references']!=source['predecessors']:raise ValueError('frozen input provenance differs')
    if c.read(c.verify(build['build_worker_report']))!=build['B']:raise ValueError('sidecar build worker drift')
    package=build['B'];c.verify_opp(package['opp'])
    if not package['opp_comparison']['all_kernel_bytes_equal']:raise ValueError('B device Kernel identity changed')
    p=root/'bc16/B/c16/report.json';w=c.read(p)
    for key,want in (('om',OM),('host_evidence',SIDECAR)):
        if any(w[key][k]!=v for k,v in want.items()):raise ValueError('user-frozen B8 identity differs: '+key)
    entry=model(w,prior,package['opp'],'B',16,out/'B8-c16-resources');entry['worker_report']=c.identity(p)
    outer=root/'bc16/report.json';outer_data=c.read(outer)
    if outer_data['status']!='BLOCKED' or outer_data.get('integrity_error'):raise ValueError('expected preserved 7-core equality blocker')
    return dict(sidecar_source=c.identity(root/'source/report.json'),sidecar_build=c.identity(root/'build/report.json'),
        old_blocked_report=c.identity(outer),old_admission=source['admission_report'],references=prior['references'],
        B8_package=package,B8_c16=entry,historical_7core='READ_ONLY_PROVENANCE_NOT_RESOURCE_EQUALITY_OR_CORRECTNESS',
        compiler_identity_text=admission['frozen']['compiler_identity_text']),prior
