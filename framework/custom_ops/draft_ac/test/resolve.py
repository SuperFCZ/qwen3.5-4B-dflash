"""Read-only provenance resolution. Never guess generated build/OPP paths."""
import importlib.util
import json
from pathlib import Path
from common import C_ROOT,checked_file,sha256,write_json
from a2_common import load_bundle,snapshot_contract
from a4_contract import paired_gears
from run_a2 import validate_om_manifest
from opp import validate_sources


def load(path):
    path=Path(path).resolve()
    if not path.is_file(): raise ValueError(f'missing evidence: {path}')
    return json.loads(path.read_text())


def linked(parent,item):
    # Some historical report links are absolute, others are root-relative.
    p=Path(item['path']); p=p if p.is_absolute() else Path(parent)/p
    p=p.resolve()
    if not p.is_file() or sha256(p)!=item['sha256']: raise ValueError(f'evidence hash/path differs: {p}')
    return p


def head_capture(path):
    spec=importlib.util.spec_from_file_location('ac_frozen_head_common',C_ROOT/'test/common.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.load_capture(path)


def resolve(args,root):
    records={}
    def keep(label,path):
        path=Path(path).resolve(); records[label]=dict(path=str(path),sha256=sha256(path)); return path
    parent=keep('a51_suite',args.a51_suite); pair=load(parent)
    if pair.get('status')!='PASS': raise ValueError('A5.1 paired evidence is not PASS')
    stage_path=keep('a51_k256',linked(parent.parent,pair['runs']['256'])); stage=load(stage_path)
    if stage.get('status')!='PASS' or stage.get('full_a4_regression')!='PASS': raise ValueError('K256 full A4 is not PASS')
    config=Path(stage['build_config_path']).resolve()
    if sha256(config)!=stage['build_config_sha256']: raise ValueError('A build configuration changed')
    args.a_build_config=keep('a_build_config',config)
    a,c=validate_sources(config,args.c_release)
    if stage.get('build')!=a: raise ValueError('A stage build identity differs')
    a4_path=keep('a4',linked(stage_path.parent,stage['checks']['full']['summary'])); a4=load(a4_path)
    if a4.get('status')!='PASS' or a4.get('build')!=a: raise ValueError('A4 evidence build differs')
    bundles={}
    for gear in (16,64):
        for kind,suffix in (('bundle','bundle'),('native_om_manifest','native')):
            path=linked(a4_path.parent,a4['inputs'][f'c{gear}'][kind])
            expected=getattr(args,f'a{gear}_{suffix}',None)
            if expected and Path(expected).resolve()!=path: raise ValueError('explicit A input differs from accepted A4 evidence')
            setattr(args,f'a{gear}_{suffix}',keep(f'a{gear}_{suffix}',path))
        bundles[gear]=load_bundle(getattr(args,f'a{gear}_bundle'),'a4')
        validate_om_manifest(getattr(args,f'a{gear}_native'),getattr(args,f'a{gear}_bundle'),bundles[gear],'a4')
    paired_gears(bundles[16],bundles[64])
    # Follow the generated header's documented CustomOp/op_host location.
    header=Path(a['header']).resolve()
    if header.parent.name!='op_host' or header.parent.parent.name!='CustomOp': raise ValueError('unknown A build layout; supply evidence with generated CustomOp/op_host header')
    install=header.parents[2]/'opp'
    # A-only installed preflight (unchanged frozen checker).
    from opp_preflight import check_install
    a_opp=check_install(install)
    accepted_opp_path=keep('a_installed_opp_report',Path(args.a_build_config).parent/'opp-preflight.json')
    if load(accepted_opp_path)!=a_opp: raise ValueError('A installed OPP differs from accepted preflight hashes')
    release=load(args.c_release); keep('c_release',args.c_release)
    keep('c_build_state',release['build_state'])
    load(release['build_state'])
    for item in release['build_artifacts']: linked(Path(args.c_release).parent,item)
    def check_tree(value):
        if isinstance(value,dict):
            for path_key,hash_key in (('path','sha256'),('binary','binary_sha256'),('json','json_sha256')):
                if path_key in value and hash_key in value:
                    linked(Path(args.c_release).parent,dict(path=value[path_key],sha256=value[hash_key]))
            for child in value.values(): check_tree(child)
        elif isinstance(value,list):
            for child in value: check_tree(child)
    check_tree(release['installed_opp'])
    head=head_capture(args.head_capture); keep('head_capture',args.head_capture); keep('head_native',args.head_native)
    c_suite_path=keep('c_release_suite',linked(Path(args.c_release).parent,release['suite'])); c_suite=load(c_suite_path)
    if c_suite.get('status')!='PASS' or c_suite.get('capture_sha256')!=sha256(args.head_capture) or c_suite.get('native_manifest_sha256')!=sha256(args.head_native):
        raise ValueError('C Release does not match supplied capture/Native manifest')
    native=load(args.head_native)
    if native.get('status')!='PASS' or native.get('capture_sha256')!=sha256(args.head_capture) or native.get('weight_binding')!='runtime-readonly-ND':
        raise ValueError('Head Native manifest/capture binding differs')
    if [(i.get('m'),i.get('kind')) for i in native.get('models',[])]!=[(m,k) for m in range(1,16) for k in ('logits','top1')]:
        raise ValueError('need all 30 existing Head Native OMs')
    for item in native['models']:
        if item.get('status')!='PASS' or item.get('runtime_input_abi',{}).get('logical_input_names')!=['hidden','weight']: raise ValueError('Head native ABI differs')
        for name in ('om','deployment','air'): checked_file(Path(args.head_native).parent,item[name])
        dep_path=checked_file(Path(args.head_native).parent,item['deployment']); dep=load(dep_path)
        if dep.get('status')!='PASS' or len(dep.get('graphs',[]))!=1: raise ValueError('Head Native deployment incomplete')
        g=dep['graphs'][0]; m=item['m']; kind=item['kind']
        abi=dict(inputs=[dict(name='hidden',dtype='float16',shape=[m,2560]),dict(name='weight',dtype='float16',shape=[248320,2560])],
                 outputs=[dict(name='y',dtype='float16' if kind=='logits' else 'int64',shape=[m,248320] if kind=='logits' else [m])])
        if g.get('metadata',{}).get('tensor_abi')!=abi or checked_file(dep_path.parent,g['om'])!=checked_file(Path(args.head_native).parent,item['om']):
            raise ValueError('Head Native full vocabulary ABI or OM binding differs')
    for gear in (16,64):
        source=bundles[gear]['source']; h=head['gears'][f'c{gear}']
        if source['feature_layers']!=h['source']['feature_layers']: raise ValueError('A/C feature layer order differs')
        for key in ('model_sha256','config_sha256'):
            if bundles[gear]['checkpoint'][key]!=h['checkpoint'][key]: raise ValueError('A/C checkpoint differs')
    deployment_path=keep('native_deployment',args.native_deployment); deployment=load(deployment_path)
    air_path=linked(deployment_path.parent,deployment['air_manifest'])
    if args.air_manifest and Path(args.air_manifest).resolve()!=air_path: raise ValueError('AIR/Deployment mismatch')
    air=load(keep('air_manifest',air_path))
    if air.get('status')!='PASS' or deployment.get('status')!='PASS': raise ValueError('Native export incomplete')
    draft=next(g for g in deployment['graphs'] if g['name']=='draft')
    exported=next(g for g in air['graphs'] if g['name']=='draft')
    for key in ('metadata','air','runtime_input_abi','input_names','output_names'):
        if exported.get(key)!=draft.get(key): raise ValueError('Native AIR/Deployment Draft differs: '+key)
    for payload in exported['payload_files']: checked_file(air_path.parent,payload)
    from qwen35_dflash.ascend310p.draft_gears import verify_om_files,om_records
    verify_om_files(draft,deployment_path.parent)
    if len(om_records(draft))!=2: raise ValueError('requires original static C16/C64 Draft OMs')
    for gear in (16,64):
        replay_path=keep(f'replay{gear}',getattr(args,f'replay{gear}')); replay=load(replay_path)
        if replay.get('fake_acl') is not False or replay['draft_om_sha256']!=draft['om']['sha256'] or (gear==64 and replay['draft_static64_om_sha256']!=om_records(draft)[1]['sha256']):
            raise ValueError('replay/Native OM binding differs')
        directory=Path(replay['input_directory']); digest,specs=snapshot_contract((directory/'contract.txt').read_text())
        if digest!=draft['om']['sha256'] or replay['kv_output_audit']['context_rows']!=gear: raise ValueError('replay contract/gear differs')
        for name,spec in specs.items():
            file=directory/(name+'.bin')
            if file.stat().st_size!=spec['bytes'] or sha256(file)!=replay['snapshot_sha256'][name]: raise ValueError('frozen input changed: '+name)
        if bundles[gear]['source']['report_sha256']!=sha256(replay_path): raise ValueError('A capture came from another replay')
        if head['gears'][f'c{gear}']['source']['report_sha256']!=sha256(replay_path): raise ValueError('Head capture came from another replay')
    config=air.get('factory_config')
    if args.factory_config:
        supplied=load(keep('factory_config_supplied',args.factory_config))
        if config!=supplied: raise ValueError('explicit Factory JSON differs from hash-bound AIR factory_config')
    if not isinstance(config,dict) or not config: raise ValueError('AIR factory_config missing; inspect the original export metadata before proceeding')
    if config.get('draft_quantization')!='w8a16' or config.get('draft_weight_prepack')!='nz':
        raise ValueError('AIR factory_config is not W8 offline NZ; variant-matrix config needs its recorded per-variant metadata, do not guess')
    args.factory_config=root/'factory-resolved.json'; write_json(args.factory_config,config)
    keep('factory_config_resolved',args.factory_config)
    result=dict(status='PASS',scope='read-only provenance; GE/ATC/NPU NOT_RUN',records=records,a_installed_opp=a_opp,
                c_installed_opp=release['installed_opp'],feature_layers=bundles[16]['source']['feature_layers'])
    write_json(root/'preflight.json',result)
    return result,bundles
