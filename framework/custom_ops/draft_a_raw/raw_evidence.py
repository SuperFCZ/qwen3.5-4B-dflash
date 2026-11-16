"""Read-only H/B/R identity admission. No compiler, model import or device call."""
from pathlib import Path
import copy
import shutil
import raw_common as c
from a3_launch import load_build_config
from opp_preflight import check_install
from gate_prefetch_contract import match_plans

NAMES=[f'layer-{i}-gate_up' for i in range(5)]
FIELDS=('abi','policy','k_tile_policy','core_limit','dequant_mode','pipeline_mode','kv_m80_mode','scale_mode','k_tile','tiling_abi','launch_version','gate_copy_mode')
# These six worker-input references are relative to H, as confirmed by the
# receiver's diagnostic output. Do not infer bases for arbitrary nested records.
H_STATE_REFERENCES=(('native_boundary',),('report','native_boundary'),
    ('report','draft_interface'),('report','provenance'),
    ('report','preflight_imports'),('report','coordinator_process'))


def canonical_historical_state(root,state):
    """New evidence copy only; keep the original worker-input and args intact."""
    root=Path(root).resolve(strict=True)
    normalized=copy.deepcopy(state)
    for route in H_STATE_REFERENCES:
        parent=normalized
        for key in route[:-1]:parent=parent[key]
        item=parent[route[-1]]
        if not isinstance(item,dict) or set(item)!={'path','bytes','sha256'}:
            raise ValueError('historical identity schema differs: '+'.'.join(route))
        path=c.linked(root,item)  # Verify the declared size/hash before rebasing.
        if root not in path.parents:raise ValueError('historical state reference escapes H: '+str(path))
        bound=dict(item,path=str(path))  # Preserve the expected size/hash, not new values.
        c.verify(bound)
        parent[route[-1]]=bound
    return normalized


def bind_head_libraries(installed):
    """Require each alias target to have its own matching frozen inventory row."""
    root=Path(installed['install_root'])
    if not root.is_absolute() or installed.get('status')!='PASS' or installed.get('release_only') is not True:
        raise ValueError('historical Head release identity missing')
    records=installed['libraries'];declared={}
    if not records:raise ValueError('historical Head library inventory empty')
    for item in records:
        if set(item)!={'path','bytes','sha256'} or not Path(item['path']).is_absolute():
            raise ValueError('historical Head library identity schema differs')
        old=declared.setdefault(item['path'],item)
        if old!=item:raise ValueError('conflicting historical library identities: '+item['path'])
    bindings=[]
    for item in records:
        target,links=c.alias_target(item['path'])
        expected=dict(item,path=str(target))
        # A matching hash on the current filesystem alone cannot authorize a
        # different target. The original C Release must also name this target.
        if declared.get(str(target))!=expected:
            raise ValueError('alias lacks matching frozen canonical library record: '+item['path'])
        bound=dict(kind=c.HEAD_ALIAS_KIND,install_root=str(root),resolved_install_root=str(root.resolve(strict=True)),
            alias=item['path'],links=links,file=expected)
        c.verify_head_alias(bound);bindings.append(bound)
    return bindings


def head_release_evidence(root,state):
    reference=state['provenance']['records']['c_release']
    release_path=c.linked(root,reference)
    release_identity=c.identity(release_path)
    if (release_path!=Path(state['args']['c_release']).resolve(strict=True) or
            release_identity['sha256']!=reference['sha256'] or
            c.read(release_path)['installed_opp']!=state['provenance']['c_installed_opp']):
        raise ValueError('historical C Release provenance differs')
    return dict(release=release_identity,libraries=bind_head_libraries(state['provenance']['c_installed_opp']),
        role='HISTORICAL_C_RELEASE_SOURCE_PACKAGE_NOT_H_JOINT_RUNTIME_OPP')


def configs(off,raw):
    if any(off.get(k)!=raw.get(k) for k in (*FIELDS,'source_sha256')):raise ValueError('non-Raw configuration differs')
    if tuple(off.get(k) for k in ('dequant_mode','scale_mode','k_tile','kv_m80_mode','gate_copy_mode'))!=('batched','broadcast',256,'a-ub','loop'):
        raise ValueError('not accepted W8A16 Group128 baseline')
    if off.get('gate_prefetch_mode')!='off' or raw.get('gate_prefetch_mode')!='raw':raise ValueError('wrong B/R selection')


def header_pair(before,after):
    old=Path(before).read_bytes();new=Path(after).read_bytes()
    line=b'#define DFLASH_GROUP_QUANT_GATE_PREFETCH_MODE 0U'
    if old.count(line)!=1 or old.replace(line,line[:-2]+b'1U')!=new:raise ValueError('generated Header differs beyond Raw macro')


def pin(root,expected):
    files={}
    for name,digest in expected.items():
        item=c.identity(root/name)
        if item['sha256']!=digest:raise ValueError('accepted report hash differs: '+str(root/name))
        files[name]=item
    return files


def pair(root,expected,out):
    import run_gate_copy_pair as gp
    files=pin(root,expected);suite=c.read(root/'suite.json');comp=c.read(root/'comparison.json')
    if (suite.get('status')!='PASS' or comp.get('status')!='PASS' or suite.get('experiment')!='prefetch'
        or comp.get('full_draft_token_kv')!='NOT_RUN' or comp.get('adoption')!='NOT_DECIDED'):
        raise ValueError('five-layer acceptance/scope missing')
    if [r['name'] for r in comp['cases']]!=NAMES or suite.get('selected_cases')!=NAMES:raise ValueError('five layers missing')
    if c.linked(root,suite['comparison'])!=root/'comparison.json':raise ValueError('pair comparison link')
    runs={}
    for mode in ('off','raw'):
        item=suite['runs'][mode];p=c.linked(root,item);child=c.read(p)
        if child['status']!='PASS' or child['selected_cases']!=NAMES:raise ValueError('child scope')
        cfgpath=Path(child['build_config_path']).resolve();cfg=load_build_config(cfgpath)
        if c.sha(cfgpath)!=child['build_config_sha256'] or child['build']!=cfg or cfg['gate_prefetch_mode']!=mode:raise ValueError('child/header binding')
        for name,digest in cfg['source_sha256'].items():
            if c.sha(c.QUANT/name)!=digest:raise ValueError('accepted candidate source drift: '+name)
        header=Path(cfg['header']).resolve()
        if header.parent.name!='op_host' or header.parent.parent.name!='CustomOp':raise ValueError('unknown generated header layout')
        buildroot=header.parents[2];opp=buildroot/'opp'
        preflight=c.linked(p.parent.parent,item['installed_opp'])
        if c.read(preflight)!=check_install(opp):raise ValueError('accepted installed OPP differs')
        runs[mode]=dict(suite=c.identity(p),config=c.identity(cfgpath),configuration=cfg,
            header=c.identity(header),kernel_header=c.identity(cfg['kernel_header']),installed_report=c.identity(preflight),
            installed=check_install(opp),runner=c.identity(buildroot/'test/dflash_group_quant_linear_test'),
            sdk_report=c.identity(c.linked(p.parent.parent,item['sdk'])))
        sdk=c.read(c.verify(runs[mode]['sdk_report']));sdkroot=Path(sdk['cann_root']).resolve()
        if sdk.get('status')!='SOURCE_INSPECTED' or sdk.get('experiment')!='prefetch':raise ValueError('accepted SDK review missing')
        sdk_files=[]
        for key in ('versions','parameters','dav_m200','event_headers'):
            if not sdk.get(key):raise ValueError('incomplete accepted SDK identities: '+key)
            for f in sdk[key]:
                actual=c.linked(sdkroot,f)
                if sdkroot not in actual.parents:raise ValueError('SDK file escapes selected root')
                sdk_files.append(c.identity(actual))
        runs[mode]['sdk_files']=sdk_files
        runs[mode]['kernel_files']=[c.identity(f[key]) for f in runs[mode]['installed']['kernels'] for key in ('binary','json')]
        runs[mode]['private_api_observed']=[c.identity(f) for f in sorted((opp/'vendors/customize').rglob('*')) if f.is_file() and (f.suffix in ('.so','.h') or '.so.' in f.name)]
        apiheader=opp/'vendors/customize/op_api/include/aclnn_d_flash_group_quant_linear.h'
        runs[mode]['api_header']=c.identity(apiheader)
        if not all(name in apiheader.read_text() for name in ('aclnnDFlashGroupQuantLinearGetWorkspaceSize','aclnnDFlashGroupQuantLinear')):raise ValueError('wrong accepted API header')
        for source in (header.parent.parent/'op_kernel/d_flash_group_quant_linear.cpp',header.parent/'d_flash_group_quant_linear.cpp'):
            rel=source.parent.name+'/'+source.name
            if c.sha(source)!=cfg['source_sha256'][rel]:raise ValueError('generated accepted source differs: '+str(source))
    configs(runs['off']['configuration'],runs['raw']['configuration'])
    header_pair(c.verify(runs['off']['header']),c.verify(runs['raw']['header']))
    if runs['off']['api_header']['sha256']!=runs['raw']['api_header']['sha256']:raise ValueError('B/R ACLNN public ABI header differs')
    # Existing strict comparator rechecks every original output, 5/30 samples,
    # per-layer input identity and actual SDK launch plans; writes NEW proof only.
    rebuilt=gp.compare_runs(c.verify(runs['off']['suite']),c.verify(runs['raw']['suite']),out/'comparison-rechecked.json',experiment='prefetch')
    if rebuilt!=comp:raise ValueError('pair summary differs from reverified raw evidence')
    for row in comp['cases']:match_plans(row['baseline_launch'],row['candidate_launch'])
    return dict(reports=files,runs=runs,comparison=comp,recheck=c.identity(out/'comparison-rechecked.json'))


def historical(root,pins):
    from c_profile import prepare_case,OUTPUTS
    from native_base import bound_native
    from source_lock import audit_source_lock
    files=pin(root,pins);suite=c.read(root/'suite.json');ds=c.read(root/'draft-suite.json');state=c.read(root/'worker-input.json');opp=c.read(root/'joint/opp.json')
    if suite.get('status')!='PASS' or suite.get('draft_only')!='PASS' or suite.get('single_op_graph')!='PASS' or ds.get('status')!='PASS':raise ValueError('H full Draft-only acceptance missing')
    if c.linked(root,suite['opp'])!=root/'joint/opp.json' or state['opp']!=opp:raise ValueError('H OPP link')
    if opp.get('status')!='BUILT_NOT_NPU_VALIDATED' or Path(opp['install']).resolve()!=root/'joint/opp':raise ValueError('H private package scope')
    c.verify_opp(opp)
    old_config=load_build_config(Path(state['args']['a_build_config']))
    if old_config!=opp['a_config']:raise ValueError('H generated configuration changed')
    runner=c.identity(root/'runner/ac_om_runner')
    if runner['sha256']!=suite['runner_sha256']:raise ValueError('H Runner changed')
    boundary_path=c.linked(root,state['native_boundary']);boundary=c.read(boundary_path)
    if boundary.get('status')!='PASS' or boundary.get('accepted_args')!=state['args'] or boundary.get('request_sha256')!=c.sha(root/'preflight-input.json'):raise ValueError('H import/source boundary')
    base,lock,air=bound_native(state);audit=audit_source_lock(base,lock)
    if audit['status']!='PASS':raise ValueError('Native Source Lock mismatch: '+str(audit.get('identity_diff')))
    native=c.read(c.linked(Path(state['args']['native_deployment']).parent,c.read(state['args']['native_deployment'])['air_manifest']))
    factory=c.read(state['args']['factory_config'])
    if factory!=native.get('factory_config'):raise ValueError('historical Factory differs from Native AIR')
    cases={};watch={}
    selected=[r for r in ds['cases'] if r['mode']=='a']
    if sorted(r['gear'] for r in selected)!=[16,64]:raise ValueError('H C16/C64 missing')
    for row in selected:
        if row['status']!='PASS':raise ValueError('H gear failed')
        case=prepare_case(root,row,Path(runner['path']));watch.update(case['watch'])
        folder=case['folder'];abi=case['abi']
        if len(abi['inputs'])!=15 or [x['name'] for x in abi['outputs']]!=OUTPUTS:raise ValueError('full 15/11 ABI required')
        cases[str(row['gear'])]=dict(om=c.identity(case['om']),abi=abi,
            inputs={k:c.identity(v) for k,v in case['inputs'].items()},
            golden={n:c.identity(folder/'native'/(n+'-actual-0.bin')) for n in OUTPUTS},
            h_outputs={n:c.identity(folder/'candidate'/(n+'-actual-0.bin')) for n in OUTPUTS},
            comparison=c.identity(folder/'comparison.json'),historical_timing=case['historical'])
    device=c.read(root/'device-process.json')
    for key,prefix in [('ASCEND_CUSTOM_OPP_PATH',Path(opp['install'])/'vendors/customize'),('LD_LIBRARY_PATH',Path(opp['install'])/'vendors/customize/op_api/lib')]:
        if Path(device[key].split(':')[0]).resolve()!=prefix:raise ValueError('H loader did not select bound OPP')
    head_binding=head_release_evidence(root,state)
    normalized=canonical_historical_state(root,state)
    normalized['provenance']['c_installed_opp']['libraries']=[dict(x['file']) for x in head_binding['libraries']]
    return dict(reports=files,state=normalized,state_reference_root=str(root.resolve()),
        head_release_binding=head_binding,
        runner=runner,opp=opp,opp_root=opp['install'],gears=cases,environment=suite['environment'],
        records=[c.identity(p) for p in watch]+[c.identity(base/x['path']) for x in audit['files']]+[c.identity(base/'SOURCE_LOCK.json'),c.identity(boundary_path),c.identity(root/'preflight-input.json'),c.identity(root/'device-process.json')],native_base=str(base),source_lock=audit,native_air=c.identity(air['path']),
        factory=c.identity(state['args']['factory_config']),native_deployment=c.identity(state['args']['native_deployment']),
        historical_build_config=c.identity(state['args']['a_build_config']),
        historical_generated_headers=[c.identity(old_config[k]) for k in ('header','kernel_header') if old_config.get(k)],
        historical_environment={k:device[k] for k in ('ASCEND_CUSTOM_OPP_PATH','LD_LIBRARY_PATH')})


def check(a,out):
    c.clean();pins=c.read(c.HERE/'accepted_reports.json')
    h=historical(a.history,pins['history'])
    pairs={}
    for label,path in [('forward',a.forward),('reverse',a.reverse)]:
        dest=out/label;dest.mkdir();pairs[label]=pair(path,pins['pairs'][label],dest)
        for child in pairs[label]['runs'].values():
            if Path(c.read(c.verify(child['sdk_report']))['cann_root']).resolve()!=a.cann_root:raise ValueError('accepted SDK root differs from selected CANN')
    for mode in ('off','raw'):
        x=pairs['forward']['runs'][mode]['configuration'];y=pairs['reverse']['runs'][mode]['configuration']
        if any(x[k]!=y[k] for k in (*FIELDS,'source_sha256','gate_prefetch_mode')):raise ValueError('cross-round build drift')
    from opp import validate_sources
    for mode in ('off','raw'):
        _,c_config=validate_sources(c.verify(pairs['forward']['runs'][mode]['config']),Path(h['state']['args']['c_release']))
        if c_config!=h['opp']['c_config']:raise ValueError('non-A package registration would change')
    # Checkpoints, features and per-layer bytes must be common across the isolated
    # pair and H's full model. Reuse exact captures; never invent a replacement.
    from a2_common import load_bundle,validate_layout
    from run_a2 import validate_om_manifest
    from a4_contract import paired_gears
    bundles={};records=[];projections={}
    from compare_a51 import unwrap
    a51=Path(h['state']['args']['a51_suite']);parent=c.read(a51);records.append(c.identity(a51))
    for label,folder in [('forward',a.forward),('reverse',a.reverse)]:
        source=c.read(folder/'suite.json')['source']['accepted_pair']
        if c.linked(folder,source)!=a51.resolve():raise ValueError('H/five-layer A5.1 provenance differs')
    stage=c.linked(a51.parent,parent['runs']['256']);_,full,_=unwrap(stage,256);full_report=c.read(full)
    records.extend([c.identity(stage),c.identity(full)])
    for gear in (16,64):
        path=Path(h['state']['args'][f'a{gear}_bundle']);b=load_bundle(path,'a4');bundles[gear]=b
        np=Path(h['state']['args'][f'a{gear}_native']);validate_om_manifest(np,path,b,'a4')
        records.extend([c.identity(path),c.identity(np),c.identity(h['state']['args'][f'replay{gear}'])])
        for case in b['cases']:
            validate_layout(path.parent,case,'a4')
            records.extend(c.identity(c.linked(path.parent,f)) for f in case['files'].values())
        golden_path=c.linked(full.parent,full_report['checks'][f'c{gear}']['summary']);golden=c.read(golden_path)
        if golden['bundle_sha256']!=c.sha(path) or golden['native_om_manifest_sha256']!=c.sha(np):raise ValueError('projection Golden source differs')
        rows=[]
        for case in b['cases']:
            old=next(x for x in golden['cases'] if x['name']==case['name'])
            outputs=old['outputs']['native_om']+old['bracket_outputs']['native_om']
            if old['status']!='PASS' or len(outputs)!=4:raise ValueError('projection Golden acceptance incomplete')
            ids=[c.identity(c.linked(golden_path.parent,x)) for x in outputs]
            if len({x['sha256'] for x in ids})!=1:raise ValueError('native projection repeat differs')
            rows.append(dict(name=case['name'],shape=[case[k] for k in ('m','k','n')],
                inputs={n:c.identity(c.linked(path.parent,case['files'][n])) for n in ('x.bin','w_nz.bin','s_gn.bin')},golden=ids[0]))
        projections[str(gear)]=rows;records.append(c.identity(golden_path))
    paired_gears(bundles[16],bundles[64])
    src=pairs['forward']['comparison']['cases']
    for pairrow in src:
        name=pairrow['name'];capture=next(x for x in bundles[16]['cases'] if x['name']==name)
        for field,ref in capture['files'].items():
            if ref['sha256']!=pairrow['input_identity']['capture_files'][field]['sha256']:raise ValueError('H/pair capture source differs')
    for pairvalue in pairs.values():
        for row in pairvalue['comparison']['cases']:
            want=next(x for x in src if x['name']==row['name'])
            if row['input_identity']!=want['input_identity']:raise ValueError('pair input/Golden identity drift')
    tools={}
    for name in ('msopgen','cmake','c++','atc'):
        requested=h['state']['args']['atc'] if name=='atc' else name
        path=shutil.which(str(requested))
        if not path:raise ValueError('build tool absent from selected environment: '+str(requested))
        item=c.identity(path)
        if name in ('msopgen','atc') and a.cann_root not in Path(item['path']).parents:raise ValueError('tool outside selected CANN: '+name)
        tools[name]=item
    c.clean()
    return dict(status='H_B_R_IDENTITIES_REVERIFIED_NO_DEVICE',H=h,pairs=pairs,capture_records=records,projections=projections,tools=tools,
        historical_environment=h['historical_environment'],B=pairs['forward']['runs']['off'],R=pairs['forward']['runs']['raw'],
        historical_a_raw_flag=h['opp']['a_config'].get('gate_prefetch_mode','NOT_DECLARED_LEGACY'),
        no_production_adoption=True)
