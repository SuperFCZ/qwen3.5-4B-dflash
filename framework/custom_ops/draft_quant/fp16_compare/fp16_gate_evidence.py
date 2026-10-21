"""Reuse accepted hardware Off/Raw artifacts, never Simulator variants."""
import json
from pathlib import Path
from fp16_gate_common import HERE,NAME,identity,verify,write_json,sha256
from a2_common import checked_file,load_bundle,validate_layout
from a3_launch import load_build_config
from opp_preflight import check_install
from run_a2 import validate_om_manifest
from run_gate_copy import layer_identity
from run_gate_copy_pair import source_evidence

FROZEN=('op_kernel/d_flash_group_quant_linear.cpp','op_kernel/d_flash_group_quant_linear_build_config.h',
        'op_host/d_flash_group_quant_linear.cpp','op_host/d_flash_group_quant_linear_contract.h',
        'op_host/d_flash_group_quant_linear_tiling.h','op_host/d_flash_group_quant_linear_build_config.h',
        'test/main.cpp','test/runner_common.h')


def accepted(a51,pair_path):
    watched={}
    def watch(p):
        item=identity(p);watched[item['path']]=item;return item
    for p in (a51,pair_path):watch(p)
    source=source_evidence(a51);bundle_path=Path(source['bundle']);native_path=Path(source['native']);gold_path=Path(source['golden'])
    for p in (bundle_path,native_path,gold_path):watch(p)
    bundle=load_bundle(bundle_path,'a4');native=validate_om_manifest(native_path,bundle_path,bundle,'a4');gold=json.loads(gold_path.read_text())
    if bundle['context_rows']!=16 or gold['status']!='PASS' or gold['bundle_sha256']!=sha256(bundle_path) or gold['native_om_manifest_sha256']!=sha256(native_path):raise ValueError('frozen C16 bundle/Golden differs')
    case=next(c for c in bundle['cases'] if c['name']==NAME);old=next(c for c in gold['cases'] if c['name']==NAME)
    if old['status']!='PASS':raise ValueError('frozen Gate failed')
    validate_layout(bundle_path.parent,case,'a4')
    compiled=next(c for c in native['cases'] if c['name']==NAME)
    ids=layer_identity(case,compiled,old,bundle_path.parent,native_path.parent,gold_path.parent)
    for item in [*ids['capture_files'].values(),*ids['native'].values(),*ids['golden']]:watch(item['path'])
    golden=Path(ids['golden'][0]['path'])
    if any(Path(item['path']).read_bytes()!=golden.read_bytes() for item in ids['golden']):raise ValueError('frozen native repeat/tail/postcheck drift')
    pair=json.loads(pair_path.read_text())
    if pair.get('status')!='PASS' or pair.get('experiment')!='prefetch' or set(pair.get('runs',{}))!={'off','raw'}:raise ValueError('need real accepted Off/Raw pair')
    comp=checked_file(pair_path.parent,pair['comparison']);watch(comp);comp=json.loads(comp.read_text())
    if comp.get('status')!='PASS' or not any(c['name']==NAME and c['status']=='PASS' for c in comp['cases']):raise ValueError('pair correctness missing')
    modes={}
    for mode in ('off','raw'):
        stage_path=Path(pair['runs'][mode]['path'])
        if sha256(stage_path)!=pair['runs'][mode]['sha256']:raise ValueError('accepted stage changed')
        watch(stage_path);stage=json.loads(stage_path.read_text());cfg_path=Path(stage['build_config_path']);watch(cfg_path)
        cfg=load_build_config(cfg_path)
        if stage['status']!='PASS' or stage['build']!=cfg or sha256(cfg_path)!=stage['build_config_sha256'] or cfg['gate_prefetch_mode']!=mode or cfg['gate_copy_mode']!='loop' or cfg['k_tile']!=256 or cfg['scale_mode']!='broadcast':raise ValueError('accepted Off/Raw configuration differs')
        if stage['checks'][NAME]['status']!='PASS':raise ValueError('accepted two-call correctness missing')
        for rel in FROZEN:
            path=HERE.parent/rel
            if sha256(path)!=cfg['source_sha256'].get(rel):raise ValueError('frozen algorithm/runner changed: '+rel)
            watch(path)
        for p in (cfg['header'],cfg['kernel_header']):watch(p)
        timed_path=checked_file(stage_path.parent,stage['timing']);watch(timed_path);timed=json.loads(timed_path.read_text())
        if timed['status']!='PASS' or timed['bundle_sha256']!=sha256(bundle_path) or timed['native_om_manifest_sha256']!=sha256(native_path):raise ValueError('accepted timing uses other capture')
        runner=Path(timed['runners']['custom']['path'])
        if sha256(runner)!=timed['runners']['custom']['sha256']:raise ValueError('accepted runner changed')
        watch(runner)
        opp=Path(cfg['header']).parents[2]/'opp';installed=check_install(opp)
        for p in opp.rglob('*'):
            if p.is_file():watch(p)
        modes[mode]=dict(runner=str(runner),opp=str(opp),config=cfg,installed=installed,stage=identity(stage_path))
    for key in ('core_limit','pipeline_mode','dequant_mode','kv_m80_mode','k_tile','scale_mode','source_sha256'):
        if modes['off']['config'][key]!=modes['raw']['config'][key]:raise ValueError('Off/Raw differs beyond prefetch: '+key)
    return dict(source=source,identity=ids,files=ids['capture_files'],golden=identity(golden),modes=modes,watched=watched)
