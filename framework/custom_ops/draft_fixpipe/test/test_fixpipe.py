"""CPU evidence/rounding/traffic checks. No device or SDK implementation import."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
import pytest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('group128_fixpipe_audit',ROOT/'audit.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
from fixpipe_math import counterexamples,evaluate,traffic
from aq_common import identity,verify


def test_weight_rounding_move_is_not_bitwise_equivalent():
    c=counterexamples()['cases'][0]
    assert c['result']['baseline_bits']=='0x4482'
    assert c['result']['alternative_bits']=='0x4481'
    assert c['group_output_fp16'] is False
    assert counterexamples()['hardware']=='NOT_RUN'


def test_group_fp16_and_fp32_reordering_have_independent_counterexamples():
    cases=counterexamples()['cases']
    assert cases[1]['result']['baseline_fp16']==1 and cases[1]['result']['alternative_fp16']==0
    assert cases[2]['result']['baseline_fp16']==.25 and cases[2]['result']['alternative_fp16']==0
    # Embed in 20 complete groups with zeros, rather than merely asserting equivalence.
    for c in cases:
        padded=[(s,terms+[(0.,0)]*(128-len(terms))) for s,terms in c['groups']]
        padded += [(1.,[(0.,0)]*128)]*(20-len(padded))
        assert evaluate(padded,c['group_output_fp16'])==c['result']


def test_exact_case_does_not_imply_universal_disagreement():
    r=evaluate([(0.5,[(1.,2),(2.,-1)])]*20)
    assert r['bitwise_equal'] and r['baseline_fp16']==0
    with pytest.raises(ValueError):evaluate([(1.,[(.1,3)])])
    with pytest.raises(ValueError):evaluate([(1.,[(1.,128)])])
    with pytest.raises(ValueError):evaluate([(float('inf'),[(1.,3)])])
    with pytest.raises(ValueError):evaluate([(1.,[(float('nan'),3)])])


def test_gm_bytes_reduction_counts_and_no_latency_projection():
    t=traffic()
    assert t['groups']==20 and t['n_tiles']==304
    assert t['partial_fp32_bytes_per_n_tile']==4096 and t['final_fp16_bytes']==622592
    assert t['scaled_fp32_partial_spill']['gm_write_bytes']==24903680
    assert t['scaled_fp32_partial_spill']['extra_gm_bytes']==49807360
    assert t['gm_accumulator_variant']['extra_gm_bytes']==99614720
    assert t['logical_reduction_additions']==5914624
    assert t['onchip_scaled_fp32_return']['feasibility']=='UNRESOLVED'
    assert t['latency'].startswith('UNRESOLVED')
    with pytest.raises(ValueError):traffic(k=2561)


def fixture(tmp):
    cann=tmp/'cann';cann.mkdir();(cann/'version.info').write_text('Version=9.0.0\n')
    samples={
      'include/ascendc/dav_m200/kernel_fixpipe.h':'#if __CCE_AICORE__ == 200\n// Fixpipe fixture, not a hardware assertion\n#endif\n',
      'python/tbe/tik/tik_cube_api.py':'raise RuntimeError("MUST_NOT_IMPORT")\ndef fixpipe(): pass\n',
      'include/ascendc/matmul/matmul.h':'// GetTensorC SetQuantVector SetAntiQuantVector enPartialSum\n',
      'data/platform_config/Ascend310P3.ini':'SoC_version=Ascend310P3\nAIC_version=dav_2002\n'}
    for name,content in samples.items():
        p=cann/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(content)
    return cann


def test_collect_does_not_promote_names_to_support_or_import_tik(tmp_path):
    cann=fixture(tmp_path);out=tmp_path/'result'
    unrelated=cann/'aarch64-linux/data/fusion_strategy/built-in/ascend310b1_l2cache.json'
    unrelated.parent.mkdir(parents=True)
    with unrelated.open('wb') as f:f.truncate(audit.MAX_SOURCE_BYTES+1)
    args=SimpleNamespace(cann_root=cann,output_dir=out,antiquant_audit=None,antiquant_review=None)
    before={str(p):identity(p) for p in cann.rglob('*') if p.is_file()}
    with patch('aq_sdk.shutil.which',return_value=None):
        assert audit.run(args)==0
    assert before=={str(p):identity(p) for p in cann.rglob('*') if p.is_file()}
    r=json.loads((out/'group128-capability.json').read_text())
    assert r['scan']['status']=='COMPLETE_WITHIN_DECLARED_SCOPE'
    assert r['scan']['skipped_by_reason']['UNRELATED_FUSION_STRATEGY_JSON']['count']==1
    for p in r['paths'].values():assert p['compatibility']=='UNRESOLVED'
    frontends=r['paths']['FixPipe Output Path']['frontends']
    assert set(frontends)=={'ascendc_explicit','tik','matmul_sdk'}
    for v in frontends.values():
        assert all(c['status']=='UNRESOLVED' for c in v['claims'].values())
        assert len(v['claims'])==10
    assert any('tik' in f['frontend_hints'] for f in r['files'])
    assert any('ascendc_explicit' in f['frontend_hints'] for f in r['files'])
    assert any('matmul_sdk' in f['frontend_hints'] for f in r['files'])
    assert r['npu']==r['simulator']==r['performance']==r['cann_compile']=='NOT_RUN'
    template=json.loads((out/'fixpipe-review-template.json').read_text())
    assert template['audit']==identity(out/'group128-capability.json')
    assert 'W8A8' in r['paths']['FixPipe Output Path']['activation_quantized_variant']


def test_tamper_old_sdk_snapshot_and_wrong_root_rejected(tmp_path):
    cann=fixture(tmp_path)
    with patch('aq_sdk.shutil.which',return_value=None):_,path=audit.collect_input(cann,tmp_path/'input')
    audit.verify_input(path,cann)
    with pytest.raises(ValueError,match='different SDK'):audit.verify_input(path,tmp_path/'other')
    r=json.loads(path.read_text());f=r['files'][0]
    Path(f['snapshot']['path']).write_text('tampered')
    with pytest.raises(ValueError,match='identity'):audit.verify_input(path,cann)


def test_source_mutation_and_escape_rejected(tmp_path):
    cann=fixture(tmp_path);output=tmp_path/'fix';output.mkdir()
    files=audit.collect_fix(cann,output)
    Path(files[0]['source']['path']).write_text('changed')
    with pytest.raises(ValueError):verify(files[0]['source'])
    outside=tmp_path/'outside.h';outside.write_text('Fixpipe')
    (cann/'escape.h').symlink_to(outside)
    second=tmp_path/'fix2';second.mkdir()
    with pytest.raises(ValueError,match='escapes'):audit.collect_fix(cann,second)


def test_permission_error_not_silently_ignored(tmp_path):
    cann=fixture(tmp_path);out=tmp_path/'out';out.mkdir()
    def denied(*a,**kw):kw['onerror'](PermissionError('SDK unreadable'))
    with patch.object(audit.os,'walk',side_effect=denied):
        with pytest.raises(PermissionError):audit.collect_fix(cann,out)
    scan=json.loads((out/'fixpipe-scan.json').read_text())
    assert scan['status']=='INCOMPLETE' and scan['error']['type']=='PermissionError'


def test_existing_input_review_never_proves_group128_or_fixpipe():
    r=audit.conclusions({'status':'SDK_REVIEW_REQUIRED'},{'reviewer':'source-cited external review'})
    assert r['AntiQuant Input Path']['compatibility']=='SOURCE_REVIEWED_K128_ONLY'
    assert r['AntiQuant Input Path']['group128_twenty_groups']=='UNRESOLVED'
    assert r['FixPipe Output Path']['compatibility']=='UNRESOLVED'


@pytest.mark.parametrize('path',[
    'aarch64-linux/data/fusion_strategy/built-in/ascend310b1_l2cache.json',
    'data/cache/ascend310p_l2_cache.json','data/irrelevant.json',
])
def test_irrelevant_large_json_filtered_before_stat_or_read(tmp_path,path):
    cann=fixture(tmp_path);large=cann/path;large.parent.mkdir(parents=True,exist_ok=True)
    with large.open('wb') as f:f.truncate(audit.MAX_SOURCE_BYTES+1)
    original_stat=Path.stat;original_open=Path.open
    def guarded_stat(p,*a,**kw):
        assert p!=large,'excluded JSON must not be statted'
        return original_stat(p,*a,**kw)
    def guarded_open(p,*a,**kw):
        assert p!=large,'excluded JSON must not be opened or hashed'
        return original_open(p,*a,**kw)
    out=tmp_path/'out';out.mkdir();scan={}
    with patch.object(Path,'stat',guarded_stat),patch.object(Path,'open',guarded_open):
        files=audit.collect_fix(cann,out,scan)
    reason=audit.scan_rule(Path(path))[1]
    assert scan['status']=='COMPLETE_WITHIN_DECLARED_SCOPE'
    assert scan['skipped_by_reason'][reason]['count']==1
    assert str(large) in scan['skipped_by_reason'][reason]['samples']
    assert scan['enumerated_files']==scan['snapshotted_files']+scan['skipped_files']
    assert all(f['source']['path']!=str(large) for f in files)
    assert any('kernel_fixpipe.h' in f['source']['path'] for f in files)


@pytest.mark.parametrize('path',[
    'include/ascendc/kernel_fixpipe.h','include/ascendc/matmul/api.hpp',
    'src/dav_m200/move.cpp','src/move.cc','python/tbe/tik/tik_cube_api.py',
    'data/platform_config/Ascend310P3.ini',
    'opp/op_impl/weight_quant_batch_matmul_v3/config.json',
    'opp/op_impl/fixpipe/config.json',
    'opp/op_impl/ai_core/tbe/config/ascend310p/aic-ascend310p-ops-info.json',
    'opp/op_info_cfg/ai_core/ascend310p.json',
    'data/platform_config/Ascend310P3.json',
    'data/product_capabilities.json',
    'data/fusion_strategy/built-in/matmul_quantization.json',
])
def test_required_source_families_and_json_registries_admitted(path):
    assert audit.scan_rule(Path(path))[0]


@pytest.mark.parametrize('suffix',['.h','.json'])
def test_related_oversized_source_preserves_partial_evidence_and_blocks(tmp_path,suffix):
    cann=fixture(tmp_path);large=cann/('zzz/fixpipe/oversize'+suffix);large.parent.mkdir(parents=True)
    with large.open('wb') as f:f.truncate(audit.MAX_SOURCE_BYTES+1)
    out=tmp_path/'report';args=SimpleNamespace(cann_root=cann,output_dir=out,antiquant_audit=None,antiquant_review=None)
    with patch.object(audit,'collect_input') as input_scan:
        assert audit.run(args)==1;input_scan.assert_not_called()
    report=json.loads((out/'group128-capability.json').read_text());scan=report['scan']
    assert report['status']=='BLOCKED_AUDIT' and scan['status']=='INCOMPLETE'
    assert report['files'] and scan['snapshotted_files']==len(report['files'])
    issue=scan['incomplete'][0]
    assert issue['path']==str(large) and issue['bytes']==audit.MAX_SOURCE_BYTES+1
    assert issue['limit_bytes']==16*1024*1024 and issue['source']==identity(large)
    assert issue['reason']=='SOURCE_SIZE_LIMIT_EXCEEDED'
    assert not (out/'fixpipe-review-template.json').exists()
    for f in report['files']:verify(f['source']);verify(f['snapshot'])
    assert report['scan_file']==identity(out/'fixpipe-scan.json')
    assert all(p['compatibility']=='UNRESOLVED' for p in report['paths'].values())
    assert report['npu']==report['cann_compile']==report['simulator']=='NOT_RUN'


def test_normal_sources_and_generic_product_json_keep_bound_snapshots(tmp_path):
    cann=fixture(tmp_path);p=cann/'data/product_capabilities.json'
    p.parent.mkdir(exist_ok=True);p.write_text('{"fixture_dtype_constraint": "float32"}')
    out=tmp_path/'out';out.mkdir();scan={};files=audit.collect_fix(cann,out,scan)
    f=next(f for f in files if f['source']['path']==str(p))
    assert f['admission_rule']=='SUPPORT_REGISTRY_JSON' and f['hits']==[]
    assert f['source']['sha256']==f['snapshot']['sha256']
    assert scan['status']=='COMPLETE_WITHIN_DECLARED_SCOPE'


def test_read_failure_keeps_scan_incomplete_and_partial_files(tmp_path):
    cann=fixture(tmp_path);blocked=cann/'zzz/fixpipe.h';blocked.parent.mkdir();blocked.write_text('Fixpipe')
    original=Path.open
    def denied(p,*args,**kwargs):
        if p==blocked:raise PermissionError('source access denied')
        return original(p,*args,**kwargs)
    out=tmp_path/'out';out.mkdir();files=[];scan={}
    with patch.object(Path,'open',denied):
        with pytest.raises(PermissionError):audit.collect_fix(cann,out,scan,files)
    assert files and scan['status']=='INCOMPLETE'
    assert scan['error']['phase']=='read' and scan['error']['path']==str(blocked)
    assert json.loads((out/'fixpipe-scan.json').read_text())==scan
