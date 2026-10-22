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
    args=SimpleNamespace(cann_root=cann,output_dir=out,antiquant_audit=None,antiquant_review=None)
    before={str(p):identity(p) for p in cann.rglob('*') if p.is_file()}
    with patch('aq_sdk.shutil.which',return_value=None):
        assert audit.run(args)==0
    assert before=={str(p):identity(p) for p in cann.rglob('*') if p.is_file()}
    r=json.loads((out/'group128-capability.json').read_text())
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


def test_existing_input_review_never_proves_group128_or_fixpipe():
    r=audit.conclusions({'status':'SDK_REVIEW_REQUIRED'},{'reviewer':'source-cited external review'})
    assert r['AntiQuant Input Path']['compatibility']=='SOURCE_REVIEWED_K128_ONLY'
    assert r['AntiQuant Input Path']['group128_twenty_groups']=='UNRESOLVED'
    assert r['FixPipe Output Path']['compatibility']=='UNRESOLVED'
