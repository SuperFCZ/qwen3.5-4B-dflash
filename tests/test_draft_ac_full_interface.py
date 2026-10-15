"""Read-only interface regressions. No frontend registration or device execution."""
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from test_draft_ac_integration import ac_tools

SCHEMA=('npu::npu_chunk_gated_delta_rule(Tensor query, Tensor key, Tensor value, Tensor g, Tensor beta, '
        'Tensor effective_length, int chunk_size=64, Tensor? initial_state=None, bool output_final_state=True, '
        'bool use_qk_l2norm_in_kernel=True) -> (Tensor, Tensor)')


def module(monkeypatch):
    ac_tools(monkeypatch)
    return importlib.import_module('full_interface')


@pytest.mark.parametrize('damage',[None,'legacy9','wrong_position','wrong_type','keyword_only','return_type'])
def test_locked_gdr_schema_has_no_legacy_fallback(monkeypatch,damage):
    tool=module(monkeypatch); schema=SCHEMA
    if damage=='legacy9': schema=schema.replace('Tensor effective_length, ','')
    elif damage=='wrong_position': schema=schema.replace('Tensor beta, Tensor effective_length','Tensor effective_length, Tensor beta')
    elif damage=='wrong_type': schema=schema.replace('Tensor effective_length','int effective_length')
    elif damage=='keyword_only': schema=schema.replace('Tensor effective_length','*, Tensor effective_length')
    elif damage=='return_type': schema=schema.replace('-> (Tensor, Tensor)','-> Tensor')
    op=SimpleNamespace(_schema=torch._C.parse_schema(schema))
    if damage:
        with pytest.raises(RuntimeError,match='schema drifted'): tool.check_gdr_schema(op)
    else: assert tool.check_gdr_schema(op)==str(op._schema)


@pytest.mark.parametrize('valid_schema,prototype_status',[(True,'PASS'),(False,'PASS'),(True,'NOT_FOUND'),(True,'FAIL')])
def test_report_collects_dispatcher_and_opp_even_when_one_fails(monkeypatch,valid_schema,prototype_status):
    tool=module(monkeypatch)
    from qwen35_dflash.ascend310p import custom_op_export as original
    schema=SCHEMA if valid_schema else SCHEMA.replace('Tensor effective_length, ','')
    fake=SimpleNamespace(ops=SimpleNamespace(npu=SimpleNamespace(npu_chunk_gated_delta_rule=SimpleNamespace(default=SimpleNamespace(_schema=torch._C.parse_schema(schema)))),loaded_libraries=set()),
                         _C=SimpleNamespace(_dispatch_dump=lambda name:'CPU fixture RegisterSchema.cpp'),__version__='CPU fixture')
    api=SimpleNamespace(NPU_CHUNK_GATED_DELTA_RULE_TORCH_OP=original.NPU_CHUNK_GATED_DELTA_RULE_TORCH_OP,
        validate_gdr_ge_prototype_environment=lambda:dict(status=prototype_status),
        validate_adn_attention_ge_prototype_environment=lambda:dict(status='PASS'))
    npu=SimpleNamespace(npu_chunk_gated_delta_rule=lambda *a,**k:pytest.fail('must not execute GDR'))
    result=tool.inspect_interfaces(fake,npu,api,{})
    assert result['status']==('PASS' if valid_schema and prototype_status=='PASS' else 'BLOCKED')
    assert len(result['dispatcher']['arguments'])==(10 if valid_schema else 9)
    assert result['checks']['gdr_ge']['status']==prototype_status
    assert result['dispatcher']['registration_dump']=='CPU fixture RegisterSchema.cpp'


def test_full_preflight_failure_stops_before_success_handoff(tmp_path,monkeypatch):
    module(monkeypatch)
    worker=importlib.import_module('preflight_worker');resolver=importlib.import_module('resolve')
    gate=importlib.import_module('full_interface')
    from process_boundary import encode_args
    args=SimpleNamespace(output_dir=tmp_path,full_draft=True)
    request=tmp_path/'preflight-input.json';request.write_text(json.dumps(encode_args(args)))
    monkeypatch.setattr(resolver,'resolve',lambda *a:({},{}))
    calls=[]
    def blocked(args,path):
        calls.append(path);path.write_text(json.dumps(dict(status='BLOCKED',reason='legacy nine-argument GDR')))
        raise RuntimeError('schema drifted')
    monkeypatch.setattr(gate,'require_full_interfaces',blocked)
    assert worker.run(request)==1
    report=json.loads((tmp_path/'preflight-result.json').read_text())
    assert report['status']=='FAIL' and len(calls)==1 and 'preflight' not in report
    assert not (tmp_path/'joint').exists()


def test_blocked_report_keeps_hash_bound_native_export_history(tmp_path,monkeypatch):
    import sys
    tool=module(monkeypatch)
    config=dict(draft_quantization='w8a16',verify_gdr='chunk')
    factory=tmp_path/'factory.json';factory.write_text(json.dumps(config))
    historical=dict(environment={'torch_npu':'historical-private-build'},operator_preflight={'operators':[{'torch_schema':SCHEMA}]},
                    factory_config=config,graphs=[])
    air=tmp_path/'nondefault-air.json';air.write_text(json.dumps(historical))
    dep=tmp_path/'deployment.json';dep.write_text(json.dumps(dict(air_manifest=dict(path=air.name,sha256=tool.sha256(air)))))
    args=SimpleNamespace(factory_config=factory,native_deployment=dep)
    monkeypatch.setitem(sys.modules,'torch_npu',SimpleNamespace())
    monkeypatch.setitem(sys.modules,'torchair',SimpleNamespace())
    monkeypatch.setattr(tool,'inspect_interfaces',lambda *a:dict(status='BLOCKED',dispatcher={'schema':'legacy9'}))
    output=tmp_path/'report.json'
    with pytest.raises(RuntimeError,match='incompatible or unverified'): tool.require_full_interfaces(args,output)
    report=json.loads(output.read_text())
    assert report['native_export']['environment']==historical['environment']
    assert report['native_export']['operator_preflight']==historical['operator_preflight']
    assert report['native_export']['air']['sha256']==tool.sha256(air)
