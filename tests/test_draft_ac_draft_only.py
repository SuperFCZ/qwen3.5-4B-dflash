"""CPU scope/identity regressions, never CANN/NPU acceptance evidence."""
import copy
import importlib
import json
import sys
from types import SimpleNamespace
import pytest
import torch
from test_draft_ac_integration import ac_tools, ROOT
from rms_norm_test_support import adn_rms_norm_cpu


def module(monkeypatch):
    ac_tools(monkeypatch)
    return importlib.import_module('draft_only')


def test_registration_is_exactly_draft_not_target(monkeypatch):
    tool = module(monkeypatch)
    from qwen35_dflash.ascend310p import custom_op_export as custom
    seen = []
    def prepare(spec, air):
        assert spec.torch_op in tool.DRAFT_OPS
        seen.append(spec)
        return SimpleNamespace(spec=spec, schema='fixture', fake_kernel='fixture', converter_policy='fixture')
    monkeypatch.setattr(custom, 'prepare_custom_op_export', prepare)
    report = tool.prepare_draft_interfaces(dict(draft_quantization='w8a16', draft_quant_matmul='weight_quant'), object())
    assert len(report) == 3 and {s.torch_op for s in seen} == tool.DRAFT_OPS
    assert {s.ge_op_type for s in seen} == {'AdnRmsNorm', 'ScatterNdUpdate', 'WeightQuantBatchMatmulV2'}


@pytest.mark.parametrize('missing', [None, 'npu::npu_scatter_nd_update', 'npu::adn_rms_norm'])
def test_legacy_gdr_is_reported_but_required_draft_interface_still_gates(tmp_path, monkeypatch, missing):
    tool = module(monkeypatch)
    config = dict(draft_quantization='w8a16', draft_quant_matmul='weight_quant')
    air = tmp_path/'old-air.json'; air.write_text('{}')
    monkeypatch.setattr(tool, 'native_history', lambda args: (config, {}, air))
    monkeypatch.setitem(sys.modules, 'torch_npu', SimpleNamespace())
    monkeypatch.setitem(sys.modules, 'torchair', SimpleNamespace())
    from qwen35_dflash.ascend310p import custom_op_export as custom
    seen = []
    def prepare(spec, air):
        seen.append(spec.torch_op)
        if spec.torch_op == missing: raise RuntimeError('required Draft op unavailable')
        return SimpleNamespace(spec=spec, schema='fixture', fake_kernel='fixture', converter_policy='fixture')
    monkeypatch.setattr(custom, 'prepare_custom_op_export', prepare)
    from test_draft_ac_full_interface import SCHEMA
    legacy = SimpleNamespace(default=SimpleNamespace(_schema=torch._C.parse_schema(
        SCHEMA.replace('Tensor effective_length, ', ''))))
    # Python read-only dispatcher stand-in, never torch.library registration.
    monkeypatch.setattr(torch.ops.npu, 'npu_chunk_gated_delta_rule', legacy, raising=False)
    path = tmp_path/'interface.json'
    if missing:
        with pytest.raises(RuntimeError, match='Draft-only interface'): tool.require_draft_interfaces(None, path)
    else:
        result = tool.require_draft_interfaces(None, path)
        assert result['status'] == 'PASS'
    result = json.loads(path.read_text())
    assert result['target_interface'] == 'BLOCKED'
    assert result['draft_only'] == result['end_to_end'] == 'NOT_RUN'
    assert set(seen) <= tool.DRAFT_OPS
    assert torch.ops.npu.npu_chunk_gated_delta_rule is legacy
    assert len(legacy.default._schema.arguments) == 9


def source_fixture(tool):
    from qwen35_dflash.ascend310p.incremental import DraftGraph
    from qwen35_dflash.ascend310p.contracts import AirGraphSpec
    # Metadata-only scope fixture. A separate test executes the real tiny Draft.
    graph = DraftGraph.__new__(DraftGraph); torch.nn.Module.__init__(graph)
    metadata = {k: {'sha256': 'original'} for k in tool.IDENTITIES}
    metadata.update(incremental_contract={'all_target_and_draft_fields': ['retained']},
                    tensor_abi={'inputs': [], 'outputs': []}, adn_rms_norm_ge_op_type='AdnRmsNorm')
    ops = tool.draft_operators(dict(draft_quantization='w8a16', draft_quant_matmul='weight_quant'))
    source = AirGraphSpec(name='draft', role='draft', model=graph, example_args=(),
                          input_names=('features',), output_names=('draft_top1',), metadata=metadata, custom_ops=ops)
    native = dict(metadata=copy.deepcopy(metadata), input_names=['features'], output_names=['draft_top1'])
    return source, native


@pytest.mark.parametrize('damage', [None, *range(8), 'contract', 'abi', 'target_op', 'model', 'order'])
def test_exact_original_model_and_complete_target_identity(monkeypatch, damage):
    from dataclasses import replace
    from qwen35_dflash.ascend310p.contracts import CustomOpExportSpec
    tool = module(monkeypatch); source, native = source_fixture(tool)
    if isinstance(damage, int): native['metadata'][tool.IDENTITIES[damage]] = 'changed'
    elif damage == 'contract': native['metadata']['incremental_contract']['another_target_field'] = 'changed'
    elif damage == 'abi': native['metadata']['tensor_abi']['inputs'] = ['changed']
    elif damage == 'target_op': source = replace(source, custom_ops=source.custom_ops + (CustomOpExportSpec('npu::npu_chunk_gated_delta_rule','ChunkGatedDeltaRule'),))
    elif damage == 'model': source = replace(source, model=torch.nn.Identity())
    elif damage == 'order': native['input_names'] = ['anchor']
    if damage is None: assert tool.select_draft([source], native) is source
    else:
        with pytest.raises(ValueError): tool.select_draft([source], native)


def test_unchanged_factory_never_prepares_or_executes_target(monkeypatch):
    tool = module(monkeypatch); source, native = source_fixture(tool)
    from qwen35_dflash.ascend310p import quant_factory
    def factory(config): return (SimpleNamespace(name='target_verify'), source)
    factory.prepare_export = lambda *args: pytest.fail('public Target preparation must not be called')
    monkeypatch.setattr(quant_factory, 'create_quant_incremental_graphs', factory)
    assert tool.load_draft({}, native) is source
    def bad_factory(config):
        torch.nn.Identity()(torch.zeros(1))
        return (source,)
    monkeypatch.setattr(quant_factory, 'create_quant_incremental_graphs', bad_factory)
    with pytest.raises(RuntimeError, match='must not execute'): tool.load_draft({}, native)
    # The diagnostic guard must always be removed, including after failure.
    assert torch.nn.Identity()(torch.zeros(1)).item() == 0


@pytest.mark.usefixtures('adn_rms_norm_cpu')
def test_real_draft_forward_never_calls_target_or_gdr(monkeypatch):
    from test_incremental_air_om import TinyTarget, draft_model, rotary
    from qwen35_dflash.ascend310p.incremental import incremental_graph_specs, TargetRowsGraph
    def forbidden(*args, **kwargs): pytest.fail('Target/GDR/Verify reached Draft replay')
    monkeypatch.setattr(TargetRowsGraph, 'forward', forbidden)
    target = TinyTarget(capacity=128).eval()
    specs = incremental_graph_specs(target, draft_model(), capacity=128, metadata={}, gdr=forbidden,
                                    attention=forbidden, rotary=rotary)
    source = next(s for s in specs if s.name == 'draft')
    with torch.no_grad(): result = source.model(*source.example_args)
    assert result[0].shape == (1, 15)
    assert len(result) == 1 + len(source.metadata['incremental_contract']['draft_states'])


def test_air_scope_rejects_target_before_atc(tmp_path, monkeypatch):
    tool = module(monkeypatch)
    from qwen35_dflash.ascend310p import draft_ac_constants
    from qwen35_dflash.ascend310p import utils
    seen = []
    monkeypatch.setattr(draft_ac_constants, 'validate_ac_graph', lambda graph: seen.append(graph))
    (tmp_path/'dynamo.pbtxt').write_text('fixture')
    air = dict(graphs=[dict(role='diagnostic', metadata={'draft_ac': {'scope': 'static real Draft diagnostic'}})])
    monkeypatch.setattr(utils, 'count_ge_ir_nodes', lambda paths: {'WeightQuantBatchMatmulV2': 26})
    assert tool.audit_draft_air(air, tmp_path)['status'] == 'PASS' and seen
    for name in tool.TARGET_GE:
        monkeypatch.setattr(utils, 'count_ge_ir_nodes', lambda paths: {name: 1})
        with pytest.raises(ValueError, match='Target operators'): tool.audit_draft_air(air, tmp_path)


def test_draft_preflight_disposable_worker_does_not_call_full_gate(tmp_path, monkeypatch):
    tool = module(monkeypatch)
    worker = importlib.import_module('preflight_worker'); resolver = importlib.import_module('resolve')
    gate = importlib.import_module('full_interface')
    from process_boundary import encode_args
    args = SimpleNamespace(output_dir=tmp_path, full_draft=False, draft_only=True)
    request = tmp_path/'preflight-input.json'; request.write_text(json.dumps(encode_args(args)))
    def resolve(args, root): (root/'preflight.json').write_text('{}')
    monkeypatch.setattr(resolver, 'resolve', resolve)
    monkeypatch.setattr(gate, 'require_full_interfaces', lambda *a: pytest.fail('wrong preflight scope'))
    calls = []
    monkeypatch.setattr(tool, 'require_draft_interfaces', lambda args, path: calls.append(path))
    assert worker.run(request) == 0
    assert calls == [tmp_path/'draft-interface-preflight.json']


def test_cli_scopes_are_mutually_exclusive(monkeypatch):
    import subprocess
    module(monkeypatch)
    result = subprocess.run([sys.executable, '-B', str(ROOT/'framework/custom_ops/draft_ac/test/run.py'),
                             '--full-draft', '--draft-only'], text=True, capture_output=True)
    assert result.returncode != 0 and 'not allowed with argument' in result.stderr


def historical_fixture(tool, tmp_path):
    from test_incremental_air_om import TinyTarget, draft_model, rotary, manifest_graphs
    from qwen35_dflash.ascend310p.incremental import incremental_graph_specs
    from dataclasses import replace
    draft = draft_model()
    draft.layers.extend(copy.deepcopy(draft.layers[0]) for _ in range(3))
    draft.config = replace(draft.config, num_hidden_layers=5,
                           layer_types=('full_attention',)*5)
    metadata = {key: {'sha256': 'CPU fixture'} for key in tool.IDENTITIES}
    specs = incremental_graph_specs(TinyTarget(capacity=128).eval(), draft, capacity=128,
                                    metadata=metadata, gdr=None, attention=None, rotary=rotary)
    # These are tiny FP16 CPU contract fixtures, not a claim about W8/device
    # acceptance. Real W8/NZ payload checks stay in the unchanged resolve().
    graphs = manifest_graphs(specs)
    config = dict(draft_quantization='w8a16', draft_quant_matmul='weight_quant')
    air = tmp_path/'historical-nondefault-air.json'
    air.write_text(json.dumps(dict(factory_config=config, graphs=graphs)))
    from common import sha256
    dep = tmp_path/'deployment.json'
    deployment = dict(graphs=copy.deepcopy(graphs), air_manifest=dict(path=air.name, sha256=sha256(air)))
    dep.write_text(json.dumps(deployment))
    factory = tmp_path/'factory.json'; factory.write_text(json.dumps(config))
    return SimpleNamespace(native_deployment=dep, factory_config=factory), air, deployment


@pytest.mark.parametrize('damage', [None, 'air_hash', 'target_contract', 'kv_abi', 'factory', 'identity'])
def test_historical_full_bundle_and_hash_binding_are_not_replaced(tmp_path, monkeypatch, damage):
    tool = module(monkeypatch)
    args, air, deployment = historical_fixture(tool, tmp_path)
    untouched_air = air.read_bytes()
    if damage == 'air_hash': air.write_text(air.read_text() + ' ')
    elif damage == 'target_contract': deployment['graphs'][0]['metadata']['incremental_contract']['capacity'] = 64
    elif damage == 'kv_abi': deployment['graphs'][-1]['metadata']['tensor_abi']['outputs'][-1]['dtype'] = 'float32'
    elif damage == 'identity': deployment['graphs'][-1]['metadata']['target_input_identity'] = {}
    elif damage == 'factory': args.factory_config.write_text('{}')
    args.native_deployment.write_text(json.dumps(deployment))
    frozen = (air.read_bytes(), args.native_deployment.read_bytes(), args.factory_config.read_bytes())
    if damage:
        with pytest.raises(ValueError): tool.native_history(args)
    else:
        _, native, bound = tool.native_history(args)
        assert bound == air and len(native['input_names']) == 15 and len(native['output_names']) == 11
        assert air.read_bytes() == untouched_air
    assert frozen == (air.read_bytes(), args.native_deployment.read_bytes(), args.factory_config.read_bytes())


@pytest.mark.parametrize('output', ['draft_top1', *[f'd{i}_{kind}' for i in range(5) for kind in ('key', 'value')]])
def test_every_native_id_and_kv_output_is_still_bitwise(tmp_path, monkeypatch, output):
    module(monkeypatch)
    import importlib.util
    path = ROOT/'framework/custom_ops/draft_ac/test/run.py'
    spec = importlib.util.spec_from_file_location('draft_only_test_runner', path)
    runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
    monkeypatch.setenv('AI_RUN_DIR', str(tmp_path))
    golden = {name: b'\x00\x00' for name in ['draft_top1', *[f'd{i}_{k}' for i in range(5) for k in ('key', 'value')]]}
    actual = dict(golden); actual[output] = b'\x01\x00'
    with pytest.raises(ValueError, match='bitwise'): runner.compare(golden, actual, tmp_path)
    assert golden[output] == b'\x00\x00'
    report = json.loads((tmp_path/'comparison.json').read_text())
    assert report['status'] == 'FAIL' and report['outputs'][output]['first_byte_difference'] == 0


def test_eight_draft_modes_use_only_frozen_native_goldens(tmp_path, monkeypatch):
    """Orchestration test with explicit fake exporter/runner, no device fallback."""
    import importlib.util
    import numpy as np
    tool = module(monkeypatch)
    path = ROOT/'framework/custom_ops/draft_ac/test/run.py'
    spec = importlib.util.spec_from_file_location('draft_only_eight_runner', path)
    runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
    from qwen35_dflash.ascend310p import draft_ac_export as export, draft_gears
    import a2_capture
    import resolve
    monkeypatch.setitem(sys.modules, 'torchair', SimpleNamespace())
    # All original artifacts deliberately live outside the new evidence root.
    root = tmp_path/'new'; root.mkdir()
    old = tmp_path/'old'; old.mkdir()
    monkeypatch.setenv('AI_RUN_DIR', str(root))
    def save(name, payload):
        p = old/name; p.write_text(json.dumps(payload)); return p
    def rec(path): return dict(path=path.name, bytes=path.stat().st_size, sha256=runner.sha256(path))
    config = {'draft_quantization': 'w8a16', 'draft_quant_matmul': 'weight_quant'}
    om16 = save('c16.om', {'frozen': 16}); om64 = save('c64.om', {'frozen': 64})
    native = dict(name='draft', metadata={'draft_quantization': 'w8a16', 'incremental_contract': {'complete': True}},
                  om=rec(om16), static_gear_oms=[{'om': rec(om16)}, {'om': rec(om64)}])
    air = save('air.json', dict(factory_config=config, graphs=[native]))
    dep = save('deployment.json', dict(status='PASS', graphs=[native], air_manifest=rec(air)))
    capture = save('head.json', {'weight': {'file': {'sha256': 'bound-head'}}})
    c16 = save('capture16.json', {}); c64 = save('capture64.json', {})
    r16 = save('replay16.json', {}); r64 = save('replay64.json', {})
    args = SimpleNamespace(factory_config=save('factory.json', config), native_deployment=dep,
        a16_bundle=c16, a64_bundle=c64, replay16=r16, replay64=r64, head_capture=capture,
        device_id=0, atc='fake-only', draft_only=True)
    (root/'draft-interface-device.json').write_text('{"target_interface":"BLOCKED"}')
    monkeypatch.setattr(resolve, 'validate_offline_nz', lambda *a: None)  # Separately covered by payload/identity tests.
    monkeypatch.setattr(draft_gears, 'verify_om_files', lambda *a: None)
    monkeypatch.setattr(draft_gears, 'om_records', lambda g: [g['om'], g['static_gear_oms'][1]['om']])
    names = ['features', 'start_position', 'valid_rows', 'anchor', 'proposal_count'] + [f'd{i}_{k}' for i in range(5) for k in ('key', 'value')]
    outputs = ['draft_top1'] + names[5:]
    graph = SimpleNamespace(config=None, head=SimpleNamespace(weight='bound-head'))
    source = SimpleNamespace(model=graph, input_names=tuple(names), metadata=native['metadata'])
    monkeypatch.setattr(tool, 'load_draft', lambda cfg, n: source)
    projections = [SimpleNamespace(qweight=f'q{i}', scales=torch.ones(1,1)) for i in range(26)]
    parents = [SimpleNamespace(linear=p) for p in projections]
    monkeypatch.setattr(export, 'projection_modules', lambda g: [(f'p{i}', p, 'linear') for i,p in enumerate(parents)])
    monkeypatch.setattr(export, 'tensor_hash', lambda value: value if isinstance(value, str) else 'scale')
    cases = [dict(name=f'p{i}', files={'q_nk.bin': {'sha256': f'q{i}'}, 's_gn.bin': {'sha256': 'scale'}}) for i in range(26)]
    bundles = {gear: dict(source={'feature_layers': [1,5]}, cases=cases) for gear in (16,64)}
    def frozen(path, config, layers):
        gear = 16 if path == r16 else 64
        return names, [np.zeros(1, dtype='float16') for _ in names], None, dict(
            context_rows=gear, draft_om_sha256=rec(om16)['sha256'], draft_static64_om_sha256=rec(om64)['sha256'])
    monkeypatch.setattr(a2_capture, 'frozen_inputs', frozen)
    monkeypatch.setattr(torch, 'from_numpy', lambda array: SimpleNamespace(to=lambda device: array))
    monkeypatch.setattr(torch, 'npu', SimpleNamespace(empty_cache=lambda: None), raising=False)
    abi = dict(inputs=[], outputs=[dict(name=n) for n in outputs])
    monkeypatch.setattr(export, 'integration_spec', lambda s,m,i,g: SimpleNamespace(metadata={'tensor_abi':abi}))
    def fake_export(spec, folder, atc, **kwargs):
        assert kwargs == {'draft_only': True}
        return folder/'candidate.om', {}
    monkeypatch.setattr(runner, 'export_one', fake_export)
    seen = []
    def execute(binary, om, abi, inputs, folder, device):
        folder.mkdir()
        seen.append((om, folder))
        return {'timing': {'execute_sync': {'median_ms': 1, 'p95_ms': 2}}}, {n: b'\x00\x00' for n in outputs}
    monkeypatch.setattr(runner, 'execute', execute)
    original = {p: p.read_bytes() for p in old.iterdir()}
    results = runner.full_draft(args, root, None, bundles)
    assert [(r['gear'],r['mode']) for r in results] == [(g,m) for g in (16,64) for m in ('off','a','c','ac')]
    native_calls = [p for p,f in seen if f.name == 'native']
    assert native_calls == [om16]*4 + [om64]*4
    assert original == {p: p.read_bytes() for p in old.iterdir()}
    report = json.loads((root/'draft-suite.json').read_text())
    assert report['target_interface'] == 'BLOCKED' and report['end_to_end'] == 'NOT_RUN'
    for folder in root.glob('draft-*-c*'):
        assert len(json.loads((folder/'weight-bindings.json').read_text())['projections']) == 26
        assert len(json.loads((folder/'comparison.json').read_text())['outputs']) == 11
