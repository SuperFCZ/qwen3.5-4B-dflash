"""Source identity and isolated extension tests. No server/NPU claim."""
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from test_draft_ac_integration import ac_tools, ROOT


def tool(monkeypatch):
    ac_tools(monkeypatch)
    return importlib.import_module('source_lock')


def locked_tree(tmp_path):
    entries = []
    for i in range(10):
        path = tmp_path/f'file{i}.py'; path.write_text(f'value = {i}\n')
        entries.append(dict(source_file=path.name, source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    # Preserve duplicate declaration semantics of the original factory verifier.
    payload = {'files': entries, 'repeat': entries[0]}
    path = tmp_path/'SOURCE_LOCK.json'; path.write_text(json.dumps(payload))
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), verified_file_count=11)


@pytest.mark.parametrize('damage', [None, 'tamper', 'missing', 'symlink', 'escape', 'lock_hash', 'count', 'historical_path', 'missing_identity'])
def test_all_source_pairs_and_complete_historical_identity(tmp_path, monkeypatch, damage):
    module = tool(monkeypatch); historical = locked_tree(tmp_path)
    if damage == 'tamper': (tmp_path/'file0.py').write_text('tampered')
    elif damage == 'missing': (tmp_path/'file1.py').unlink()
    elif damage == 'symlink':
        (tmp_path/'file1.py').unlink(); (tmp_path/'file1.py').symlink_to(tmp_path/'file2.py')
    elif damage == 'escape':
        p = tmp_path/'SOURCE_LOCK.json'; data=json.loads(p.read_text());data['files'][1]['source_file']='../escape.py';p.write_text(json.dumps(data))
    elif damage == 'lock_hash': historical['sha256'] = '0'*64
    elif damage == 'count': historical['verified_file_count'] = 10
    elif damage == 'historical_path': historical['path'] = '/different-checkout/SOURCE_LOCK.json'
    elif damage == 'missing_identity': historical = None
    report = module.audit_source_lock(tmp_path, historical)
    assert len(report['files']) == report['declaration_count'] == 11
    assert report['status'] == ('PASS' if damage is None else 'FAIL')
    if damage == 'tamper': assert report['failed_declaration_count'] == 2
    if damage in ('lock_hash', 'count', 'historical_path', 'missing_identity'):
        assert report['failed_declaration_count'] == 0 and report['identity_diff']


def test_collects_all_failures_not_first(tmp_path, monkeypatch):
    module = tool(monkeypatch); historical=locked_tree(tmp_path)
    (tmp_path/'file0.py').write_text('changed')
    (tmp_path/'file9.py').unlink()
    report = module.audit_source_lock(tmp_path, historical)
    assert report['failed_declaration_count'] == 3
    assert {f['error'] for f in report['files'] if f['status']=='FAIL'} == {'HASH_MISMATCH','MISSING_FILE'}


def test_real_repository_matches_original_factory_and_archived_recovery(monkeypatch, tmp_path):
    module = tool(monkeypatch)
    from qwen35_dflash.ascend310p.quant_factory import _verify_quant_source_lock
    identity = _verify_quant_source_lock()  # Original implementation, no patching.
    before = json.loads((ROOT/'framework/custom_ops/draft_ac/source_lock_recovery/audit-before.json').read_text())
    assert identity['sha256'] == before['lock_sha256']
    assert identity['verified_file_count'] == before['declaration_count'] == 201
    report = module.require_source_lock(dict(metadata={'quant_source_lock': identity}), tmp_path/'source.json')
    assert report['status'] == report['factory_verifier'] == 'PASS'
    assert report['unique_file_count'] == 191 and report['failed_declaration_count'] == 0
    assert report['diagnostic_extensions'] and not report['identity_diff']
    patch = ROOT/'framework/custom_ops/draft_ac/source_lock_recovery/changes-through-5e5b0d7.patch'
    assert hashlib.sha256(patch.read_bytes()).hexdigest() == before['patch_sha256']


def test_source_failure_precedes_torch_npu_and_converter_preparation(tmp_path, monkeypatch):
    module = tool(monkeypatch)
    draft = importlib.import_module('draft_only')
    native = dict(metadata={'quant_source_lock': {'sha256': 'wrong'}})
    air = tmp_path/'air.json'; air.write_text('{}')
    monkeypatch.setattr(draft, 'native_history', lambda args: ({}, native, air))
    monkeypatch.setattr(draft, 'prepare_draft_interfaces', lambda *a: pytest.fail('must fail before converter registration'))
    def forbidden_factory_check(): pytest.fail('historical mismatch must stop before the factory verifier')
    from qwen35_dflash.ascend310p import quant_factory
    monkeypatch.setattr(quant_factory, '_verify_quant_source_lock', forbidden_factory_check)
    with pytest.raises(RuntimeError, match='interface preflight blocked'):
        draft.require_draft_interfaces(None, tmp_path/'draft-interface-preflight.json')
    report = json.loads((tmp_path/'source-lock-preflight.json').read_text())
    assert report['status'] == 'FAIL' and report['identity_diff']
    assert len(report['files']) == 201


def test_preflight_worker_stops_before_joint_opp_on_source_failure(tmp_path, monkeypatch):
    tool(monkeypatch)
    worker = importlib.import_module('preflight_worker'); draft = importlib.import_module('draft_only')
    resolver = importlib.import_module('resolve')
    from process_boundary import encode_args
    args = SimpleNamespace(output_dir=tmp_path, draft_only=True, full_draft=False)
    request = tmp_path/'preflight-input.json'; request.write_text(json.dumps(encode_args(args)))
    monkeypatch.setattr(resolver, 'resolve', lambda *a: None)
    air = tmp_path/'old-air.json'; air.write_text('{}')
    monkeypatch.setattr(draft, 'native_history', lambda a: ({}, {'metadata': {'quant_source_lock': {}}}, air))
    assert worker.run(request) == 1
    assert not (tmp_path/'joint').exists() and not (tmp_path/'worker-input.json').exists()
    result = json.loads((tmp_path/'preflight-result.json').read_text())
    assert result['status'] == 'FAIL'
    assert json.loads((tmp_path/'source-lock-preflight.json').read_text())['status'] == 'FAIL'


def test_diagnostic_import_does_not_replace_locked_exporter_or_registry(monkeypatch):
    from qwen35_dflash.ascend310p import exporter, runtime_input_export, custom_op_export, compiler
    original = (exporter.canonical_runtime_input_abi, exporter.prepare_custom_op_export,
                compiler.compile_air_bundle, custom_op_export.prepare_custom_op_export)
    from qwen35_dflash.ascend310p import draft_ac_export_pipeline as extension
    assert extension.canonical_runtime_input_abi is not original[0]
    assert exporter.canonical_runtime_input_abi is runtime_input_export.canonical_runtime_input_abi
    assert original == (exporter.canonical_runtime_input_abi, exporter.prepare_custom_op_export,
                        compiler.compile_air_bundle, custom_op_export.prepare_custom_op_export)
    from qwen35_dflash.ascend310p.contracts import CustomOpExportSpec
    calls=[]
    monkeypatch.setattr(extension, '_prepare_ac', lambda spec, air: calls.append(spec.torch_op))
    extension.prepare_custom_op_export(CustomOpExportSpec('dflash_ac::head_top1','DFlashDraftLmHeadTop1'), None)
    assert calls == ['dflash_ac::head_top1']
    with pytest.raises(NotImplementedError, match='no exact export adapter'):
        custom_op_export.prepare_custom_op_export(CustomOpExportSpec('dflash_ac::head_top1','DFlashDraftLmHeadTop1'), None)


def test_extension_compiler_cannot_skip_exact_ac_audit(tmp_path, monkeypatch):
    from qwen35_dflash.ascend310p import draft_ac_export_pipeline as extension, compiler
    path=tmp_path/'air.json'
    path.write_text(json.dumps({'graphs':[{'role':'diagnostic','metadata':{'draft_ac':{'policy':'bad'}}}]}))
    monkeypatch.setattr(compiler,'compile_air_bundle',lambda *a,**k:pytest.fail('ATC must not run'))
    with pytest.raises(ValueError, match='policy'): extension.compile_air_bundle(path)
