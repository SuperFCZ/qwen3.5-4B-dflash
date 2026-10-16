"""Real 319f51d source fixtures in fresh CPU interpreters, never NPU evidence."""
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import pytest

ROOT=Path(__file__).resolve().parents[1]
TOOLS=ROOT/'framework/custom_ops/draft_ac/test'
REV='319f51d0c51352c7f27f739e845ba93101f1ce8f'


@pytest.fixture(scope='module')
def history(tmp_path_factory):
    base=tmp_path_factory.mktemp('native-319f51d')
    # Test data only, not a new server checkout. No checkout/setup scripts run.
    data=subprocess.check_output(['git','archive',REV],cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        for member in archive.getmembers():
            assert not member.name.startswith('/') and '..' not in Path(member.name).parts
        archive.extractall(base)
    return base


def request(tmp_path,base):
    output=tmp_path/'evidence';output.mkdir()
    identity=dict(path=str(base/'SOURCE_LOCK.json'),sha256='7872c0f687aca12421786ed9508b8416a987ec38d5d8bc116dee39861c119395',verified_file_count=205)
    graph=dict(name='draft',metadata={'quant_source_lock':identity})
    air=tmp_path/'native-air.json';air.write_text(json.dumps(dict(status='PASS',graphs=[graph])))
    dep=tmp_path/'deployment.json';dep.write_text(json.dumps(dict(status='PASS',graphs=[graph],air_manifest=dict(path=air.name,sha256=hashlib.sha256(air.read_bytes()).hexdigest()))))
    state=dict(args=dict(draft_only=True,native_base=str(base),native_deployment=str(dep),output_dir=str(output)))
    path=tmp_path/'state.json';path.write_text(json.dumps(state))
    return path


def child(state,code,*,extension=ROOT):
    prelude=f'''
import json, sys
from pathlib import Path
sys.path.insert(0, {str(TOOLS)!r})
from native_base import activate, extension_inventory
state=json.loads(Path({str(state)!r}).read_text())
extension=Path({str(extension)!r})
'''
    result=subprocess.run([sys.executable,'-B','-c',prelude+code],cwd=ROOT,text=True,capture_output=True,
        env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','TORCH_DEVICE_BACKEND_AUTOLOAD':'0'})
    assert result.returncode==0, result.stdout+'\n'+result.stderr
    return result


def test_actual_historical_factory_and_extensions_are_separate(history,tmp_path):
    state=request(tmp_path,history)
    child(state,'''
boundary, report=activate(state,extension)
from native_base import check_extension_compatibility
compat=check_extension_compatibility({'draft_weight_prepack_manifest':'not-loaded-in-interface-check'})
assert compat['status']=='PASS' and 'output_dir' not in compat['native_prepack_signature']
identity=boundary.before_factory({'metadata':{'quant_source_lock':report['identity']}})
assert identity['verified_file_count']==205
modules=boundary.verify_loaded()
for name in ('quant_factory','incremental','draft_constants','exporter','weight_prepack'):
    assert modules['qwen35_dflash.ascend310p.'+name]['path'].startswith(str(boundary.base))
for name, record in modules.items():
    if record['kind']=='extension': assert record['path'].startswith(str(extension))
from qwen35_dflash.ascend310p import quant_factory
assert quant_factory.create_quant_incremental_graphs.prepare_export.__module__==quant_factory.__name__
try: (boundary.base/'forbidden-write').write_text('must not happen')
except PermissionError: pass
else: raise AssertionError('historical writes must be rejected')
assert not list(boundary.base.rglob('__pycache__'))
''')


@pytest.mark.parametrize('damage',['preloaded','mixed_module','identity','extension_changed','lock_changed'])
def test_boundary_rejects_identity_and_import_mix(history,tmp_path,damage):
    state=request(tmp_path,history)
    setup=''
    if damage=='preloaded':
        setup="import qwen35_dflash.ascend310p.quant_factory\n"
        action='activate(state,extension)'
    elif damage=='mixed_module':
        setup="boundary,report=activate(state,extension)\nfrom types import SimpleNamespace\nsys.modules['qwen35_dflash.ascend310p.quant_factory']=SimpleNamespace(__file__=str(extension/'framework/python/qwen35_dflash/ascend310p/quant_factory.py'))\n"
        action='boundary.verify_loaded()'
    elif damage=='identity':
        p=tmp_path/'native-air.json';air=json.loads(p.read_text());air['graphs'][0]['metadata']['quant_source_lock']['verified_file_count']=201;p.write_text(json.dumps(air))
        dep=tmp_path/'deployment.json';d=json.loads(dep.read_text());d['graphs']=air['graphs'];d['air_manifest']['sha256']=hashlib.sha256(p.read_bytes()).hexdigest();dep.write_text(json.dumps(d))
        action='activate(state,extension)'
    elif damage=='extension_changed':
        setup="""from native_base import bound_native
base,identity,_=bound_native(state)
expected=dict(status='PASS',native_base=str(base),identity=identity,extension_root=str(extension),extensions=extension_inventory(extension))
expected['extensions']['framework/python/qwen35_dflash/ascend310p/draft_ac_export.py']='0'*64
"""
        action='activate(state,extension,expected=expected)'
    else:
        # A separate minimal bad-root fixture: never modify the shared historical data.
        bad=tmp_path/'bad-base';bad.mkdir();(bad/'SOURCE_LOCK.json').write_bytes((history/'SOURCE_LOCK.json').read_bytes()+b' ')
        state.unlink(); import shutil
        # Recreate the request in a second directory bound to the bad root.
        second=tmp_path/'request2';second.mkdir();state=request(second,bad)
        action='activate(state,extension)'
    child(state,setup+f'''
try: {action}
except (ValueError,RuntimeError,ImportError): pass
else: raise AssertionError('boundary accepted {damage}')
''')


def test_historical_prepack_fileconstant_path_cpu(history,tmp_path):
    state=request(tmp_path,history)
    child(state,f'''
boundary,report=activate(state,extension)
sys.path.insert(0,{str(ROOT/'tests')!r})
import pytest
from test_draft_ac_fileconstant_descriptors import test_actual_air_save_hook_repairs_sparse_fileconstant_before_strict_prepack
with pytest.MonkeyPatch.context() as monkeypatch:
    test_actual_air_save_hook_repairs_sparse_fileconstant_before_strict_prepack(monkeypatch,Path(state['args']['output_dir']),False)
boundary.verify_loaded()
''')


def test_second_process_requires_same_native_and_extension_snapshot(history,tmp_path):
    state=request(tmp_path,history); snapshot=tmp_path/'snapshot.json'
    child(state,f'''
boundary,report=activate(state,extension)
report['status']='PASS'
Path({str(snapshot)!r}).write_text(json.dumps(report))
''')
    child(state,f'''
expected=json.loads(Path({str(snapshot)!r}).read_text())
boundary,report=activate(state,extension,expected=expected)
assert report['extensions']==expected['extensions']
assert report['native_python_sources']==expected['native_python_sources']
''')
    child(state,f'''
expected=json.loads(Path({str(snapshot)!r}).read_text())
state['args']['atc']='not-approved-by-preflight'
try: activate(state,extension,expected=expected)
except ValueError: pass
else: raise AssertionError('device accepted changed export arguments')
''')


def test_extension_actual_file_mutation_is_detected(history,tmp_path):
    import shutil
    extension=tmp_path/'extension'; directory=extension/'framework/python/qwen35_dflash/ascend310p';directory.mkdir(parents=True)
    for name in ('draft_ac_export','draft_ac_constants','draft_ac_payload','draft_ac_export_pipeline','draft_ac_runtime_input_export','draft_ac_weight_prepack'):
        shutil.copyfile(ROOT/f'framework/python/qwen35_dflash/ascend310p/{name}.py',directory/f'{name}.py')
    state=request(tmp_path,history)
    child(state,'''
boundary,report=activate(state,extension)
p=extension/'framework/python/qwen35_dflash/ascend310p/draft_ac_export.py'
p.write_text(p.read_text()+chr(10)+'# changed after selection'+chr(10))
try: boundary.verify_loaded()
except RuntimeError as error: assert 'Extension files changed' in str(error)
else: raise AssertionError('extension drift accepted')
''',extension=extension)


def test_wrapper_rejects_bad_history_before_resolve_or_opp(tmp_path):
    bad=tmp_path/'bad';bad.mkdir();(bad/'SOURCE_LOCK.json').write_text('{}')
    state=request(tmp_path,bad);data=json.loads(state.read_text())
    request_path=Path(data['args']['output_dir'])/'preflight-input.json';request_path.write_text(json.dumps(data))
    result=subprocess.run([sys.executable,'-B',str(TOOLS/'native_worker.py'),'preflight',str(request_path)],
        cwd=ROOT,text=True,capture_output=True)
    assert result.returncode==1
    output=request_path.parent
    assert json.loads((output/'native-base-preflight.json').read_text())['status']=='FAIL'
    assert not (output/'joint').exists() and not (output/'preflight-result.json').exists()


def test_server_shell_never_creates_evidence_inside_native_base(tmp_path):
    base=tmp_path/'native';base.mkdir();output=base/'new'/'evidence'
    result=subprocess.run(['bash',str(ROOT/'framework/custom_ops/draft_ac/run_server.sh'),
        '--draft-only','--native-base',str(base),'--preflight-only'],cwd=ROOT,text=True,capture_output=True,
        env={**os.environ,'AC_OUTPUT_DIR':str(output),'MODEL_PYTHON':sys.executable})
    assert result.returncode!=0 and 'outside read-only Native Base' in result.stderr
    assert list(base.iterdir())==[]


def test_module_object_cannot_masquerade_with_historical_filename(history,tmp_path):
    state=request(tmp_path,history)
    child(state,'''
boundary,report=activate(state,extension)
from qwen35_dflash.ascend310p import quant_factory
from types import SimpleNamespace
sys.modules[quant_factory.__name__]=SimpleNamespace(__file__=quant_factory.__file__)
try: boundary.verify_loaded()
except RuntimeError as error: assert 'mixed module origin' in str(error)
else: raise AssertionError('forged module object accepted')
''')


def test_coordinator_rechecks_accepted_boundary_before_opp(history,tmp_path):
    state=request(tmp_path,history)
    child(state,'''
from native_base import validate_preflight_boundary,digest
boundary,report=activate(state,extension)
root=Path(state['args']['output_dir'])
path=root/'preflight-input.json';path.write_text(json.dumps(state))
report.update(status='PASS',request_sha256=digest(path))
(root/'native-base-preflight.json').write_text(json.dumps(report))
record=validate_preflight_boundary(root,state,accepted_args=state['args'])
assert record['sha256']==digest(root/'native-base-preflight.json')
changed=dict(state['args'],atc='different')
try: validate_preflight_boundary(root,state,accepted_args=changed)
except ValueError: pass
else: raise AssertionError('changed accepted args passed coordinator gate')
assert not (root/'joint').exists()
''')


def test_original_receiver_fallback_is_manifest_bound_and_cannot_shadow_base(history,tmp_path):
    state=request(tmp_path,history)
    child(state,'''
boundary,report=activate(state,extension)
from qwen35_dflash.ascend310p.input_manifest import _file_group
from qwen35_dflash.ascend310p.quant_factory import QUANT_BASE_REVISION
from native_base import digest,check_extension_compatibility
root=Path(state['args']['output_dir'])/'receiver';root.mkdir()
wrapper=root/'export_model_wrapper_qwen3_5.py';wrapper.write_text("MARKER='bound-receiver'"+chr(10))
(root/'internal_dflash_bridge.py').write_text("raise AssertionError('receiver must not shadow Native')"+chr(10))
group=_file_group((wrapper,))
manifest=root.parent/'input-manifest.json'
manifest.write_text(json.dumps(dict(schema_version=1,artifact_kind='qwen35-quant-air-om-input-manifest',status='PASS',quant_branch_base_revision=QUANT_BASE_REVISION,groups={'receiver_wrapper':group})))
native={'metadata':dict(quant_source_lock=report['identity'],quant_input_manifest_sha256=digest(manifest),target_input_identity={'receiver_wrapper':group['sha256']},receiver_models_dir=str(root))}
config=dict(input_manifest=str(manifest),receiver_models_dir=str(root),draft_weight_prepack_manifest='not-loaded')
compat=check_extension_compatibility(config,native)
assert compat['receiver']['root']==str(root)
import models.export_model_wrapper_qwen3_5 as receiver
import models.internal_dflash_bridge as bridge
assert receiver.MARKER=='bound-receiver'
assert Path(bridge.__file__).is_relative_to(boundary.base)
assert boundary.verify_loaded()[receiver.__name__]['kind']=='receiver'
assert boundary.before_factory(native,require_receiver=True)==report['identity']
''')
