"""CPU tracing/selection tests only. No substitute for TorchAir/ATC/NPU gates."""
import ast
import copy
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest
import torch
from test_draft_ac_integration import ROOT,ac_tools

TOOLS=ROOT/'framework/custom_ops/draft_ac/test'


def load_entry(monkeypatch):
    ac_tools(monkeypatch)
    spec=importlib.util.spec_from_file_location('generator_test_run',TOOLS/'run.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('guard',['none','legacy','fixed'])
def test_dynamo_fullgraph_isolates_generator_in_write_guard(tmp_path,guard):
    legacy=subprocess.check_output(['git','show','a6b5498:framework/custom_ops/draft_ac/test/native_base.py'],cwd=ROOT,text=True)
    node=next(n for n in ast.parse(legacy).body if isinstance(n,ast.FunctionDef) and n.name=='protect_writes')
    body='\n'.join(legacy.splitlines()[node.lineno-1:node.end_lineno])
    program=f'''
import sys,os
from pathlib import Path
import torch
from contextlib import contextmanager
sys.path.insert(0,{str(TOOLS)!r})
root=Path({str(tmp_path)!r});protected=root/'native';protected.mkdir();(protected/'original').write_text('unchanged')
mode={guard!r}
if mode=='legacy':
    exec(compile({body!r},'a6b5498/native_base.py','exec'))
    protect_writes(protected)
elif mode=='fixed':
    from native_base import protect_writes
    protect_writes(protected)
class Probe(torch.nn.Module):
    def forward(self,x): return x.sin()+1
captured=[]
def backend(graph,inputs):
    captured.append(graph.code)
    return graph.forward
@contextmanager
def host_context():
    yield
try:
    with host_context():
        out=torch.compile(Probe(),backend=backend,fullgraph=True)(torch.ones(2))
except torch._dynamo.exc.Unsupported as error:
    assert mode=='legacy' and 'Attempt to trace generator' in str(error)
else:
    assert mode!='legacy' and len(captured)==1
    torch.testing.assert_close(out,torch.ones(2).sin()+1,rtol=0,atol=0)
if mode=='fixed':
    for action in (lambda:(protected/'new').write_text('bad'),lambda:os.mkdir(protected/'dir'),lambda:os.remove(protected/'original'),lambda:os.rename(protected/'original',root/'moved')):
        try: action()
        except PermissionError: pass
        else: raise AssertionError('write guard was weakened')
    assert (protected/'original').read_text()=='unchanged'
    (root/'allowed').write_text('allowed outside Native Base')
print('CPU_TRACE_ONLY',mode,torch.__version__)
'''
    result=subprocess.run([sys.executable,'-B','-c',program],cwd=ROOT,text=True,capture_output=True,
                          env={**os.environ,'TORCHDYNAMO_VERBOSE':'1','TORCH_DEVICE_BACKEND_AUTOLOAD':'0'})
    assert result.returncode==0,result.stdout+result.stderr


def test_exception_report_preserves_generator_location_and_chain(monkeypatch,tmp_path):
    ac_tools(monkeypatch)
    tool=importlib.import_module('generator_diagnostics')
    def generator():
        yield 1
        raise ValueError('inner')
    try:
        try: list(generator())
        except ValueError as error: raise RuntimeError('outer') from error
    except RuntimeError as error: tool.save_exception(error,tmp_path)
    report=json.loads((tmp_path/'exception.json').read_text())
    assert [e['type'] for e in report['exceptions']]==['RuntimeError','ValueError']
    frame=next(f for f in report['exceptions'][1]['frames'] if f['function']=='generator')
    assert frame['generator'] and frame['file']==__file__ and frame['line']>frame['first_line']
    assert 'ValueError: inner' in (tmp_path/'traceback.txt').read_text()


@pytest.mark.parametrize('air_only',[False,True])
def test_single_scope_stops_after_fc_and_preserves_native_comparison(tmp_path,monkeypatch,air_only):
    import numpy as np
    tool=load_entry(monkeypatch)
    import resolve
    from a2_common import sha256
    monkeypatch.setattr(resolve,'head_capture',lambda *a:pytest.fail('Head or other graphs must not run'))
    bundle_file=tmp_path/'capture.json';bundle_file.write_text('{}')
    for name,data in (('x.bin',np.zeros((16,256),dtype='<f2')),('q_nk.bin',np.zeros((64,256),dtype='i1')),('s_gn.bin',np.ones((2,64),dtype='<f2'))):
        data.tofile(tmp_path/name)
    om=tmp_path/'native.om';om.write_bytes(b'fixture')
    rec=lambda p:dict(path=p.name,bytes=p.stat().st_size,sha256=sha256(p))
    fc=dict(name='fc',files={n:rec(tmp_path/n) for n in ('x.bin','q_nk.bin','s_gn.bin')})
    bundle=dict(cases=[dict(name='must-not-run'),fc])
    native=tmp_path/'native.json';native.write_text(json.dumps(dict(status='PASS',bundle_sha256=sha256(bundle_file),cases=[dict(name='fc',status='PASS',om=rec(om))])))
    args=SimpleNamespace(device_id=0,a16_bundle=bundle_file,a16_native=native,atc='fake')
    monkeypatch.setattr(tool,'validate_layout',lambda *a:None)
    monkeypatch.setattr(tool,'case_shape',lambda *a:(16,256,64))
    monkeypatch.setattr(torch.Tensor,'to',lambda self,device:self if device=='npu:0' else pytest.fail('unexpected cast'))
    monkeypatch.setattr(torch,'npu',SimpleNamespace(empty_cache=lambda:None),raising=False)
    exports=[];executions=[];compares=[]
    def export(spec,folder,atc,**kwargs):
        exports.append((folder.name,kwargs));return folder/'candidate.om',{'air':'CPU fixture'}
    monkeypatch.setattr(tool,'export_one',export)
    def execute(runner,model,abi,paths,folder,device):
        executions.append((model,folder.name));return {},{'y':b'\x00\x00'}
    monkeypatch.setattr(tool,'execute',execute)
    monkeypatch.setattr(tool,'compare',lambda golden,actual,folder:compares.append((golden,actual)))
    cases,heads=tool.probes(args,tmp_path,None,{16:bundle},only_first=True,air_only=air_only)
    assert exports==[('a-c16-fc',{'air_only':True} if air_only else {})]
    assert not heads and len(cases)==1 and cases[0]['name']=='c16-fc'
    assert cases[0]['status']==('AIR_EXPORTED' if air_only else 'PASS')
    if air_only: assert not executions and not compares
    else:
        assert executions==[(om,'native'),(tmp_path/'a-c16-fc/candidate.om','custom')]
        assert len(compares)==1


def reuse_fixture(tool,tmp_path):
    source=tmp_path/'old';source.mkdir();(source/'joint/opp').mkdir(parents=True);(source/'runner').mkdir()
    args=dict(draft_only=True,native_base=str(tmp_path/'native'),output_dir=str(source),a_build_config=str(tmp_path/'a.json'),c_release=str(tmp_path/'c.json'))
    def save(path,value): path.write_text(json.dumps(value));return path
    def rec(path): return dict(path=str(path.relative_to(source)),bytes=path.stat().st_size,sha256=tool.sha256(path))
    request=save(source/'preflight-input.json',{'args':args})
    boundary=save(source/'native-base-preflight.json',dict(status='PASS',accepted_args=args,request_sha256=tool.sha256(request)))
    opp=dict(status='BUILT_NOT_NPU_VALIDATED',install=str(source/'joint/opp'),preflight={'status':'PASS'},a_config={},c_config={})
    opp_path=save(source/'joint/opp.json',opp)
    runner=source/'runner/ac_om_runner';runner.write_bytes(b'CPU fake runner')
    save(source/'suite.json',dict(status='FAIL',opp=rec(opp_path),runner_sha256=tool.sha256(runner)))
    save(source/'worker-input.json',dict(args=args,path_args=['native_base','output_dir','a_build_config','c_release'],native_boundary=rec(boundary),opp=opp))
    vendor=source/'joint/opp/vendors/customize'
    save(source/'device-process.json',dict(ASCEND_CUSTOM_OPP_PATH=str(vendor),LD_LIBRARY_PATH=str(vendor/'op_api/lib')))
    return source


@pytest.mark.parametrize('damage',[None,'runner','opp_record','arguments','environment','installed_opp'])
def test_reuse_requires_exact_installed_opp_and_runner(monkeypatch,tmp_path,damage):
    ac_tools(monkeypatch);tool=importlib.import_module('single_graph')
    source=reuse_fixture(tool,tmp_path)
    monkeypatch.setattr(tool,'preflight',lambda *a:dict(status='PASS' if damage!='installed_opp' else 'FAIL'))
    monkeypatch.setattr(tool,'validate_sources',lambda *a:({},{}))
    if damage=='runner': (source/'runner/ac_om_runner').write_bytes(b'changed')
    elif damage=='opp_record': (source/'joint/opp.json').write_text('{}')
    elif damage=='arguments':
        path=source/'worker-input.json';data=json.loads(path.read_text());data['args']['native_base']='changed';path.write_text(json.dumps(data))
    elif damage=='environment': (source/'device-process.json').write_text('{"ASCEND_CUSTOM_OPP_PATH":"/wrong"}')
    if damage:
        with pytest.raises((ValueError,KeyError)):tool.load_reuse(source)
    else:
        args,reuse=tool.load_reuse(source);tool.verify_reuse(reuse)
        assert reuse['runner']==str(source/'runner/ac_om_runner') and args.native_base==tmp_path/'native'
        (source/'runner/ac_om_runner').write_bytes(b'changed after verification')
        with pytest.raises(ValueError,match='changed'): tool.verify_reuse(reuse)


def test_single_coordinator_launches_fresh_preflight_then_one_graph_without_build(tmp_path,monkeypatch):
    ac_tools(monkeypatch);tool=importlib.import_module('single_graph')
    from process_boundary import decode_args
    source=reuse_fixture(tool,tmp_path)
    monkeypatch.setattr(tool,'preflight',lambda *a:dict(status='PASS'))
    monkeypatch.setattr(tool,'validate_sources',lambda *a:({},{}))
    import opp,native_base
    monkeypatch.setattr(opp,'build',lambda *a:pytest.fail('must reuse OPP; cannot rebuild'))
    output=tmp_path/'new';calls=[]
    def run(command,**kwargs):
        stage=command[-2];calls.append((stage,kwargs['env']))
        if stage=='preflight':
            request=json.loads((output/'preflight-input.json').read_text())
            (output/'preflight-result.json').write_text(json.dumps(request))
        else:
            assert stage=='single' and command[-3].endswith('native_worker.py')
            state=json.loads((output/'worker-input.json').read_text())
            assert state['args']['single_graph']=='a-c16-fc'
            assert kwargs['env']['ASCEND_CUSTOM_OPP_PATH']==state['reuse']['environment']['ASCEND_CUSTOM_OPP_PATH']
            (output/'suite.json').write_text(json.dumps({'status':'PASS','scope':'CPU worker mock'}))
        return subprocess.CompletedProcess(command,0)
    monkeypatch.setattr(tool.subprocess,'run',run)
    monkeypatch.setattr(tool,'validate_preflight_result',lambda request,result,root:(decode_args(result),{'records':{}}))
    monkeypatch.setattr(native_base,'validate_preflight_boundary',lambda *a,**k:{'path':'mock','sha256':'fixture'})
    assert tool.coordinator(SimpleNamespace(source_evidence=source,output_dir=output,air_only=False,trace_bytecode=True))==0
    assert [s for s,e in calls]==['preflight','single']
    assert all(e['TORCHDYNAMO_VERBOSE']=='1' and e['TORCH_LOGS']=='+dynamo,trace_source,trace_bytecode' for s,e in calls)
    assert not (output/'joint').exists() and not list(output.glob('draft-*-c*'))
