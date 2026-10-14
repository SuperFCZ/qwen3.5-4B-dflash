"""File materialization/order tests; CPU bytes only, never NPU correctness."""
import json
from pathlib import Path
from types import ModuleType,SimpleNamespace
import sys
import pytest
import torch
from test_draft_ac_integration import fixture
from test_weight_quant_layout import graph_type
from qwen35_dflash.ascend310p.draft_ac_payload import CHUNK_BYTES,PayloadError,tensor_digest,verify_payload
from qwen35_dflash.ascend310p.draft_ac_constants import _bound
from qwen35_dflash.ascend310p.runtime_input_export import canonical_runtime_input_abi


@pytest.mark.parametrize('style',['absolute','relative','directory_prefixed'])
@pytest.mark.parametrize('damage',[None,'missing','size','hash','escape','offset','directory','changed_tensor'])
def test_payload_paths_and_failure_reasons(tmp_path,monkeypatch,style,damage):
    monkeypatch.chdir(tmp_path); root=tmp_path/'graph'; root.mkdir()
    g,values,_=fixture();n=g.op[2];n.type='FileConstant';t=values['s:0']
    file=root/'scale';file.write_bytes(t.numpy().tobytes());expected=tensor_digest(t)
    directory=Path('graph') if style=='directory_prefixed' else root
    raw=str(file) if style=='absolute' else 'graph/scale' if style=='directory_prefixed' else 'scale'
    n.attr['file_path'].s=raw.encode();n.attr['offset'].i=0
    codes=dict(missing='MISSING_FILE',size='FILE_SIZE_MISMATCH',hash='FILE_HASH_MISMATCH',escape='PATH_ESCAPE',
               offset='UNSUPPORTED_NONZERO_OFFSET',directory='NOT_REGULAR_FILE',changed_tensor='IMMUTABLE_TENSOR_CHANGED')
    if damage=='missing': file.unlink()
    elif damage=='size': file.write_bytes(b'0')
    elif damage=='hash': file.write_bytes(b'0'*t.numel()*2)
    elif damage=='escape': n.attr['file_path'].s=b'../outside'
    elif damage=='offset': n.attr['offset'].i=32
    elif damage=='directory': file.unlink();file.mkdir()
    elif damage=='changed_tensor': t[0,0]+=1
    if damage:
        with pytest.raises(PayloadError) as caught: verify_payload(n,t,directory,expected)
        assert caught.value.diagnostic['reason_code']==codes[damage]
        assert 'attributes' in caught.value.diagnostic and 'output_descriptors' in caught.value.diagnostic
    else:
        result=verify_payload(n,t,directory,expected)
        assert result['sha256']==expected and result['bytes']==t.numel()*2 and result['path']==str(file)


@pytest.mark.parametrize('damage',[None,'size','hash'])
def test_const_still_checks_original_bytes(tmp_path,damage):
    g,values,_=fixture();node=g.op[1];tensor=values['w:0']
    if damage=='size': node.attr['value'].t.data=b'x'
    elif damage=='hash': node.attr['value'].t.data=b'x'*tensor.numel()
    if damage:
        with pytest.raises(PayloadError,match='CONST_'): verify_payload(node,tensor,tmp_path)
    else: assert verify_payload(node,tensor,tmp_path)['sha256']==tensor_digest(tensor)


@pytest.mark.parametrize('relative',[False,True])
@pytest.mark.parametrize('external,damage',[(False,d) for d in (None,'post_export_hash','tensor_change')]+
    [(True,d) for d in (None,'skip_writer','file_hash','post_export_hash','tensor_change','writer_inputs','writer_error','binding_change')])
def test_hook_checks_after_materialization_before_dump_and_restores(tmp_path,monkeypatch,external,damage,relative,eager=False):
    root=tmp_path/'graph';root.mkdir();monkeypatch.setenv('AI_RUN_DIR',str(tmp_path))
    g,values,config=fixture();x=torch.zeros(16,256,dtype=torch.float16);q=values['w:0'];s=values['s:0']
    inputs=[x,q,s]; names={id(q):'weight',id(s):'scale'}
    for i,node in enumerate(g.op[:3]):
        node.type='Data';node.attr.clear();node.attr['index'].i=i
    calls=[];states=[]
    def convert(inputs,graph,path,weights):
        for i,t in enumerate(inputs):
            if id(t) not in weights: continue
            n=graph.op[i];n.attr.clear();n.type='FileConstant' if external else 'Const'
            if external:
                n.attr['file_path'].s=(weights[id(t)] if relative else str(Path(path)/weights[id(t)])).encode();n.attr['offset'].i=0
            else:
                n.attr['value'].t.desc.CopyFrom(n.output_desc[0]);n.attr['value'].t.data=t.numpy().tobytes()
        if external and eager: sdk._save_weight2file(inputs,path,weights,2)
        return external,2
    def save(inputs,path,weights,count):
        calls.append('save')
        assert not any(n.name=='custom_ac_nz' for n in g.op) # Nothing verified/rewritten before SDK's write.
        if damage=='writer_error': raise OSError('SDK save failed')
        for t in inputs:
            if id(t) in weights: (Path(path)/weights[id(t)]).write_bytes(t.numpy().tobytes())
        if damage=='file_hash': (Path(path)/'scale').write_bytes(b'0'*s.numel()*2)
    sdk=SimpleNamespace(_convert_data_to_const=convert,_save_weight2file=save)
    torchair=ModuleType('torchair');monkeypatch.setitem(sys.modules,'torchair',torchair)
    monkeypatch.setitem(sys.modules,'torchair._utils.export_utils',sdk)
    def export():
        with canonical_runtime_input_abi(torchair,public_inputs=[x],public_names=['x'],draft_ac=config) as audit:
            sdk._convert_data_to_const(inputs,g,str(root),names)
            states.append(audit['draft_ac']['status'])
            if damage=='tensor_change': s[0,0]+=1
            if external and not eager and damage!='skip_writer':
                sdk._save_weight2file(inputs[::-1] if damage=='writer_inputs' else inputs,str(root),names,2)
                assert audit['draft_ac']['status']=='PASS' # Before SDK dump/GE save.
                assert Path(next(n for n in g.op if n.name=='s').attr['file_path'].s.decode()).is_absolute()
            if damage=='binding_change': next(n for n in g.op if n.name=='s').attr['file_path'].s=b'elsewhere'
            if damage=='post_export_hash':
                if external: (root/'scale').write_bytes(b'0'*s.numel()*2)
                else: next(n for n in g.op if n.name=='s').attr['value'].t.data=b'0'*s.numel()*2
        return audit
    if damage:
        with pytest.raises((ValueError,OSError)): export()
        report=json.loads((root/'ac-payloads.json').read_text());assert report['status']=='FAIL'
        assert json.loads((root/'ac-constants.json').read_text())['status']=='FAIL'
        if damage=='skip_writer': assert report['failure']['reason_code']=='NOT_MATERIALIZED'
    else:
        audit=export();assert audit['status']=='PASS' and audit['draft_ac']['status']=='PASS'
        assert states==['PENDING_MATERIALIZATION' if external and not eager else 'PASS']
        assert calls==(['save'] if external else [])
        assert json.loads((root/'ac-payloads.json').read_text())['events'][-1]['stage']=='after_export'
        if external: assert 'source' in json.loads((root/'ac-torchair-fileconstant-api.json').read_text())['_save_weight2file']
    assert sdk._convert_data_to_const is convert and sdk._save_weight2file is save


def test_full_head_payload_streams_exact_fp16_bytes(tmp_path,monkeypatch):
    # Real ABI (1,271,398,400 bytes), sparse disk backing; no full tensor clone/to-bytes.
    n,k=248320,2560;file=tmp_path/'head'
    with file.open('wb') as f: f.truncate(n*k*2)
    with file.open('r+b') as f:
        f.write(bytes.fromhex('007e007c00800100'));f.seek(-2,2);f.write(bytes.fromhex('ff7b'))
    weight=torch.from_file(str(file),shared=False,size=n*k,dtype=torch.float16).view(n,k)
    g=graph_type()();node=g.op.add(name='head',type='FileConstant');node.output_desc.add(name='weight',dtype='DT_UNDEFINED')
    node.attr['file_path'].s=str(file).encode()
    _,bound,desc=_bound({'head':node},'head:0',{'head:0':weight},torch.float16,[n,k])
    assert list(desc.shape.dim)==[n,k]
    original=torch.Tensor.cpu; largest=[]
    def bounded_cpu(self,*a,**kw):
        assert self.numel()*self.element_size()<=CHUNK_BYTES
        largest.append(self.numel()*self.element_size());return original(self,*a,**kw)
    monkeypatch.setattr(torch.Tensor,'cpu',bounded_cpu)
    from qwen35_dflash.ascend310p.draft_ac_export import C,GE,POLICY
    from qwen35_dflash.ascend310p.draft_ac_constants import normalize_ac_constants
    monkeypatch.setenv('AI_RUN_DIR',str(tmp_path))
    hidden=g.op.add(name='hidden',type='Data');hidden.output_desc.add(name='hidden',dtype='DT_FLOAT16').shape.dim.extend([15,k])
    op=g.op.add(name='head_op',type=GE[C],input=['hidden:0','head:0'])
    op.input_desc.add().CopyFrom(hidden.output_desc[0]);op.input_desc.add().CopyFrom(desc)
    audit=normalize_ac_constants(g,{'head:0':bound},dict(policy=POLICY,mode='c',a_count=0,c_count=1,audit_count=0),tmp_path)
    result=audit['nodes'][0]['head_weight']
    assert result['bytes']==n*k*2 and max(largest)==CHUNK_BYTES
    from qwen35_dflash.ascend310p.utils import sha256_file
    assert result['sha256']==sha256_file(file)


def test_fileconstant_symlink_cannot_escape_export_directory(tmp_path):
    root=tmp_path/'graph';root.mkdir();outside=tmp_path/'outside'
    g,values,_=fixture();tensor=values['s:0'];outside.write_bytes(tensor.numpy().tobytes())
    (root/'alias').symlink_to(outside)
    node=g.op[2];node.type='FileConstant';node.attr['file_path'].s=b'alias'
    with pytest.raises(PayloadError,match='PATH_ESCAPE'): verify_payload(node,tensor,root)
    assert outside.read_bytes()==tensor.numpy().tobytes()


def test_eager_file_materializer_uses_the_same_strict_hash_gate(tmp_path,monkeypatch):
    test_hook_checks_after_materialization_before_dump_and_restores(tmp_path,monkeypatch,True,None,True,eager=True)
