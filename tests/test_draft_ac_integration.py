"""CPU interface/immutable lowering tests only. GE, ATC and NPU are NOT_RUN."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest
import torch
from test_weight_quant_layout import graph_type
from qwen35_dflash.ascend310p.draft_ac_export import A,C,AUDIT,GE,POLICY,ensure_ops,linear_meta,head_meta,ACLinear,ACOps,prepare_export,tensor_hash
from qwen35_dflash.ascend310p.draft_ac_constants import normalize_ac_constants,validate_ac_graph
from qwen35_dflash.ascend310p.contracts import CustomOpExportSpec

ROOT=Path(__file__).resolve().parents[1]


def test_fake_only_no_cpu_numerical_fallback():
    ensure_ops()
    x=torch.empty(16,256,device='meta',dtype=torch.float16)
    w=torch.empty(64,256,device='meta',dtype=torch.int8); s=torch.empty(2,64,device='meta',dtype=torch.float16)
    assert torch.ops.dflash_ac.group_quant_linear(x,w,s).shape==(16,64)
    h=torch.empty(15,2560,device='meta',dtype=torch.float16); head=torch.empty(248320,2560,device='meta',dtype=torch.float16)
    ids,logits=torch.ops.dflash_ac.head_audit(h,head)
    assert ids.dtype==torch.int64 and ids.shape==(15,) and logits.shape==(15,248320)
    with pytest.raises(NotImplementedError): torch.ops.dflash_ac.group_quant_linear(torch.zeros(16,256,dtype=torch.float16),torch.zeros(64,256,dtype=torch.int8),torch.ones(2,64,dtype=torch.float16))
    for m in (0,16):
        with pytest.raises(ValueError): head_meta(torch.empty(m,2560,dtype=torch.float16),head)


def test_gn_is_transposed_not_reshaped_and_anchor_rows_unchanged():
    from models.dflash_v1.draft_quantization import GroupQuantLinear
    scale=torch.arange(128,dtype=torch.float16).reshape(64,2)
    original=GroupQuantLinear(torch.zeros(64,256,dtype=torch.int8),scale,bits=8,in_features=256,ops=None,matmul_backend='weight_quant')
    wrapper=ACLinear(original)
    assert wrapper.weight is original.qweight and wrapper.scale.is_contiguous()
    assert torch.equal(wrapper.scale,scale.t()) and not torch.equal(wrapper.scale,scale.reshape(2,64))
    source=SimpleNamespace(other='preserved'); proxy=ACOps(source)
    assert proxy.other=='preserved'
    with pytest.raises(ValueError,match='15 post-anchor'): proxy.top1(torch.empty(1,16,2560),None)
    with pytest.raises(ValueError,match='GN'): linear_meta(torch.empty(16,256,dtype=torch.float16),wrapper.weight,scale)


@pytest.mark.parametrize('op',[A,C,AUDIT])
def test_named_converter_ports_and_exact_output_arity(op):
    registry={}; seen=[]
    def custom_op(op_type,*,inputs,outputs,attrs):
        seen.append((op_type,inputs,outputs,attrs)); return ['out'+str(i) for i in range(len(outputs))]
    def register(target):
        def wrap(fn): registry[target]=fn; return fn
        return wrap
    session=prepare_export(CustomOpExportSpec(op,GE[op]),SimpleNamespace(ge=SimpleNamespace(custom_op=custom_op),register_fx_node_ge_converter=register))
    result=next(iter(registry.values()))(*range(3 if op==A else 2))
    assert session.converter_calls==1 and seen[0][0]==GE[op]
    assert list(seen[0][1])==(['x','w_nz','s'] if op==A else ['hidden','weight'])
    assert result==(['out0','out1'] if op==AUDIT else 'out0')


def fixture():
    g=graph_type()(); q=(torch.arange(64*256)%255-128).to(torch.int8).reshape(64,256)
    s=(torch.arange(128)%7+1).half().reshape(2,64)
    def node(name,kind,value):
        op=g.op.add(name=name,type=kind); d=op.output_desc.add(name='y',dtype='DT_INT8' if value.dtype==torch.int8 else 'DT_FLOAT16');d.shape.dim.extend(value.shape);d.layout='ND'
        if kind=='Const': op.attr['value'].t.desc.CopyFrom(d);op.attr['value'].t.data=value.numpy().tobytes()
        return op
    x=node('x','Data',torch.zeros(16,256,dtype=torch.float16));w=node('w','Const',q);scale=node('s','Const',s)
    op=g.op.add(name='custom',type=GE[A],input=['x:0','w:0','s:0'])
    for source in (x,w,scale): op.input_desc.add().CopyFrom(source.output_desc[0])
    g.op.add(name='out',type='NetOutput',input=['custom:0'])
    config=dict(policy=POLICY,mode='a',a_count=1,c_count=0,audit_count=0,projections=[dict(n=64,k=256,weight_sha256=tensor_hash(q),scale_gn_sha256=tensor_hash(s))])
    return g,{'w:0':q,'s:0':s},config


def test_real_nz_payload_and_gn_hashes(tmp_path,monkeypatch):
    monkeypatch.setenv('AI_RUN_DIR',str(tmp_path));folder=tmp_path/'graph';folder.mkdir()
    g,values,config=fixture(); before=values['w:0'].clone()
    report=normalize_ac_constants(g,values,config,folder)
    nz=next(n for n in g.op if n.name=='custom_ac_nz')
    assert list(nz.output_desc[0].shape.dim)==[8,4,16,32]
    raw=torch.frombuffer(bytearray(nz.attr['value'].t.data),dtype=torch.int8).reshape(8,4,16,32)
    assert torch.equal(raw.permute(1,2,0,3).reshape(64,256),before)
    assert report['nodes'][0]['roundtrip']=='BIT_EXACT' and all(n.name!='w' for n in g.op)
    assert torch.equal(values['w:0'],before)


@pytest.mark.parametrize('damage',['runtime','payload','dtype','shape','inventory','gn','count','file_escape'])
def test_immutable_audit_rejects_damage(tmp_path,monkeypatch,damage):
    monkeypatch.setenv('AI_RUN_DIR',str(tmp_path));folder=tmp_path/'graph';folder.mkdir()
    g,values,config=fixture(); w=g.op[1]
    if damage=='runtime': w.type='Data'
    elif damage=='payload': w.attr['value'].t.data=b'bad'
    elif damage=='dtype': w.output_desc[0].dtype=1
    elif damage=='shape': w.output_desc[0].shape.dim[:]=[256,64]
    elif damage=='inventory': config['projections'][0]['weight_sha256']='bad'
    elif damage=='gn': values['s:0']=values['s:0'].t()
    elif damage=='count': config['a_count']=26
    elif damage=='file_escape': w.type='FileConstant';w.attr['file_path'].s=b'../outside.bin'
    with pytest.raises(ValueError): normalize_ac_constants(g,values,config,folder)


def test_unknown_fileconstant_descriptor_repaired_from_exact_binding(tmp_path,monkeypatch):
    monkeypatch.setenv('AI_RUN_DIR',str(tmp_path));folder=tmp_path/'graph';folder.mkdir()
    g,values,config=fixture();w=g.op[1];w.type='FileConstant';w.output_desc[0].dtype=0;w.output_desc[0].shape.dim[:]=[]
    path=folder/'weight.bin';path.write_bytes(values['w:0'].numpy().tobytes());w.attr['file_path'].s=str(path).encode()
    assert normalize_ac_constants(g,values,config,folder)['status']=='PASS'


def test_compile_requires_exact_custom_inventory():
    config=dict(policy=POLICY,a_count=1,c_count=0,audit_count=0)
    g=dict(metadata=dict(draft_ac=config),runtime_input_abi=dict(draft_ac=dict(status='PASS',policy=POLICY,counts={GE[A]:1,GE[C]:0,GE[AUDIT]:0})),custom_op_audit=[dict(ge_op_type=GE[A],ge_node_occurrences=1)])
    validate_ac_graph(g)
    g['custom_op_audit'][0]['ge_node_occurrences']=0
    with pytest.raises(ValueError): validate_ac_graph(g)


def test_static_runner_cpp_declarations(tmp_path):
    header=tmp_path/'extras.h';header.write_text('#include <cstddef>\nextern "C" const char* aclGetRecentErrMsg();\nextern "C" int aclrtMemset(void*,std::size_t,int,std::size_t);\n')
    result=subprocess.run(['c++','-std=c++17','-Wall','-Wextra','-Werror','-fsyntax-only','-I',str(ROOT/'framework/runtime/cpp/tests/fake_acl'),'-include',str(header),str(ROOT/'framework/custom_ops/draft_ac/test/runner.cpp')],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr

@pytest.mark.parametrize('mode,acount,ccount',[('off',0,0),('a',26,0),('c',0,1),('ac',26,1)])
def test_full_draft_opt_in_preserves_public_abi_and_masks(monkeypatch,mode,acount,ccount):
    from qwen35_dflash.ascend310p.incremental import DraftGraph
    from qwen35_dflash.ascend310p.contracts import AirGraphSpec
    from qwen35_dflash.ascend310p.draft_ac_export import integration_spec,projection_modules
    from qwen35_dflash.ascend310p.incremental_plan import expected_signatures
    from models.dflash_v1.draft_quantization import GroupQuantLinear
    graph=DraftGraph.__new__(DraftGraph);torch.nn.Module.__init__(graph)
    graph.layers=torch.nn.ModuleList([torch.nn.Module() for _ in range(5)])
    graph.ops=SimpleNamespace(); original_ops=graph.ops
    for name,parent,attr in projection_modules(graph):
        setattr(parent,attr,GroupQuantLinear(torch.zeros(64,256,dtype=torch.int8),torch.ones(64,2,dtype=torch.float16),bits=8,in_features=256,ops=None,matmul_backend='weight_quant'))
    contract=dict(target_states=[],draft_states=[dict(name=f'd{i}_{k}',dtype='float16',shape=[1,4,128,128]) for i in range(5) for k in ('key','value')],verify_discard_states=[],feature_width=28160)
    abi=expected_signatures(contract)['draft']; abi['inputs'][0]['shape'][1]=16
    args=tuple(torch.zeros(t['shape'],dtype=getattr(torch,t['dtype'])) for t in abi['inputs'])
    metadata=dict(incremental_contract=contract,tensor_abi=abi,draft_quantization='w8a16',draft_weight_storage='nz',draft_weight_prepack='nz')
    source=AirGraphSpec('draft','draft',graph,args,input_names=tuple(t['name'] for t in abi['inputs']),output_names=tuple(t['name'] for t in abi['outputs']),metadata=metadata,
                        custom_ops=(CustomOpExportSpec('npu::npu_weight_quant_batchmatmul','WeightQuantBatchMatmulV2',minimum_occurrences=26),))
    saved=copy.deepcopy(source.metadata)
    selected=integration_spec(source,mode,args,16)
    assert source.metadata==saved and selected.metadata['tensor_abi']==abi
    assert 'incremental_contract' not in selected.metadata and selected.metadata['source_incremental_contract']==contract
    assert selected.metadata['draft_ac']['a_count']==acount and selected.metadata['draft_ac']['c_count']==ccount
    assert len(selected.metadata['draft_ac']['projections'])==acount
    assert sum(isinstance(getattr(p,a),ACLinear) for _,p,a in projection_modules(graph))==acount
    assert (graph.ops is original_ops)==(ccount==0)
    assert type(graph).forward is DraftGraph.forward # anchor/proposal_count/functional cache update are original code
    damaged=list(args);damaged[0]=torch.zeros(1,64,28160,dtype=torch.float16)
    with pytest.raises(ValueError,match='ABI'): integration_spec(source,mode,tuple(damaged),16)


def ac_tools(monkeypatch):
    path=ROOT/'framework/custom_ops/draft_ac/test'
    monkeypatch.syspath_prepend(str(path))
    spec=importlib.util.spec_from_file_location('common',path/'common.py'); common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
    monkeypatch.setitem(sys.modules,'common',common)
    # Load entry tools under their own names only in this CPU test process.
    import opp
    import resolve
    return opp,resolve


def test_provenance_hash_and_path_is_authoritative(tmp_path,monkeypatch):
    _,tool=ac_tools(monkeypatch);p=tmp_path/'old.json';p.write_text('{}')
    item=dict(path=str(p),sha256=tool.sha256(p))
    assert tool.linked(tmp_path,item)==p
    item['path']='old.json';assert tool.linked(tmp_path,item)==p
    p.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='hash/path'): tool.linked(tmp_path,item)


@pytest.mark.parametrize('damage',[None,'missing_op','empty_object','missing_json','invalid_json','prefix','old_opp'])
def test_joint_opp_requires_three_pairs_and_headers(tmp_path,monkeypatch,damage):
    tool,_=ac_tools(monkeypatch);monkeypatch.delenv('LD_LIBRARY_PATH',raising=False)
    install=tmp_path/('bad.o-root' if damage=='prefix' else 'joint');vendor=install/'vendors/customize'
    kernelroot=vendor/'op_impl/ai_core/tbe/kernel/ascend310p'
    for _,(name,_) in tool.OPS.items():
        folder=kernelroot/name;folder.mkdir(parents=True)
        (folder/'kernel.o').write_bytes(b'object');(folder/'kernel.json').write_text('{}')
        inc=vendor/'op_api/include';inc.mkdir(parents=True,exist_ok=True);(inc/f'aclnn_{name}.h').write_text('header')
    proto=vendor/'op_proto/lib';proto.mkdir(parents=True);(proto/'libcust_opsproto.so').write_bytes(b'GE')
    first=kernelroot/'d_flash_group_quant_linear/kernel.o';ambient=''
    if damage=='missing_op': first.unlink()
    elif damage=='empty_object': first.write_bytes(b'')
    elif damage=='missing_json': first.with_suffix('.json').unlink()
    elif damage=='invalid_json': first.with_suffix('.json').write_text('broken')
    elif damage=='old_opp':
        old=tmp_path/'old';(old/'op_impl/ai_core/tbe/kernel/ascend310p/d_flash_group_quant_linear').mkdir(parents=True);ambient=str(old)
    if damage:
        with pytest.raises((ValueError,FileNotFoundError)): tool.preflight(install,ambient)
    else:
        result=tool.preflight(install);assert len(result['kernels'])==3 and result['status']=='PASS'

@pytest.mark.parametrize('air_name',['air-manifest.json','air-manifest-w8a16.json'])
@pytest.mark.parametrize('damage',[None,'stage_hash','capture_path','replay_hash','factory','air_hash','air_missing'])
def test_resolve_accepted_evidence_chain_without_guessing_paths(tmp_path,monkeypatch,damage,air_name):
    opp,tool=ac_tools(monkeypatch)
    def save(path,obj):
        path=tmp_path/path;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj));return path
    def ref(path): return dict(path=str(path),sha256=tool.sha256(path))
    def relative(path,root): return dict(path=str(path.relative_to(root)),bytes=path.stat().st_size,sha256=tool.sha256(path))
    header=tmp_path/'build/CustomOp/op_host/config.h';header.parent.mkdir(parents=True);header.write_text('CPU fixture')
    cfg=save(Path('evidence/build-config.json'),dict(header=str(header)))
    a=dict(header=str(header));monkeypatch.setattr(tool,'validate_sources',lambda *args:(a,{}))
    accepted=save(Path('evidence/opp-preflight.json'),dict(status='PASS'))
    import opp_preflight
    monkeypatch.setattr(opp_preflight,'check_install',lambda p:dict(status='PASS'))
    source={};bundle_records={};bundle_data={};head=dict(gears={})
    config=dict(draft_quantization='w8a16',draft_weight_prepack='nz')
    checkpoint=dict(model_sha256='checkpoint',config_sha256='config')
    om=save(Path('native/om.json'),{});om64=save(Path('native/om64.json'),{})
    omrec=relative(om,tmp_path/'native');om64rec=relative(om64,tmp_path/'native')
    graph=dict(name='draft',metadata={},om=omrec,input_names=[],output_names=[],runtime_input_abi={},air={},payload_files=[])
    air=save(Path('native')/air_name,dict(status='PASS',graphs=[graph],factory_config=config))
    if air_name!='air-manifest.json':
        # An unrelated default-name file must never override Deployment's link.
        save(Path('native/air-manifest.json'),dict(status='FAIL',factory_config={}))
    dep=save(Path('native/deployment.json'),dict(status='PASS',air_manifest=relative(air,tmp_path/'native'),graphs=[graph]))
    from qwen35_dflash.ascend310p import draft_gears
    monkeypatch.setattr(draft_gears,'verify_om_files',lambda *a:None)
    monkeypatch.setattr(draft_gears,'om_records',lambda *a:[omrec,om64rec])
    replays={}
    for gear in (16,64):
        directory=tmp_path/f'snapshot{gear}';directory.mkdir();(directory/'contract.txt').write_text('fixture')
        replay=save(Path(f'replay{gear}.json'),dict(fake_acl=False,draft_om_sha256=omrec['sha256'],draft_static64_om_sha256=om64rec['sha256'],input_directory=str(directory),snapshot_sha256={},kv_output_audit=dict(context_rows=gear)))
        replays[gear]=replay
        item=dict(source=dict(feature_layers=[1,5],report_sha256=tool.sha256(replay)),checkpoint=checkpoint)
        bundle=save(Path(f'capture{gear}.json'),item);bundle_data[gear]=item
        native=save(Path(f'a-native{gear}.json'),{})
        bundle_records[f'c{gear}']=dict(bundle=ref(bundle),native_om_manifest=ref(native));head['gears'][f'c{gear}']=item
    monkeypatch.setattr(tool,'snapshot_contract',lambda text:(omrec['sha256'],{}))
    monkeypatch.setattr(tool,'load_bundle',lambda path,scope:tool.load(path))
    monkeypatch.setattr(tool,'paired_gears',lambda *args:None)
    monkeypatch.setattr(tool,'validate_om_manifest',lambda *args:None)
    monkeypatch.setattr(tool,'head_capture',lambda *args:head)
    a4=save(Path('a4.json'),dict(status='PASS',build=a,inputs=bundle_records))
    stage=save(Path('stage.json'),dict(status='PASS',full_a4_regression='PASS',build_config_path=str(cfg),build_config_sha256=tool.sha256(cfg),build=a,checks=dict(full=dict(summary=ref(a4)))))
    paired=save(Path('pair.json'),dict(status='PASS',runs={'256':ref(stage)}))
    cap=save(Path('head-capture.json'),{})
    models=[]
    for m in range(1,16):
        for kind in ('logits','top1'):
            abi=dict(inputs=[dict(name='hidden',dtype='float16',shape=[m,2560]),dict(name='weight',dtype='float16',shape=[248320,2560])],outputs=[dict(name='y',dtype='float16' if kind=='logits' else 'int64',shape=[m,248320] if kind=='logits' else [m])])
            d=save(Path(f'native/head{m}-{kind}.json'),dict(status='PASS',graphs=[dict(metadata=dict(tensor_abi=abi),om=omrec)]))
            models.append(dict(m=m,kind=kind,status='PASS',runtime_input_abi=dict(logical_input_names=['hidden','weight']),om=omrec,air=omrec,deployment=relative(d,tmp_path/'native')))
    native=save(Path('native/head.json'),dict(status='PASS',capture_sha256=tool.sha256(cap),weight_binding='runtime-readonly-ND',models=models))
    csuite=save(Path('c-suite.json'),dict(status='PASS',capture_sha256=tool.sha256(cap),native_manifest_sha256=tool.sha256(native)))
    state=save(Path('state.json'),{})
    release=save(Path('release.json'),dict(build_state=str(state),build_artifacts=[],installed_opp={},suite=ref(csuite)))
    args=SimpleNamespace(a51_suite=paired,c_release=release,head_capture=cap,head_native=native,native_deployment=dep,air_manifest=None,replay16=replays[16],replay64=replays[64],factory_config=None)
    if damage=='stage_hash': stage.write_text('{}')
    elif damage=='capture_path': args.a16_bundle=tmp_path/'wrong.json'
    elif damage=='replay_hash':
        data=tool.load(replays[16]);data['fake_acl']=True;save(Path('replay16.json'),data)
    elif damage=='factory': args.factory_config=save(Path('wrong.json'),{})
    elif damage=='air_hash': air.write_text('{}')
    elif damage=='air_missing': air.unlink()
    output=tmp_path/'new';output.mkdir()
    if damage:
        with pytest.raises(ValueError): tool.resolve(args,output)
    else:
        result,_=tool.resolve(args,output)
        assert result['status']=='PASS' and args.a_build_config==cfg and tool.load(args.factory_config)==config
        assert result['records']['air_manifest']==ref(air)
        assert args.a16_native==tmp_path/'a-native16.json'
        assert cfg.read_text()==json.dumps(dict(header=str(header)))


def test_torch_export_retains_custom_node_and_immutable_buffers():
    ensure_ops()
    class Projection(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('weight',torch.zeros(64,256,dtype=torch.int8))
            self.register_buffer('scale',torch.ones(2,64,dtype=torch.float16))
        def forward(self,x): return torch.ops.dflash_ac.group_quant_linear(x,self.weight,self.scale)
    exported=torch.export.export(Projection(),(torch.zeros(16,256,dtype=torch.float16),))
    calls=[n.target for n in exported.graph.nodes if n.op=='call_function']
    assert calls==[torch.ops.dflash_ac.group_quant_linear.default]
    assert tuple(exported.graph_signature.user_inputs)==('x',)
    assert set(exported.graph_signature.buffers)=={'weight','scale'}


@pytest.mark.parametrize('preflight_only',[True,False])
def test_entry_never_loads_npu_before_fresh_joint_loader(tmp_path,monkeypatch,preflight_only):
    ac_tools(monkeypatch)
    spec=importlib.util.spec_from_file_location('ac_entry',ROOT/'framework/custom_ops/draft_ac/test/run.py')
    entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
    monkeypatch.delitem(sys.modules,'torch_npu',raising=False)
    monkeypatch.setenv('ASCEND_CUSTOM_OPP_PATH','receiver-adn-gdr')
    monkeypatch.setenv('LD_LIBRARY_PATH','receiver-libraries')
    root=tmp_path/'new';args=SimpleNamespace(output_dir=root,preflight_only=preflight_only,a_build_config=tmp_path/'a',c_release=tmp_path/'c')
    def resolve(args,root):
        entry.write_json(root/'preflight.json',dict(status='PASS'))
        return dict(records={}),{}
    monkeypatch.setattr(entry,'resolve',resolve)
    builds=[]
    def build(args):
        builds.append(args);return dict(preflight=dict(vendor=str(root/'joint/opp/vendors/customize')))
    monkeypatch.setattr(entry,'build',build)
    class Exec(BaseException): pass
    calls=[]
    def execve(binary,command,env): calls.append((binary,command,env));raise Exec()
    monkeypatch.setattr(entry.os,'execve',execve)
    if preflight_only:
        assert entry.run(args)==0 and not builds and not calls
        assert json.loads((root/'suite.json').read_text())['single_op_graph']=='NOT_RUN'
    else:
        with pytest.raises(Exec): entry.run(args)
        assert len(builds)==len(calls)==1 and '--device-worker' in calls[0][1]
        assert calls[0][2]['LD_LIBRARY_PATH'].startswith(str(root/'joint/opp/vendors/customize/op_api/lib'))
        assert calls[0][2]['ASCEND_CUSTOM_OPP_PATH'].endswith('receiver-adn-gdr')
        assert calls[0][2]['TORCH_DEVICE_BACKEND_AUTOLOAD']=='0'
        assert json.loads((root/'suite.json').read_text())['single_op_graph']=='NOT_RUN'
