"""Synthetic arrays here are unit fixtures, never real-model acceptance evidence."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import real_mlp_common as c
import prepare_real as prep
import run_real_gate as gate
import run_mlp_reference as mlp
import capture_real


def test_shared_activation_exact_historical_recipe():
    x=np.linspace(-12,8,16*2560).reshape(16,2560).astype('<f2')
    a,s,detail=c.shared_activation(x)
    original,old_s,_=c.activation(x)
    assert a.tobytes()==original.tobytes() and s==old_s
    assert detail['rowwise_scale_supported'] is False
    assert len(set(detail['hypothetical_row_scales']))>1
    assert detail['scale_scope']==[16,2560]
    assert detail['clipped_ratio']==0


def test_row_scales_not_expressible_by_per_channel_words():
    # A per-N coefficient cannot equal sA[row]*sW[N] for two unequal row scales.
    sa=np.array([.1,.2]);sw=np.array([.3,.7])
    combined=sa[:,None]*sw
    assert np.all(combined[0]!=combined[1])
    assert c.encode(combined[0].astype('<f4'))[0].shape==(2,)


def test_zero_and_nonfinite_activation():
    q,s,d=c.shared_activation(np.zeros((16,2560),'<f2'))
    assert not q.any() and s==1 and d['all_zero']
    with pytest.raises(ValueError):c.shared_activation(np.full((16,2560),np.nan,'<f2'))


def test_effective_weight_rounding_and_group_boundary():
    q=np.tile(np.arange(-128,128,dtype='i1'),(2,1));s=np.array([[.25,.125],[.03125,.0625]],'<f2')
    w=c.effective_weight(q,s)
    assert w.shape==(2,256)
    assert w[0,127]==np.float16(q[0,127]*.25)
    assert w[0,128]==np.float16(q[0,128]*.03125)
    assert w[0,0]==-32
    with pytest.raises(ValueError):c.effective_weight(q,s.T[:1])


def test_codec_and_cpu_fullk_exact():
    x=np.array([[1.,-2,3,4],[0,0,0,0]],'<f2')
    q=np.array([[1,2,-3,4],[-127,126,0,3]],'i1');sw=np.array([.01,.02])
    result,detail=prep.fullk_cpu(x,q,sw)
    exact=result['a'].astype(np.int64)@q.astype(np.int64).T
    assert np.array_equal(result['dot'],exact)
    assert np.all(result['words']>>32==1<<14)
    assert np.all((result['words']&8191)==0)
    expected=(exact.astype('<f4')*result['effective']).astype('<f2')
    assert result['y'].tobytes()==expected.tobytes()
    assert detail['timing_scope'].startswith('NUMPY_CPU')


def test_full_channel_bnz_roundtrip_boundaries():
    # All 49,807,360 logical elements, not just prefix/suffix of one Tile.
    total=0
    for t in range(76):
        k=np.arange(2560,dtype=np.int32)[:,None];n=np.arange(256,dtype=np.int32)[None,:]
        b=((k*17+n*31+t*7)%256-128).astype('i1')
        z=c.pack_tile(b)
        assert np.array_equal(c.unpack_tile(z),b)
        assert z[79,15,15,31]==b[2559,255]
        total+=b.size
    assert total==49807360
    assert 9728//256==38


def test_checkpoint_weight_quant_matches_frozen_function():
    w=np.array([[0,0,0,0],[1,-2,3,0]],'<f2')
    q,s,d=c.quantize_weight(w)
    assert s[0]==1 and not q[0].any() and q[1,2]==127
    assert d['clipped']==0


def test_fp64_and_swiglu_boundaries():
    y=np.array([[1,-2,.5,2]],'<f2')
    silu,z=c.swiglu_cpu(y)
    ref=(y[:,:2].astype('f8')/(1+np.exp(-y[:,:2].astype('f8')))).astype('<f2')
    assert silu.tobytes()==ref.tobytes()
    assert z.tobytes()==(ref.astype('f8')*y[:,2:].astype('f8')).astype('<f2').tobytes()
    w=np.array([[1,2],[3,4],[5,6]],'<f2')
    assert np.array_equal(c.fp64_mm(z,w,1),z.astype('f8')@w.astype('f8').T)


def test_down_shape_and_fullk_bound_are_separate():
    assert c.SHAPES['down']==(16,9728,2560)
    assert 9728*127*127 < 2**31
    with pytest.raises(ValueError):c.pack_tile(np.zeros((9728,256),'i1'))


def captures():
    return [dict(context_rows=g,activation_sha256=f'x{g}-{i}',replay=dict(sha256=f'r{g}-{i}')) for g in (16,64) for i in range(2)]


def test_c16_c64_distinct_sample_requirement():
    rows=captures();c.coverage(rows)
    for incomplete in (rows[:2],rows[:3],rows[1:]):
        with pytest.raises(ValueError):c.coverage(incomplete)
    rows[1]['activation_sha256']=rows[0]['activation_sha256']
    with pytest.raises(ValueError):c.coverage(rows)


def test_tensor_view_reads_pinned_i32_bf16_and_bounds(tmp_path):
    raw=np.array([1,-1],'<i4').tobytes()+np.array([0x3f80,0xc000],'<u2').tobytes()
    meta={'packed':dict(dtype='I32',shape=[1,2],data_offsets=[0,8]),'scale':dict(dtype='BF16',shape=[1,2],data_offsets=[8,12])}
    def write_meta(m):
        header=json.dumps(m).encode();p=tmp_path/'weights';p.write_bytes(len(header).to_bytes(8,'little')+header+raw);return c.identity(p)
    item=write_meta(meta)
    assert c.tensor_view(item,'packed').tolist()==[[1,-1]]
    assert c.tensor_view(item,'scale').tolist()==[[1.,-2.]]
    meta['scale']['data_offsets'][1]=14
    with pytest.raises(ValueError):c.tensor_view(write_meta(meta),'scale')


def test_file_identity_and_no_output_overwrite(tmp_path):
    p=tmp_path/'data';p.write_bytes(b'original');item=c.identity(p)
    p.write_bytes(b'tampered')
    with pytest.raises(ValueError):c.verify(item)
    with pytest.raises(FileExistsError):c.fresh(tmp_path)
    with pytest.raises(ValueError):c.fresh(HERE/'output')


def test_checkpoint_wrong_hash_blocks(tmp_path):
    (tmp_path/'config.json').write_text('{}');(tmp_path/'model.safetensors').write_bytes(b'wrong')
    with pytest.raises(ValueError):c.checkpoint(tmp_path)


@pytest.mark.parametrize('stage',list(gate.STATES))
def test_cli_every_stage(stage):
    args=[stage,'--prepare-report','/tmp/p','--paired-report','/tmp/b','--output-dir','/tmp/out']
    predecessors=list(gate.STATES)[:list(gate.STATES).index(stage)]
    if predecessors:
        with pytest.raises(SystemExit):gate.parse(args)
    for s in predecessors:args+=['--'+s+'-report','/tmp/'+s]
    parsed=gate.parse(args)
    assert parsed.phase==stage
    if 'build' in predecessors:assert parsed.build_report==Path('/tmp/build').resolve()


@pytest.mark.parametrize('module',['capture_real','prepare_real','run_real_gate','run_mlp_reference','measure_preprocessing','summarize_real'])
def test_help_in_independent_process(module):
    result=subprocess.run([sys.executable,'-B',str(HERE/(module+'.py')),'--help'],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_native_and_capture_required_cli():
    with pytest.raises(SystemExit):mlp.parse(['benchmark','--prepare-report','/tmp/p','--gate-smoke-report','/tmp/g','--output-dir','/tmp/o'])
    with pytest.raises(SystemExit):capture_real.parse([])


def fixture_row(tmp_path,monkeypatch):
    plan=dict(total=16784384,blocks=7,cores=7,core_policy=dict(aic=8,aiv=8,all=7,selected=7,arch=2002))
    monkeypatch.setattr(c.pe.c.old_protocol,'plan',lambda *a:plan)
    def file(name,value,binary=False):
        p=tmp_path/name
        if binary:p.write_bytes(value)
        else:c.write(p,value)
        return c.identity(p)
    raw=np.zeros((16,19456),'<f2').tobytes()
    ref=np.zeros((16,19456),'<f8').tobytes()
    case=dict(files={'cpu_fullk_fp16.bin':file('gold',raw,True),'gate_up_reference_f64.bin':file('ref',ref,True),'gate_up_w8a16.bin':file('native',raw,True)})
    runner=file('runner',b'elf fixture',True)
    bound={'bnz':dict(build={'variants':{'candidate':{'build':dict(runner=runner)}}},operator={'op':'DFlashFullKBNz'},t1={},evidence={'baseline':{'policy':plan['core_policy']}})}
    row=dict(runner=runner,profiled=False,log=file('log',b'log',True),
        admission=file('admission',dict(status='ADMITTED_BEFORE_EXECUTION',workspace_bytes=plan['total'],declared_total=plan['total'],blocks=7,cores=7,core_policy=plan['core_policy'],host_proof_checked=False)))
    ex=dict(status='HOST_TILING_ONLY',npu='NOT_RUN',device=0,workspace_bytes=plan['total'],output_mode=0,core_limit=0,tile_id=-1)
    row['execution']=file('execution',ex)
    return row,case,bound,ex,file


def test_formal_host_exact_contract(tmp_path,monkeypatch):
    row,case,bound,ex,file=fixture_row(tmp_path,monkeypatch)
    assert gate.validate(row,case,bound,0,host=True)['status']=='FORMAL_HOST_ADMITTED'
    for key,value in [('status','HOST_ONLY'),('device',1),('workspace_bytes',1)]:
        wrong=dict(ex);wrong[key]=value;row['execution']=file('execution',wrong)
        with pytest.raises(ValueError):gate.validate(row,case,bound,0,host=True)


def test_new_numeric_scope_never_reuses201():
    ref=np.zeros((16,19456),'<f2');actual=ref.copy();actual[2,100]=np.float16(.001)
    obs=c.observe(actual.tobytes(),ref.tobytes(),-1,0)
    assert obs['mismatch_count']==1 and obs['differences'][0]['row']==2
    assert obs['numeric_status']=='NUMERIC_MISMATCH_UNRESOLVED'
    for bad in (np.full(ref.shape,np.nan,'<f2'),np.full(ref.shape,np.inf,'<f2')):
        with pytest.raises(ValueError):c.observe(bad.tobytes(),ref.tobytes(),-1,0)
    with pytest.raises(ValueError):c.observe(actual.tobytes(),ref.tobytes(),-1,1,ref.tobytes())


def test_statistics_count_and_nonfinite_rejected():
    with pytest.raises(ValueError):c.stats([1.]*89,90)
    with pytest.raises(ValueError):c.stats([float('nan')]*90,90)
    assert c.stats([1.]*90,90)['median_ms']==1


def test_historical_files_not_generated_or_op_rebuilt():
    text=(HERE/'run_real_gate.py').read_text()
    assert 'FROZEN_NORMAL_RUNNER_OPP_REVERIFICATION_NO_RECOMPILE' in text
    assert 'admitted_full.bin' in text and 'cpu_fullk_fp16.bin' in text
    assert not list(HERE.rglob('*.cpp'))


@pytest.mark.parametrize('broken',[None,'guards','readonly','repeat_equal','poison_verified','sync_completed','cpu_fallback','timed_log_stable','workspace_bytes','numeric_status'])
def test_full_new_input_safety_contract(tmp_path,monkeypatch,broken):
    row,case,bound,ex,file=fixture_row(tmp_path,monkeypatch)
    adm=c.read(c.verify(row['admission']));adm['host_proof_checked']=True;row['admission']=file('admission',adm)
    ex.pop('npu')
    ex.update(status='EXPLORATORY_EXECUTION_SAFE',op='DFlashFullKBNz',soc='Ascend310P3',shape=[16,2560,19456],
        tile_count=76,group_count=1,guards=True,readonly=True,repeat_equal=True,repetitions=2,cpu_fallback=False,
        guard_coverage='FULL_K_TILE_384_WORD_THREE_BOUNDARIES',prepare_calls=2,verify_calls=2,poison_verified=True,
        poison_patterns=[127,255],timed_prepare_io=False,timed_log_stable=True,timed_log_checked=False,
        sync_completed=True,resource_admission='BEFORE_EXECUTION',ub_guard_protocol='UB_AND_CORE_ARENA_POISON_ON_FAILURE',
        layout='nd',input_count=3,finite_active_output=True,inactive_poison_intact=True,active_coverage_checked=True,
        guard_coverage_unchanged=True,arena_detector='XOR_OR_U32',reference_kind='CPU_FULL_K_REFERENCE',
        kernel_guards_unchanged=False,strict_golden_match=False,production='NOT_APPROVED',numeric_status='BITWISE_REFERENCE_OBSERVATION',
        timing=dict(status='NOT_RUN',protocol='checked-v1',warmup=0,repetitions=0,per_timed_call_readback=True,
            per_timed_call_poison=True,correctness_before_calls=2,correctness_after_calls=0,timed_tail_checked=False,
            execute_sync={'samples_ms':[]},prepare={'samples_ms':[]}))
    raw=c.verify(case['files']['cpu_fullk_fp16.bin']).read_bytes()
    row['outputs']=[file('out0',raw,True),file('out1',raw,True)]
    row['numeric']=[file('num0',c.observe(raw,raw,-1,0)),file('num1',c.observe(raw,raw,-1,1,raw))]
    if broken:
        ex[broken]=not ex[broken] if isinstance(ex[broken],bool) else 'WRONG'
    row['execution']=file('execution',ex)
    if broken:
        with pytest.raises(ValueError):gate.validate(row,case,bound,0)
    else:
        value=gate.validate(row,case,bound,0)
        assert value['numeric_status']=='BITWISE_REFERENCE_OBSERVATION'
        assert value['quality']=='UNRESOLVED' and not value['historical_201_mismatches_reused']


def test_cross_scheme_bitwise_not_required():
    x=np.array([1,2],'<f2');y=np.array([1.1,2.1],'<f2')
    assert c.errors(x,y)['relative_l2']>0
    assert 'NOT_RUN_DOWN_KERNEL_UNSUPPORTED' in (HERE/'run_mlp_reference.py').read_text()


def test_checkpoint_gate_then_up_order(monkeypatch):
    monkeypatch.setitem(c.SHAPES,'gate_up',(16,4,4))
    def tensor(file,name):
        if name.endswith('weight_scale'):
            return np.full((2,1),.5 if 'gate_proj' in name else .25,'<f2')
        # offset binary0x81 encodes+1;0x7f encodes-1, four K codes per word.
        word=0x81818181 if 'gate_proj' in name else 0x7f7f7f7f
        return np.full((2,1),word,dtype='<u4').view('<i4')
    monkeypatch.setattr(c,'tensor_view',tensor)
    # Keep the production Group128 shape rule; use K128 packed rows instead.
    monkeypatch.setitem(c.SHAPES,'gate_up',(16,128,4))
    def sized(file,name):
        v=tensor(file,name)
        return v if name.endswith('weight_scale') else np.repeat(v,32,axis=1)
    monkeypatch.setattr(c,'tensor_view',sized)
    q,s=c.checkpoint_projection({'files':{'model.safetensors':{}}},'gate_up')
    assert np.all(q[:2]==1) and np.all(q[2:]==-1)
    assert s.tolist()==[[.5,.5,.25,.25]]


def test_positive_scale_underflow_rejected():
    with pytest.raises(ValueError,match='underflows'):
        prep.fullk_cpu(np.ones((2,4),'<f2'),np.ones((2,4),'i1'),np.array([1e-320,1e-320]))


def test_capture_fake_acl_rejected_before_data_use(tmp_path,monkeypatch):
    report=tmp_path/'replay.json';c.write(report,dict(fake_acl=True))
    ck={'source':{'pinned':'unit-fixture'}}
    b={'checkpoint':dict(source=ck['source'],execution='cann-weight-quant-batchmatmul-v2',matmul_backend='weight_quant',activation_dtype='float16'),
       'sources':{},'source':dict(report=str(report),report_sha256=c.identity(report)['sha256'])}
    monkeypatch.setattr(c,'load_bundle',lambda p:b)
    with pytest.raises(ValueError,match='real replay'):
        c.capture_binding(tmp_path/'manifest.json',ck)
