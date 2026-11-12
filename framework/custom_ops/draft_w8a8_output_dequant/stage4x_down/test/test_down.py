"""Synthetic CPU fixtures test contracts only; server inputs must be real NPU outputs."""
import copy
import re
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import x_common as c
import x_transform as transform
import reference_x as ref
import resources_x as resource
import protocol_x as protocol
import run_down


def fixture():
    p=dict(usedCoreNum=1,singleCoreM=16,singleCoreN=256,singleCoreK=9728,baseM=16,baseN=256,baseK=128,
        depthA1=76,depthB1=8,stepM=1,stepN=1,stepKa=76,stepKb=4,dbL0A=2,dbL0B=2,dbL0C=1,
        shareMode=0,shareL1Size=417792,shareL0CSize=16384,shareUbSize=2048,transLength=34816,isBias=0,Ka=9728,Kb=9728,M=16,N=256)
    platform=dict(platform_source='NO_ARGUMENT_MANAGER_ON_SELECTED_DEVICE',soc='Ascend310P3',arch=2002,aic=8,aiv=8,all=8,system_workspace_bytes=2097152,
        capacity=dict(UB=262144,L1=1048576,L0A=65536,L0B=65536,L0C=65536))
    row=dict(name='FULL_K_DOWN_N256',requested_n=256,requested_base_n=256,scratch=202240,setup_accepted=True,get_tiling_rc=0,fields=p)
    return platform,row


def test_actual_shape_only_transform_and_guard_sync_preserved():
    transform.verify_repository()
    for kind in transform.NAMES:
        old=next((transform.BASE/kind/'op_kernel').glob('*.cpp')).read_text()
        new=next((HERE/kind/'op_kernel').glob('*.cpp')).read_text()
        calls=lambda s:re.findall(r'(?:PipeBarrier<[^>]+>\([^;]+|(?:SetFlag|WaitFlag)<[^>]+>\([^;]+|mm\.(?:Iterate|End|SetLocalWorkspace|GetTensorC)\([^;]+)',s)
        assert calls(old)==calls(new)
        assert '9728*256' in new and 'm*2560+tile*256' in new
        assert '202240' in new and 'AllDelta' in new


def test_all_down_packed_elements_and_output_coverage():
    count=0
    for t in range(10):
        b=((np.arange(9728,dtype='i4')[:,None]*13+np.arange(256,dtype='i4')+t*7)%255-127).astype('i1')
        z=ref.pack(b);assert z.shape==(304,16,16,32)
        assert np.array_equal(ref.unpack(z),b)
        assert z[303,15,15,31]==b[9727,255]
        count+=z.nbytes
    assert count==24903680
    visited=np.zeros((16,2560),np.uint8)
    for core in range(7):
        for t in range(core,10,7):
            for nb in range(16):visited[:,t*256+nb*16:t*256+(nb+1)*16]+=1
    assert np.all(visited==1)


def test_quant_scale_and_int64_reference_generic_full_k():
    x=np.zeros((16,9728),'<f2');x[:,0]=1;x[15,-1]=127
    a,s,detail=c.w.shared_activation(x)
    assert s==1 and detail['scale_scope']==[16,9728] and a[15,-1]==127
    q=np.zeros((9728,2),'i1');q[0]=[1,-1];q[-1]=[-127,127]
    words,eff=c.w.encode(np.array([.001,.002],'<f4'))
    dot,y=c.w.compute(a,q,words)
    exact=a.astype('i8')@q.astype('i8')
    assert np.array_equal(dot,exact)
    assert y.tobytes()==(exact.astype('<f4')*eff).astype('<f2').tobytes()
    assert 9728*127*127<2**31


def test_host_resources_require_new_k_l1_scale_and_capacities():
    platform,row=fixture();out=resource.validate(platform,row)
    assert out['status']=='HOST_RESOURCE_CHECKS_PASS'  # CPU fixture, NOT real Host evidence
    assert out['details']['l1_conservative_bound']==417792
    assert out['details']['total_ub']==220672
    for key,value in [('singleCoreK',2560),('Ka',2560),('baseN',128),('shareL1Size',368640),('depthB1',99),('stepKa',20),('transLength',202240)]:
        changed=copy.deepcopy(row);changed['fields'][key]=value
        assert resource.validate(platform,changed)['status']=='BLOCKED',key
    for key,value in [('L1',400000),('L0B',32768),('UB',200000)]:
        changed=copy.deepcopy(platform);changed['capacity'][key]=value
        assert resource.validate(changed,row)['status']=='BLOCKED'


@pytest.mark.parametrize('phase',list(run_down.STATES))
def test_all_cli_stage_prerequisites(phase):
    args=[phase,'--output-dir','/tmp/unit-not-executed']
    prior=[x for x in list(run_down.STATES)[:list(run_down.STATES).index(phase)] if not (phase=='mlp' and x=='benchmark')]
    if prior:
        with pytest.raises(SystemExit):run_down.parse(args)
    for name in prior:args+=['--'+name+'-report','/tmp/'+name]
    assert run_down.parse(args).phase==phase


def test_numerical_difference_inventory_and_safety():
    golden=np.zeros((16,2560),'<f2');actual=golden.copy();actual[15,2559]=np.float16(.125)
    out=ref.numeric(actual.tobytes(),golden.tobytes())
    assert out['active_elements']==40960 and out['mismatch_count']==1
    assert out['differences'][0]['row']==15 and out['differences'][0]['channel']==2559
    with pytest.raises(ValueError):protocol.require_numeric([{'validation':{'status':'NUMERIC_MISMATCH_UNRESOLVED'}}])
    with pytest.raises(ValueError):ref.numeric(actual.tobytes(),golden.tobytes(),first=golden.tobytes())
    actual[0,0]=np.nan
    with pytest.raises(ValueError):ref.numeric(actual.tobytes(),golden.tobytes())


def test_tile0_poison_and_scope(tmp_path):
    golden=np.zeros((16,2560),'<f2');p=tmp_path/'gold';golden.tofile(p);case=dict(files={'golden_fp16.bin':c.identity(p)})
    a=ref.expected(case,0,0);b=ref.expected(case,0,1)
    assert ref.numeric(a,a,0,0)['active_elements']==4096
    ref.numeric(b,b,0,1,first=a)
    corrupt=bytearray(a);corrupt[-2:]=b'\x00\x00'
    with pytest.raises(ValueError):ref.numeric(bytes(corrupt),a,0,0)


def test_fault_witness_output_scope():
    for mode in range(2,8):
        for repeat in (0,1):
            raw=protocol.fault_expected(mode,repeat)
            assert len(raw)==81920 and raw[32:]==bytes([255 if repeat else 127])*(81920-32)


def test_uint32_guard_every_word_bit_and_boundaries():
    expected=0x5a39c671
    for boundary in range(3):
        values=np.full(384,expected,dtype='u4')
        for index in range(384):
            for bit in range(32):
                values[index]^=np.uint32(1<<bit)
                assert int(np.bitwise_or.reduce(values^expected))==1<<bit
                values[index]^=np.uint32(1<<bit)
        assert not np.bitwise_or.reduce(values^expected)
    for side in range(2):
        for index in range(128):
            for bit in range(32):
                values=np.full(128,expected,dtype='u4');values[index]^=np.uint32(1<<bit)
                assert int(np.bitwise_or.reduce(values^expected))==1<<bit


def test_unique_host_target_and_abi():
    assert 'CMakeFiles/stage4x_host.dir' in (HERE/'down_dependency_identity.py').read_text()
    text=(HERE/'candidate/runner/runner.cpp').read_text()
    assert 'aclnnDFlashDownFullKBNzGetWorkspaceSize' in text
    assert 'r.x.Make({16,9728}' in text and 'r.s.Make({2560},{2560}' in text
    assert 'r.w.Make({10,304,16,16,32}' in text
    assert 'r.y.Make({16,2560}' in text


def test_help_subprocess():
    result=subprocess.run([sys.executable,'-B',str(HERE/'run_down.py'),'--help'],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_cpp_numeric_contract_matches_python(tmp_path):
    import shutil
    compiler=shutil.which('c++')
    if not compiler:pytest.skip('host C++ compiler unavailable')
    source=tmp_path/'check.cpp'
    source.write_text('#include "output_contract.h"\nint main(int argc,char**argv){std::vector<uint8_t>a(81920),b(81920);a[81918]=1;auto n=fullk_explore::Inspect(a,b,{},-1,0x7f7f);fullk_explore::WriteNumeric(argv[1],n);}\n')
    exe=tmp_path/'check'
    subprocess.run([compiler,'-std=c++17','-I'+str(HERE/'candidate/runner'),str(source),'-o',str(exe)],check=True,capture_output=True)
    report=tmp_path/'numeric.json';subprocess.run([exe,report],check=True)
    actual=bytearray(81920);actual[-2]=1
    assert c.read(report)==ref.numeric(bytes(actual),bytes(81920))


def test_safety_resource_protocol_and_golden_timing_gate(tmp_path,monkeypatch):
    def file(name,value):
        p=tmp_path/name
        if isinstance(value,bytes):p.write_bytes(value)
        else:c.write(p,value)
        return c.identity(p)
    policy=dict(aic=8,aiv=8,all=7,selected=7,arch=2002)
    pl=dict(total=16784384,blocks=7,cores=7,core_policy=policy)
    monkeypatch.setattr(protocol,'plan',lambda *a:pl)
    data=bytes(81920);case=dict(files={'golden_fp16.bin':file('gold',data),'reference_f64.bin':file('ref',bytes(40960*8))},same_input_w8a16_down=file('native',data),original_w8a16_mlp=file('mlp',data))
    ex=dict(soc='Ascend310P3',shape=[16,9728,2560],tile_count=10,group_count=1,guards=True,readonly=True,repeat_equal=True,repetitions=2,device=0,cpu_fallback=False,
        output_mode=0,core_limit=0,tile_id=-1,guard_coverage='FULL_K_TILE_384_WORD_THREE_BOUNDARIES',prepare_calls=2,verify_calls=2,poison_verified=True,poison_patterns=[127,255],
        timed_prepare_io=False,timed_log_stable=True,timed_log_checked=False,sync_completed=True,resource_admission='BEFORE_EXECUTION',ub_guard_protocol='UB_AND_CORE_ARENA_POISON_ON_FAILURE',layout='nd',input_count=3,op='DFlashDownFullKBNz',
        status='EXPLORATORY_EXECUTION_SAFE',finite_active_output=True,inactive_poison_intact=True,active_coverage_checked=True,reference_kind='CPU_FULL_K_REFERENCE',guard_coverage_unchanged=True,arena_detector='XOR_OR_U32',
        initial_input_payload_bytes=25079808,initial_copyin_wall_ms=1.,output_copyout_wall_ms=[1.,1.],workspace_bytes=pl['total'],numeric_status='BITWISE_REFERENCE_OBSERVATION',
        timing=dict(status='NOT_RUN',protocol='checked-v1',warmup=0,repetitions=0,per_timed_call_readback=True,per_timed_call_poison=True,correctness_before_calls=2,correctness_after_calls=0,timed_tail_checked=False,execute_sync={'samples_ms':[]},prepare={'samples_ms':[]}))
    row=dict(mode=0,limit=0,tile=-1,variant='candidate',execution=file('execution',ex),log=file('log',b'log'),
        admission=file('admission',dict(status='ADMITTED_BEFORE_EXECUTION',workspace_bytes=pl['total'],declared_total=pl['total'],blocks=7,cores=7,core_policy=policy,host_proof_checked=True)),
        outputs=[file('actual0',data),file('actual1',data)],numeric=[file('num0',ref.numeric(data,data)),file('num1',ref.numeric(data,data))])
    result=protocol.validate(row,case,dict(formal_core_policy=policy),{},0)
    assert result['status']=='DOWN_GOLDEN_AND_SAFETY_MATCH'
    protocol.require_numeric([dict(validation=result)])
    for key in ('guards','readonly','repeat_equal','poison_verified','sync_completed'):
        changed=dict(ex);changed[key]=False;row['execution']=file('execution',changed)
        with pytest.raises(ValueError):protocol.validate(row,case,dict(formal_core_policy=policy),{},0)
