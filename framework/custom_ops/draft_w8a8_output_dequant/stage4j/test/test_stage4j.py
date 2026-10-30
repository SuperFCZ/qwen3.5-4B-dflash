"""CPU doubles test addressing/contracts, never CANN execution or hardware events."""
import copy
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest

HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import j_common as c
import j_reference as ref
import run_stage4j as run
import source_audit
import build_n128

def t1():
    fields={k:0 for k in c.FIELDS}
    fields.update(usedCoreNum=1,singleCoreM=16,singleCoreN=128,singleCoreK=128,baseM=16,baseN=128,baseK=128,depthA1=1,depthB1=1,dbL0A=2,dbL0B=2,dbL0C=1,transLength=16384,stepM=1,stepN=1,stepKa=1,stepKb=1,shareUbSize=1024,Ka=128,Kb=128,M=16,N=128)
    return dict(fields=fields,capacity=dict(UB=262144,L1=1048576,L0A=65536,L0B=65536,L0C=262144),system_workspace_bytes=4096,scratch=226816)

@pytest.fixture(scope='module')
def packed(tmp_path_factory):
    root=tmp_path_factory.mktemp('n128');old=root/'old';old.mkdir()
    a=((np.arange(40960)%31)-15).astype('i1').reshape(20,16,128)
    q=np.empty((19456,2560),'i1');sw=np.empty((20,19456),'<f2')
    for t in range(304):
        q[t*64:(t+1)*64]=((np.arange(64)[:,None]*7+np.arange(2560)*3+t*11)%27-13).astype('i1')
        sw[:,t*64:(t+1)*64]=((np.arange(20)[:,None]+np.arange(64)[None,:]/64+1+t/304)*.0003).astype('<f2')
    nz=q.reshape(1216,16,80,32).transpose(2,0,1,3).copy()
    b64=q.reshape(304,64,20,128).transpose(0,2,3,1).copy();w64=np.empty((304,20,64),'<u8')
    for t in range(304):
        for g in range(20):w64[t,g]=c.common.stage2a.encode_doc_candidate((np.float64(.03125)*sw[g,t*64:(t+1)*64].astype(np.float64)).astype('<f4'))[0]
    src=dict(a=a,q=q,nz=nz,sw=sw,sa=.03125);parts64=np.empty((304,20,16,64),'<f2');gold=np.empty((16,19456),'<f2')
    hb=__import__('hashlib').sha256();hs=__import__('hashlib').sha256()
    for t in range(152):
        b,w=c.layout_i.verify_tile(src,b64,w64,t);hb.update(b.tobytes());hs.update(w.tobytes());parts,out=ref.compute(a,b,w)
        parts64[2*t]=parts[:,:,:64];parts64[2*t+1]=parts[:,:,64:];gold[:,t*128:(t+1)*128]=out
    oldmaps={'a_i8.bin':a,'b_i8.bin':b64,'quant_u64_doc_candidate.bin':w64,'partials_fp16.bin':parts64,'golden_fp16.bin':gold}
    files={}
    for n,v in oldmaps.items():v.tofile(old/n);files[n]=c.identity(old/n)
    oldcase=dict(files=files,source={'fixture':'CPU only'});c.write(old/'case.json',oldcase)
    proof=dict(stage4b_case=c.identity(old/'case.json'),layout=dict(packed_B_sha256=hb.hexdigest(),packed_scale_sha256=hs.hexdigest()))
    with patch.object(ref.reference,'verify_case',return_value=oldmaps),patch.object(ref.reference,'load_source',return_value=src):
        case=ref.prepare(root/'data',proof)
    return root,case,src,oldmaps

def test_all_layouts_historical_bits_and_no_requantization(packed):
    root,case,src,oldmaps=packed;maps=ref.verify_case(case,numerical=True)
    assert len(case['tiles'])==152
    assert case['files']['golden_fp16.bin']['path']==str(root/'old/golden_fp16.bin')
    assert maps['a_i8.bin'].tobytes()==src['a'].tobytes()
    assert sum(x['groups'] for x in case['tiles'])==3040
    for t in range(152):
        assert np.array_equal(maps['b_i8.bin'][t],np.concatenate([oldmaps['b_i8.bin'][2*t],oldmaps['b_i8.bin'][2*t+1]],axis=2))
    # Blind byte concatenation has group-major/K-major order wrong.
    assert maps['b_i8.bin'][75].tobytes()!=oldmaps['b_i8.bin'][150:152].tobytes()
    idx=c.layout_i.local_nz_indices(128)
    assert np.array_equal(np.sort(idx.ravel()),np.arange(2048))
    assert case['layout']['gate_up_boundary']['tile']==76

def test_layout_rejects_wrong_nz_scale_and_nonfinite(packed):
    _,_,src,old=packed
    bad=dict(src,nz=src['nz'].copy());bad['nz'][79,1215,15,31]^=1
    with pytest.raises(ValueError,match='tile=151 group=19'):c.layout_i.verify_tile(bad,old['b_i8.bin'],old['quant_u64_doc_candidate.bin'],151)
    words=old['quant_u64_doc_candidate.bin'].copy();words[152,10,0]^=np.uint64(8192)
    with pytest.raises(ValueError,match='Scale'):c.layout_i.verify_tile(src,old['b_i8.bin'],words,76)

def test_ordered_partial_rounding_and_codec():
    a=np.ones((20,16,128),'i1');b=np.zeros((20,128,128),'i1');b[0]=127;b[1]=-127;b[19]=1
    w=np.full((20,128),np.uint64(1<<46)|np.uint64(0x3f000000),'<u8')
    parts,out=ref.compute(a,b,w);s=np.zeros((16,128),'<f4')
    for g in range(20):s=np.add(s,parts[g].astype('<f4'),dtype=np.float32)
    assert out.tobytes()==s.astype('<f2').tobytes() and np.all(out==64)
    bad=w.copy();bad[1,4]^=np.uint64(1)
    with pytest.raises(ValueError,match='control'):ref.compute(a,b,bad)
    bad=w.copy();bad[1,4]=np.uint64(1<<46)|np.uint64(0x7f800000)
    with pytest.raises(ValueError,match='effective'):ref.compute(a,b,bad)

def test_fp16_accumulator_or_reordered_sum_is_not_reference():
    a=np.ones((20,16,128),'i1');b=np.ones((20,128,128),'i1');b[-1]=-1
    scales=np.full((20,128),1/128,'<f4');scales[0]=16;scales[-1]=16
    words=np.empty((20,128),'<u8')
    for g in range(20):
        for lo in (0,64):words[g,lo:lo+64]=c.common.stage2a.encode_doc_candidate(scales[g,lo:lo+64])[0]
    parts,out=ref.compute(a,b,words)
    wrong=np.zeros((16,128),'<f2')
    for part in parts:wrong=(wrong+part).astype('<f2')
    assert np.all(out==18) and not np.array_equal(out,wrong)
    # A midpoint-crossing example distinguishes per-group FP16 conversion from
    # keeping all scaled INT32 partials in FP32 until the end.
    b[:]=1;b[:,0,0]=3;scales[:]=.001234567
    for g in range(20):
        for lo in (0,64):words[g,lo:lo+64]=c.common.stage2a.encode_doc_candidate(scales[g,lo:lo+64])[0]
    parts,out=ref.compute(a,b,words);total=np.zeros((16,128),'<f4')
    eff=(words & np.uint64(0xffffffff)).astype('<u4').view('<f4')
    for g in range(20):total=np.add(total,(a[g].astype(np.int64)@b[g].astype(np.int64)).astype('<f4')*eff[g],dtype=np.float32)
    assert np.any(out.view('<u2')!=total.astype('<f2').view('<u2'))

@pytest.mark.parametrize('cores,limit',[(7,0),(8,0),(7,1),(7,2)])
def test_schedule_coverage_and_arena(cores,limit):
    blocks,owners=c.schedule(cores,limit)
    assert sorted(t for row in owners for t in row)==list(range(152))
    for t in c.BOUNDED_TILES:
        _,bounded=c.schedule(cores,limit,t);assert [x for row in bounded for x in row]==[t]
        assert t in bounded[t%blocks]
    intervals=[(i*5120+512,i*5120+4608) for i in range(blocks)]
    assert all(a[1]<b[0] for a,b in zip(intervals,intervals[1:]))
    with pytest.raises(ValueError):c.schedule(cores,cores+1)

def outputs(maps,mode,blocks,tile):
    out=[]
    for pattern in (0x7f7f,0xffff):
        a=np.full((16,19456),pattern,'<u2').view('<f2')
        for t in (range(152) if tile<0 else [tile]):a[:,t*128:(t+1)*128]=ref.expected(maps,t,mode,blocks)
        out.append(a.tobytes())
    return out

@pytest.mark.parametrize('tile,mode',[(-1,0),(-1,21),(0,1),(75,20),(76,0),(151,20)])
def test_strict_output_scope_poison_and_diagnostics(packed,tile,mode):
    _,case,_,_=packed;maps=ref.verify_case(case);out=outputs(maps,mode,7,tile)
    result=ref.compare(maps,mode,7,tile,out)
    assert result['status']=='BITWISE_MATCH' and result['checked_elements']==(311296 if tile<0 else 2048)
    active=0 if tile<0 else tile;bad=bytearray(out[1]);bad[active*256]^=1
    failure=ref.compare(maps,mode,7,tile,[out[0],bytes(bad)])
    assert failure['status']=='FAIL_NUMERICAL' and failure['tiles'][0]['first_differences'][1]['channel']==active*128
    if tile>=0:
        bad=bytearray(out[1]);bad[((tile+1)%152)*256]=0
        assert not ref.compare(maps,mode,7,tile,[out[0],bytes(bad)])['inactive_poison_unchanged']

def test_preparation_tamper_rejected(packed):
    _,case,_,_=packed;bad=copy.deepcopy(case);bad['tiles'][76]['n_range']=[0,128]
    with pytest.raises(ValueError,match='index'):ref.verify_case(bad)
    bad=copy.deepcopy(case);bad['files']['b_i8.bin']['sha256']='0'*64
    with pytest.raises(ValueError,match='identity'):ref.verify_case(bad)
    bad=copy.deepcopy(case);bad['tiles'].pop()
    with pytest.raises(ValueError,match='incomplete'):ref.verify_case(bad)

def test_exact_source_delta_and_no_performance():
    record,_=source_audit.audit();assert record['status']=='EXACT_ALLOWED_TRANSFORM'
    text=(HERE/'op_kernel/d_flash_w8_a8_n128_gate.cpp').read_text()
    assert text.count('CheckGuard(')==13 and text.count('mm.SetLocalWorkspace(scratch)')==1
    assert text.index('for(uint32_t g=0;g<20')<text.index('mm.SetLocalWorkspace(scratch)')
    assert 'PipeBarrier<PIPE_ALL>();mm.End();PipeBarrier<PIPE_ALL>();' in text
    subprocess.run(['bash','-n',str(HERE/'run.sh')],check=True)

def test_t1_contract_cpp_rejects_field_and_resource_drift(tmp_path):
    contract=t1();(tmp_path/'t1_contract.h').write_text(c.contract_header(contract))
    getters='\n'.join(f'int get_{k}() const {{return {v};}}' for k,v in contract['fields'].items())
    cpp='#include <cstdint>\n#include "t1_contract.h"\nstruct T{'+getters+'};\nint main(){T t;return S4JAcceptedT1(t,262144,1048576,65536,65536,262144,4096,226816)?0:1;}'
    compiler=shutil.which('c++');assert compiler
    (tmp_path/'test.cpp').write_text(cpp)
    def check(expected):
        subprocess.run([compiler,'-std=c++17',str(tmp_path/'test.cpp'),'-o',str(tmp_path/'test')],check=True)
        p=subprocess.run([str(tmp_path/'test')],capture_output=True);assert p.returncode==expected
    check(0)
    for bad in (cpp.replace('return 16384;','return 8192;'),cpp.replace('262144,1048576','253952,1048576'),cpp.replace('4096,226816','8192,226816')):
        (tmp_path/'test.cpp').write_text(bad);check(1)
    broken=t1();broken['fields']['transLength']=None
    with pytest.raises(ValueError):c.contract_header(broken)

def plantext(limit,mode,tile):
    blocks,_=c.schedule(7,limit);user=blocks*5120
    return f'STAGE4J_CORE_PROBE aic=8 aiv=8 all=7\nSTAGE4J_ARCH_PROBE arch=2002\nSTAGE4J_CORE_POLICY aic=8 aiv=8 all=7 selected=7\nSTAGE4J_SCHEDULE coreLimit={limit} blocks={blocks} cores=7 system=4096 arena=4096 stride=5120 user={user} total={4096+user} mode={mode}\nscratch=226816 userUB=24576 ubGuards=2560 reserve=8192 single=16/128/128 base=16/128/128\nSTAGE4J_SCOPE tile={tile}\nSTAGE4J_T1_CONTRACT match=1 effectiveSharedUb=0\n'+''.join(f'STAGE4J_T1 {k}={v}\n' for k,v in t1()['fields'].items())

def test_log_plan_rejects_core_and_tiling_drift():
    text=plantext(0,1,75);p=c.plan_from_log(text,1,0,75,t1())
    assert p['blocks']==7 and sum(map(len,p['owners']))==1
    for bad in (text.replace('aiv=8','aiv=0'),text.replace('baseN=128','baseN=64'),text.replace('tile=75','tile=76'),text.replace('total=39936','total=1'),text.replace('match=1','match=0')):
        if bad==text:continue
        with pytest.raises(ValueError):c.plan_from_log(bad,1,0,75,t1())

def test_pipeline_missing_proof_fails():
    with pytest.raises(ValueError,match='prior'):run.validate_proof(None,{},None,'host-only')
    with pytest.raises(ValueError,match='prepare'):run.prepared(None,{})
    assert run.sequence('bounded')==[(0,m,t) for t in [0,75,76,151] for m in (1,20,0)]
    assert run.sequence('full')==[(0,21,-1),(0,0,-1),(2,21,-1),(2,0,-1)]

def test_build_binds_generated_t1_header_and_private_identity(tmp_path):
    root=tmp_path/'build';root.mkdir();sdk=tmp_path/'sdk';sdk.mkdir()
    tools={}
    for n in ('msopgen','cmake','c++'):
        p=tmp_path/n;p.write_text(n);tools[n]=str(p)
    pf=tmp_path/'preflight.json';c.write(pf,dict(tools={n:dict(file=c.identity(p)) for n,p in tools.items()}))
    r=dict(evidence=dict(preflight=c.identity(pf),t1=t1()))
    args=SimpleNamespace(cann_root=sdk)
    def fake_command(argv,log,env,record,key,*extra):
        log.write_text('CPU build fixture; no CANN execution\n')
        if key=='msopgen':
            for sub in ('op_host','op_kernel'):(root/'CustomOp'/sub).mkdir(parents=True)
        if key=='build':
            out=root/'CustomOp/build_out';out.mkdir();(out/'custom_opp_cpu_fixture.run').write_text('not executable')
        if key=='runner_configure':
            dest=root/'runner-build';dest.mkdir();cache=[]
            for name in ('custom','ascendcl','nnopbase','acl_op_compiler'):
                base=root/'opp' if name=='custom' else sdk;base.mkdir(exist_ok=True)
                lib=base/('lib'+name+'.so');lib.write_text(name);cache.append(f'LIB_{name}:FILEPATH={lib}')
            (dest/'CMakeCache.txt').write_text('\n'.join(cache));(dest/'compile_commands.json').write_text('[]')
        if key=='runner_build':(root/'runner-build/stage4j_runner').write_text('CPU fixture')
    host=sdk/'host.so';host.write_text('CPU fixture')
    with patch.object(build_n128.sys,'platform','linux'),patch.object(build_n128.platform,'machine',return_value='aarch64'),patch.object(build_n128.shutil,'which',side_effect=tools.get),patch.object(build_n128,'command',side_effect=fake_command),patch.object(build_n128,'core_header_binding',return_value={}),patch.object(build_n128,'installed',return_value=dict(host=c.identity(host))):
        build_n128.build(args,root,r)
    generated=root/'CustomOp/op_host/t1_contract.h'
    assert generated.read_text()==c.contract_header(t1())
    assert c.identity(generated) in r['generated_sources']
    assert r['opp']==str(root/'opp') and r['status']=='STAGE4J_BUILT_NPU_NOT_RUN'
    assert r['runner']['path']==str(root/'runner-build/stage4j_runner')
    assert set(r['linked_libraries'])=={'LIB_custom','LIB_ascendcl','LIB_nnopbase','LIB_acl_op_compiler'}

@pytest.mark.parametrize('fail',[False,True])
def test_phases_order_and_failure_stops_device_calls(packed,tmp_path,fail):
    root,case,_,_=packed;maps=ref.verify_case(case);runner=tmp_path/'runner';runner.write_text('CPU fixture')
    prep=tmp_path/'prepare.json';c.write(prep,{})
    report=dict(settings=c.SETTINGS,sources=[],device=0,evidence=dict(t1=t1(),formal_core_policy=dict(aic=8,aiv=8,all=7,selected=7,arch=2002)),prepare_report=c.identity(prep),case=c.identity(root/'data/case.json'))
    bp=tmp_path/'build.json';c.write(bp,dict(**report,status='STAGE4J_BUILT_NPU_NOT_RUN',phase='complete',opp=str(tmp_path/'opp'),installed={},runner=c.identity(runner),cann_compile='CPU_FIXTURE',installation='CPU_FIXTURE'))
    args=SimpleNamespace(build_report=bp,cann_root=tmp_path/'sdk',device=0,host_report=None,bounded_report=None)
    calls=[]
    def invoke(argv,log,environment,r,key,timeout):
        host=argv[-1]=='--host-only';base=argv[:-1] if host else argv;mode,limit,tile=map(int,base[-3:]);calls.append((limit,mode,tile))
        text=plantext(limit,mode,tile);log.write_text(text);plan=c.plan_from_log(text,mode,limit,tile,t1());folder=log.parent
        ex=dict(status='HOST_TILING_ONLY' if host else 'EXECUTION_COMPLETED_NUMERICS_UNCHECKED',op=c.NAME,soc='Ascend310P3',shape=[16,2560,19456],tile_count=152,group_count=20,output_mode=mode,core_limit=limit,tile_id=tile,device=0,workspace_bytes=50000,
            resource_admission='BEFORE_EXECUTION',ub_guard_protocol='UB_AND_CORE_ARENA_POISON_ON_FAILURE',guards=True,readonly=True,repeat_equal=True,repetitions=2,cpu_fallback=False,layout='nd',input_count=3,poison_verified=True,poison_patterns=[127,255],timing=dict(status='NOT_RUN',warmup=0,repetitions=0),host_only=host,npu='NOT_RUN' if host else 'COMPLETED')
        c.write(folder/'execution.json',ex);c.write(folder/'resource-admission.json',run.admission_expected(ex,plan,host))
        if not host:
            out=outputs(maps,mode,plan['blocks'],tile)
            if fail:
                bad=bytearray(out[1]);bad[0]^=1;out[1]=bytes(bad)
            for i,raw in enumerate(out):(folder/f'actual-{i}.bin').write_bytes(raw)
    with patch.object(run,'installed',return_value={}),patch.object(run,'command',side_effect=invoke):
        for stage in ('host-only','bounded','full'):
            args.stage=stage;dest=tmp_path/stage;dest.mkdir();r=copy.deepcopy(report)
            if fail and stage=='bounded':
                with pytest.raises(ValueError,match='no next phase'):run.smoke(args,dest,r,case)
                assert calls==[(1,21,-1),(0,1,0)]
                assert r['runs'][0]['status']=='FAIL_NUMERICAL';break
            run.smoke(args,dest,r,case);c.write(dest/'report.json',r)
            assert r['status']==run.STATES[stage]
            if stage=='host-only':args.host_report=dest/'report.json'
            if stage=='bounded':args.bounded_report=dest/'report.json'
        if not fail:
            assert len(calls)==17
            tampered=run.read(args.bounded_report);tampered['runs'].pop();c.write(args.bounded_report,tampered)
            with pytest.raises(ValueError,match='coverage'):run.validate_proof(args.bounded_report,r,maps,'bounded')
