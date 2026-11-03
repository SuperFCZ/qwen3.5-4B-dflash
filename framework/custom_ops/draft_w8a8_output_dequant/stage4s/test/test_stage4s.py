"""Source/address/ABI models only; never NPU execution evidence."""
import sys,shutil,subprocess
from pathlib import Path
import numpy as np
import pytest
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import source_s as s
import s_common as c
import run_stage4s as run

def test_exact_map_all_rows_tiles_and_maximum_extent():
 seen=np.zeros(311296,dtype='u1')
 for tile in range(76):
  for m in range(16):
   actual=[];expected=[]
   for block in range(16):
    for element in range(16):
     src=m*16+block*(1+15)*16+element;dst=m*19456+tile*256+block*16+element
     actual.append((src,dst));expected.append((block*256+m*16+element,dst));seen[dst]+=1
   assert actual==expected and max(x[0] for x in actual)<4096
   assert m*32+(16*1+15*15)*32<=8192
 assert np.all(seen==1)
 assert 15*32+(16+15*15)*32==8192
 assert 76*16==1216 and 76*16*16==19456


def test_single_source_delta_math_guard_and_events(tmp_path):
 original=(s.BASE/'op_kernel/d_flash_w8_a8_n256_gate.cpp').read_bytes();changed=s.transform(original)
 assert changed.replace(s.NEW.encode(),s.OLD.encode())==original
 for token in (b'GroupDelta',b'CheckGuard',b'PipeBarrier',b'WaitFlag',b'SetFlag',b'mm.End()',b'InitBuffer',b'SetQuantVector',b'SetLocalWorkspace'):
  assert changed.count(token)==original.count(token)
 a=s.materialize(tmp_path/'source');s.verify_audit(a)
 with pytest.raises(ValueError):s.transform(changed)

@pytest.fixture(scope='module')
def executable(tmp_path_factory):
 root=tmp_path_factory.mktemp('strided-model');sys.path.insert(0,str(s.BASE/'test'));import test_stage4p as prior
 model=prior.model.__wrapped__(tmp_path_factory);_,inc=model
 p=inc/'kernel_operator.h'
 with p.open('a') as f:f.write(r'''
namespace AscendC {
struct DataCopyParams{uint16_t blockCount,blockLen,srcStride,dstStride;};
inline size_t stridedCalls=0;
template<class T>void DataCopy(GlobalTensor<T>d,LocalTensor<T>src,const DataCopyParams& p){
 Require(p.blockCount==16&&p.blockLen==1&&p.srcStride==15&&p.dstStride==0);++stridedCalls;
 const size_t unit=32/sizeof(T);
 Require((p.blockCount*p.blockLen+(p.blockCount-1)*p.srcStride)*unit<=src.size);
 Require((p.blockCount*p.blockLen+(p.blockCount-1)*p.dstStride)*unit<=d.size);
 for(size_t i=0;i<p.blockCount;++i)DataCopy(d[i*(p.blockLen+p.dstStride)*unit],src[i*(p.blockLen+p.srcStride)*unit],p.blockLen*unit);
}
}
''')
 kernels={}
 for diag in (False,True):
  data=(s.BASE/'op_kernel/d_flash_w8_a8_n256_gate.cpp').read_bytes()
  if diag:
   from p_transform import kernel,J
   data=kernel((J/'op_kernel/d_flash_w8_a8_n128_gate.cpp').read_bytes(),fault=True)
  target=root/('fault.cpp' if diag else 'normal.cpp');target.write_bytes(s.transform(data))
  body=prior.BODY.replace('d_flash_w8_a8_n256_gate(', 'd_flash_w8_a8_n256_gate(')
  # Each completed tile uses16 strided API calls, physical writes remain256 blocks.
  body=body.replace('Require(ubBytes==253952);','Require(ubBytes==253952);if(fault==0)Require(stridedCalls==copies/16);stridedCalls=0;')
  main=root/('main-f.cpp' if diag else 'main-n.cpp');main.write_text('#include "'+str(target)+'"\n'+body)
  exe=root/('fault' if diag else 'normal');p=subprocess.run([shutil.which('c++'),'-std=c++17','-O2','-I'+str(inc),str(main),'-o',str(exe)],capture_output=True,text=True)
  assert p.returncode==0,p.stderr;kernels[diag]=exe
 return kernels,prior

@pytest.mark.parametrize('blocks,tile,mode',[(7,-1,0),(2,-1,0),(7,0,1),(7,37,20),(7,38,0),(7,75,20)])
def test_actual_candidate_matches_independent_partial_model(executable,tmp_path,blocks,tile,mode):
 kernels,prior=executable
 prior.test_actual_kernel_coverage_and_partial(kernels[False],tmp_path,blocks,tile,mode)


def test_fault_sweep_and_abort_path_remain_valid(executable,tmp_path):
 import struct
 kernels,_=executable
 for mode in range(21):
  out=tmp_path/str(mode);subprocess.run([str(kernels[True]),str(out),'1','0',str(mode),'0'],check=True,timeout=120)
  raw=out.read_bytes()
  if mode:
   assert raw[:32]==struct.pack('<8I',0x534e4641,mode-1,0x80000001,640,4,1,63,mode)
   assert raw[32:]==bytes([127])*(622592-32)
  else:assert np.isfinite(np.frombuffer(raw,'<f2').reshape(16,19456)[:,:256]).all()


def test_fail_closed_pairing_and_materiality():
 rows=[dict(round=i,position=j,variant=n,status='FROZEN_GOLDEN_MATCH',profiled=False,formal_latency_evidence=True,plan={},workspace_bytes=123,
 timing={k:c.q.statistics_ms([1]*30,30) for k in ('execute_sync','prepare')}) for i,o in enumerate(c.ORDER) for j,n in enumerate(o)]
 assert c.comparison(rows)['decision']=='NO_GO_NO_MATERIAL_IMPROVEMENT'
 rows[0]['profiled']=True
 with pytest.raises(ValueError):c.comparison(rows)
 with pytest.raises(ValueError):run.prior(None,'faults',{}, {})

def test_wrong_stride_or_missing_last_block_is_detected():
 correct=[m*16+b*256+x for m in range(16) for b in range(16) for x in range(16)]
 wrong=[m*16+b*(1+14)*16+x for m in range(16) for b in range(16) for x in range(16)]
 assert correct!=wrong and sorted(correct)==list(range(4096))
 assert len(set(wrong))<4096
 assert (16*1+15*16)*32+15*32>8192 # gap16 overruns final row


def test_prior_failed_or_missing_device_proof_cannot_authorize_measurement(tmp_path):
 path=tmp_path/'r.json';c.write(path,dict(status='BLOCKED',phase='complete'))
 with pytest.raises(ValueError):run.prior(path,'faults',dict(sources=[],evidence={},device=0,build_report={}),{})
 assert run.sequence('faults')==[(1,m,0) for m in range(21)]
 assert run.sequence('bounded')==[(0,m,t) for t in (0,37,38,75) for m in (1,20,0)]
