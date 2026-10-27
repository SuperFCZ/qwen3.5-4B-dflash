"""CPU-only Local NZ probe contracts. No CANN/NPU emulation claims."""
import copy,json,shutil,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import local_common as c
import run_stage3b2 as run
from local_review import review,RANGES,NEEDED

@pytest.fixture
def entries(tmp_path):
 old=tmp_path/'old';c.common.stage2a.make_cases(old)
 upstream=dict(stage2a_manifest=c.identity(old/'manifest.json'),stage2a_case_files=c.common.checked_cases(old/'manifest.json'))
 cases=c.reference.prepare(tmp_path/'basic',upstream,'basic')+c.reference.prepare(tmp_path/'nonpower',upstream,'nonpower')
 items=[]
 for case in cases:
  folder=Path(case['files']['a_i8.bin']['path']).parent;gold=Path(case['files']['doc_candidate_fp16.bin']['path']).read_bytes();outs=[]
  for i in range(2):p=folder/f'actual-{i}.bin';p.write_bytes(gold);outs.append(c.identity(p))
  items.append(dict(name=case['name'],case=c.identity(folder/'case.json'),outputs=outs))
 return items


def test_original_gm_bitwise_and_scale_witnesses(entries):
 for entry in entries:
  outs=[c.verify(x).read_bytes() for x in entry['outputs']]
  r=c.compare(entry,outs);assert r['status']=='CANDIDATE_BITWISE_MATCH'
  if entry['name'] in ('per_channel_powers','per_channel_blocks'):
   assert r['scale_controls']['vs_first_channel_broadcast']['bit_mismatches']>0
   assert r['scale_controls']['vs_unscaled']['bit_mismatches']>0

@pytest.mark.parametrize('mode',['unscaled','scalar','channel_shift','one_bit','repeat'])
def test_wrong_scale_or_rounding_never_pass(entries,mode):
 entry=next(e for e in entries if e['name']=='per_channel_blocks');case=json.loads(c.verify(entry['case']).read_text())
 acc,_,eff,_,gold=c.reference.verify_case(case)
 scale=np.ones(64,'<f4') if mode=='unscaled' else np.full(64,eff[0],'<f4') if mode=='scalar' else np.roll(eff,16)
 wrong=c.common.stage2a.convert(acc,scale).tobytes()
 if mode in ('one_bit','repeat'):
  bits=bytearray(gold.tobytes());bits[0]^=1;wrong=bytes(bits)
 assert c.compare(entry,[gold.tobytes() if mode=='repeat' else wrong,wrong])['status']=='FAIL_NUMERICAL'


def test_evidence_tamper_fails(entries):
 entry=entries[0];p=c.verify(entry['outputs'][0]);p.write_bytes(bytes(2048))
 with pytest.raises(ValueError,match='identity'):c.compare(entry,[bytes(2048)]*2)


def test_nz_mapping_is_bit_preserving_and_all_addresses_unique():
 # Include zero, signed zero, subnormals, NaNs: this is a bit-layout test only.
 bits=np.arange(1024,dtype='<u2').reshape(16,64);bits[0,:6]=[0,0x8000,1,0x7c00,0x7e01,0xfc00]
 nz=bits.reshape(16,4,16).transpose(1,0,2).copy()
 assert np.array_equal(c.nz_to_nd_model(nz),bits)
 src=[];dst=[]
 for nb in range(4):
  for m in range(16):
   a=nb*256+m*16;b=m*64+nb*16
   assert a*2%32==b*2%32==0
   src+=list(range(a,a+16));dst+=list(range(b,b+16))
 assert sorted(src)==sorted(dst)==list(range(1024))
 with pytest.raises(ValueError):c.nz_to_nd_model(np.zeros((16,64),'<u2'))


def execution():
 return dict(status='EXECUTION_COMPLETED_NUMERICS_UNCHECKED',op=c.NAME,soc='Ascend310P3',shape=[16,128,64],
 guards=True,readonly=True,repeat_equal=True,repetitions=2,device=0,cpu_fallback=False,layout='nd',input_count=3,
 poison_verified=True,poison_patterns=[127,255],timing=dict(status='NOT_RUN',warmup=0,repetitions=0))


def test_safety_gates():
 ex=execution();c.check_execution(ex,0)
 for key in ('guards','readonly','repeat_equal','poison_verified'):
  bad=copy.deepcopy(ex);bad[key]=False
  with pytest.raises(ValueError):c.check_execution(bad,0)
 bad=execution();bad['op']=c.common.NAME
 with pytest.raises(ValueError):c.check_execution(bad,0)
 bad=execution();bad['timing']['warmup']=5
 with pytest.raises(ValueError):c.check_execution(bad,0)


def test_smoke_stops_on_first_bad_output_without_fallback(entries,tmp_path):
 runner=tmp_path/'runner';runner.write_text('CPU fixture');bp=tmp_path/'build.json'
 evidence=dict(cases=entries);built=dict(status='STAGE3B2_BUILT_NPU_NOT_RUN',settings={},sources=[],evidence=evidence,device=0,
   installed={},opp=str(tmp_path/'opp'),runner=c.identity(runner),cann_compile='fixture',installation='fixture')
 c.write(bp,built);root=tmp_path/'smoke';root.mkdir();calls=[]
 def command(argv,log,*rest):
  calls.append(argv);c.write(log.parent/'execution.json',execution())
  log.write_text('HostC=VECIN_NZ_DT_FLOAT16 dequant=TENSOR single=16/64/128 base=16/64/128 localOutput=2048 reserve=8192')
  for i in range(2):(log.parent/f'actual-{i}.bin').write_bytes(bytes(2048))
 args=SimpleNamespace(local_build_report=bp,device=0,cann_root=tmp_path/'sdk')
 report=dict(settings={},sources=[],evidence=evidence,device=0)
 with patch.object(run,'installed',return_value={}),patch.object(run,'command',side_effect=command):
  with pytest.raises(ValueError,match='stop, no fallback'):run.smoke(args,root,report)
 assert len(calls)==1 and report['cases'][0]['status']=='FAIL_NUMERICAL'


def test_static_collection_does_not_grant_source_review(tmp_path):
 records=[]
 for i,name in enumerate(RANGES):
  src=tmp_path/f'source-{name}';src.write_text('\n'.join(NEEDED[name]));snap=tmp_path/f'snapshot-{name}';snap.write_bytes(src.read_bytes())
  records.append(dict(relative=name,source=c.identity(src),snapshot=c.identity(snap)))
 result=review(dict(files=records),tmp_path/'review')
 assert result['status'].endswith('UNRESOLVED') and result['host_tiling']=='NOT_RUN'
 assert result['source_review']=='NOT_A_SOURCE_REVIEWED_CERTIFICATE'
 Path(records[0]['snapshot']['path']).write_text('changed')
 with pytest.raises(ValueError):review(dict(files=records),tmp_path/'bad-review')


def test_actual_kernel_is_local_nz_and_has_no_manual_dequant():
 kernel=(HERE/'op_kernel/d_flash_w8_a8_local_dequant.cpp').read_text();host=(HERE/'op_host/host_common.h').read_text()
 assert 'MatmulType<TPosition::VECIN,CubeFormat::NZ,half>' in kernel
 assert 'mm.GetTensorC(nz,0,true)' in kernel and 'mm.IterateAll' not in kernel
 assert 'DataCopy(out[m*64+nb*16],nz[nb*256+m*16],16)' in kernel
 assert all(x not in kernel for x in ('Cast(', 'Mul(', 'Muls(', 'Brcb('))
 assert 'SetCType(TPosition::VECIN,CubeFormat::NZ,DataType::DT_FLOAT16)' in host
 assert 'ub-localOutput-reserve' in host
 subprocess.run(['bash','-n',str(HERE/'run.sh')],check=True)

def test_runner_cpp_syntax_only(tmp_path):
 compiler=shutil.which('c++')
 if not compiler:pytest.skip('no local compiler')
 (tmp_path/'acl').mkdir();(tmp_path/'acl/acl.h').write_text('''#pragma once
#include <cstddef>
#include <cstdint>
using aclrtStream=void*;using aclDataType=int;using aclFormat=int;
constexpr int ACL_SUCCESS=0,ACL_MEM_MALLOC_HUGE_FIRST=0,ACL_MEMCPY_HOST_TO_DEVICE=1,ACL_MEMCPY_DEVICE_TO_HOST=2,ACL_FLOAT16=1,ACL_INT8=2,ACL_UINT64=4,ACL_FORMAT_ND=2;
const char* aclGetRecentErrMsg();const char* aclrtGetSocName();
int aclrtMalloc(void**,size_t,int);int aclrtFree(void*);int aclrtMemset(void*,size_t,int,size_t);int aclrtMemcpy(void*,size_t,const void*,size_t,int);
int aclInit(const char*);int aclFinalize();int aclrtSetDevice(int);int aclrtResetDevice(int);int aclrtCreateStream(aclrtStream*);int aclrtDestroyStream(aclrtStream);int aclrtSynchronizeStream(aclrtStream);
''')
 (tmp_path/f'aclnn_{c.SNAKE}.h').write_text('''#pragma once
struct aclTensor;struct aclOpExecutor;
aclTensor* aclCreateTensor(const int64_t*,uint64_t,int,const int64_t*,int64_t,int,const int64_t*,uint64_t,void*);
int aclDestroyTensor(const aclTensor*);
int aclnnDFlashW8A8LocalDequantGetWorkspaceSize(const aclTensor*,const aclTensor*,const aclTensor*,aclTensor*,uint64_t*,aclOpExecutor**);
int aclnnDFlashW8A8LocalDequant(void*,uint64_t,aclOpExecutor*,aclrtStream);
''')
 result=subprocess.run([compiler,'-std=c++17','-fsyntax-only','-I'+str(tmp_path),'-I'+str(c.common.QUANT/'test'),str(HERE/'test/runner.cpp')],capture_output=True,text=True)
 assert result.returncode==0,result.stderr



def test_real_case_keeps_extra_references_and_full_x_quantization(tmp_path):
 x=((np.arange(16*2560)%63-31)/32).astype('<f2').reshape(16,2560);x[0,-1]=3
 nk=((np.arange(64*128)*37)%256-128).astype('i1').reshape(64,128)
 sw=np.linspace(.0003,.003,64).astype('<f2')
 arrays,detail=c.r2c.derive(x,nk,sw)
 for name,a in arrays.items():a.tofile(tmp_path/name)
 case=dict(name='layer-0-gate_up-group0-n0',shape=[16,128,64],original_shape=[16,2560,19456],group=0,
   k_range=[0,128],n_range=[0,64],codec_status=c.common.CODEC,files={n:c.identity(tmp_path/n) for n in arrays},source_files={},**detail)
 c.write(tmp_path/'case.json',case);outputs=[]
 for i in range(2):p=tmp_path/f'actual-{i}.bin';arrays['doc_candidate_fp16.bin'].tofile(p);outputs.append(c.identity(p))
 entry=dict(name='real_gate_up',case=c.identity(tmp_path/'case.json'),outputs=outputs)
 # Only fixture disk slice access is substituted; production verifier and arithmetic execute.
 with patch.object(c.r2c,'read_frozen',return_value=(x,nk,sw)):
  core=c.core_case(case);assert set(core['files'])==set(c.common.FILES)
  r=c.compare(entry,[c.verify(o).read_bytes() for o in outputs]);assert r['status']=='CANDIDATE_BITWISE_MATCH'
  case['group']=1;c.write(tmp_path/'case.json',case);entry['case']=c.identity(tmp_path/'case.json')
  with pytest.raises(ValueError,match='first K128'):c.compare(entry,[c.verify(o).read_bytes() for o in outputs])
