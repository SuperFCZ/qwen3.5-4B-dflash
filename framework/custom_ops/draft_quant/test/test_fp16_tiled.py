"""Isolated tiled FP16 CPU proofs; no CANN/NPU claim."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE/'fp16_tiled'))
import tiled_sources as src
import tiled_contract as contract
import tiled_experiment as run


def generate(tmp_path):
    project=tmp_path/'project';(project/'op_host').mkdir(parents=True);(project/'op_kernel').mkdir()
    audit=src.generate(HERE,project,tmp_path/'derived')
    return project,audit


def test_source_adaptation_keeps_cube_recipe_and_buffers(tmp_path):
    kernel=(HERE/'op_kernel/d_flash_group_quant_linear.cpp').read_text();host=(HERE/'op_host/d_flash_group_quant_linear.cpp').read_text()
    project,audit=generate(tmp_path)
    derived=(project/'op_kernel/d_flash_gate_fp16_tiled.cpp').read_text()
    start='            if constexpr (StageA) mm.SetTensorA('
    assert src.section(kernel,start,'extern "C" __global__')==src.section(derived,start,'extern "C" __global__')
    start='    AscendC::TPipe pipe;';end='    AscendC::GlobalTensor<half> xGm,'
    assert src.section(kernel,start,end)==src.section(derived,start,end)
    h=(project/'op_host/d_flash_gate_fp16_tiled.cpp').read_text().replace(src.NEW,src.OLD)
    start='    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());';end='    // Account using real tile sizes/depths'
    assert src.section(host,start,end)==src.section(h,start,end)
    assert all(word not in derived for word in ('AscendC::Cast(', 'AscendC::Brcb(', 'AscendC::Mul(', 'CopyRawTile('))
    assert 'DataCopy(wLocal, wTilesGm[offset], kTileN*tileK)' in derived
    assert 'globalM!=16' not in derived and 'globalK!=2560' not in derived
    assert '{304,10,64,256}' in h and 'const int64_t m=16, k=2560, n=19456;' in h
    assert 'baseline_dequant_mode' in h and 'weight_tiles_per_n_tile' in h
    assert kernel==(HERE/'op_kernel/d_flash_group_quant_linear.cpp').read_text()
    with pytest.raises(ValueError):src.kernel_source(kernel.replace('const uint32_t bank = prefetch ?', 'const uint32_t other = prefetch ?'))


def test_packed_layout_first_last_group_and_n_tile_boundaries(tmp_path):
    values=(np.arange(128*512,dtype=np.uint32)%65536).astype('<u2').reshape(128,512)
    file=tmp_path/'nk.bin';values.tofile(file)
    result=contract.pack_tiles(contract.identity(file),tmp_path/'tiles.bin',128,512)
    packed=np.fromfile(tmp_path/'tiles.bin','<u2')
    for n in (0,63,64,127):
        for k in (0,127,128,255,256,511):
            offset=((n//64)*2+k//256)*64*256+(n%64)*256+k%256
            assert packed[offset]==values[n,k]
    assert result['npu_correctness']=='NOT_RUN'
    with pytest.raises(FileExistsError):contract.pack_tiles(contract.identity(file),tmp_path/'tiles.bin',128,512)
    record=contract.identity(file);file.write_bytes(b'changed')
    with pytest.raises(ValueError):contract.pack_tiles(record,tmp_path/'bad.bin',128,512)


def test_actual_derived_kernel_cpu_model_full_gate(tmp_path):
    compiler=shutil.which('c++')
    if not compiler:pytest.skip('no C++ compiler')
    project,audit=generate(tmp_path);binary=tmp_path/'cpu-model'
    flags=['-DDFLASH_GROUP_QUANT_K_TILE=256','-DDFLASH_GROUP_QUANT_SCALE_MODE=1','-DDFLASH_GROUP_QUANT_PIPELINE_MODE=1','-DDFLASH_GROUP_QUANT_GATE_PREFETCH_MODE=0','-DDFLASH_GROUP_QUANT_GATE_COPY_MODE=0']
    p=subprocess.run([compiler,'-std=c++17','-O2','-ffp-contract=off',*flags,'-DDERIVED_KERNEL='+json.dumps(audit['kernel']['path']),'-I'+str(HERE/'test/cpu_kernel_include'),'-I'+str(HERE/'op_kernel'),str(HERE/'fp16_tiled/cpu_model.cpp'),'-o',str(binary)],capture_output=True,text=True)
    assert p.returncode==0,p.stderr
    r=subprocess.run([str(binary)],capture_output=True,text=True,timeout=120)
    assert r.returncode==0,r.stdout+r.stderr
    assert 'NPU NOT_RUN' in r.stdout


def test_plan_comparison_is_exact_and_not_data_prep_semantics(tmp_path):
    from test_a5 import scale_launch
    from a3_launch import pipeline_plan,K_PREFIX
    log=tmp_path/'off.log';scale_launch(log,(16,2560,19456),'broadcast');lines=log.read_text().splitlines()
    a=json.loads(lines[0].split(' ',1)[1]);c=json.loads(lines[1].split(' ',1)[1])
    a.update(tile_k=256,**pipeline_plan(2560,19456,'serial',256,'off',16));a['user_ub_bytes']+=10496;a['matmul_ub_bytes']=262144-a['user_ub_bytes']
    c.update(base_k=256,weight_dequantizations_per_n_tile=10)
    kt=dict(version=1,requested_k_tile=256,selected_k_tile=256,selection_reason='configured',l1_bytes=81920,l1_capacity=1048576,l0a_bytes=16384,l0a_capacity=65536,l0b_bytes=65536,l0b_capacity=65536,l0c_bytes=4096,l0c_capacity=262144)
    text=lines[0].split(' ',1)[0]+' '+json.dumps(a)+'\n'+lines[1].split(' ',1)[0]+' '+json.dumps(c)+'\n'+lines[2]+'\n'+K_PREFIX+json.dumps(kt)+'\n'
    text+='DFLASH_GROUP_QUANT_GATE_PREFETCH {"version":1,"mode":"off","enabled":false}\nDFLASH_GROUP_QUANT_GATE_COPY {"version":1,"mode":"loop","selected":"loop"}\n'
    log.write_text(text)
    cfg=dict(core_limit=0,dequant_mode='batched',pipeline_mode='serial',kv_m80_mode='a-ub',scale_mode='broadcast',k_tile_policy='m80-synthetic-k128-v1',gate_prefetch_mode='off')
    off=contract.launch_plan(log,cfg,2097152,'prefetch')
    candidate=text.replace('DFLASH_GROUP_QUANT_','DFLASH_GATE_FP16_PLAN_')
    for k,v in src.PLAN_FIELDS.items():candidate=candidate.replace('"'+k+'"','"'+v+'"')
    candidate+='DFLASH_GATE_FP16_PATH '+json.dumps(dict(layout='NT_KT_N64_K256',cast_calls_per_k=0,brcb_calls_per_k=0,mul_calls_per_k=0,gm_weight_bytes_per_k=32768,pipe_all_per_k=3))+'\n'
    custom=tmp_path/'custom.log';custom.write_text(candidate)
    read=contract.custom_plan(custom,cfg,2097152);contract.require_same_plan(off,read['allocation_plan'])
    for key in ('block_dim','matmul_ub_bytes','user_ub_bytes'):
        changed=copy.deepcopy(read['allocation_plan']);changed[key]+=1
        with pytest.raises(ValueError):contract.require_same_plan(off,changed)
    changed=copy.deepcopy(off);changed['cube']['base_k']=128
    with pytest.raises(ValueError):contract.require_same_plan(off,changed)
    assert read['actual_data_path']['cast_calls_per_k']==0


def test_accounting_does_not_call_delta_pure_dequant():
    a=contract.accounting()
    assert a['off_weight_payload_bytes']==50585600 and a['fp16_weight_payload_bytes']==99614720
    assert a['off']['pipe_v_per_k']==6 and a['custom_fp16']['pipe_v_per_k']==0
    assert a['common']['pipe_all_per_k']==3 and a['common']['matmul_ub_for_256k']==202240
    assert 'not measured' in a['scope']


def test_failed_same_plan_never_enters_timing(tmp_path):
    args=SimpleNamespace(output_dir=tmp_path/'new',prior_report=tmp_path/'old/comparison.json',a51_evidence=tmp_path/'a51/suite.json',prefetch_evidence=tmp_path/'pair/suite.json',prepare_only=False,cann_root=tmp_path/'sdk',device_id=0,msprof='msprof')
    file=tmp_path/'dummy';file.write_bytes(b'fixture');item=contract.identity(file)
    source=dict(watched={},modes={'off':dict(config={},runner=str(file),opp=str(tmp_path/'opp'))})
    with patch.object(run,'accepted',return_value=source),patch.object(run,'reuse_prior',return_value=({'files':{'w_fp16_nk.bin':item}},{'device':{'id':0}},{})),patch.object(run,'pack_tiles',return_value={'file':item}),patch.object(run,'build',return_value=(file,{},dict(files=[]))),patch.object(run,'invoke_off',return_value=({'launch':{'block_dim':7}},b'bytes')) as off,patch.object(run,'invoke_custom',return_value=({'plan':{'allocation_plan':{'block_dim':1}}},b'bytes')) as custom:
        assert run.run(args)==1
        assert off.call_count==custom.call_count==1
    r=json.loads((args.output_dir/'comparison.json').read_text())
    assert r['performance']=='NOT_RUN' and r['npu_correctness']=='GATE_FAILED_OR_INCOMPLETE' and r['rounds']==[]


def test_success_only_crosses_custom_and_off_and_profiles_separately(tmp_path):
    args=SimpleNamespace(output_dir=tmp_path/'new',prior_report=tmp_path/'old/comparison.json',a51_evidence=tmp_path/'a51/suite.json',prefetch_evidence=tmp_path/'pair/suite.json',prepare_only=False,cann_root=tmp_path/'sdk',device_id=0,msprof='msprof')
    file=tmp_path/'dummy';file.write_bytes(b'fixture');item=contract.identity(file)
    plan={'block_dim':7};calls=[]
    source=dict(watched={},modes={'off':dict(config={},runner=str(file),opp=str(tmp_path/'opp'))})
    history=dict(device={'id':0},timing_summary={m:contract.stats([t]*180) for m,t in (('fp16',.78),('off',2.5),('raw',2.3))})
    def invoke_off(mode,folder,*rest):
        assert mode=='off';timed=rest[5];prof=len(rest)>6;calls.append(('off',timed,prof))
        return dict(launch=plan,timing=contract.stats([999. if prof else 2.5]*30)),b'golden'
    def invoke_custom(folder,*rest):
        timed=rest[6];prof=len(rest)>7;calls.append(('custom_fp16',timed,prof))
        return dict(plan={'allocation_plan':plan},timing=contract.stats([999. if prof else 1.5]*30)),b'golden'
    with patch.object(run,'accepted',return_value=source),patch.object(run,'reuse_prior',return_value=({'files':{'w_fp16_nk.bin':item}},history,{})),patch.object(run,'pack_tiles',return_value={'file':item}),patch.object(run,'build',return_value=(file,{},dict(files=[]))),patch.object(run,'invoke_off',side_effect=invoke_off),patch.object(run,'invoke_custom',side_effect=invoke_custom),patch.object(run.shutil,'which',return_value=str(file)):
        assert run.run(args)==0
    assert calls==[('off',False,False),('custom_fp16',False,False),('off',True,False),('custom_fp16',True,False),('custom_fp16',True,False),('off',True,False),('off',True,True),('custom_fp16',True,True)]
    r=json.loads((args.output_dir/'comparison.json').read_text())
    assert r['net_data_preparation_delta']['median_ms']==1 and len(r['timing_summary']['off']['samples_ms'])==60
    assert r['simulator']=='NOT_RUN' and r['dequant_only']=='NOT_RUN_SECOND_STAGE'


def test_derived_runner_cpp_syntax_without_cann(tmp_path):
    compiler=shutil.which('c++')
    if not compiler:pytest.skip('no C++ compiler')
    _,audit=generate(tmp_path);include=tmp_path/'include';(include/'acl').mkdir(parents=True)
    (include/'acl/acl.h').write_text('''#pragma once
#include <cstddef>
#include <cstdint>
using aclrtStream=void*;
constexpr int ACL_SUCCESS=0,ACL_MEM_MALLOC_HUGE_FIRST=0,ACL_MEMCPY_HOST_TO_DEVICE=1,ACL_MEMCPY_DEVICE_TO_HOST=2,ACL_FLOAT16=1,ACL_FORMAT_ND=2;
const char* aclGetRecentErrMsg(); const char* aclrtGetSocName();
int aclrtMalloc(void**,size_t,int);int aclrtFree(void*);int aclrtMemset(void*,size_t,int,size_t);int aclrtMemcpy(void*,size_t,const void*,size_t,int);
int aclInit(const char*);int aclFinalize();int aclrtSetDevice(int);int aclrtResetDevice(int);int aclrtCreateStream(aclrtStream*);int aclrtDestroyStream(aclrtStream);int aclrtSynchronizeStream(aclrtStream);
''')
    header=include/'custom.h';header.write_text('''struct aclTensor;struct aclOpExecutor;
aclTensor* aclCreateTensor(const int64_t*,uint64_t,int,const int64_t*,int64_t,int,const int64_t*,uint64_t,void*);
int aclDestroyTensor(const aclTensor*);
int aclnnDFlashGateFp16TiledGetWorkspaceSize(const aclTensor*,const aclTensor*,aclTensor*,uint64_t*,aclOpExecutor**);
int aclnnDFlashGateFp16Tiled(void*,uint64_t,aclOpExecutor*,aclrtStream);
''')
    result=subprocess.run([compiler,'-std=c++17','-fsyntax-only','-I'+str(include),'-I'+str(HERE/'test'),'-DDFLASH_NATIVE_MM_HEADER='+json.dumps(str(header)),audit['runner']['path']],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
