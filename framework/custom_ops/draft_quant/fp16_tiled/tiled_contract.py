"""Packing, source-level accounting and same-tiling runtime audit."""
import json
import math
from pathlib import Path
import sys
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'fp16_compare'))
from fp16_gate_common import identity,verify,write_json,SHAPE,OUTPUTS,stats,compare
from tiled_sources import PLAN_FIELDS
from run_gate_copy import launch_plan,copy_evidence


def pack_tiles(nk_record,output,n=19456,k=2560):
    path=verify(nk_record)
    if n%64 or k%256 or min(n,k)<=0 or nk_record['bytes']!=n*k*2:raise ValueError('packed tile shape/size mismatch')
    if output.exists():raise FileExistsError(output)
    dense=np.memmap(path,mode='r',dtype='<u2',shape=(n,k))
    tiles=np.memmap(output,mode='w+',dtype='<u2',shape=(n//64,k//256,64,256))
    for nt in range(n//64):
        for kt in range(k//256):tiles[nt,kt]=dense[nt*64:(nt+1)*64,kt*256:(kt+1)*256]
    tiles.flush()
    for nt in range(n//64):
        for kt in range(k//256):
            if not np.array_equal(tiles[nt,kt],dense[nt*64:(nt+1)*64,kt*256:(kt+1)*256]):raise ValueError('tile packing changed FP16 bits')
    del dense,tiles;verify(nk_record)
    return dict(status='CPU_BITWISE_PACK_VERIFIED',source=nk_record,file=identity(output),shape=[n//64,k//256,64,256],
                element_offset='((n//64)*(K//256)+(k//256))*64*256+(n%64)*256+(k%256)',npu_correctness='NOT_RUN')


def accounting():
    tiles=304*10
    return dict(scope='source-level counts and requested payload, not measured DMA traffic or device cycles',
        total_k_tiles=tiles,off=dict(bytes_per_k=16640,dma_calls_per_k=10,cast_calls_per_k=8,brcb_calls_per_k=2,mul_calls_per_k=2,pipe_v_per_k=6),
        custom_fp16=dict(bytes_per_k=32768,dma_calls_per_k=1,cast_calls_per_k=0,brcb_calls_per_k=0,mul_calls_per_k=0,pipe_v_per_k=0),
        off_weight_payload_bytes=tiles*16640,fp16_weight_payload_bytes=tiles*32768,
        common=dict(pipe_all_per_k=3,pipe_all_after_final_output_per_n=1,mte2_s_set_wait_pairs_per_k=1,mte2_v_set_wait_pairs_per_k=1,
                    startup_duplicate_and_pipe_v_per_active_core=1,codes_ub_reserved=16384,scale_ub_reserved=256,w16_ub=32768,broadcast_ub=2304,sdk_reserved=8192,
                    user_ub_budget=59904,matmul_ub_for_256k=202240),
        differences='FP16 tile packing, doubled payload and fewer DMA/scalar/vector operations + removal of dequant PIPE_V; no pure dequant-time claim',
        full_serialized_cube_tiling_comparison='NOT_AVAILABLE_IN_FROZEN_OFF_LOG; compare every recorded Cube/resource field plus byte-identical Host recipe')


def custom_plan(log,config,workspace):
    text=log.read_text()
    rows=[json.loads(line.split('DFLASH_GATE_FP16_PATH ',1)[1]) for line in text.splitlines() if 'DFLASH_GATE_FP16_PATH ' in line]
    expected=dict(layout='NT_KT_N64_K256',cast_calls_per_k=0,brcb_calls_per_k=0,mul_calls_per_k=0,gm_weight_bytes_per_k=32768,pipe_all_per_k=3)
    if not rows or any(r!=expected for r in rows):raise ValueError('missing/wrong custom FP16 data path evidence')
    # Resource-plan adapter only: never rewrite the actual log or claim that
    # reserved baseline dequant buffers execute Cast/Brcb/Mul in this kernel.
    normalized=text.replace('DFLASH_GATE_FP16_PLAN_', 'DFLASH_GROUP_QUANT_')
    for old,new in PLAN_FIELDS.items():normalized=normalized.replace('"'+new+'"','"'+old+'"')
    adapted=log.parent/'allocation-comparison.log';adapted.write_text(normalized)
    plan=launch_plan(adapted,config,workspace,'prefetch');copy_evidence(adapted,'loop')
    return dict(allocation_plan=plan,actual_data_path=expected,adapter_scope='baseline allocation comparison only; no dequant execution claim')


def require_same_plan(off,candidate):
    if off!=candidate:raise ValueError('Custom FP16 Cube/UB/blockDim/launch plan differs from current W8 Off')


def verify_execution(ex,device,timed):
    expected=dict(status='PASS',runtime='AscendCL custom FP16 ACLNN',op='DFlashGateFp16Tiled',cpu_fallback=False,input_readonly=True,guards_intact=True,
        repetitions=2,m=16,k=2560,n=19456,soc_name='Ascend310P3',device_id=device,dtype='float16',weight_layout='NT_KT_N64_K256',tile_n=64,tile_k=256)
    if any(type(ex.get(k)) is not type(v) or ex.get(k)!=v for k,v in expected.items()):raise ValueError('custom FP16 execution contract/guards differ')
    if timed:
        from a2_common import validate_execution
        validate_execution(ex,SHAPE,device,'AscendCL custom FP16 ACLNN',5,30,'continuous-v1',256)
        samples=ex['timing'].get('prepare',{}).get('samples_ms',[])
        if len(samples)!=30 or any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in samples):raise ValueError('missing custom prepare timing samples')
    else:
        t=ex.get('timing',{})
        if t.get('status')!='NOT_RUN' or t.get('warmup')!=0 or t.get('repetitions')!=0:raise ValueError('precheck must not time')
