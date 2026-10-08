#!/usr/bin/env python3
"""Capture production post-final-norm hidden rows and original FP16 head bytes."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
from common import ABI,K,N,REPO,record,sha256,write_json
from a2_capture import frozen_inputs,embedding_rows,require_draft_row_update
sys.path[:0]=[str(REPO/'framework/python'),str(REPO)]

def normalized_rows(result,torch):
    if tuple(result.shape)!=(1,16,K) or result.dtype!=torch.float16 or result.device.type!='npu':
        raise ValueError('final norm must be NPU FP16 [1,16,2560]')
    hidden=result[:,1:,:].detach().contiguous()
    if not torch.isfinite(hidden).all().item(): raise ValueError('nonfinite real hidden')
    return hidden


def save_head(target,requested,embedding_key,destination):
    import torch
    from safetensors import safe_open
    index=target/'model.safetensors.index.json'
    if index.is_file():
        mapping=json.loads(index.read_text())['weight_map']
    else:
        with safe_open(target/'model.safetensors',framework='pt',device='cpu') as f:
            mapping={key:'model.safetensors' for key in f.keys()}
    names=[key for key in mapping if key=='lm_head.weight' or key.endswith('.lm_head.weight')]
    config=json.loads((target/'config.json').read_text())
    tied=config.get('text_config',config).get('tie_word_embeddings',False)
    key=requested or (names[0] if len(names)==1 else (embedding_key if not names and tied else None))
    if key is None or key not in mapping or (key not in names and not (tied and key==embedding_key)):
        raise ValueError('head tensor ambiguous/missing; specify actual lm_head key; no untied embedding substitution')
    shard=(target/mapping[key]).resolve()
    if not shard.is_relative_to(target): raise ValueError('head shard escapes checkpoint')
    with safe_open(shard,framework='pt',device='cpu') as handle, destination.open('wb') as out:
        tensor=handle.get_slice(key)
        if tensor.get_shape()!=[N,K]: raise ValueError('head must cover full [248320,2560] vocabulary')
        for first in range(0,N,1024):
            original=tensor[first:min(first+1024,N)]
            if not torch.is_floating_point(original): raise ValueError('head must be original floating weights, not integer codes')
            part=original.to(torch.float16).contiguous()
            if not torch.isfinite(part).all().item(): raise ValueError('nonfinite real checkpoint head')
            out.write(part.numpy().tobytes())
    return dict(shape=[N,K],dtype='float16',layout='fp16_nk',tensor=key,tied_embedding=key==embedding_key,
                shard=str(shard),shard_sha256=sha256(shard),config_sha256=sha256(target/'config.json'))


def capture(args):
    if os.environ.get('ASCEND310P_SIMULATION_ONLY')=='1': raise ValueError('real NPU capture required')
    import torch
    import torch_npu
    from models.dflash_v1.draft_quantization import load_quantized_draft
    from models.dflash_v1.modeling_dflash import DFlashDraftModel
    from qwen35_dflash.ascend310p.incremental import DraftGraph
    from qwen35_dflash.ascend310p.quant_factory import AirDFlashOps
    from qwen35_dflash.ascend310p.utils import require_run_output
    root=require_run_output(args.output_dir); root.mkdir(parents=True,exist_ok=False)
    report=dict(abi=ABI,status='RUNNING',runtime='native NPU DraftGraph final norm capture',cpu_fallback=False,
                gears={},repeat_equal=False,input_readonly=False,native_head_execution='NOT_RUN',
                custom_head_validation='NOT_RUN',full_draft_validation='NOT_RUN',end_to_end='NOT_RUN')
    path=root/'manifest.json'; write_json(path,report)
    try:
        device=f'npu:{args.device_id}'; torch.npu.set_device(device)
        if '310P' not in torch.npu.get_device_name(args.device_id).upper(): raise ValueError('requires Ascend310P')
        update=require_draft_row_update(torch)
        head=save_head(args.target_dir.resolve(),args.head_key,args.embedding_key,root/'head.bin')
        report['weight']=dict(head,file=record(root/'head.bin',root))
        for gear in (16,64):
            draft=load_quantized_draft(DFlashDraftModel,args.draft_dir,variant='w8a16',ops=AirDFlashOps(quant_matmul_backend='weight_quant'),
                                      device=device,dtype=torch.float16)
            if (draft.config.hidden_size,draft.config.vocab_size,draft.config.num_hidden_layers)!=(K,N,5):
                raise ValueError('pinned W8 Draft shape differs')
            names,arrays,layers,source=frozen_inputs(getattr(args,f'c{gear}_replay_report'),draft.config,args.feature_layers)
            if source['context_rows']!=gear: raise ValueError('need actual C16 and C64 frozen inputs')
            embedding,embedding_source=embedding_rows(args.target_dir,args.embedding_key,[source['anchor'],draft.config.mask_token_id],draft.config)
            anchor,mask=source['anchor'],draft.config.mask_token_id
            class FrozenEmbedding(torch.nn.Module):
                def __init__(self):
                    super().__init__(); self.register_buffer('rows',embedding.to(device))
                def forward(self,ids):
                    if not ((ids==anchor)|(ids==mask)).all().item(): raise ValueError('unexpected embedding token')
                    return self.rows[(ids!=anchor).long()]
            graph=DraftGraph(draft,FrozenEmbedding(),torch.nn.Identity(),row_update=update,consume_source=True,feature_layers=layers).eval()
            inputs=tuple(torch.from_numpy(a.copy()).to(device) for a in arrays)
            captured=[]
            class Complete(Exception): pass
            def hook(module,values,result):
                hidden=normalized_rows(result,torch)
                captured.append(hidden.cpu().numpy().tobytes()); raise Complete()
            handle=graph.norm.register_forward_hook(hook)
            try:
                with torch.inference_mode():
                    for repeat in range(2):
                        try: graph(*inputs)
                        except Complete: pass
                        torch.npu.synchronize()
                        if len(captured)!=repeat+1: raise ValueError('final norm hook did not execute exactly once')
                        for name,value in zip(names,inputs):
                            if hashlib.sha256(value.cpu().contiguous().numpy().tobytes()).hexdigest()!=source['replay_input_sha256'][name]:
                                raise ValueError('frozen input modified')
            finally: handle.remove()
            if captured[0]!=captured[1]: raise ValueError('hidden capture repeat drift')
            file=root/f'c{gear}-hidden.bin'; file.write_bytes(captured[0])
            report['gears'][f'c{gear}']=dict(source=source,embedding=embedding_source,checkpoint=draft.draft_quantization_audit,
                hidden=record(file,root),hidden_repeat_sha256=hashlib.sha256(captured[1]).hexdigest())
            del inputs,graph,draft,embedding; gc.collect(); torch.npu.empty_cache()
        for key in ('config_sha256','model_sha256'):
            if report['gears']['c16']['checkpoint'][key]!=report['gears']['c64']['checkpoint'][key]: raise ValueError('checkpoint drift')
        if report['gears']['c16']['source']['feature_layers']!=report['gears']['c64']['source']['feature_layers']:
            raise ValueError('feature order changed between gears')
        report.update(status='PASS',repeat_equal=True,input_readonly=True,environment=dict(torch=str(torch.__version__),torch_npu=str(torch_npu.__version__)))
    except Exception as error:
        report.update(status='FAIL',error=f'{type(error).__name__}: {error}'); raise
    finally: write_json(path,report)
    print(f'PASS: Head capture only; custom/native OM NOT_RUN; {path}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('draft-dir','target-dir','c16-replay-report','c64-replay-report','output-dir'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--feature-layers',type=lambda s:tuple(map(int,s.split(','))))
    p.add_argument('--embedding-key',default='model.language_model.embed_tokens.weight')
    p.add_argument('--head-key'); p.add_argument('--device-id',type=int,default=0)
    capture(p.parse_args())
if __name__=='__main__': main()
