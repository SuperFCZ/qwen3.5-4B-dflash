"""Explicit A/C graph frontends. Export-only: no CPU or eager numerical fallback.

Normal factories never import/register these operators. A tensors remain logical NK
until the save hook binds immutable bytes to an audited physical NZ constant.
"""
from __future__ import annotations
import copy
from dataclasses import replace
import hashlib
import inspect
import json
import threading
import torch
from torch import nn
from .contracts import AirGraphSpec,CustomOpExportSpec

A='dflash_ac::group_quant_linear'; C='dflash_ac::head_top1'; AUDIT='dflash_ac::head_audit'
GE={A:'DFlashGroupQuantLinear',C:'DFlashDraftLmHeadTop1',AUDIT:'DFlashDraftLmHeadTop1Audit'}
POLICY='dflash-ac-immutable-graph-v1'
_LIBS=[]; _LOCK=threading.Lock()


def tensor_hash(value):
    value=value.detach().cpu().contiguous()
    return hashlib.sha256(memoryview(value.numpy()).cast('B')).hexdigest()


def linear_meta(x,weight,scale):
    if x.dtype!=torch.float16 or weight.dtype!=torch.int8 or scale.dtype!=torch.float16 or x.ndim!=2 or weight.ndim!=2:
        raise ValueError('A requires FP16 [M,K], immutable INT8 [N,K], FP16 contiguous [K/128,N]')
    n,k=weight.shape
    if k%128 or n%16 or x.shape[1]!=k or scale.shape!=(k//128,n) or not scale.is_contiguous(): raise ValueError('A shape/GN/group128 mismatch')
    if isinstance(x.shape[0],int) and x.shape[0] not in (16,32,64,80): raise ValueError('unsupported A M')
    return x.new_empty((x.shape[0],n))


def head_meta(hidden,weight,audit=False):
    if hidden.dtype!=torch.float16 or weight.dtype!=torch.float16 or hidden.ndim!=2 or hidden.shape[1]!=2560 or weight.shape!=(248320,2560):
        raise ValueError('C full-vocabulary FP16 ABI mismatch')
    if isinstance(hidden.shape[0],int) and not 1<=hidden.shape[0]<=15: raise ValueError('C M must be 1..15')
    ids=hidden.new_empty((hidden.shape[0],),dtype=torch.int64)
    return (ids,hidden.new_empty((hidden.shape[0],248320))) if audit else ids


def ensure_ops():
    with _LOCK:
        if _LIBS: return
        lib=torch.library.Library('dflash_ac','DEF')
        lib.define('group_quant_linear(Tensor x, Tensor weight, Tensor scale) -> Tensor')
        lib.define('head_top1(Tensor hidden, Tensor weight) -> Tensor')
        lib.define('head_audit(Tensor hidden, Tensor weight) -> (Tensor, Tensor)')
        torch.library.register_fake(A)(linear_meta)
        torch.library.register_fake(C)(head_meta)
        torch.library.register_fake(AUDIT)(lambda h,w:head_meta(h,w,True))
        def unavailable(*args,**kwargs):
            raise RuntimeError('dflash_ac is export-only; execute its compiled OM. No eager/CPU fallback is installed')
        # A receiver that insists on eager execution must fail explicitly.
        for name in ('group_quant_linear','head_top1','head_audit'): lib.impl(name,unavailable,'PrivateUse1')
        _LIBS.append(lib)


def converter_capabilities(spec,torchair):
    """Inspect without emitting GE nodes; signature support is not GE/ATC PASS."""
    from .custom_op_export import _custom_op_call_mode
    if spec.torch_op not in GE or spec.ge_op_type!=GE[spec.torch_op] or spec.overload!='default':
        raise ValueError('unknown A/C ABI')
    def describe(fn):
        info=dict(callable=callable(fn),module=getattr(fn,'__module__',None),name=getattr(fn,'__qualname__',None))
        signature=None
        if callable(fn):
            try: signature=inspect.signature(fn); info['signature']=str(signature)
            except (TypeError,ValueError) as e: info['signature_error']=str(e)
            try: info['wrapper_signature']=str(inspect.signature(fn,follow_wrapped=False))
            except (TypeError,ValueError): pass
            try: info['source_file']=inspect.getsourcefile(fn)
            except TypeError: pass
        return info,signature
    call=getattr(getattr(torchair,'ge',None),'custom_op',None)
    registrar=getattr(torchair,'register_fx_node_ge_converter',None)
    call_info,signature=describe(call); reg_info,reg_signature=describe(registrar)
    names=['x','w_nz','s'] if spec.torch_op==A else ['hidden','weight']
    outputs=['token_id','logits'] if spec.torch_op==AUDIT else ['y' if spec.torch_op==A else 'token_id']
    report=dict(torchair_version=str(getattr(torchair,'__version__','unknown')),torchair_file=getattr(torchair,'__file__',None),
                torch_target=spec.torch_target,ge_op_type=spec.ge_op_type,input_ports=names,output_ports=outputs,
                custom_op=call_info,registrar=reg_info,errors=[],scope='API capability only; GE/ATC/NPU NOT_RUN')
    if not callable(call): report['errors'].append('missing callable torchair.ge.custom_op')
    if not callable(registrar): report['errors'].append('missing callable torchair.register_fx_node_ge_converter')
    if callable(call):
        mode=_custom_op_call_mode(call); report['legacy_call_mode']=mode
        # A wrapper may have *args AND explicit keyword ports. Do not mistake
        # this documented named interface for registered-IR tensor varargs.
        if signature and {'inputs','outputs','attrs'}<=signature.parameters.keys(): mode='named-only'
        report['selected_call_mode']=mode
        if signature:
            try:
                if mode=='named-only': signature.bind(spec.ge_op_type,inputs=dict.fromkeys(names),outputs=outputs,attrs={})
                else: signature.bind(spec.ge_op_type,*([None]*len(names)))
            except TypeError as e: report['errors'].append(f'unsupported ge.custom_op signature for {mode}: {e}')
        else:
            report['signature_policy']='uninspectable; existing framework registered-IR positional convention; no retry'
    if reg_signature:
        try: reg_signature.bind(None)
        except TypeError as e: report['errors'].append(f'unsupported converter registrar signature: {e}')
    report['status']='UNSUPPORTED' if report['errors'] else 'CALL_FORM_SELECTED'
    return report


def prepare_export(spec,torchair):
    from .custom_op_export import CustomOpExportSession
    diagnostic=converter_capabilities(spec,torchair)
    def fail(message,error=None):
        raise RuntimeError(message+'; A/C interface='+json.dumps(diagnostic,sort_keys=True)) from error
    if diagnostic['errors']: fail('; '.join(diagnostic['errors']))
    ensure_ops(); op=getattr(torch.ops.dflash_ac,spec.torch_op.split('::')[1]).default
    call=torchair.ge.custom_op; registrar=torchair.register_fx_node_ge_converter
    mode=diagnostic['selected_call_mode']; names=diagnostic['input_ports']; outputs=diagnostic['output_ports']
    session=CustomOpExportSession(spec,str(op._schema),'framework-meta-only',mode,'framework-registered-ge-ir')
    def emit(args):
        session.converter_calls+=1
        try:
            # Registered IR supplies the exact frozen port schema for positional
            # calls. No attrs, casts, native replacement or alternate retry.
            result=(call(spec.ge_op_type,*args) if mode=='registered-ir-positional' else
                    call(spec.ge_op_type,inputs=dict(zip(names,args)),outputs=list(outputs),attrs={}))
        except Exception as e: fail(f'{spec.ge_op_type} GE emission failed using {mode}: {e}',e)
        if len(outputs)==1:
            if isinstance(result,(list,tuple)):
                if len(result)!=1: fail('custom GE output arity differs: expected '+str(outputs))
                result=result[0]
            if result is None: fail('custom GE output is missing: '+str(outputs))
        elif not isinstance(result,(list,tuple)) or len(result)!=len(outputs) or any(v is None for v in result):
            fail('custom GE output arity differs: expected '+str(outputs))
        return result
    # Explicit schema signatures reject missing/extra/unknown inputs rather
    # than letting zip silently truncate; FX keyword calls preserve port order.
    if spec.torch_op==A:
        def converter(x,weight,scale,meta_outputs=None):
            return emit((x,weight,scale))
    else:
        def converter(hidden,weight,meta_outputs=None):
            return emit((hidden,weight))
    try:
        decorator=registrar(op)
    except Exception as e: fail(f'converter registration failed for {spec.torch_target}: {e}',e)
    if not callable(decorator): fail('converter registrar did not return a callable decorator')
    try: decorator(converter)
    except Exception as e: fail(f'converter registration failed for {spec.torch_target}: {e}',e)
    return session


class ACLinear(nn.Module):
    def __init__(self,original):
        super().__init__()
        if original.bits!=8 or original.matmul_backend!='weight_quant' or original.group_size!=128: raise ValueError('only frozen W8 Group128 linears')
        self.register_buffer('weight',original.qweight)
        self.register_buffer('scale',original.scales.t().contiguous()) # GN bytes, not NG-backed view
        self.k=original.in_features; self.n=original.out_features
    def forward(self,value):
        result=torch.ops.dflash_ac.group_quant_linear.default(value.reshape(-1,self.k),self.weight,self.scale)
        return result.reshape(*value.shape[:-1],self.n)


class ACOps:
    """Private proxy on DraftGraph only; no mutation of shared Target/layer ops."""
    def __init__(self,original): self.original=original
    def __getattr__(self,name): return getattr(self.original,name)
    def top1(self,hidden,weight):
        if hidden.ndim!=3 or hidden.shape[:2]!=(1,15): raise ValueError('Draft C requires exactly the 15 post-anchor rows')
        return torch.ops.dflash_ac.head_top1.default(hidden.reshape(15,2560).contiguous(),weight).reshape(1,15)


def projection_modules(graph):
    if len(graph.layers)!=5: raise ValueError('A/C requires the frozen five-layer W8 Draft')
    result=[('fc',graph,'fc')]
    for i,layer in enumerate(graph.layers):
        result.extend((f'layer-{i}-{kind}',layer,attr) for kind,attr in
                      (('kv','kv_linear'),('q','q_proj'),('o','o_proj'),('gate_up','gate_up_linear'),('down','down_proj')))
    return result


def apply_mode(graph,mode):
    if mode not in ('off','a','c','ac'): raise ValueError('mode must be off/a/c/ac')
    if mode=='off': return []
    from .incremental import DraftGraph
    if not isinstance(graph,DraftGraph): raise ValueError('must wrap the original DraftGraph, not runtime-weight functional_call')
    ensure_ops(); records=[]
    if mode in ('a','ac'):
        for name,parent,attr in projection_modules(graph):
            original=getattr(parent,attr); wrapped=ACLinear(original)
            records.append(dict(name=name,n=wrapped.n,k=wrapped.k,weight_sha256=tensor_hash(wrapped.weight),
                                scale_gn_sha256=tensor_hash(wrapped.scale),group_size=128))
            setattr(parent,attr,wrapped)
    if mode in ('c','ac'): graph.ops=ACOps(graph.ops)
    return records


def integration_spec(spec,mode,example_args,gear):
    """Draft-only diagnostic envelope; keep original forward/ABI, never rewrite Target/Verify.

    Intentionally not a production deployment plan. Full-model runner adoption is
    gated separately after these static Draft OMs pass bitwise frozen replay.
    """
    if spec.name!='draft' or spec.constant_input_names or gear not in (16,64): raise ValueError('requires immutable-NZ Draft and static gear')
    if spec.metadata.get('draft_quantization')!='w8a16' or not spec.metadata.get('draft_weight_storage'): raise ValueError('requires existing W8 offline-NZ source spec')
    if len(example_args)!=len(spec.input_names): raise ValueError('Draft input arity differs')
    from .incremental_plan import expected_signatures
    expected=expected_signatures(spec.metadata['incremental_contract'])['draft']
    expected['inputs'][0]['shape'][1]=gear
    actual=dict(inputs=[dict(name=n,dtype=str(t.dtype).removeprefix('torch.'),shape=list(t.shape)) for n,t in zip(spec.input_names,example_args)],
                outputs=copy.deepcopy(spec.metadata['tensor_abi']['outputs']))
    if actual!=expected or tuple(t['name'] for t in expected['outputs'])!=spec.output_names:
        raise ValueError('static Draft ABI differs from original incremental contract')
    projections=apply_mode(spec.model,mode)
    native=tuple(o for o in spec.custom_ops if not (mode in ('a','ac') and o.ge_op_type=='WeightQuantBatchMatmulV2'))
    custom=list(native)
    if mode in ('a','ac'): custom.append(CustomOpExportSpec(A,GE[A],minimum_occurrences=26))
    if mode in ('c','ac'): custom.append(CustomOpExportSpec(C,GE[C]))
    meta={k:copy.deepcopy(v) for k,v in spec.metadata.items() if k not in
          ('incremental_contract','dynamic_input_axes','draft_compile_policy','custom_op_export_contracts','custom_op_export_contract')}
    if mode in ('a','ac'):
        for key in ('draft_weight_storage','draft_weight_prepack_manifest','draft_weight_prepack'): meta.pop(key,None)
    meta['source_incremental_contract']=copy.deepcopy(spec.metadata['incremental_contract'])
    meta['tensor_abi']=dict(inputs=[dict(name=n,dtype=str(t.dtype).removeprefix('torch.'),shape=list(t.shape))
                                    for n,t in zip(spec.input_names,example_args)],outputs=copy.deepcopy(spec.metadata['tensor_abi']['outputs']))
    for out in meta['tensor_abi']['outputs']:
        if out['name']=='draft_top1': out['shape']=[1,15]
    meta['custom_op_export_contracts']=[dict(torch_target=o.torch_target,ge_op_type=o.ge_op_type,minimum_occurrences=o.minimum_occurrences) for o in custom]
    meta['draft_ac']=dict(policy=POLICY,mode=mode,gear=gear,a_count=26 if mode in ('a','ac') else 0,
                         c_count=1 if mode in ('c','ac') else 0,audit_count=0,projections=projections,
                         scope='static real Draft diagnostic; NOT production plan / NOT end-to-end')
    return replace(spec,name=f'draft_ac_{mode}_c{gear}',role='diagnostic',dynamic=False,example_args=tuple(example_args),metadata=meta,custom_ops=tuple(custom))
