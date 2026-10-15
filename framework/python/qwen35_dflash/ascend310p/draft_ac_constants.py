"""Scoped immutable A/C constant binding at TorchAir save; fail closed on unknown ABI."""
from __future__ import annotations
import hashlib
from pathlib import Path
import torch
from .draft_ac_export import A,C,AUDIT,GE,POLICY
from .weight_quant_layout import _dtype,_resolved_desc,_storage_desc,_nz_shape,_source
from .weight_prepack import pack_int8_nz,_const_value_desc,CONST_SHAPE_LOCK
from .utils import atomic_write_json


def _bound(nodes,edge,values,dtype,shape):
    node,port=_source(nodes,edge); tensor=values.get(edge)
    if port or node.type not in ('Const','FileConstant') or node.input or tensor is None or tensor.dtype!=dtype or list(tensor.shape)!=list(shape):
        raise ValueError(f'A/C requires immutable typed tensor at {edge}; runtime weights/views are forbidden')
    desc=_resolved_desc(nodes,edge,{edge:dict(dtype='DT_INT8' if dtype==torch.int8 else 'DT_FLOAT16',shape=list(shape))})
    if desc.layout not in ('','ND'): raise ValueError('logical A/C constant must start as ND')
    if node.type=='FileConstant': desc.layout='ND' # Canonicalize the proven logical descriptor before snapshotting.
    if node.type=='Const':
        value_desc=node.attr['value'].t.desc
        if _dtype(value_desc) not in ('DT_UNDEFINED',_dtype(desc)) or (list(value_desc.shape.dim) and list(value_desc.shape.dim)!=list(shape)):
            raise ValueError('immutable Const value descriptor conflicts with bound tensor')
        value_desc.CopyFrom(desc)
    node.output_desc[0].CopyFrom(desc)
    return node,tensor,desc


def _verify_payload(node,tensor,directory,expected=None):
    from .draft_ac_payload import verify_payload
    result=verify_payload(node,tensor,directory,expected)
    if node.type=='FileConstant':
        # GE/ATC must read the exact verified file, independent of its cwd.
        node.attr['file_path'].s=result['path'].encode('utf-8')
    return result


def normalize_ac_constants(graph,immutable,config,directory,*,expected_hashes=None):
    if config.get('policy')!=POLICY or config.get('mode') not in ('off','a','c','ac'): raise ValueError('unknown A/C save policy')
    expected_hashes=expected_hashes or {}
    directory=Path(directory); nodes={op.name:op for op in graph.op}
    selected=[op for op in graph.op if op.type in GE.values()]
    expected={GE[A]:config['a_count'],GE[C]:config['c_count'],GE[AUDIT]:config.get('audit_count',0)}
    if any(sum(op.type==kind for op in selected)!=count for kind,count in expected.items()): raise ValueError('A/C exact node count differs')
    records=[]; used=set()
    for op in selected:
        required=3 if op.type==GE[A] else 2
        if len(op.input)!=required or len(op.input_desc)!=required: raise ValueError('A/C port arity differs')
        if op.type==GE[A]:
            weight=immutable.get(op.input[1])
            if weight is None or weight.ndim!=2: raise ValueError('A weight is not immutable NK')
            n,k=map(int,weight.shape)
            if k%128 or n%16: raise ValueError('A group/N alignment differs')
            wn,q,wd=_bound(nodes,op.input[1],immutable,torch.int8,[n,k])
            sn,s,sd=_bound(nodes,op.input[2],immutable,torch.float16,[k//128,n])
            if not s.is_contiguous(): raise ValueError('A GN scale must have contiguous GN bytes')
            scale_payload=_verify_payload(sn,s,directory,expected_hashes.get(op.input[2]))
            # Read/check source first. Create a NEW Const so any other consumer
            # still sees the original ND tensor; never relabel its ND bytes.
            source_payload=_verify_payload(wn,q,directory,expected_hashes.get(op.input[1]))
            packed=pack_int8_nz(q).cpu().contiguous(); payload=packed.numpy().tobytes()
            restored=packed.reshape(k//32,n//16,16,32).permute(1,2,0,3).reshape(n,k)
            if not torch.equal(restored,q.detach().cpu()): raise ValueError('A NZ inverse is not bit-exact')
            desc=_storage_desc(wd,layout='FRACTAL_NZ',shape=_nz_shape(n,k),origin_shape=[n,k]); desc.name='w_nz'
            nz=graph.op.add(); nz.name=op.name+'_ac_nz'; nz.type='Const'
            if nz.name in nodes: raise ValueError('duplicate A NZ constant')
            nz.output_desc.add().CopyFrom(desc); nz.attr['value'].t.desc.CopyFrom(_const_value_desc(desc))
            nz.attr['value'].t.data=payload; nz.attr[CONST_SHAPE_LOCK].b=True
            op.input[1]=nz.name+':0'; op.input_desc[1].CopyFrom(desc); op.input_desc[2].CopyFrom(sd); op.input_desc[2].name='s'
            used.add(wn.name)
            records.append(dict(name=op.name,ge_type=op.type,n=n,k=k,weight_sha256=source_payload['sha256'],
                                nz_sha256=hashlib.sha256(payload).hexdigest(),nz_bytes=len(payload),storage_shape=list(packed.shape),
                                scale_gn_sha256=scale_payload['sha256'],scale_shape=list(s.shape),roundtrip='BIT_EXACT'))
        else:
            if config.get('runtime_head'):
                if config.get('scope')!='single-op-C' or config['a_count'] or config['c_count']+config.get('audit_count',0)!=1:
                    raise ValueError('runtime head is restricted to the explicit single-op graph probe')
                wn,port=_source(nodes,op.input[1])
                if wn.type!='Data' or port or _dtype(wn.output_desc[0])!='DT_FLOAT16' or list(wn.output_desc[0].shape.dim)!=[248320,2560]:
                    raise ValueError('runtime C weight ABI differs')
                records.append(dict(name=op.name,ge_type=op.type,head_weight=dict(kind='runtime-readonly',sha256=config['head_weight_sha256'])))
                continue
            wn,w,wd=_bound(nodes,op.input[1],immutable,torch.float16,[248320,2560])
            payload=_verify_payload(wn,w,directory,expected_hashes.get(op.input[1]))
            op.input_desc[1].CopyFrom(wd); op.input_desc[1].name='weight'
            records.append(dict(name=op.name,ge_type=op.type,head_weight=payload))
    # No dead ND copies retained after replacement. Shared/control uses survive.
    referenced={edge.rpartition(':')[0] for node in graph.op for edge in node.input if edge}
    for i in range(len(graph.op)-1,-1,-1):
        if graph.op[i].name in used-referenced: del graph.op[i]
    from collections import Counter
    projections=config.get('projections',[])
    if config['a_count']:
        expected_pairs=Counter((r['n'],r['k'],r['weight_sha256'],r['scale_gn_sha256']) for r in projections)
        actual=Counter((r['n'],r['k'],r['weight_sha256'],r['scale_gn_sha256']) for r in records if r['ge_type']==GE[A])
        if actual!=expected_pairs or len(projections)!=config['a_count']: raise ValueError('A checkpoint/projection inventory differs')
    if graph.ByteSize()>(2048-200)*1024*1024: raise ValueError('A/C graph exceeds protobuf budget; no runtime-weight fallback')
    report=dict(policy=POLICY,status='PASS',scope='immutable constants/descriptor audit; GE/ATC execution NOT_RUN',counts=expected,nodes=records)
    atomic_write_json(directory/'ac-constants.json',report)
    return report


def validate_ac_graph(graph):
    config=graph.get('metadata',{}).get('draft_ac')
    if config is None: return
    if config.get('policy')!=POLICY: raise ValueError('unknown A/C graph policy')
    audit=graph.get('runtime_input_abi',{}).get('draft_ac',{})
    expected={GE[A]:config['a_count'],GE[C]:config['c_count'],GE[AUDIT]:config.get('audit_count',0)}
    if audit.get('status')!='PASS' or audit.get('policy')!=POLICY or audit.get('counts')!=expected: raise ValueError('missing exact A/C immutable audit')
    observed={r['ge_op_type']:r['ge_node_occurrences'] for r in graph.get('custom_op_audit',[])}
    if any(observed.get(kind,0)!=count for kind,count in expected.items()): raise ValueError('A/C GE audit count mismatch')
    # All 26 sources are required in full-Draft mode; diagnostic single-op specs
    # carry their own exact inventory. No global native count rule is weakened.
    if config.get('scope','').startswith('static real Draft'):
        from .incremental_plan import expected_signatures
        expected_abi=expected_signatures(graph['metadata']['source_incremental_contract'])['draft']
        expected_abi['inputs'][0]['shape'][1]=config['gear']
        if graph['metadata']['tensor_abi']!=expected_abi:
            raise ValueError('A/C public ABI no longer matches the frozen contract')
        if graph['input_names']!=[t['name'] for t in expected_abi['inputs']] or graph['output_names']!=[t['name'] for t in expected_abi['outputs']]:
            raise ValueError('A/C public input/output order changed')
        native=observed.get('WeightQuantBatchMatmulV2',0)
        if native+expected[GE[A]]!=26 or len(graph['input_names'])!=15 or len(graph['output_names'])!=11:
            raise ValueError('full Draft projection or public state ABI changed')
