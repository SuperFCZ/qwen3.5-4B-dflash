"""Optional, strict frozen Native full-K semantics audit. Never executes an OM."""
import io
import json
from pathlib import Path
import sys
from s3a_reference import e2c,common,identity,verify

REPO=common.HERE.parents[3]
SOURCE_FILES=('framework/custom_ops/draft_quant/test/a2_export.py','framework/custom_ops/draft_quant/test/a2_common.py',
 'framework/custom_ops/draft_quant/test/native_om.cpp','framework/custom_ops/draft_quant/fp16_compare/fp16_gate_evidence.py',
 'models/dflash_v1/weight_quant_matmul.py')

class Unknown(ValueError):pass
class Mismatch(ValueError):pass


def single(v):
    if isinstance(v,list):
        if len(v)!=1:raise Unknown('ambiguous repeated graph value')
        return v[0]
    return v


def attributes(node):
    from profile_pbtxt import read
    attrs=node.get('attributes',[])
    if not isinstance(attrs,list):raise Unknown('unsupported graph attribute representation')
    result={}
    for a in attrs:
        key=single(a.get('key'));v=single(a.get('value'))
        if key in result:raise Unknown('duplicate graph attribute')
        if isinstance(v,dict) and set(v)=={'s'}:
            text=single(v['s'])
            if isinstance(text,str) and len(text)<4096 and text.lstrip().startswith(('b:','i:')):v=read(io.StringIO(text))
        if isinstance(v,dict) and len(v)==1 and next(iter(v)) in ('b','i','s'):
            field=next(iter(v));v=single(next(iter(v.values())))
            if field=='b':
                if v in ('true','false'):v=v=='true'
                elif type(v) is int and v in (0,1):v=bool(v)
                elif type(v) is not bool:raise Unknown('invalid boolean attribute '+str(key))
        result[key]=v
    return result


def direct_desc(node):
    values=node.get('output_desc',[])
    if not values:
        values=[d for d in node.get('serialized_descriptors',[]) if single(d.get('name'))=='y']
    if len(values)!=1:raise Unknown('missing/ambiguous explicit output descriptor: '+node['name'])
    d=values[0];shape=single(d.get('shape',{}))
    if not isinstance(shape,dict) or 'dim' not in shape:raise Unknown('missing output dims: '+node['name'])
    return single(d.get('dtype')),shape['dim'],single(d.get('layout'))


def pure_linear_graph(nodes,weight_name,weight_edge=None,scale_edge=None):
    sys.path.insert(0,str(common.OPS/'draft_ac/test'))
    from c_profile_graph import edge_name
    table={n['name']:n for n in nodes}
    if len(table)!=len(nodes):raise Unknown('duplicate graph node names')
    outputs=[n for n in nodes if n['type']=='NetOutput'];weights=[n for n in nodes if n['type']=='WeightQuantBatchMatmulV2']
    if len(outputs)!=1 or len(weights)!=1:raise Mismatch('not exactly one output and one WeightQuant projection')
    w=weights[0]
    if w['name']!=weight_name:raise Mismatch('graph WeightQuant node differs from layout audit')
    data=lambda n:[x for x in n.get('inputs',[]) if x and not str(x).startswith('^') and not str(x).endswith(':-1')]
    tail=data(outputs[0])
    if len(tail)!=1:raise Mismatch('Native has multiple output values')
    cursor=edge_name(tail[0]);chain=[];seen=set()
    while cursor!=w['name']:
        if cursor in seen or cursor not in table:raise Unknown('output dependency chain missing/cyclic')
        seen.add(cursor);n=table[cursor];chain.append(dict(name=cursor,type=n['type']))
        if n['type'] not in ('Identity','Reshape'):
            if n['type'] in ('Add','AddV2','Mul','Silu','SiLU','Gelu','GELU','Relu','Sigmoid','Tanh'):raise Mismatch('post-projection operation: '+n['type'])
            raise Unknown('unsupported output dependency operation: '+n['type'])
        dtype,shape,_=direct_desc(n)
        if dtype!='DT_FLOAT16' or any(type(d) is not int or d<=0 for d in shape):raise Mismatch('post-projection dtype/shape differs')
        import math
        if math.prod(shape)!=16*19456:raise Mismatch('reshape changes projection element count')
        edges=data(n)
        if not edges:raise Unknown('view node has no data input')
        cursor=edge_name(edges[0])
    dtype,shape,layout=direct_desc(w)
    if dtype is None or layout is None:raise Unknown('missing explicit WeightQuant output dtype/layout')
    if dtype!='DT_FLOAT16' or shape!=[16,19456] or layout!='ND':raise Mismatch('WeightQuant output is not full-K FP16 M16/N19456')
    inputs=w.get('inputs',[])
    if len(inputs)<3 or any(x and not str(x).endswith(':-1') and not str(x).startswith('^') for x in inputs[3:]):raise Mismatch('bias/offset/output quantization or unknown extra input present')
    if weight_edge is not None and inputs[1]!=weight_edge:raise Mismatch('WeightQuant weight edge differs from frozen layout audit')
    if scale_edge is not None and inputs[2]!=scale_edge:raise Mismatch('WeightQuant Scale edge differs from frozen layout audit')
    a=attributes(w)
    for key,expected in {'antiquant_group_size':128,'inner_precise':0,'transpose_x':False,'transpose_weight':True}.items():
        if key not in a:raise Unknown('missing WeightQuant attribute '+key)
        if type(a[key]) is not type(expected) or a[key]!=expected:raise Mismatch('WeightQuant attribute differs: '+key)
    cursor=edge_name(inputs[0]);seen=set();input_chain=[]
    while True:
        if cursor in seen or cursor not in table:raise Unknown('activation dependency chain missing/cyclic')
        seen.add(cursor);n=table[cursor];input_chain.append(dict(name=cursor,type=n['type']))
        if n['type']=='Data':
            dtype,shape,layout=direct_desc(n)
            if dtype is None or layout is None:raise Unknown('missing explicit activation dtype/layout')
            if dtype!='DT_FLOAT16' or shape!=[16,2560] or layout!='ND':raise Mismatch('activation is not original full FP16 X')
            break
        if n['type'] not in ('Reshape','Identity'):raise Mismatch('activation preprocessing changes values: '+n['type'])
        edges=data(n)
        if not edges:raise Unknown('activation view lacks input')
        cursor=edge_name(edges[0])
    if len([n for n in nodes if n['type']=='Data'])!=1:raise Mismatch('Native does not have exactly one external input')
    return dict(weight_node=w['name'],attributes=a,output_chain=chain,input_chain=input_chain,output_dtype='DT_FLOAT16',output_shape=[16,19456],
                semantics='pure full-K W8A16 WeightQuant projection; only shape-preserving value views after projection')


def audit(args,input_files,frozen=None):
    r=dict(native_semantics_status='UNRESOLVED',native_comparison_status='SKIPPED',
        reason='NATIVE_COMPARISON_SKIPPED_SEMANTICS_UNRESOLVED',
        checks={name:dict(status='UNKNOWN',detail='not yet established') for name in ('case','input_identity','group_layout','air_om_binding','pure_linear_graph','output_and_repeat','numerical_scope')},evidence=[],
        generator_sources=[],
        generator_source_scope='current source establishes intended export contract; actual frozen graph and metadata must independently match')
    def note(name,value,status='PASS'):r['checks'][name]=dict(status=status,detail=value)
    try:
        r['generator_sources']=[identity(REPO/p) for p in SOURCE_FILES]
        if frozen is not None:
            r['expected_frozen_native']=[]
            for item in [*frozen.get('identity',{}).get('native',{}).values(),*frozen.get('identity',{}).get('golden',[])]:
                observed=dict(identity=item,status='UNKNOWN');r['expected_frozen_native'].append(observed)
                try:verify(item);observed['status']='PASS'
                except FileNotFoundError as ex:observed['reason']=str(ex);raise
                except ValueError as ex:
                    observed.update(status='FAIL',reason=str(ex));raise Mismatch('frozen Native identity changed: '+str(ex)) from ex
        from fp16_gate_evidence import accepted
        from a2_common import checked_file,load_bundle,case_shape,validate_native_graph
        from run_a2 import validate_om_manifest
        src=accepted(args.a51_evidence,args.prefetch_evidence)
        r['evidence']=[identity(args.a51_evidence),identity(args.prefetch_evidence),*src['identity']['native'].values()]
        if any(src['files'][k]!=v for k,v in input_files.items()):raise Mismatch('Native Capture inputs differ from Stage3A')
        bundle_path=Path(src['source']['bundle']);native_path=Path(src['source']['native']);gold_path=Path(src['source']['golden'])
        r['evidence'] += [identity(p) for p in (bundle_path,native_path,gold_path)]
        bundle=load_bundle(bundle_path,'a4');native=validate_om_manifest(native_path,bundle_path,bundle,'a4')
        case=next(c for c in bundle['cases'] if c['name']=='layer-0-gate_up')
        if case_shape(case,'a4')!=(16,2560,19456) or case.get('layer')!=0 or case.get('projection')!='gate_up' or case.get('group_size')!=128:raise Mismatch('projection scope differs')
        note('case',dict(name=case['name'],shape=[16,2560,19456],group_size=128))
        note('input_identity',input_files)
        compiled=next(c for c in native['cases'] if c['name']==case['name'])
        air_path=checked_file(native_path.parent,compiled['air_manifest']);dep_path=checked_file(native_path.parent,compiled['deployment'])
        air=json.loads(air_path.read_text());dep=json.loads(dep_path.read_text())
        if len(air.get('graphs',[]))!=1 or len(dep.get('graphs',[]))!=1:raise Mismatch('not standalone Native graph')
        graph=air['graphs'][0];dg=dep['graphs'][0]
        validate_native_graph(graph,case,native['bundle_sha256'],'a4');validate_native_graph(dg,case,native['bundle_sha256'],'a4')
        if graph['air']!=dg['air']:raise Mismatch('deployment compiled a different AIR')
        command=dg.get('atc_command')
        if not isinstance(command,list):raise Unknown('missing frozen ATC command')
        if [v for v in command if v.startswith('--precision_mode')]!=['--precision_mode=must_keep_origin_dtype']:raise Mismatch('Native precision-mode contract differs')
        for key in ('metadata','input_names','output_names','runtime_input_abi'):
            if graph.get(key)!=dg.get(key):raise Mismatch('AIR/deployment semantics differ: '+key)
        layout=graph['runtime_input_abi']['weight_quant_layout'];nodes=layout.get('nodes',[])
        if len(nodes)!=1:raise Unknown('missing single-node layout audit')
        node=nodes[0]
        for key,value in {'group_size':128,'scale_layout':'GN','scale_shape':[20,19456],'weight_shape':[19456,2560],
                          'weight_layout':'NK','weight_format':'FRACTAL_NZ','transpose_weight':True}.items():
            if key not in node:raise Unknown('missing layout field: '+key)
            if node[key]!=value:raise Mismatch('layout field differs: '+key)
        note('group_layout',node)
        note('air_om_binding',dict(native_manifest=identity(native_path),air_manifest=identity(air_path),deployment=identity(dep_path),om=src['identity']['native']['om']))
        actual_air=checked_file(air_path.parent,graph['air']);r['evidence'] += [identity(p) for p in (air_path,dep_path,actual_air)]
        records=[v for v in graph.get('payload_files',[]) if Path(v.get('path','')).name=='dynamo.pbtxt']
        if len(records)!=1:raise Unknown('missing/ambiguous SHA-bound dynamo.pbtxt; cannot prove no post-processing')
        pb=checked_file(air_path.parent,records[0]);r['evidence'].append(identity(pb))
        if pb.stat().st_size>256*1024*1024:raise Unknown('graph exceeds bounded 256 MiB parser scope')
        sys.path.insert(0,str(common.OPS/'draft_ac/test'))
        from c_profile_graph import read_pbtxt
        if not isinstance(node.get('weight'),str) or not isinstance(node.get('scale'),str):raise Unknown('missing audited weight/Scale graph edges')
        try:graph_contract=pure_linear_graph(read_pbtxt(pb),node['name'],node['weight'],node['scale'])
        except Mismatch:raise
        except (ValueError,KeyError,TypeError) as ex:raise Unknown('graph interpretation unresolved: '+str(ex)) from ex
        note('pure_linear_graph',graph_contract)
        gold=json.loads(gold_path.read_text());row=next(c for c in gold['cases'] if c['name']==case['name'])
        execution=row.get('executions',{}).get('native_om',{})
        for key,value in {'status':'PASS','runtime':'AscendCL native OM','io_validated':True,'input_readonly':True,'guards_intact':True,
                          'm':16,'k':2560,'n':19456,'repetitions':2}.items():
            if key not in execution:raise Unknown('missing Native execution field '+key)
            if execution[key]!=value:raise Mismatch('Native execution field differs '+key)
        ids=src['identity']['golden']
        if len(ids)!=4 or any(v['bytes']!=622592 or v['sha256']!=ids[0]['sha256'] for v in ids):raise Mismatch('full-K Golden bytes/repeat differ')
        for v in ids:verify(v)
        r['evidence']+=ids
        note('output_and_repeat',dict(dtype='FP16',shape=[16,19456],bytes=622592,execution=execution,golden=ids))
        parity=row.get('comparison',{}).get('custom_vs_native_om',{})
        if not parity:raise Unknown('missing frozen W8A16 custom/Native parity evidence')
        if parity.get('bitwise_equal') is not True or parity.get('finite') is not True:raise Mismatch('frozen W8A16 parity failed')
        note('numerical_scope',dict(frozen_w8a16_parity=parity,scheme='original FP16 activation / INT8 weight / FP16 GN Group128 Scale; final FP16 linear output',
            accumulation='Native reduction implementation/order remains opaque; existing identical-input W8A16 parity is evidence for this frozen result, not a CPU bitwise guarantee',comparison='cross-scheme diagnostic only'))
        if not all(c['status']=='PASS' for c in r['checks'].values()):raise Unknown('incomplete Native semantic evidence')
        for item in r['evidence']:
            try:verify(item)
            except FileNotFoundError:raise
            except ValueError as ex:raise Mismatch('Native evidence changed during audit: '+str(ex)) from ex
        r.update(native_semantics_status='VERIFIED',native_comparison_status='SKIPPED',reason='awaiting completed full-K accumulation',golden=ids[0])
    except Mismatch as ex:
        r.update(native_semantics_status='MISMATCH',reason='NATIVE_SEMANTICS_MISMATCH',error=str(ex));note('blocking_condition',str(ex),'FAIL')
    except (OSError,ValueError,KeyError,TypeError,StopIteration) as ex:
        r.update(error=f'{type(ex).__name__}: {ex}');note('missing_or_unresolved',str(ex),'UNKNOWN')
    return r
