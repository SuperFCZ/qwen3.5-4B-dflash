#!/usr/bin/env python3
"""Offline C16 attribution only. No subprocess, SDK, OM execution or profiler capture."""
import argparse
import ast
import json
import math
from pathlib import Path
import re
from common import sha256,write_json,checked_file
from c_profile_graph import read_pbtxt,first
from c_profile_trace import collect,stats,union,norm
from profile_recovery_checks import snapshot

# Historical architecture, not interchangeable checkpoints. Classification still
# REQUIRES actual input/output shapes, never occurrence order or duration.
SHAPES={
 'w8_a':{'GATE_UP':(16,2560,19456),'DOWN':(16,9728,2560),
         'FC':(16,12800,2560),'KV':(32,2560,2048),'Q':(16,2560,4096),'O':(16,4096,2560)},
 'fp16':{'GATE_UP':(16,2560,18432),'DOWN':(16,9216,2560),
         'FC':(16,20480,2560),'KV':(32,2560,2048),'Q':(16,2560,4096),'O':(16,4096,2560)}}
REGIONS=('GATE_UP','DOWN','OTHER_LINEAR','HEAD','OTHER_KNOWN','UNKNOWN')


def shapes(value, _depth=0):
    """Decode CSV JSON strings without accepting partial/malformed dimensions."""
    if _depth > 8:return []
    if isinstance(value,(list,tuple)):
        if not value:return []
        if all(type(x) is int and x>0 for x in value):return [list(value)]
        parsed=[shapes(x,_depth+1) for x in value]
        return [s for group in parsed for s in group] if all(parsed) else []
    if not isinstance(value,str):return []
    value=value.strip()
    if not value:return []
    if value[0] in ('"', "'", '[', '('):
        try:parsed=json.loads(value)
        except ValueError:
            try:parsed=ast.literal_eval(value)
            except (ValueError,SyntaxError,RecursionError):return []
        return shapes(parsed,_depth+1) if isinstance(parsed,(str,list,tuple)) and parsed!=value else []
    result=[]
    for group in value.split(';'):
        if not re.fullmatch(r'\s*[1-9][0-9]*(?:\s*[,xX*]\s*[1-9][0-9]*)*\s*',group):return []
        result.append([int(x) for x in re.split(r'\s*[,xX*]\s*',group.strip())])
    return result


def tokens(value):
    """Formats/dtypes may have the same outer JSON quoting as shapes."""
    for _ in range(8):
        if not isinstance(value,str):break
        try:parsed=json.loads(value)
        except ValueError:break
        if parsed==value:break
        value=parsed
    return [str(x).strip() for x in value] if isinstance(value,list) else str(value or '').split(';')


def logical_matrix(shape,fields,side):
    """Only ND is inherently logical. NZ requires an exact graph TensorDesc.

    A physical NZ extent cannot distinguish e.g. logical M15 from padded M16.
    Never reconstruct logical dimensions just by multiplying physical axes.
    """
    formats=tokens(fields.get(side+'formats',fields.get(side+'format','')))
    fmt=norm(formats[0]) if formats else ''
    return matrix(shape) if fmt in ('','nd','formatnd') else None


def matrix(shape):
    if len(shape)<2 or any(x<=0 for x in shape) or math.prod(shape[:-2])!=1:return None
    return tuple(shape[-2:])


def role(task,model):
    kind=norm(task.get('op_type',''))
    if task.get('graph_shape_conflict'):return 'UNKNOWN','frozen graph vs runtime descriptor conflict/unproven layout'
    if task.get('label') in ('head_matmul','head_argmax','custom_c'):return 'HEAD','profile vocabulary/Head evidence'
    evidence=task.get('shape_evidence',{})
    fields={norm(k):v for k,v in evidence.get('fields',{}).items()}
    a=shapes(fields.get('inputshapes',fields.get('inputshape')))
    y=shapes(fields.get('outputshapes',fields.get('outputshape')))
    if 'matmul' in kind or 'dflashgroupquantlinear' in kind:
        if evidence.get('status')=='AMBIGUOUS':return 'UNKNOWN','conflicting shape metadata'
        aa=logical_matrix(a[0],fields,'input') if a else None;yy=logical_matrix(y[0],fields,'output') if y else None
        if aa and yy and aa[0]==yy[0]:
            mkn=(aa[0],aa[1],yy[1])
            if mkn==(15,2560,248320):return 'HEAD','actual M/K/N='+str(mkn)
            for name,want in SHAPES[model].items():
                if mkn==want:return name if name in ('GATE_UP','DOWN') else 'OTHER_LINEAR',name+' actual M/K/N='+str(mkn)
        return 'UNKNOWN','Matmul role lacks unambiguous actual logical M/K/N'
    if any(x in kind for x in ('norm','softmax','attention','scatter','cacheupdate','transdata','transpose','memcpy','cast','reshape','silu','sigmoid','mul','add','memset')):
        return 'OTHER_KNOWN','explicit non-Matmul op type'
    return 'UNKNOWN','unmapped op type/shape'



def graph_shapes(report,model):
    """Fallback only from exact node name/type and explicit frozen descriptors."""
    row=next(c for c in report['cases'] if c['model']==model);identity=row['model_identity']
    watch=report['source_watch']
    if model=='fp16':
        dep=Path(identity['deployment']).resolve()
        if sha256(dep)!=watch[str(dep)]:raise ValueError('FP16 deployment changed')
        d=json.loads(dep.read_text());air=(dep.parent/d['air_manifest']['path']).resolve()
        if sha256(air)!=d['air_manifest']['sha256']:raise ValueError('FP16 AIR binding changed')
    else:
        folder=Path(identity['evidence'])/'draft-a-c16';air=checked_file(folder,identity['artifacts']['air'])
    if sha256(air)!=watch[str(air)]:raise ValueError('frozen AIR manifest changed')
    wanted='draft' if model=='fp16' else 'draft_ac_a_c16'
    graphs=[g for g in json.loads(air.read_text())['graphs'] if g['name']==wanted]
    if len(graphs)!=1:raise ValueError('not one Draft graph')
    records=[p for p in graphs[0]['payload_files'] if Path(p['path']).name=='dynamo.pbtxt']
    if len(records)!=1:raise ValueError('no unique frozen pbtxt')
    path=(air.parent/records[0]['path']).resolve()
    digest=sha256(path)
    if not path.is_relative_to(air.parent) or digest!=records[0]['sha256'] or digest!=watch[str(path)]:raise ValueError('frozen pbtxt changed/escaped')
    def shape(desc):
        if norm(first(desc.get('layout'),'')) not in ('','nd','formatnd'):return []
        value=first(desc.get('shape'),{})
        return value.get('dim',[]) if isinstance(value,dict) else []
    try:nodes=read_pbtxt(path)
    except Exception as error:return {},dict(status='UNPARSED',path=str(path),error=str(error))
    finally:
        if sha256(path)!=digest:raise ValueError('pbtxt changed while parsing')
    result={};duplicates=set()
    for node in nodes:
        a=node['input_desc'];y=node['output_desc']
        inputs=shape(a[0]) if a else [];outputs=shape(y[0]) if y else []
        if not inputs or not outputs:
            aa=[x for x in node['serialized_descriptors'] if first(x.get('name')) in ('x','x1')]
            yy=[x for x in node['serialized_descriptors'] if first(x.get('name'))=='y']
            if len(aa)==1 and len(yy)==1:inputs,outputs=shape(aa[0]),shape(yy[0])
        if not inputs or not outputs or matrix(inputs) is None or matrix(outputs) is None:continue
        if node['name'] in result:duplicates.add(node['name'])
        result[node['name']]=dict(type=node['type'],fields={'Input Shapes':[inputs],'Output Shapes':[outputs]})
    for name in duplicates:result.pop(name,None)
    return result,dict(status='PARSED',path=str(path),sha256=digest,usable_nodes=len(result))


def descriptor_matches(actual, logical, fmt):
    if not actual:return True
    # Only verify physical storage against an already-known logical descriptor.
    if actual==logical and norm(fmt) in ('','nd','formatnd','fractalnz','formatfractalnz'):return True
    if norm(fmt) not in ('fractalnz','formatfractalnz'):return False
    if matrix(logical) is None or len(actual)<4 or math.prod(actual[:-4])!=1:return False
    m,n=matrix(logical)
    return actual[-4:]==[(n+15)//16,(m+15)//16,16,16]


def add_graph_shapes(profile,mapping):
    for trace in profile['timelines']:
        for call in trace.get('all_iterations',[]):
            for task in call['tasks']:
                evidence=task.get('shape_evidence',{})
                source=mapping.get(task['name'])
                if not source or norm(source['type'])!=norm(task['op_type']) or evidence.get('status')=='AMBIGUOUS':continue
                fields={norm(k):v for k,v in evidence.get('fields',{}).items()}
                valid=True
                for side in ('Input','Output'):
                    observed=shapes(fields.get(side.lower()+'shapes',fields.get(side.lower()+'shape')))
                    logical=source['fields'][side+' Shapes'][0]
                    fmt=tokens(fields.get(side.lower()+'formats',fields.get(side.lower()+'format','')))[0]
                    dtype=tokens(fields.get(side.lower()+'datatypes',fields.get(side.lower()+'datatype','')))[0]
                    if observed:
                        if norm(fmt) in ('fractalnz','formatfractalnz') and norm(dtype) not in ('float16','dtfloat16','fp16'):valid=False
                        if not descriptor_matches(observed[0],logical,fmt):valid=False
                    elif evidence.get('status')!='MISSING':valid=False
                if valid:
                    task['shape_evidence']=dict(status='FROZEN_GRAPH_EXACT_NAME_TYPE',fields=source['fields'],original=evidence)
                else:task['graph_shape_conflict']='physical/logical shape, format or dtype unproven; original evidence retained'


def intervals_overlap(groups):
    # Per-region unions do not partition time: explicitly retain shared spans.
    events=[]
    for region,intervals in groups.items():
        for start,end in intervals:
            events.extend([(start,region,1),(end,region,-1)])
    active={};last=None;shared=0.
    for time,region,delta in sorted(events):
        if last is not None and sum(v>0 for v in active.values())>1:shared+=time-last
        active[region]=active.get(region,0)+delta;last=time
    return shared


def analyze(profile,model):
    trace=profile.get('selected_timeline')
    if not trace:return dict(status='UNRESOLVED_WINDOWS',iterations=[],reason='inspect raw API inventory; never infer windows by dividing task totals')
    iterations=[]
    for call in trace['all_iterations']:
        groups={k:[] for k in REGIONS};counts={k:0 for k in REGIONS};tasks=[]
        for task in call['tasks']:
            region,basis=role(task,model);start=task['start_us'];end=start+task['duration_us']
            groups[region].append((start,end));counts[region]+=1
            tasks.append({**task,'linear_region':region,'classification_basis':basis})
        all_intervals=[i for group in groups.values() for i in group]
        occupied=union(all_intervals);duration_sum=sum(b-a for a,b in all_intervals)
        iterations.append(dict(index=call['index'],phase=call['phase'],execute_api=call['execute_api'],sync_api=call['sync_api'],
            regions_us={k:union(v) for k,v in groups.items()},counts=counts,tasks=tasks,
            host_window_us=call['host_window_us'],host_launch_api_us=call['host_launch_api_us'],host_sync_api_us=call['host_sync_api_us'],
            device_window=call.get('device_window'),device_occupancy_us=occupied,device_envelope_us=call['device_envelope_us'],device_gap_us=call['device_gap_us'],
            task_sum_minus_union_us=duration_sum-occupied,cross_region_overlap_us=intervals_overlap(groups),
            unknown_occupancy_ratio=union(groups['UNKNOWN'])/occupied if occupied else 0.,
            boundary_crossing_tasks=call.get('boundary_crossing_tasks',[])))
    selected=[r for r in iterations if r['phase']=='timed']
    if len(selected)!=30 or [r['index'] for r in selected]!=list(range(7,37)):raise ValueError('not the exact 30 measured calls')
    layers=5 if model=='w8_a' else 6
    expected={'GATE_UP':layers,'DOWN':layers,'OTHER_LINEAR':1+3*layers}
    inventories=[]
    for row in selected:
        listing={k:sorted(t['name'] for t in row['tasks'] if t['linear_region']==k) for k in expected}
        head_mm=[t for t in row['tasks'] if t['linear_region']=='HEAD' and 'matmul' in norm(t['op_type'])]
        head_arg=[t for t in row['tasks'] if t['linear_region']=='HEAD' and 'argmax' in norm(t['op_type'])]
        row['head_components']={label:dict(count=len(ts),names=[t['name'] for t in ts],duration_us=union([(t['start_us'],t['start_us']+t['duration_us']) for t in ts])) for label,ts in [('matmul',head_mm),('argmax',head_arg)]}
        row['linear_node_inventory']=listing
        inventories.append(listing)
    stable=all(x==inventories[0] for x in inventories)
    inventory=stable and all(all(r['counts'][k]==v and len(set(r['linear_node_inventory'][k]))==v for k,v in expected.items()) for r in selected)
    head_complete=all(r['head_components']['matmul']['count']==1 and r['head_components']['argmax']['count']==1 for r in selected)
    device=trace.get('device_assignment','UNRESOLVED_DEVICE_ASSIGNMENT')
    status=('UNRESOLVED_DEVICE_ASSIGNMENT' if device!='DEVICE_STEPS_VERIFIED' else
            'INCOMPLETE_KERNEL_CLASSIFICATION' if not inventory or not head_complete or any(r['counts']['UNKNOWN'] for r in selected) else 'ATTRIBUTION_VERIFIED')
    keys=('host_window_us','host_launch_api_us','host_sync_api_us','device_occupancy_us','device_envelope_us','device_gap_us',
          'task_sum_minus_union_us','cross_region_overlap_us','unknown_occupancy_ratio')
    return dict(status=status,host_windows='WINDOWS_VERIFIED',device_assignment=device,expected_linear_counts=expected,stable_node_inventory=stable,head_inventory='COMPLETE' if head_complete else 'INCOMPLETE',ffn_inventory='MATCHES_EXPECTED_LAYERS' if inventory else 'INCOMPLETE_OR_DIFFERENT_SHAPES',
                iterations=iterations,measured_indices=list(range(7,37)),
                region_statistics_us={k:stats([r['regions_us'][k] for r in selected]) for k in REGIONS},
                window_statistics={k:stats([r[k] for r in selected]) for k in keys},
                warning='Kernel unions can overlap. Statistics are profiled durations, not unprofiled latency or causal savings.')


def run(args):
    report_path=args.report.resolve();old=json.loads(report_path.read_text());root=args.output_dir.resolve()
    if old.get('status')!='DIAGNOSTIC_ONLY' or old.get('historical_fp16_status')!='FAIL_REPLAY_CHECKS':raise ValueError('requires original diagnostic-only report')
    rows={c['model']:c for c in old['cases']}
    if set(rows)!={'fp16','w8_a'} or len(old['cases'])!=2:raise ValueError('two-model inventory differs')
    source_dirs={m:Path(c['profile_origin']).resolve()/'profiler' for m,c in rows.items()}
    if root.is_relative_to(report_path.parent) or any(root.is_relative_to(p) or p.is_relative_to(root) for p in source_dirs.values()):raise ValueError('new output must be outside historical evidence')
    root.mkdir(parents=True,exist_ok=False);digest=sha256(report_path);before={}
    result=dict(status='DIAGNOSTIC_ONLY',device_execution='NOT_RUN_OFFLINE_ONLY',end_to_end='NOT_RUN',
                fp16_correctness=old['historical_fp16_status'],
                a_build_config=old.get('reuse',{}).get('opp',{}).get('a_config'),
                source_report=dict(path=str(report_path),sha256=digest),models={},analysis_status='RUNNING')
    try:
        for model,path in source_dirs.items():
            before[model]=snapshot(path,profiler=True)
            expected=(old['recovery']['fp16_profiler_integrity'] if model=='fp16' else rows[model]['profiler_inventory'])['files']
            if before[model]!=expected:raise ValueError(model+' archived profiler SHA256 inventory differs')
            graph=rows[model].get('compiled_graph',{})
            if graph.get('status')!='PARSED':graph=rows[model].get('pre_atc_graph',{})
            mapping,mapping_report=graph_shapes(old,model)
            profile=collect(path,graph);add_graph_shapes(profile,mapping);analysis=analyze(profile,model)
            result['models'][model]=dict(profiler=str(path),unprofiled_latency=rows[model]['unprofiled_latency'],graph_shape_evidence=mapping_report,profile=profile,analysis=analysis)
        statuses={v['analysis']['status'] for v in result['models'].values()}
        result['analysis_status']=next((s for s in ('UNRESOLVED_WINDOWS','UNRESOLVED_DEVICE_ASSIGNMENT','INCOMPLETE_KERNEL_CLASSIFICATION') if s in statuses),'ATTRIBUTION_VERIFIED')
        result['host_windows']='WINDOWS_VERIFIED' if all(v['analysis'].get('host_windows')=='WINDOWS_VERIFIED' for v in result['models'].values()) else 'UNRESOLVED_WINDOWS'
    except Exception as error:result['analysis_status']='FAILED';result['error']=str(error);raise
    finally:
        try:
            if sha256(report_path)!=digest or any(snapshot(source_dirs[m],profiler=True)!=inventory for m,inventory in before.items()):raise ValueError('historical evidence changed during offline analysis')
        except Exception as error:
            result['analysis_status']='FAILED';result['integrity_error']=str(error);raise
        finally:
            write_json(root/'linear-timeline.json',result)
            lines=['# C16 W8 Linear 离线归因','',f"状态：{result['analysis_status']}；DIAGNOSTIC_ONLY；本次未执行任何设备命令。",
                   '按同一 Host PID/TID 的真实 Execute/Sync 配对；使用第 7..36 次（零起始）共 30 个计时窗口。FP16 原 FAIL_REPLAY_CHECKS 保留。','']
            for model,r in result['models'].items():
                lines+=[f'## {model}','',json.dumps(r['analysis'].get('region_statistics_us',{}),ensure_ascii=False),
                        json.dumps(r['analysis'].get('window_statistics',{}),ensure_ascii=False),
                        'FFN/其他 Linear 节点数核对：'+r['analysis'].get('ffn_inventory','UNKNOWN'),
                        '设备归属：'+r['analysis'].get('device_assignment','UNKNOWN')+'；Head：'+r['analysis'].get('head_inventory','UNKNOWN')]
                audit=(r['profile'].get('selected_timeline') or {}).get('assignment_evidence',{})
                lines+=['', 'Step 时间戳序列化检查：'+json.dumps(audit.get('step_time_summary',{}),ensure_ascii=False),
                        '剩余失败原因：'+json.dumps(audit.get('remaining_failure_reasons',{}),ensure_ascii=False),
                        '', '| Iteration | Duration − (End − Start), μs | 允许误差区间, μs | 状态 |',
                        '| --- | --- | --- | --- |']
                for check in audit.get('step_time_checks',[]):
                    lines.append(f"| {check['iteration_id']} | {check.get('signed_error_us','UNKNOWN')} | [{check.get('allowed_error_min_us','UNKNOWN')}, {check.get('allowed_error_max_us','UNKNOWN')}] | {check['status']} |")
            lines+=['','Host 窗口通过不代表设备归属通过。Step Trace 仅作为范围证据，不计入 occupancy；跨 Host 边界 Kernel 保留完整区间。',
                    '逐轮 API 原始名、时间戳、connection_id、Kernel 分类依据及重叠见 JSON。UNKNOWN 不能凭耗时/顺序改成 Gate/Up；各区域 union 之和不是 wall time。']
            (root/'linear-analysis.md').write_text('\n'.join(lines)+'\n')
    print(result['analysis_status']+': '+str(root/'linear-timeline.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    run(p.parse_args())
