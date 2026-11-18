"""Read only sealed profiling inputs. No device, profiler, ATC or subprocess calls."""
import argparse
import csv
import json
import traceback
from collections import Counter,defaultdict
from decimal import Decimal
from math import fsum
from pathlib import Path
from signals import UNKNOWN,Graph,desc,edge_name,notify,projection,stats,trace
import analyze as v2
from audit import PROVISIONAL,complete,device,partition,require

STRICT='STRICT_DEVICE_ATTRIBUTION'
TIMED=range(7,37)


def check(path,item):
    require(v2.identity(path)==item,'identity drift: '+str(path));return path


def linked(item):
    path=Path(item['path']);require(path.is_absolute(),'unbound relative identity');return v2.read(check(path,item))


def col(fields,*names):
    for key in names:
        val=fields.get(key)
        if val not in (None,'','N/A','-'):return str(val)
    return None


def sig(value):
    key=trace.norm(value)
    return 'wait' if key=='notifywait' else 'record' if key in ('notifyrecord','notifywrite') else None


def csv_index(table):
    index=defaultdict(list);notify_rows=[]
    for row in table['rows']:
        fields={trace.norm(k):v for k,v in row['fields'].items() if k}
        key=(col(fields,'physicstreamid','streamid'),col(fields,'taskid'))
        if all(key):index[key].append((row,fields))
        if any(sig(fields.get(k,'')) for k in ('opname','optype','tasktype','opstate')):notify_rows.append(row)
    return index,notify_rows


def task_data(timeline,rows,table):
    raw=v2.read(timeline);events=raw.get('traceEvents',[]) if isinstance(raw,dict) else raw
    done,tracks,hw=complete(events);_,_,tasks=partition(done,tracks,hw)
    by_id,notify_csv=csv_index(table);tasks={t['event_index']:t for t in tasks}
    require(len(tasks)==len(rows) and len({int(r['event_index']) for r in rows})==len(rows),'hardware inventory drift')
    output=[]
    for row in rows:
        idx=int(row['event_index']);event=tasks[idx];args=device.args(event)
        lo=Decimal(row['start_us']);hi=Decimal(row['end_us'])
        require(lo==Decimal(event['raw_timestamp_us']) and hi-lo==Decimal(event['dur']),'timeline/v2 interval drift')
        stream=col(args,'physicstreamid','streamid') or str(event['tid']);tid=col(args,'taskid');model=col(args,'modelid')
        raw_fields={k:args.get(k) for k in ('opname','optype','tasktype','opstate')}
        sign={sig(x) for x in [event.get('name',''),*raw_fields.values()] if x and sig(x)}
        matched=[]
        for item,f in by_id.get((stream,tid),[]):
            if model and col(f,'modelid') and col(f,'modelid')!=model:continue
            csv_signal={sig(f.get(k,'')) for k in ('tasktype','optype','opname','opstate')}-{None}
            if row['op_name']!=item['op_name'] and event.get('name')!=item['op_name'] and not sign.intersection(csv_signal):continue
            if col(args,'batchid') and col(f,'batchid') and col(args,'batchid')!=col(f,'batchid'):continue
            matched.append((item,f))
        if not sign and matched:
            types={sig(x.get('tasktype','')) or sig(x.get('optype','')) or sig(x.get('opname','')) or sig(x.get('opstate','')) for _,x in matched}
            if len(types)==1 and None not in types:sign=types
        e=dict(idx=idx,call=int(row['call']) if row['call']!='' else None,state=row['state'],pid=event['pid'],model=model,
               task_id=tid,stream=stream,start=str(lo),end=str(hi),lo=lo,hi=hi,us=float(hi-lo),op=row['op_name'],category=row['category'],
               kind=row['op_type'],signal=next(iter(sign)) if len(sign)==1 else None,
               compute=(trace.norm(args.get('tasktype','')) in ('aicore','aivector','aicpu','aic','aiv','mixaic','mixaiv') or row['category']=='GroupQuantLinear') and not sign,
               raw=dict(name=event.get('name'),**raw_fields,args=event.get('args',{})),notify_id=col(args,'notifyid','notifyidentifier'),
               csv_join='unique filtered stream/task candidate; not a clock match' if len(matched)==1 else UNKNOWN,
               csv_lines=[item['line'] for item,_ in matched],
               csv_fields=matched[0][0]['fields'] if len(matched)==1 and sign else None)
        output.append(e)
    return output,notify_csv


def descriptors(table,name):
    unique={}
    for row in table['rows']:
        if row['op_name']!=name:continue
        fields={k:v for k,v in row['fields'].items() if k and any(t in trace.norm(k) for t in ('shape','datatype','format'))}
        key=json.dumps(fields,sort_keys=True)
        if fields and key not in unique:unique[key]=dict(line=row['line'],**fields)
    return list(unique.values())


def per_stats(events):
    selected=[e for e in events if e['call'] in TIMED]
    counts=Counter(e['call'] for e in selected)
    return dict(tasks=stats([e['us'] for e in selected]),time_per_execute=stats([fsum(e['us'] for e in selected if e['call']==i) for i in TIMED]),
                counts_per_execute=[counts[i] for i in TIMED])


def paths_report(net,names,field,table,proj):
    results=[];passthru={'reshape','transpose','transposed','transdata','cast','identity','squeeze','expanddims','gather','gatherv2','slice','stridedslice','split','splitv'}
    convert={'transdata','cast','memcpy','memcpyasync'}
    attn=passthru|{'mul','muls','add','addv2','realdiv','sub','maskedfill','select','selectv2','tile','broadcastto','concat','concatd','permute'}
    for name,events in sorted(names.items()):
        node=net.nodes.get(name)
        record=dict(name=name,timing=per_stats(events),node=desc(node),csv=descriptors(table,name),status=UNKNOWN)
        results.append(record)
        if node is None:continue
        if field=='head':
            # Only input 1 of this MatMul can provide weight-side dataflow evidence.
            inputs=node['inputs'];parent=net.nodes.get(edge_name(inputs[1])) if len(inputs)>1 else None
            chain=[] if parent is None else [[name,parent['name']]+path[1:] for path in net.paths(parent['name'],True,
                lambda x:x in convert,lambda x:x in passthru)]
            if parent and trace.norm(parent['type']) in convert:chain.insert(0,[name,parent['name']])
            layout=[] if parent is None else [[name]+p for p in net.paths(parent['name'],True,lambda x:x=='transdata',lambda x:x in passthru|convert)]
            if parent and trace.norm(parent['type'])=='transdata':layout.insert(0,[name,parent['name']])
            record.update(weight_input=desc(parent),weight_conversion_paths=chain,explicit_weight_transdata=layout,
                weight_layout_finding='EXPLICIT_WEIGHT_TRANSDATA' if layout else 'UNRESOLVED: no explicit weight-side TransData on inspected paths',
                conversion_evidence='EXPLICIT_WEIGHT_PATH_CONVERSION' if chain else 'NO_EXPLICIT_WEIGHT_PATH_CONVERSION_OBSERVED; fused/internal preparation UNRESOLVED',
                adjacent_conversions=[desc(net.nodes[x]) for x in net.up[name]+net.down[name] if trace.norm(net.nodes[x]['type']) in convert],
                top1_paths=net.paths(name,False,lambda x:'argmax' in x or x.startswith('topk'),lambda x:x in passthru))
            record['status']='EXACT_OM_NODE';continue
        # A score MatMul must feed a softmax, and the softmax must feed a value MatMul.
        # Branch shape/count alone is not enough to identify either operand's source.
        if 'softmax' not in trace.norm(node['type']):continue
        score=net.paths(name,True,lambda x:'matmul' in x,lambda x:x in attn)
        value=net.paths(name,False,lambda x:'matmul' in x,lambda x:x in attn)
        record.update(score_paths=score,value_paths=value)
        anchors={x['name']:x for x in proj if x['projection']!=UNKNOWN}
        def branch(node_name):
            evidence=[]
            for parent in net.up[node_name]:
                paths=([[parent]] if parent in anchors else net.paths(parent,True,lambda x:x=='dflashgroupquantlinear',lambda x:x in attn or 'norm' in x or 'rotary' in x))
                evidence.append([dict(projection=anchors[p[-1]]['projection'],path=p) for p in paths if p[-1] in anchors])
            return evidence
        if len(score)==len(value)==1:
            qk=score[0][-1];pv=value[0][-1]
            qbranches=branch(qk);vbranches=branch(pv)
            record.update(score_inputs=qbranches,value_inputs=vbranches)
            # Q and KV on distinct QK operands; KV on a PV operand other than the softmax branch.
            qsets=[{item['projection'] for item in b} for b in qbranches]
            pvparents=net.up[pv];last=value[0][-2]
            layers=[i for i in range(5) if len(qsets)==2 and {frozenset(x) for x in qsets}=={frozenset([f'layer-{i}-q']),frozenset([f'layer-{i}-kv'])} and
                    len(pvparents)==2 and any(p!=last and {x['projection'] for x in b}=={f'layer-{i}-kv'} for p,b in zip(pvparents,vbranches))]
            record['status']='CONNECTED_MATMUL_SOFTMAX_MATMUL' if len(layers)!=1 else 'SAME_LAYER_Q_KV_DATAFLOW_SUPPORTS_ATTENTION_CANDIDATE'
            record['limitation']='Q/KV ancestry is not a proof of K versus V split-port semantics or transpose flags; do not label all occurrences from counts.'
            record['layer']=layers[0] if len(layers)==1 else None
    return results


def load_gear(root,out,verified,cap,gear,src,smoke,deployment):
    directory=root/'analysis-v2'/('c'+gear)
    def artifact(name):
        path=directory/name;return check(path,verified[str(path)])
    p=v2.read(artifact('provisional.json'));strict=v2.read(artifact('strict.json'))
    require(p['scope']==PROVISIONAL and p['strict_status']==strict['status'],'mixed attribution')
    for item in p['inputs']:check(Path(item['path']),item)
    windows=v2.read(artifact('host-windows.json'))
    require([w['call'] for w in windows]==list(range(38)),'incomplete call inventory')
    with artifact('all-device-tasks.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    require(len(rows)==p['device_events'],'v2 task inventory differs')
    allowed={x['path']:x for x in p['inputs']};timeline=Path(linked(cap['windows'])['file'])
    check(timeline,allowed[str(timeline)])
    folder=root/('profile-c'+gear)/'profiler';paths=list(folder.rglob('op_summary*.csv'))
    require(len(paths)==1,'ambiguous op_summary');check(paths[0],allowed[str(paths[0])])
    tables,_=trace.csv_tables(folder);table=tables[0]
    events,notify_csv=task_data(timeline,rows,table)
    graph_path=check(Path(cap['graph']['path']),allowed[cap['graph']['path']])
    try:nodes=v2.graph.read_om_json(graph_path);graph_status='PARSED'
    except ValueError as err:nodes=[];graph_status='UNRESOLVED: '+str(err)
    net=Graph(nodes)
    cases=[c for key,c in src['binding']['cases'].items() if key.startswith(gear+'/')]
    require(len(cases)==26 and all([smoke['plans']['A/'+gear+'/'+c['name']]['schedule'][k] for k in ('m','k','n')]==c['shape'] for c in cases),'missing/foreign projection contracts')
    audit=deployment['graphs'][0]['runtime_input_abi']['draft_ac'];require(audit['status']=='PASS' and audit['counts']['DFlashGroupQuantLinear']==26,'invalid immutable audit')
    by_name=defaultdict(list)
    for e in events:
        if e['call'] in TIMED:by_name[e['category'],e['op']].append(e)
    proj=[]
    immutable=audit.get('nodes',[])
    group_names={name for category,name in by_name if category=='GroupQuantLinear'}|{r['name'] for r in immutable if r.get('ge_type')=='DFlashGroupQuantLinear'}
    for name in sorted(group_names):
        record=projection(name,net.nodes.get(name),descriptors(table,name),immutable,cases)
        record['timing']=per_stats(by_name['GroupQuantLinear',name]);proj.append(record)
    wait=notify(events,windows);v2.write(out/'notify.json',wait);v2.write(out/'op-summary-notify-rows.json',notify_csv)
    head=paths_report(net,{name:ev for (cat,name),ev in by_name.items() if cat=='Head MatMul'},'head',table,proj)
    for item in head:
        neighbors={n['name'] for n in item.get('adjacent_conversions',[])}
        neighbors.update(path[-1] for path in item.get('weight_conversion_paths',[]))
        item['observed_conversion_tasks']={name:per_stats(ev) for (_,name),ev in by_name.items() if name in neighbors}
    attention=paths_report(net,{name:ev for (cat,name),ev in by_name.items() if cat in ('Unmapped MatMul','Softmax')},'attention',table,proj)
    for item in attention:
        item['connected_softmax_nodes']=[dict(softmax=s['name'],role=role,status=s['status'],layer=s.get('layer')) for s in attention
            for key,role in [('score_paths','upstream score candidate'),('value_paths','downstream value candidate')] if any(path[-1]==item['name'] for path in s.get(key,[]))]
    for name,items in [('projection',proj),('head',head),('attention',attention)]:v2.write(out/(name+'.json'),items)
    inputs=dict(timeline=str(timeline),op_summary=str(paths[0]),graph=str(graph_path),v2_tasks=str(artifact('all-device-tasks.csv')))
    return dict(gear=gear,strict=strict['status'],graph_status=graph_status,inputs=inputs,wait=wait,proj=proj,head=head,attention=attention,
                v2_wait_mean=next((x['mean_us_per_execute'] for x in p['top20_task_names'] if x['op_name']=='NOTIFY_WAIT'),None),
                csv_wait_record_rows=len(notify_csv),baseline=p['baseline'])


def average(items,key):
    return fsum(x[key] for x in items)/len(items)


def report_markdown(out,gears):
    reports={k:[] for k in ('notify-wait-analysis','group-quant-projection-analysis','head-matmul-analysis','attention-analysis','summary-v3')}
    summary=reports['summary-v3'];waitstats=[];picks=[]
    for g in gears:
        gear=g['gear']
        for lines in reports.values():lines.extend(['',f'## C{gear}',f"STRICT_DEVICE_ATTRIBUTION: {g['strict']}. Timing below: {PROVISIONAL}."])
        calls=g['wait']['calls'][7:37];w=reports['notify-wait-analysis'];a=g['attention']
        data={k:average(calls,k) for k in ('count','sum_us','wait_union_us','groupquant_union_us','intersection_us','wait_union_without_groupquant_us','wait_union_fraction_of_host')}
        waitstats.append(data)
        streams=Counter((event['pid'],event['stream']) for event in g['wait']['waits'] if event['call'] in TIMED)
        w+=['Only calls 7..36 in averages. All 38 calls and ALL wait/record events (including unassigned) are in c'+gear+'/notify.json.',
            f"All timeline waits: {g['wait']['wait_count']}; records: {g['wait']['record_count']}; raw notify CSV rows: {g['csv_wait_record_rows']}.",
            f"Timed wait counts by (PID, stream): {dict(streams)}.",
            *[f'- {key}: {val:.3f} per Execute' for key,val in data.items()],
            f"v2 name-only NOTIFY_WAIT sum: {g['v2_wait_mean']} µs/Execute. v3 additionally inspects raw event/CSV labels.",
            f"Calls containing waits: {sum(x['count']>0 for x in calls)}/30. Critical-path membership and producer/object: UNRESOLVED.",
            'Temporal neighbors, explicit notify IDs when present, per-wait overlaps and offsets from launch/Sync return are evidence, not causality. Uncovered wait time is not removable time.']
        pl=reports['group-quant-projection-analysis'];pl+=['Node | Projection | Role | M/K/N | weight / scale B | task count | mean / P95 / max µs per task','---|---|---|---|---|---:|---']
        for x in g['proj']:
            t=x['timing']['tasks']
            pl.append(f"{x['name']} | {x['projection']} | {x['role']} | {x['mkn']} | {x['weight_bytes']} / {x['scale_bytes']} | {t['count']} | {t['mean_us']} / {t['p95_us']} / {t['max_us']}")
        pl+=['Timing bands alone are not identities. Different K/N, input row counts and bytes imply different work; they do not separate dequant, transfer and Cube costs. See projection.json for exact source evidence.']
        for role in sorted({x['role'] for x in g['proj']}):
            group=[x for x in g['proj'] if x['role']==role and x['timing']['tasks']['count']]
            if group:
                volume=sorted({(x['weight_bytes'],2*x['mkn'][0]*x['mkn'][1]*x['mkn'][2]) for x in group if x['mkn']})
                pl.append(f"- {role}: {len(group)} nodes; observed mean/task range {min(x['timing']['tasks']['mean_us'] for x in group):.2f}–{max(x['timing']['tasks']['mean_us'] for x in group):.2f} µs. (INT8 weight bytes, nominal 2MKN ops): {volume}.")
        for key,items in [('head-matmul-analysis',g['head']),('attention-analysis',a)]:
            lines=reports[key];lines.append(f"OM graph: {g['graph_status']}. Physical/runtime descriptors and exact graph dataflow are separate.")
            if key.startswith('attention'):
                supported=[s for s in a if s.get('layer') is not None]
                layers=sorted({s['layer'] for s in supported})
                lines.append(f"Layers with a graph-supported QK→Softmax→PV chain: {layers}. Five-layer coverage: {'GRAPH_SUPPORTED' if layers==list(range(5)) else UNKNOWN}. "
                             "This does not label the remaining tasks by count; see the exact upstream/downstream paths per node.")
            for x in items:
                msg=f"{x['name']}: {x['timing']['tasks']['count']} tasks; {x['timing']['time_per_execute']['mean_us']:.3f} µs/Execute; {x['status']}"
                if key.startswith('head'):
                    msg+=f"; weight layout={x.get('weight_layout_finding',UNKNOWN)}; paths to ArgMax/TopK (TopK alone does not prove K=1)={x.get('top1_paths',[])}; neighbors={[v['name'] for v in x.get('adjacent_conversions',[])]}"
                else:msg+=f"; layer={x.get('layer')}; connected softmax={x.get('connected_softmax_nodes',[])}"
                geom={side:x['node'][side] for side in ('input_desc','output_desc')} if x['node'] else UNKNOWN
                lines.extend([msg,'',f"CSV shapes/formats/dtypes: {json.dumps(x['csv'],ensure_ascii=False)[:1200]}",
                              f"OM descriptors (full in JSON): {json.dumps(geom,ensure_ascii=False)[:1600]}"])
        roles=defaultdict(list)
        for x in g['proj']:
            if x['role']!=UNKNOWN:roles[x['role']].append(x['timing']['time_per_execute']['mean_us'])
        ranking=sorted(((fsum(v),k) for k,v in roles.items()),reverse=True)
        head_time=fsum(x['timing']['time_per_execute']['mean_us'] for x in g['head'])
        summary.extend([f"Unprofiled median: {g['baseline']['median_ms']:.4f} ms. Wait sum {data['sum_us']:.3f}, wait union {data['wait_union_us']:.3f}, wait ∩ GroupQuant union {data['intersection_us']:.3f} µs/Execute.",
                        f'Confirmed role task sums (µs/Execute): {ranking}; Head MatMul: {head_time:.3f}.',
                        'Sources: '+json.dumps(g['inputs'],ensure_ascii=False)])
        candidates=ranking+([(head_time,'Head MatMul')] if g['head'] else [])
        picks.append(max(candidates)[1] if candidates else UNKNOWN)
        summary.append('First compute A/B candidate among identified roles (by aggregate provisional time): '+picks[-1])
    delta=waitstats[1]['sum_us']-waitstats[0]['sum_us']
    summary.extend(['',f'C64 minus C16 wait sum: {delta:.3f} µs/Execute; not extra Draft latency.',
        f'Common candidate for a controlled test (not verified optimization): {picks[0] if picks[0]==picks[1] else UNKNOWN}.',
        'First sync study: identify the wait stream, its producer and the record; test whether the repeated waits span useful GroupQuant work or extend beyond it toward Sync return. Do not remove or shorten waits.',
        'Next minimal experiment (design only): first benchmark one identified heavy projection or the frozen native Head, one real C16 input and same weights, changing one implementation variable. Require the existing numerical/guard gate; use fresh unprofiled paired samples. Confirm any candidate again on the full Draft OM. For waits use a separate bounded trace with explicit producer/consumer notify identifiers if these exports lack them.',
        'No measured causal savings. Task Duration Sum is not Draft wall time.'])
    for name,lines in reports.items():
        (out/(name+'.md')).write_text('# '+name+'\n'+'\n'.join(lines)+'\n')


def main(root):
    root=root.resolve(strict=True);out=root/'analysis-v3';out.mkdir(exist_ok=False)
    r=dict(status='BLOCKED',scope='OFFLINE_ONLY',npu='NOT_RUN',msprof='NOT_RUN')
    try:
        v2root=root/'analysis-v2';prev=v2.read(v2root/'report.json');require(prev['status']=='OFFLINE_V2_RECORDED','v2 not accepted')
        verified={x['path']:x for x in prev['files']};first=linked(prev['input_report']);require(first['status']=='A_OM_PROFILE_ANALYZED','v1 not accepted')
        admit=linked(first['predecessors']['admit']);require(admit['status']=='A_OM_PROFILE_ADMITTED','original admit missing')
        history=admit['binding']['history'];src=linked(history['source']);smoke=linked(history['projection-smoke']);export=linked(history['export'])
        require(smoke['status']=='FLOW_PROJECTION_SMOKE_PASS' and export['status']=='FLOW_EXPORT_PASS' and export['predecessors']['projection-smoke']==history['projection-smoke'],'wrong projection/export evidence')
        gears=[]
        for gear in ('16','64'):
            dest=out/('c'+gear);dest.mkdir();cap=linked(first['predecessors']['profile-c'+gear])
            model=export['models']['A/'+gear];require(model['om']==admit['binding']['cases'][gear]['om'],'different OM')
            require(cap['status']==f'A_C{gear}_RUNNER_CAPTURE_VERIFIED' and cap['predecessors']['admit']==first['predecessors']['admit'],'different capture')
            result=load_gear(root,dest,verified,cap,gear,src,smoke,linked(model['deployment']))
            result['inputs'].update(projection_contract=history['source']['path'],projection_smoke=history['projection-smoke']['path'],deployment=model['deployment']['path'])
            gears.append(result)
        report_markdown(out,gears);r.update(status='OFFLINE_V3_RECORDED',v2=v2.identity(v2root/'report.json'),source=history['source'],projection_smoke=history['projection-smoke'],export=history['export'])
        r['files']=[v2.identity(p) for p in sorted(out.rglob('*')) if p.is_file()]
    except Exception as err:r['error']=repr(err);(out/'traceback.txt').write_text(traceback.format_exc())
    v2.write(out/'report.json',r);print(r['status'],r.get('error',''),out);return int(r['status']=='BLOCKED')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True,type=Path);raise SystemExit(main(parser.parse_args().root))
