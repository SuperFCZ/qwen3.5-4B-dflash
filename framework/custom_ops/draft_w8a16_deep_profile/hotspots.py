"""Offline summaries. A CSV counter is not assigned by row order or clipped to a host window."""
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from math import isfinite, fsum
from pathlib import Path
from statistics import median
from deep_common import deep,c,require
from signals import Graph,notify,projection
from audit import PROVISIONAL,complete,host_windows,partition,enclose,trace
import c_profile_graph as graph

TIMED=range(7,37)
STRICT='STRICT_DEVICE_ATTRIBUTION'
UNKNOWN='UNRESOLVED'
ABSENT='UNAVAILABLE'


def counter_columns(rows):
    names={key for row in rows for key in row['fields'] if key}
    pipelines=('cube','mac','mte1','mte2','mte3','vector','vec','scalar','aicore')
    return sorted(key for key in names if ('stall' in key.lower() or traffic_field(key) or
                  any(p in trace.norm(key) for p in pipelines) and any(x in key.lower() for x in ('time','ratio','utilization','cycle','%'))))


def traffic_field(key):
    name=trace.norm(key)
    return ('read' in name and any(scope in name for scope in ('dram','hbm','ddr','mainmemory','gmread'))
            and ('byte' in name or 'bandwidth' in name))


def numerical(value):
    try:number=float(str(value).strip().rstrip('%'))
    except (ValueError,TypeError):return None
    return number if isfinite(number) and number>=0 else None


def same_time(value,stamp):
    try:return Decimal(str(value).strip())==Decimal(stamp)
    except InvalidOperation:return False


def counters(events,table):
    fields=counter_columns(table['rows']);lookup=defaultdict(list)
    for row in table['rows']:
        f={trace.norm(k):v for k,v in row['fields'].items() if k}
        lookup[(deep.col(f,'physicstreamid','streamid'),deep.col(f,'taskid'),row['op_name'])].append((row,f))
    joined={};uses=Counter()
    for e in events:
        if not e['task_id']:continue
        choices=lookup.get((e['stream'],e['task_id'],e['op']),[])
        choices=[(row,f) for row,f in choices if (deep.col(f,'modelid') in (None,e['model']) or e['model'] is None)
                 and deep.col(f,'deviceid') in (None,'0') and
                 (not deep.col(f,'tasktype') or not e.get('raw',{}).get('tasktype') or trace.norm(f['tasktype'])==trace.norm(e['raw']['tasktype']))
                 and (not row['op_type'] or not e['kind'] or row['op_type']==e['kind'])]
        exact=[row for row,f in choices if same_time(f.get('taskstarttimeus'),e['start'])]
        # Never compare durations to guess an owner. Reused IDs require equal recorded starts.
        row=exact[0] if len(exact)==1 else choices[0][0] if len(choices)==1 and not deep.col(choices[0][1],'taskstarttimeus') else None
        if row is not None:joined[e['idx']]=row;uses[row['line']]+=1
    values=defaultdict(lambda:defaultdict(list));links=[]
    for e in events:
        if e['call'] not in TIMED:continue
        row=joined.get(e['idx'])
        if row is None or uses[row['line']]!=1:continue
        metrics={key:numerical(row['fields'].get(key)) for key in fields}
        f={trace.norm(k):v for k,v in row['fields'].items() if k}
        group=(e['label'],deep.col(f,'tasktype') or ABSENT,deep.col(f,'coreid','aicoreid','coretype') or ABSENT)
        links.append(dict(event=e['idx'],call=e['call'],stream=e['stream'],task=e['task_id'],op=e['op'],group=group,line=row['line'],metrics=metrics))
        for key,val in metrics.items():
            if val is not None:values[group][key].append(val)
    available={key:sum(link['metrics'][key] is not None for link in links) for key in fields}
    families={name:[k for k in fields if any(tag in trace.norm(k) for tag in tags) and available[k]] for name,tags in
              [('Cube/MAC',('cube','mac')),('MTE1',('mte1',)),('MTE2',('mte2',)),('MTE3',('mte3',)),('Vector',('vec',)),('Scalar',('scalar',)),('Stall',('stall',))]}
    return dict(source=table['file'],columns=fields,joined=len(links),timed_tasks=sum(e['call'] in TIMED for e in events),
                unique_csv_links=links,per_class={' / '.join(group):{k:dict(n=len(vals),median_per_task=median(vals)) for k,vals in columns.items()} for group,columns in values.items()},
                coverage=available,families={name:cols or ABSENT for name,cols in families.items()},
                explicit_memory_read_fields=[key for key in fields if traffic_field(key)],
                stall=('MEASURED_FOR_LINKED_ROWS' if any('stall' in k.lower() and val is not None for link in links for k,val in link['metrics'].items())
                       else ABSENT),
                policy='Single op_summary only. Units as printed in column header; unjoined/N/A cells are missing. Not a critical-path decomposition.')


def choose_owners(tasks,windows,timeline):
    rows=enclose(tasks,windows)
    if timeline['device_assignment']!='DEVICE_STEPS_VERIFIED':return rows,PROVISIONAL
    owners={};require(len(timeline['all_iterations'])==38,'missing strict calls')
    for call in timeline['all_iterations']:
        for item in call['tasks']:
            idx=item['event_index'];require(idx not in owners,'strict duplicate owner');owners[idx]=call['index']
    require(set(owners)<={r['event_index'] for r in rows},'unknown strict task')
    for row in rows:
        row['call']=owners.get(row['event_index']);row['state']=STRICT if row['call'] is not None else 'OUTSIDE_STRICT_MODEL_TASKS'
    return rows,STRICT


def analyze_case(folder,capture,binding,mode,gear,force_host=False):
    folder.mkdir();case=binding['variants'][mode]['cases'][gear];timeline=c.read(c.verify(capture['windows']))
    require(capture['strict_status']==timeline['device_assignment'] and timeline['status']=='WINDOWS_IDENTIFIED','capture/window attribution drift')
    raw=Path(capture['folder'])/'profiler';path=Path(timeline['file'])
    require(raw in path.resolve().parents and c.identity(path) in capture['files'],'timeline outside sealed capture')
    raw_json=deep.v2.read(path);events=raw_json.get('traceEvents',[]) if isinstance(raw_json,dict) else raw_json
    done,tracks,hw=complete(events);windows=host_windows(done,hw,timeline,capture['runner_pid'])
    _,_,tasks=partition(done,tracks,hw);tables,meta=trace.csv_tables(raw)
    require(len(tables)==1 and not tables[0]['rejected_rows'] and len(tasks)==timeline['assignment_evidence']['real_event_count'],'trace/CSV inventory drift')
    require(c.identity(raw/tables[0]['file']) in capture['files'],'unbound op_summary')
    rows,method=(enclose(tasks,windows),PROVISIONAL) if force_host else choose_owners(tasks,windows,timeline)
    graph_path=c.verify(capture['graph']);graph_status='PARSED'
    try:nodes=graph.read_om_json(graph_path)
    except ValueError as err:nodes=[];graph_status='UNAVAILABLE: '+str(err)
    net=Graph(nodes);kinds=defaultdict(set)
    for n in nodes:kinds[n['name']].add(n['type'])
    try:heads=set(graph.summarize(nodes,'compiled_om')['head_anchors'])
    except ValueError:heads=set()
    for row,e in zip(rows,tasks):
        name,kind,category,basis=deep.v2.label(e,meta,kinds,heads)
        row.update(call='' if row['call'] is None else row['call'],op_name=name,op_type=kind,category=category,basis=basis)
    hardware,notify_csv=deep.task_data(path,rows,tables[0])
    deployment=c.read(c.verify(case['deployment']));audit=deployment['graphs'][0]['runtime_input_abi']['draft_ac']
    require(audit['counts']['DFlashGroupQuantLinear']==26,'different graph')
    contracts=c.read(c.verify(binding['flow_source']))['binding']['cases']
    cases=[v for k,v in contracts.items() if k.startswith(gear+'/')];require(len(cases)==26,'missing projection cases')
    proj={};names={}
    for record in audit['nodes']:
        if record['ge_type']!='DFlashGroupQuantLinear':continue
        name=record['name'];p=projection(name,net.nodes.get(name),deep.descriptors(tables[0],name),audit['nodes'],cases)
        require(name not in proj,'duplicate immutable node');proj[name]=p
        if p['contract_matches']==[p['projection']]:names[name]=p['projection']
    require(len(proj)==26 and len(set(names.values()))==len(names),'incomplete or aliased immutable projection inventory')
    timed=[e for e in hardware if e['call'] in TIMED];classed=defaultdict(list)
    headnames={e['op'] for e in timed if e['category']=='Head MatMul'}
    headreport=deep.paths_report(net,{name:[e for e in timed if e['op']==name] for name in headnames},'head',tables[0],list(proj.values()))
    tops={p[-1] for n in headreport for p in n.get('top1_paths',[]) if 'argmax' in trace.norm(net.nodes[p[-1]]['type'])}
    attn=deep.paths_report(net,{name:[e for e in timed if e['op']==name] for name in {e['op'] for e in timed if e['category'] in ('Softmax','Unmapped MatMul')}},'attention',tables[0],list(proj.values()))
    attn_mms={path[-1] for n in attn if n['status']=='SAME_LAYER_Q_KV_DATAFLOW_SUPPORTS_ATTENTION_CANDIDATE'
              for key in ('score_paths','value_paths') for path in n[key]}
    for e in hardware:
        e['label']=('NOTIFY_'+e['signal'].upper() if e['signal'] else 'Head Top1' if e['op'] in tops else 'Attention graph candidate' if e['op'] in attn_mms else e['category'])
        e['projection']=names.get(e['op']) if e['category']=='GroupQuantLinear' else None
        if e['category']=='GroupQuantLinear':e['label']='Linear/'+names[e['op']].split('-')[-1] if e['op'] in names else 'Linear/Unmapped'
        if e['call'] in TIMED:classed[e['label']].append(e)
    projections={}
    for case_row in cases:
        ident=case_row['name'];hits=[key for key,val in names.items() if val==ident]
        entries=[e for e in timed if e['op'] in hits and e['category']=='GroupQuantLinear']
        s=deep.per_stats(entries);projections[ident]=dict(nodes=hits,shape=case_row['shape'],timing=s,
            status='BOUND_EVERY_CALL' if len(hits)==1 and all(x>0 for x in s['counts_per_execute']) else UNKNOWN)
    winrows=[dict(call=i,raw_host_start_us=str(lo),raw_sync_end_us=str(hi)) for i,(lo,hi) in enumerate(windows)]
    wait=notify(timed,winrows[7:37])
    # Cross-pipe overlap is not additive. Report only the union/intersection and observed associations.
    c.write(folder/'notify.json',wait);c.write(folder/'counters.json',counters(hardware,tables[0]))
    c.write(folder/'head.json',headreport);c.write(folder/'attention.json',attn);c.write(folder/'projection-map.json',proj)
    fields=['idx','call','label','projection','op','kind','pid','model','stream','task_id','notify_id','start','end','us','state','csv_join']
    deep.v2.csvout(folder/'timed-tasks.csv',timed,fields)
    required=['Linear/'+x for x in ('gate_up','down','q','o','kv','fc')]+['Head MatMul','Head Top1','Attention','Attention graph candidate','Softmax','Norm','Memcpy / Cast','TransData','NOTIFY_WAIT','NOTIFY_RECORD']
    classes={k:dict(observed=True,**deep.per_stats(es)) for k,es in classed.items()}
    classes.update({k:dict(observed=False,status='NO_ATTRIBUTED_TASKS') for k in required if k not in classes})
    report=dict(mode=mode,gear=gear,graph_status=graph_status,baseline=case['baseline'],strict_status=timeline['device_assignment'],scope=method,
                timed_task_count=len(timed),strict_reasons=timeline['assignment_evidence']['remaining_failure_reasons'],
                assigned_all_calls=dict(Counter(r['state'] for r in rows)),classes=classes,projections=projections,
                wait_summary={k:fsum(x[k] for x in wait['calls'])/30 for k in ('sum_us','wait_union_us','groupquant_union_us','intersection_us','wait_union_without_groupquant_us')},
                notify_csv_rows_all=len(notify_csv),wait_object=UNKNOWN,critical_path=UNKNOWN)
    c.write(folder/'case-analysis.json',report);return report
