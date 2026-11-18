"""Analyze frozen A-only trace exports; no profiler, runner, CANN or device calls."""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from decimal import Decimal,localcontext
import math
from pathlib import Path
import sys
import traceback
from audit import HERE,PROVISIONAL,complete,device,enclose,host_windows,partition,require,step_audit,step_csv,trace
import c_profile_graph as graph

sys.path.insert(0,str(HERE.parent/'draft_a_core_scheduler_profile'))
from attribution import category


def read(path):
    return json.loads(path.read_text(),parse_float=str) if path.name.startswith('msprof') else json.loads(path.read_text())


def write(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def identity(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):digest.update(block)
    return dict(path=str(path.resolve()),bytes=path.stat().st_size,sha256=digest.hexdigest())


def checked(path,item,root):
    require(root in path.resolve(strict=True).parents and identity(path)==item,'input outside capture or changed: '+str(path))
    return path


def csvout(path,rows,fields):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)


def label(event,metadata,nodes,heads):
    args=device.args(event);raw=str(args.get('opname') or event.get('name',''))
    stream=args.get('physicstreamid',args.get('streamid'));tid=args.get('taskid')
    by_id=metadata.get(('task',str(stream),str(tid)),[]) if stream is not None and tid is not None else []
    by_name=metadata.get(('name',raw),[])
    exact=[x for x in by_id if x['op_name']==raw]
    matched=exact or by_name or by_id
    ops={(x['op_name'],x['op_type']) for x in matched}
    name=raw;kind=args.get('optype','');basis='timeline type'
    if len(ops)==1:
        name,typ=next(iter(ops))
        if kind and typ and kind!=typ:return raw,kind,'Other / Unmapped','conflicting timeline/CSV types'
        kind=typ or kind;basis='unique CSV name/type' if exact or by_name else 'unique stream/task CSV name/type'
    elif len(ops)>1 and not kind:return raw,'','Other / Unmapped','ambiguous CSV mapping'
    ge=nodes.get(name,set())
    if len(ge)==1 and (not kind or kind in ge):kind=next(iter(ge));basis='exact OM node'
    elif ge and kind not in ge:return name,kind,'Other / Unmapped','ambiguous OM node/type'
    return name,kind,category(kind,name,heads),basis


def grouped(rows,key,denominator):
    totals=defaultdict(list)
    for row in rows:totals[row[key]].append(row['duration_us'])
    return sorted(({key:name,'count':len(values),'sum_us':math.fsum(values),'mean_us_per_execute':math.fsum(values)/denominator,
                    'max_task_us':max(values),'mean_task_us':math.fsum(values)/len(values)} for name,values in totals.items()),
                  key=lambda x:(-x['sum_us'],x[key]))


def analyze_gear(root,out,main,gear):
    phase='profile-c'+gear;folder=root/phase
    cap_path=checked(folder/'profile-report.json',main['predecessors'][phase],root);cap=read(cap_path)
    require(cap['status']==f'A_C{gear}_RUNNER_CAPTURE_VERIFIED' and cap['execute']=='38_PROFILED_CALLS_GOLDEN_VERIFIED' and
            not cap.get('error') and not cap.get('integrity_error'),'capture not verified')
    old_path=checked(root/'analyze'/('c'+gear)/'profile-report.json',main['gears'][gear]['report'],root)
    old=read(old_path);frozen=read(checked(folder/'windows.json',cap['windows'],root))
    require(frozen['status']=='WINDOWS_IDENTIFIED' and cap['device_assignment']==frozen['device_assignment'],'strict report drift')
    raw=folder/'profiler';timeline=Path(frozen['file']);inventory={x['path']:x for x in cap['files']}
    require(raw in timeline.parents,'timeline not in gear capture')
    csvs=list(raw.rglob('op_summary*.csv'));require(len(csvs)==1 and len(cap['csv_sources'])==1 and
        str(csvs[0].relative_to(raw))==cap['csv_sources'][0]['file'],'unexpected op_summary')
    step_files=list(timeline.parent.glob('step_trace*.csv'))
    expected={x['path'] for x in frozen['assignment_evidence']['step_csv_evidence']['tables']}
    require({str(p) for p in step_files}==expected,'Step CSV inventory differs')
    inputs=[cap_path,old_path,folder/'windows.json',checked(Path(cap['graph']['path']),cap['graph'],root)]
    for path in [timeline,*csvs,*step_files]:inputs.append(checked(path,inventory[str(path)],root))
    ids=[identity(path) for path in inputs]
    tables,metadata=trace.csv_tables(raw);require(not tables[0]['rejected_rows'] and len(tables[0]['rows'])==cap['csv_sources'][0]['rows'],'CSV row inventory drift')
    with localcontext() as legacy:
        legacy.prec=28  # Reproduce the original trace parser's process default.
        strict_replay=trace.parse_timeline(timeline,metadata,set())
    require(strict_replay['status']=='WINDOWS_IDENTIFIED' and strict_replay['device_assignment']==frozen['device_assignment'] and
            strict_replay['assignment_evidence']==frozen['assignment_evidence'],'strict parser replay differs from the retained evidence')
    del strict_replay
    data=read(timeline);events=data['traceEvents'] if isinstance(data,dict) else data
    done,tracks,hardware=complete(events);windows=host_windows(done,hardware,frozen,cap['runner_pid'])
    markers,quarantine,tasks=partition(done,tracks,hardware)
    ev=frozen['assignment_evidence']
    require(len(tasks)==ev['real_event_count'] and {x['event_index'] for x in quarantine}=={x['event_index'] for x in ev['unconfirmed_markers']},'hardware event inventory drift')
    audit=step_audit(markers,tracks,windows,step_csv.load(timeline.parent),frozen)
    audit['other_step_exports_not_used']=[str(path) for path in sorted(raw.rglob('step_trace*.csv')) if path.parent!=timeline.parent]
    write(out/'step-audit.json',audit)
    strict=dict(status=audit['strict_status'],strict_parser_replay='MATCH',original_report=identity(old_path),task_accounting=ev['event_accounting'],
                original_attribution=old['attribution'],operator_types=old['operator_types'] if audit['strict_status']=='DEVICE_STEPS_VERIFIED' else None)
    write(out/'strict.json',strict)
    rows=enclose(tasks,windows);require(len(rows)==len({row['event_index'] for row in rows}),'nonunique event indices')
    graph_status='PARSED'
    try:graph_nodes=graph.read_om_json(Path(cap['graph']['path']))
    except ValueError as err:graph_status='UNAVAILABLE: '+str(err);graph_nodes=[]
    kinds=defaultdict(set)
    for node in graph_nodes:kinds[node['name']].add(node['type'])
    try:heads=set(graph.summarize(graph_nodes,'compiled_om')['head_anchors'])
    except ValueError as err:heads=set();graph_status+='; HEAD_ANCHORS_UNAVAILABLE: '+str(err)
    for row,event in zip(rows,tasks):
        name,kind,cat,basis=label(event,metadata,kinds,heads)
        row.update(op_name=name,op_type=kind,category=cat,basis=basis)
    fields=list(rows[0]) if rows else []
    csvout(out/'all-device-tasks.csv',rows,fields)
    csvout(out/'crossing-tasks.csv',[r for r in rows if r['state'] in ('CROSSING','AMBIGUOUS_BOUNDARY')],fields)
    csvout(out/'unassigned-tasks.csv',[r for r in rows if r['call'] is None],fields)
    stats=Counter(row['state'] for row in rows)
    timed=[row for row in rows if row['call'] is not None and 7<=row['call']<=36]
    unknown=[row for row in timed if row['category'] in ('Other / Unmapped','Unmapped MatMul')]
    csvout(out/'unmapped-timed.csv',unknown,fields)
    types=grouped(timed,'category',30);top=grouped(timed,'op_name',30)[:20]
    per_call=[]
    for index,(start,end) in enumerate(windows):
        group=[r for r in rows if r['call']==index]
        per_call.append(dict(call=index,phase=frozen['all_iterations'][index]['phase'],raw_host_start_us=str(start),
                             raw_sync_end_us=str(end),host_window_us=float(end-start),count=len(group),
                             task_union_us=trace.union([(float(Decimal(r['start_us'])-start),float(Decimal(r['end_us'])-start)) for r in group]),
                             categories=grouped(group,'category',1)))
    write(out/'host-windows.json',per_call)
    duration=math.fsum(r['duration_us'] for r in timed);unmapped_us=math.fsum(r['duration_us'] for r in unknown)
    leading=next((entry for entry in types if entry['category'] not in ('Other / Unmapped','Unmapped MatMul')),None)
    result=dict(scope=PROVISIONAL,strict_status=audit['strict_status'],graph_status=graph_status,inputs=ids,
                baseline=old['formal_baseline'],device_events=len(rows),accounting=dict(stats),timed_tasks=len(timed),
                classified_tasks=len(timed)-len(unknown),classified_fraction=(1-len(unknown)/len(timed)) if timed else None,
                classified_duration_fraction=1-unmapped_us/duration if duration else None,
                attribution_fraction=stats[PROVISIONAL]/len(rows) if rows else None,
                categories=types,top20_task_names=top,top20_individual_tasks=sorted(timed,key=lambda x:-x['duration_us'])[:20],
                largest_classified_candidate=leading,optimization_decision='NOT_ESTABLISHED: provisional ranking can nominate a controlled A/B, not prove wall-time savings')
    write(out/'provisional.json',result)
    require(ids==[identity(path) for path in inputs],'analyzed inputs drifted during read')
    first=audit['first_blocking_step']
    summary=[f'## C{gear}',f"Strict: {audit['strict_status']}; markers: {audit['marker_count']}; IDs: {audit['ids']}",
        f"Missing ID ranges (inclusive): {audit['missing_id_ranges_inclusive']}; duplicates: {audit['duplicates']}; host calls without strict anchors: {audit['missing_strict_host_calls']}",
        f"First precision blocker: {first['id'] if first else None}; first mapping error: {(audit['first_mapping_error'] or {}).get('reason')}",
        f"Independent {PROVISIONAL}: {dict(stats)}; measured calls 7..36, classified {len(timed)-len(unknown)}/{len(timed)} tasks.",
        'Category | tasks | sum µs | mean sum µs/Execute | max µs/task','---|---:|---:|---:|---:']
    summary+=['{category} | {count} | {sum_us:.3f} | {mean_us_per_execute:.3f} | {max_task_us:.3f}'.format(**x) for x in types]
    if first:
        chk=first['precision'];endpoints=first['csv']
        summary+=['',f"Step {first['id']} raw Start/End/Duration (µs): {chk['raw_start_us']} / {chk['raw_end_us']} / {chk['raw_duration_us']}",
                  f"Signed residual (µs): {chk.get('signed_error_us')}; allowed ±{chk.get('allowed_error_max_us')}; CSV: {endpoints['status']}",
                  f"CSV reason(s): {endpoints.get('reason') or [x['reason'] for x in endpoints['unsupported_rows']] or 'See exact comparisons in step-audit.json'}"]
    summary+=['','Top 20 names by provisional duration (µs/Execute):']+[f"{i}. {' '.join(x['op_name'].split())[:160]}: {x['mean_us_per_execute']:.3f}" for i,x in enumerate(top,1)]
    summary+=['',f"Classified duration / assigned timed duration: {result['classified_duration_fraction']}; largest identified category for follow-up: {leading['category'] if leading else 'UNRESOLVED'}."]
    return summary


def main(root):
    root=root.resolve(strict=True);out=root/'analysis-v2';out.mkdir(exist_ok=False)
    report=dict(status='BLOCKED',scope='OFFLINE_ONLY',output=str(out),npu='NOT_RUN',msprof='NOT_RUN')
    try:
        report['sources']=[identity(Path(__file__)),identity(HERE/'audit.py'),identity(Path(sys.modules['attribution'].__file__)),
                           *[identity(Path(module.__file__)) for module in (graph,trace,device,step_csv)]]
        path=root/'analyze/profile-report.json';report['input_report']=identity(path);main_report=read(path)
        require(main_report['status']=='A_OM_PROFILE_ANALYZED' and not main_report.get('integrity_error') and not main_report.get('error'),'original analysis not accepted')
        summary=['# Offline Step audit / independent host enclosure','Only fully enclosed unique tasks; duration sums may overlap and are not Draft wall time.']
        with localcontext() as context:
            context.prec=64
            for gear in ('16','64'):
                dest=out/('c'+gear);dest.mkdir();summary.extend(['',*analyze_gear(root,dest,main_report,gear)])
        summary+=['','Optimization: not established. Use the ranking for a single-variable validation; it is not a causal time decomposition.']
        (out/'summary.md').write_text('\n'.join(summary)+'\n')
        require(identity(path)==report['input_report'],'analysis report drift')
        report['files']=[identity(p) for p in sorted(out.rglob('*')) if p.is_file()]
        report['status']='OFFLINE_V2_RECORDED'
    except Exception as err:report['error']=repr(err);(out/'traceback.txt').write_text(traceback.format_exc())
    write(out/'report.json',report);print(report['status'],report.get('error',''),out);return int(report['status']=='BLOCKED')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True,type=Path)
    raise SystemExit(main(parser.parse_args().root))
