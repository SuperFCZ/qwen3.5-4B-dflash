"""Raw msprof task/PipeUtilization evidence, never a substitute for formal latency."""
import csv,json,math
from pathlib import Path
from bench_common import identity

def norm(x):return ''.join(c.lower() for c in str(x) if c.isalnum())
OP='dflashw8a8fullgate'
def numeric(value):
    try:return math.isfinite(float(value.strip().rstrip('%')))
    except (ValueError,AttributeError):return False

def collect(root):
    files=[];rows=[];tasks=[];skipped=[];core_rows=[]
    for path in sorted(root.rglob('*')):
        if not path.is_file():continue
        item=identity(path);files.append(item)
        if path.suffix=='.csv':
            if item['bytes']>64*1024*1024:skipped.append(dict(file=item,reason='CSV_SIZE_LIMIT'));continue
            with path.open(encoding='utf-8-sig',newline='') as f:
                for line,row in enumerate(csv.DictReader(f),2):
                    names=[v for k,v in row.items() if k and norm(k) in ('opname','optype','kernelname','taskname') and isinstance(v,str)]
                    if not any(OP in norm(v) for v in names):continue
                    fields={k:v for k,v in row.items() if k and isinstance(v,str) and v and any(t in norm(k) for t in ('mte1','mte2','mte3','duration','cube','mac','vec','scalar','coreid','taskid','streamid'))}
                    if fields:
                        record=dict(file=item,line=line,names=names,fields=fields);rows.append(record)
                        if any(norm(k) in ('coreid','aicoreid') for k in fields):core_rows.append(record)
        elif path.name.startswith('msprof_') and path.suffix=='.json':
            if item['bytes']>128*1024*1024:skipped.append(dict(file=item,reason='TIMELINE_SIZE_LIMIT'));continue
            try:data=json.loads(path.read_text(),parse_float=str)
            except (ValueError,UnicodeError) as e:skipped.append(dict(file=item,reason=str(e)));continue
            events=data.get('traceEvents',[]) if isinstance(data,dict) else data
            if not isinstance(events,list):skipped.append(dict(file=item,reason='UNKNOWN_TIMELINE_SCHEMA'));continue
            for e in events:
                if not isinstance(e,dict):continue
                args=e.get('args') or {}
                if not isinstance(args,dict):continue
                if e.get('ph')=='X' and OP in norm(e.get('name','')) and norm(args.get('Task Type','')) in ('aicore','aic','aivector'):
                    tasks.append(dict(file=item,event=e)) # raw timestamps/units, no sum of overlapping pipes
    available={kind:('AVAILABLE_RAW_FIELDS' if any(any(kind in norm(k) and not norm(k).startswith('machine') and numeric(v) for k,v in r['fields'].items()) for r in rows) else 'UNAVAILABLE') for kind in ('duration','cube','mac','mte1','mte2','mte3','vec','scalar')}
    return dict(status='RAW_OPERATOR_METRICS_FOUND' if rows or tasks else 'UNAVAILABLE',files=files,counter_rows=rows,device_tasks=tasks,
        availability=available,core_tagged_rows=core_rows,skipped=skipped,
        tail_analysis='UNRESOLVED; static tile counts are not measured per-core task durations',
        instruction_hotspots={k:'UNAVAILABLE_WITHOUT_PC_SOURCE_OR_INSTRUCTION_EVIDENCE' for k in ('Matmul','VDEQ16','Cast_Add','PIPE_ALL_wait')},
        interpretation='Original column names, values and units. No Pipe sums as wall time, no ratio-to-duration conversion, no unsupported attribution.',
        formal_latency_evidence=False)
