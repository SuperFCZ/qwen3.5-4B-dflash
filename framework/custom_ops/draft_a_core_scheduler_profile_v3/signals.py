"""Evidence-only interval and graph operations; no device entrypoints."""
from bisect import bisect_left,bisect_right
from collections import Counter,defaultdict,deque
from decimal import Decimal
from math import ceil,fsum
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'draft_a_core_scheduler_profile_v2'))
import audit
trace=audit.trace
from c_profile_graph import edge_name
from linear_attribution import shapes,logical_matrix

UNKNOWN='UNRESOLVED'


def stats(values):
    seq=sorted(values)
    return dict(count=len(seq),mean_us=fsum(seq)/len(seq),p95_us=seq[ceil(.95*len(seq))-1],max_us=seq[-1]) if seq else dict(count=0,mean_us=None,p95_us=None,max_us=None)


def union(intervals):
    result=[]
    for start,end in sorted(intervals):
        if not result or start>result[-1][1]:result.append([start,end])
        else:result[-1][1]=max(result[-1][1],end)
    return result


def measure(intervals):
    return float(sum((end-start for start,end in union(intervals)),Decimal(0)))


def overlaps(start,end,events):
    return [(max(start,e['lo']),min(end,e['hi']),e['idx']) for e in events if start<e['hi'] and e['lo']<end and min(end,e['hi'])>max(start,e['lo'])]


def compact(e):
    return {k:e[k] for k in ('idx','op','call','stream','model','pid','task_id','start','end')}


class Neighbors:
    def __init__(self,events):
        self.starts=sorted((e['lo'],e['idx'],e) for e in events)
        self.ends=sorted((e['hi'],e['idx'],e) for e in events)
        self.lo=[x[0] for x in self.starts];self.hi=[x[0] for x in self.ends]
        self.max_span=max((e['hi']-e['lo'] for e in events),default=Decimal(0))

    def near(self,e):
        before=bisect_right(self.hi,e['lo']);after=bisect_left(self.lo,e['hi'])
        # Preserve ties; neighboring tasks do not establish a dependency.
        prev=self.ends[bisect_left(self.hi,self.hi[before-1]):before] if before else []
        nxt=self.starts[after:bisect_right(self.lo,self.lo[after])] if after<len(self.lo) else []
        return dict(previous=[compact(x[2]) for x in prev],following=[compact(x[2]) for x in nxt])

    def during(self,e):
        return [x[2] for x in self.starts[bisect_left(self.lo,e['lo']-self.max_span):bisect_left(self.lo,e['hi'])] if x[2]['hi']>e['lo']]


def notify(events,windows):
    waits=[e for e in events if e['signal']=='wait'];records=[e for e in events if e['signal']=='record']
    computes=[e for e in events if e['compute']];details=[];per=[]
    indices={};streams=defaultdict(list)
    for x in computes:streams[(x['pid'],x['stream'])].append(x)
    by_stream={key:Neighbors(values) for key,values in streams.items()}
    for e in waits:
        scope=e['pid']
        if scope not in indices:
            indices[scope]=[Neighbors([x for x in items if x['pid']==scope]) for items in (records,computes)]
        rec,comp=indices[scope];over=overlaps(e['lo'],e['hi'],comp.during(e))
        nearby=rec.during(e)
        matches=[x for x in nearby if e['notify_id'] is not None and x['notify_id']==e['notify_id'] and x['model']==e['model']]
        candidates=[compact(x) for x in records if e['notify_id'] is not None and e['call'] is not None and
                    (x['pid'],x['model'],x['call'],x['notify_id'])==(e['pid'],e['model'],e['call'],e['notify_id'])]
        details.append(dict(**compact(e),raw=e['raw'],status=e['state'],csv_join=e['csv_join'],csv_lines=e['csv_lines'],
                            duration_us=e['us'],notify_id=e['notify_id'],csv_fields=e['csv_fields'],record_neighbors=rec.near(e),compute_neighbors=comp.near(e),
                            same_stream_compute=by_stream[(scope,e['stream'])].near(e) if (scope,e['stream']) in by_stream else None,
                            concurrent_record_events=[x['idx'] for x in nearby],matching_notify_id_during=[x['idx'] for x in matches],same_call_id_candidates=candidates,
                            concurrent_compute_events=[x[2] for x in over],wait_object=UNKNOWN))
    for w in windows:
        i=w['call'];start,end=map(Decimal,(w['raw_host_start_us'],w['raw_sync_end_us']))
        selected=[e for e in waits if e['call']==i]
        q=[e for e in events if e['call']==i and e['category']=='GroupQuantLinear']
        intervals=union((e['lo'],e['hi']) for e in selected);gi=union((e['lo'],e['hi']) for e in q)
        both=[(max(a,c),min(b,d)) for a,b in intervals for c,d in gi if a<d and c<b]
        per.append(dict(call=i,count=len(selected),sum_us=fsum(e['us'] for e in selected),wait_union_us=measure(intervals),
                        groupquant_union_us=measure(gi),intersection_us=measure(both),wait_union_without_groupquant_us=measure(intervals)-measure(both),
                        wait_union_fraction_of_host=measure(intervals)/float(end-start) if end>start else None,
                        first_start_offset_us=float(intervals[0][0]-start) if intervals else None,
                        last_end_before_sync_return_us=float(end-intervals[-1][1]) if intervals else None))
    byindex={e['idx']:e for e in events}
    for item in details:
        q=[byindex[idx] for idx in item['concurrent_compute_events'] if byindex[idx]['category']=='GroupQuantLinear']
        start,end=map(Decimal,(item['start'],item['end']))
        item['groupquant_events']=[e['idx'] for e in q];item['groupquant_overlap_us']=measure((lo,hi) for lo,hi,_ in overlaps(start,end,q))
    return dict(wait_count=len(waits),record_count=len(records),waits=details,records=[dict(**compact(e),raw=e['raw'],notify_id=e['notify_id']) for e in records],
                calls=per,critical_path=UNKNOWN,wait_object=UNKNOWN)


class Graph:
    def __init__(self,nodes):
        counts=Counter(n['name'] for n in nodes)
        self.nodes={n['name']:n for n in nodes if counts[n['name']]==1};self.up=defaultdict(list);self.down=defaultdict(list)
        for name,node in self.nodes.items():
            for edge in node['inputs']:
                if not isinstance(edge,str) or edge.startswith('^') or edge.endswith(':-1'):continue
                src=edge_name(edge)
                if src in self.nodes:self.down[src].append(name);self.up[name].append(src)

    def paths(self,name,reverse,target,via,limit=14):
        adjacent=self.up if reverse else self.down;todo=deque([[name]]);seen={name};found=[]
        while todo:
            path=todo.popleft()
            if len(path)>limit:continue
            for dest in adjacent[path[-1]]:
                if dest in seen:continue
                seen.add(dest);kind=trace.norm(self.nodes[dest]['type']);walk=path+[dest]
                if target(kind):found.append(walk)
                elif via(kind):todo.append(walk)
        return found


def desc(node):
    if node is None:return None
    return {key:node[key] for key in ('name','type','inputs','input_desc','output_desc','serialized_descriptors')}


def logical_desc(items):
    if not items:return None
    d=items[0];dim=d.get('shape',{});dim=dim[0] if isinstance(dim,list) and len(dim)==1 else dim
    fmt=trace.norm(d.get('layout',d.get('format','')))
    if fmt not in ('nd','formatnd') or not isinstance(dim,dict):return None
    try:values=[int(x) for x in dim.get('dim',[])]
    except (ValueError,TypeError):return None
    return tuple(values[-2:]) if len(values)>=2 and min(values)>0 and all(v==1 for v in values[:-2]) else None


def mkn(a,y):
    return (a[0],a[1],y[1]) if a and y and a[0]==y[0] else None


def shape_evidence(node,fields):
    seen=set()
    if node:
        shape=mkn(logical_desc(node['input_desc']),logical_desc(node['output_desc']))
        if shape:seen.add(shape)
    for raw in fields:
        f={trace.norm(k):v for k,v in raw.items()}
        pair=[]
        for side in ('input','output'):
            dims=shapes(f.get(side+'shapes',f.get(side+'shape')))
            pair.append(logical_matrix(dims[0],f,side) if dims else None)
        shape=mkn(*pair)
        if shape:seen.add(shape)
    return sorted(seen)


def projection(name,node,fields,audit,cases):
    candidates=[r for r in audit if r.get('name')==name and r.get('ge_type')=='DFlashGroupQuantLinear']
    shapes_seen=shape_evidence(node,fields);matches=[];shape=None;role=layer=UNKNOWN
    reason='no unique binding';node_ok=node is not None and node['type']=='DFlashGroupQuantLinear'
    if len(candidates)==1 and (node_ok or (node is None and len(shapes_seen)==1)):
        record=candidates[0]
        for case in cases:
            m,k,n=case['shape'];w=case['inputs']['w_nz.bin'];s=case['inputs']['s_gn.bin']
            if (record['nz_sha256'],record['scale_gn_sha256'],record['k'],record['n'],record['nz_bytes'])==(w['sha256'],s['sha256'],k,n,w['bytes']):
                matches.append(case)
    if len(matches)==1 and (not shapes_seen or shapes_seen==[tuple(matches[0]['shape'])]):
        case=matches[0];shape=case['shape'];layer=case['name'];role=layer.split('-')[-1]
        reason=('exact OM name/type' if node_ok else 'exact deployment audit name/type + logical profile shape')+' + immutable NZ/Scale SHA pair + accepted projection contract'
    elif len(shapes_seen)==1 and not candidates and (node_ok or node is None):
        compatible=[c for c in cases if tuple(c['shape'])==shapes_seen[0]]
        roles={c['name'].split('-')[-1] for c in compatible}
        if len(roles)==1:
            shape=list(shapes_seen[0]);role=next(iter(roles));layer=compatible[0]['name'] if len(compatible)==1 else UNKNOWN
            reason='logical input/output shape + contract; shared shapes do not identify layer'
    return dict(name=name,projection=layer,role=role,mkn=shape,basis=reason,logical_shapes=shapes_seen,
                weight_bytes=shape[1]*shape[2] if shape else None,scale_bytes=shape[1]//128*shape[2]*2 if shape else None,
                graph=desc(node),immutable_node=candidates,contract_matches=[c['name'] for c in matches],csv_descriptors=fields)
