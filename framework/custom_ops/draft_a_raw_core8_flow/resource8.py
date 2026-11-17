"""New canonical 8-core contracts. No seven-core resource equality references."""
import json
from pathlib import Path
import tempfile
from c8_common import c
from sidecar_atc import ORDER,FIELDS,STRINGS,unique_object,collect_host_evidence
from a3_launch import pipeline_plan,scale_plan,selected_k_tile
from gate_prefetch_contract import check_plan


def schedule(p):
    for k in ('available_cores','block_dim','core_limit','n','n_tiles','tile_n'):
        if type(p.get(k)) is not int:raise ValueError('integer schedule field required: '+k)
    if (p['available_cores'],p['block_dim'],p['core_limit'])!=(8,8,0):raise ValueError('requires actual 8/8 cores and core_limit=0; no fallback')
    tiles=p['n_tiles']
    if p.get('policy')!='n-tile-cyclic-v1' or p['tile_n']!=64 or p['n']%64 or tiles!=p['n']//64 or tiles<8:raise ValueError('invalid N64 tile domain')
    rows=p.get('assignments');seen=[]
    if not isinstance(rows,list) or len(rows)!=8:raise ValueError('eight assignment rows required')
    for b,row in enumerate(rows):
        wanted=list(range(b,tiles,8))
        expected=dict(block=b,tile_count=len(wanted),first_column=b*64,last_tile_column=wanted[-1]*64,tile_stride_columns=512)
        if row!=expected or any(type(x) is not int for x in row.values()):raise ValueError('assignment mismatch')
        actual=[(row['first_column']+i*row['tile_stride_columns'])//64 for i in range(row['tile_count'])]
        seen.extend(actual)
    if len(seen)!=tiles or len(set(seen))!=tiles or set(seen)!=set(range(tiles)):raise ValueError('tile overlap/omission')
    return dict(cores=8,tiles=tiles,coverage='EXACT_ONCE',scope='Host cyclic plan, not hardware trace')


def config(cfg,variant):
    if variant not in ('B','R') or cfg.get('core_limit')!=0 or cfg.get('gate_copy_mode')!='loop' or cfg.get('gate_prefetch_mode')!=('off' if variant=='B' else 'raw'):
        raise ValueError('B8/R8 configuration mismatch')


def canonical(p,cfg,variant):
    config(cfg,variant);coverage=schedule(p)
    shape=tuple(p[k] for k in ('m','k','n'));m,k,n=shape
    if any(type(x) is not int or x<=0 for x in shape):raise ValueError('invalid shape')
    tk=selected_k_tile(shape,cfg['k_tile']);base=pipeline_plan(k,n,cfg['pipeline_mode'],tk,cfg['gate_prefetch_mode'],m)
    scale=scale_plan(shape,cfg['scale_mode']);extra=20480 if shape==(80,2560,2048) and cfg['kv_m80_mode']=='a-ub' else 0
    base['user_ub_bytes']+=extra+scale['broadcast_bytes']+scale['sdk_reserved_bytes']
    if any(p.get(key)!=value for key,value in base.items()) or p.get('scale')!=scale:raise ValueError('canonical pipeline/Scale/UB mismatch')
    if p['user_ub_bytes']+p['matmul_ub_bytes']!=262144:raise ValueError('UB total differs')
    if p['tile_k']!=tk or p['tile_m']!=m or p['weight_reuse_rows']!=m:raise ValueError('M/K tile contract differs')
    if shape==(16,2560,19456):check_plan(p,cfg['gate_prefetch_mode'])
    return coverage


def shape_keys(gear):
    if gear not in (16,64):raise ValueError('unsupported gear')
    return {','.join(map(str,s)) for s in ((gear,12800,2560),(gear+16,2560,2048),(16,2560,4096),(16,4096,2560),(16,2560,19456),(16,9728,2560))}


def resources(sidecar,out,gear,cfg,variant):
    result=collect_host_evidence(sidecar,out,gear,cfg)
    if set(result['plans'])!=shape_keys(gear):raise ValueError('six canonical shapes required')
    result['core8']={key:canonical(p,cfg,variant) for key,p in result['plans'].items()}
    return result


def compare(b,r):
    if set(b)!=set(r):raise ValueError('B8/R8 shape inventory differs')
    from gate_prefetch_contract import match_plans
    for key,x in b.items():
        y=r[key];schedule(x);schedule(y)
        if key=='16,2560,19456':match_plans(x,y)
        else:
            if x['gate_prefetch']!={'version':1,'mode':'off','enabled':False} or y['gate_prefetch']!={'version':1,'mode':'raw','enabled':False}:raise ValueError('non-Gate prefetch changed')
            if {k:v for k,v in x.items() if k!='gate_prefetch'}!={k:v for k,v in y.items() if k!='gate_prefetch'}:raise ValueError('non-Gate resource/Workspace changed')


def probe_view(path,out):
    # Two GetWorkspaceSize calls may share one cached Host callback. All emitted
    # callbacks must be complete and byte-semantically identical for this shape.
    raw=Path(path).read_bytes()
    if not raw.endswith(b'\n'):raise ValueError('incomplete probe sidecar')
    lines=raw.decode().splitlines()
    if len(lines) not in (6,12):raise ValueError('probe requires one/two complete callbacks')
    blocks=[]
    for start in range(0,len(lines),6):
        block=[]
        for i,line in enumerate(lines[start:start+6]):
            obj=json.loads(line,object_pairs_hook=unique_object);name=ORDER[i]
            if set(obj)!={'record','payload'} or obj['record']!=name or set(obj['payload'])!=FIELDS[name]:raise ValueError('probe sidecar schema/order')
            for k,v in obj['payload'].items():
                if type(v) is not (str if k in STRINGS else bool if k=='enabled' else int):raise ValueError('probe sidecar type')
            block.append(obj)
        blocks.append(block)
    if any(x!=blocks[0] for x in blocks):raise ValueError('probe callback plan drift')
    Path(out).write_text(''.join('DFLASH_GROUP_QUANT_'+x['record']+' '+json.dumps(x['payload'])+'\n' for x in blocks[0]))
    return len(blocks)


def verify_resource(record,gear,cfg,variant):
    with tempfile.TemporaryDirectory(prefix='core8-resource-') as d:
        actual=resources(record['source'],Path(d)/'view',gear,cfg,variant)
    for key in ('plans','core8','source','sidecar_schema'):
        if actual[key]!=record[key]:raise ValueError('8-core resource evidence changed')
    c.verify(record['legacy_view'])
