"""Bounded structural review of 310P Matmul repository JSON, not capability proof."""
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re

MAX_METADATA_BYTES=32*1024*1024
MAX_NODES=2_000_000
MAX_DEPTH=64
MAX_FIELD_SAMPLES=128  # per category; exact total counts are retained


def is_repository(relative):
    parts=[p.lower() for p in relative.parts]
    return (len(parts)>=4 and 'opp' in parts and
            parts[-3:-1]==['tiling','ascend310p'] and
            bool(re.fullmatch(r'repository_ascend310p(?:3)?_\w*matmul\w*\.json',parts[-1])))


def normalized(value):
    return re.sub(r'[^a-z0-9]','',value.lower())


def category(key,value):
    """Classify observed fields, never hardware acceptance/rejection."""
    name=normalized(key)
    text=normalized(value) if isinstance(value,str) else ''
    if any(token in name or token in text for token in
           ('fixpipe','antiquant','dequant','quantmode','quantpre','deqscale','scaledfp32',
            'setquant','f322f32','f322f16','deqf16','enpartialsum')) or 'quant' in name or 'scale' in name:
        return 'conversion_related'
    if any(token in name for token in ('support','dtype','datatype','format','layout','soc','product','chip','accum','atomic')):
        return 'type_layout_support_related'
    if any(token in name for token in ('tiling','basem','basen','basek','singlecore','deptha1','depthb1','dbl0','blockdim','splitk')) or name in ('m','n','k'):
        return 'tiling_related'
    return None


def location(parts):
    pointer='/'+'/'.join(str(p).replace('~','~0').replace('/','~1') for p in parts)
    if len(pointer)<=2048:return dict(json_pointer=pointer)
    return dict(json_pointer_prefix=pointer[:2048],json_pointer_chars=len(pointer),
                json_pointer_sha256=hashlib.sha256(pointer.encode()).hexdigest(),
                locator_truncated=True)


def field_value(value):
    if isinstance(value,dict):return dict(type='object',member_count=len(value))
    if isinstance(value,list):return dict(type='array',length=len(value))
    if isinstance(value,Decimal):return dict(type='number',decimal_literal=str(value))
    if isinstance(value,str) and len(value)>512:
        return dict(type='string',prefix=value[:512],characters=len(value),
                    sha256=hashlib.sha256(value.encode()).hexdigest(),value_truncated=True)
    return dict(type=type(value).__name__,value=value)


def inspect(raw,source):
    """Source SHA + JSON pointers locate observations in the immutable snapshot.

    All nodes/fields are traversed if within explicit depth/node limits. Only
    report samples are capped. No field absence or name is a hardware verdict.
    """
    result=dict(status='UNPARSED',source=source,limit_bytes=MAX_METADATA_BYTES,
        node_limit=MAX_NODES,depth_limit=MAX_DEPTH,sample_limit_per_category=MAX_FIELD_SAMPLES,
        inspected_nodes=0,field_count=0,unmatched_field_count=0,unmatched_samples=[],
        matches={k:dict(count=0,fields=[]) for k in
                 ('conversion_related','type_layout_support_related','tiling_related')},
        compatibility='UNRESOLVED',classification='UNCLASSIFIED_JSON_CONTENT',
        unparsed=['Entire document pending structural parse'],
        semantic_unreviewed=['SDK support semantics/chip dispatch','Scale encoding and FP32 destination',
                             'FP16 x INT8 Cube acceptance','20-group on-chip accumulator lifetime',
                             'Embedded serialized configurations inside JSON strings are not recursively parsed'])
    def pairs(items):
        out={}
        for key,value in items:
            if key in out:raise ValueError('duplicate JSON key: '+key[:160])
            out[key]=value
        return out
    def constant(value):raise ValueError('nonstandard JSON number: '+value)
    def children(path,value):
        entries=value.items() if isinstance(value,dict) else enumerate(value)
        for key,child in entries:yield (*path,key),child
    try:
        if len(raw)>MAX_METADATA_BYTES:raise ValueError('metadata exceeds 32 MiB ceiling')
        if source['bytes']!=len(raw) or source['sha256']!=hashlib.sha256(raw).hexdigest():
            raise ValueError('metadata bytes differ from bound source')
        tree=json.loads(raw.decode('utf-8'),object_pairs_hook=pairs,parse_float=Decimal,parse_constant=constant)
        if not isinstance(tree,(dict,list)):raise ValueError('expected repository object or array')
        result['top_level']=field_value(tree)
        # Iterator stack bounds traversal memory instead of enqueuing all children.
        stack=[iter([((),tree)])]
        while stack:
            try:path,value=next(stack[-1])
            except StopIteration:stack.pop();continue
            result['inspected_nodes']+=1
            if result['inspected_nodes']>MAX_NODES or len(path)>MAX_DEPTH:
                result['first_unreviewed_location']=location(path)
                raise ValueError('metadata traversal node/depth limit exceeded')
            if isinstance(value,dict):
                for key,child in value.items():
                    result['field_count']+=1;kind=category(key,child)
                    if kind:
                        group=result['matches'][kind];group['count']+=1
                        if len(group['fields'])<MAX_FIELD_SAMPLES:
                            group['fields'].append(dict(**location((*path,key)),field=key[:512],
                                field_truncated=len(key)>512,observed=field_value(child)))
                    else:
                        result['unmatched_field_count']+=1
                        if len(result['unmatched_samples'])<MAX_FIELD_SAMPLES:
                            result['unmatched_samples'].append(dict(**location((*path,key)),observed=field_value(child)))
                stack.append(children(path,value))
            elif isinstance(value,list):
                stack.append(children(path,value))
            elif path and isinstance(path[-1],int):
                kind=category('',value)
                if kind:
                    group=result['matches'][kind];group['count']+=1
                    if len(group['fields'])<MAX_FIELD_SAMPLES:
                        group['fields'].append(dict(**location(path),origin='array_value',observed=field_value(value)))
        result.update(status='STRUCTURAL_REVIEW_COMPLETE',unparsed=[])
        counts={k:v['count'] for k,v in result['matches'].items()}
        if counts['conversion_related'] or counts['type_layout_support_related']:
            result['classification']='CAPABILITY_RELATED_FIELDS_REQUIRE_SOURCE_REVIEW'
        elif counts['tiling_related']:
            result['classification']='TILING_FIELDS_OBSERVED_NO_CAPABILITY_FIELD_HITS'
        result['classification_basis']=dict(field_counts=counts,
            rule='Actual parsed field names/values only; filename not used. No hits does not prove absence of encoded constraints.')
    except (ValueError,UnicodeError,RecursionError) as e:
        result.update(status='INCOMPLETE_METADATA_REVIEW',error=f'{type(e).__name__}: {e}',
                      unparsed=['Document or remaining traversal not verified; see error and first_unreviewed_location'])
    for group in result['matches'].values():group['samples_truncated']=group['count']>len(group['fields'])
    result['unmatched_samples_truncated']=result['unmatched_field_count']>len(result['unmatched_samples'])
    return result
