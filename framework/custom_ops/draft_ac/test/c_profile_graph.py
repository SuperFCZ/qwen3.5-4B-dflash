"""Bounded, payload-free GE/OM graph inspection. No TorchAir/model imports."""
import collections
import json
from pathlib import Path
import re

VOCAB=248320
TOKENS=re.compile(r'"(?:\\.|[^"\\])*"|[A-Za-z_][\w.]*|[-+]?\d+(?:\.\d+)?|[{}:]')


def category(kind):
    text=re.sub('[^a-z0-9]','',str(kind).lower())
    if 'dflashdraftlmheadtop1' in text: return 'custom_c'
    if 'dflashgroupquantlinear' in text: return 'custom_a'
    if 'argmax' in text: return 'argmax'
    if 'matmul' in text: return 'matmul'
    if 'transdata' in text: return 'transdata'
    if 'memcpy' in text: return 'memcpy'
    if 'cast' in text: return 'cast'
    if 'memset' in text or 'atomicaddrclean' in text: return 'memset'
    if 'reshape' in text or 'transpose' in text or 'identity' in text: return 'view'
    return 'other'


def first(value,default=None):
    return value[0] if isinstance(value,list) and value else default if value is None else value


def values(value):
    if isinstance(value,dict):
        for key,item in value.items():
            if key.lower() not in ('data','tensor_content'):
                yield from values(item)
    elif isinstance(value,list):
        for item in value: yield from values(item)
    else: yield value


def edge_name(edge):
    return re.sub(r':-?\d+$','',str(edge).lstrip('^'))


def redact(value):
    if isinstance(value,dict): return {k:redact(v) for k,v in value.items() if k.lower() not in ('data','tensor_content','weight_data')}
    if isinstance(value,list): return [redact(v) for v in value]
    if isinstance(value,str) and len(value)>4096: return '<omitted oversized attribute string>'
    return value


def node_summary(node):
    # Preserve all descriptors/attributes except tensor payload fields.
    return dict(name=first(node.get('name'),''),type=first(node.get('type',node.get('op')),''),
                inputs=node.get('input',[]),input_desc=node.get('input_desc',node.get('inputDesc',[])),
                output_desc=node.get('output_desc',node.get('outputDesc',[])),attributes=redact(node.get('attr',{})))


def read_pbtxt(path):
    """GE TextFormat, bounded lines; skip enormous Const.data without loading it."""
    nodes=[]; stack=[]; pending=None; current=None; line_no=0
    with Path(path).open(encoding='utf-8') as stream:
        while True:
            line=stream.readline(65536)
            if not line: break
            line_no+=1
            payload=bool(re.match(r'\s*(data|tensor_content)\s*:',line))
            if not line.endswith('\n') and len(line)==65536:
                if not payload: raise ValueError(f'oversized non-payload pbtxt line {line_no}')
                while line and not line.endswith('\n'): line=stream.readline(65536)
            if payload: continue
            for token in TOKENS.findall(line):
                if token==':': continue
                if token=='{':
                    obj={}
                    if current is None:
                        if pending in ('op','node'): current=obj; stack=[obj]
                    else:
                        stack[-1].setdefault(pending,[]).append(obj);stack.append(obj)
                    pending=None
                elif token=='}':
                    if current is not None:
                        stack.pop()
                        if not stack: nodes.append(node_summary(current));current=None
                    pending=None
                elif pending is None: pending=token
                else:
                    if current is not None:
                        if token.startswith('"'):
                            try: value=json.loads(token)
                            except ValueError: value=token[1:-1]
                        elif re.fullmatch(r'-?\d+',token): value=int(token)
                        else: value=token
                        stack[-1].setdefault(pending,[]).append(value)
                    pending=None
    if current is not None: raise ValueError('unterminated GE op')
    if not nodes: raise ValueError('no GE op messages parsed')
    return nodes


def read_om_json(path):
    if Path(path).stat().st_size>256*1024*1024: raise ValueError('OM JSON exceeds 256 MiB analysis limit; inspect raw artifact')
    data=json.loads(Path(path).read_text());nodes=[]
    def visit(value):
        if isinstance(value,dict):
            if isinstance(value.get('op'),list):
                nodes.extend(node_summary(n) for n in value['op'] if isinstance(n,dict) and 'name' in n)
            for key,item in value.items():
                if key not in ('op','data','tensor_content'): visit(item)
        elif isinstance(value,list):
            for item in value: visit(item)
    visit(data)
    if not nodes: raise ValueError('no OM graph op records parsed')
    return nodes


def summarize(nodes,stage):
    table={n['name']:n for n in nodes}
    if len(table)!=len(nodes): raise ValueError('ambiguous duplicate node names across graphs')
    counts=collections.Counter(n['type'] for n in nodes)
    heads={n['name'] for n in nodes if category(n['type'])=='custom_c' or
           (category(n['type']) in ('matmul','argmax') and any(str(v)==str(VOCAB) for v in values(n)))}
    # A weight descriptor may be on the operand producer rather than MatMul.
    for n in nodes:
        if category(n['type'])=='matmul':
            for edge in n['inputs']:
                producer=table.get(edge_name(edge),{})
                if any(str(v)==str(VOCAB) for v in values(producer)): heads.add(n['name'])
    neighbors={name:set() for name in table}
    for n in nodes:
        for edge in n['inputs']:
            parent=edge_name(edge)
            if parent in table: neighbors[n['name']].add(parent);neighbors[parent].add(n['name'])
    related=set(heads);front=set(heads)
    for _ in range(4):
        next_front=set()
        for name in front:
            for other in neighbors[name]-related:
                kind=table[other]['type'];cat=category(kind)
                if cat in ('argmax','transdata','cast','memcpy','memset','view') or kind in ('Const','FileConstant'):
                    related.add(other)
                    if kind not in ('Const','FileConstant'): next_front.add(other)
        front=next_front
    return dict(status='PARSED',stage=stage,node_count=len(nodes),op_counts=dict(counts),
        category_counts=dict(collections.Counter(category(n['type']) for n in nodes)),
        head_anchors=sorted(heads),head_related_nodes=[table[name] for name in sorted(related)],
        classification_basis='full-vocabulary descriptors / Custom C plus bounded conversion edges; not largest-MatMul guessing',
        limitation='pre-ATC graph is not executed kernel inventory' if stage=='pre_atc' else 'OM metadata does not expose work inside a fused kernel')
