"""Bounded, payload-free GE/OM graph inspection. No TorchAir/model imports."""
import collections
import json
import io
from pathlib import Path
import re

VOCAB=248320



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


def valid_items(value):
    for key,item in value.items():
        if not isinstance(key,str) or not key:
            raise ValueError('invalid graph field key: '+repr(key))
        yield key,item


def values(value):
    if isinstance(value,dict):
        for key,item in valid_items(value):
            if key.lower() not in ('data','tensor_content'):
                yield from values(item)
    elif isinstance(value,list):
        for item in value: yield from values(item)
    else: yield value


def edge_name(edge):
    return re.sub(r':-?\d+$','',str(edge).lstrip('^'))


def redact(value):
    if isinstance(value,dict): return {k:redact(v) for k,v in valid_items(value) if k.lower() not in ('data','tensor_content','weight_data')}
    if isinstance(value,list): return [redact(v) for v in value]
    if isinstance(value,str) and len(value)>4096: return '<omitted oversized attribute string>'
    return value


def node_summary(node):
    # Preserve all descriptors/attributes except tensor payload fields.
    return dict(name=first(node.get('name'),''),type=first(node.get('type',node.get('op')),''),
                inputs=node.get('input',[]),input_desc=redact(node.get('input_desc',node.get('inputDesc',[]))),
                output_desc=redact(node.get('output_desc',node.get('outputDesc',[]))),attributes=redact(node.get('attr',{})),
                serialized_descriptors=serialized_descriptors(node.get('attr',{})))


def serialized_descriptors(attributes):
    """TorchAir GraphDef wraps GE TensorDesc text in attr.value.s strings.

    Decode only complete, bounded TensorDesc-shaped messages. Never treat
    arbitrary string fragments/braces as outer nodes or guess truncated shapes.
    """
    from profile_pbtxt import read
    result=[]
    for text in values(attributes):
        if not isinstance(text,str) or len(text)>4096 or not re.match(r'\s*name\s*:',text):continue
        try:
            desc=read(io.StringIO(text));dtype=first(desc.get('dtype'))
            shape=first(desc.get('shape'),{})
            dims=shape.get('dim') if isinstance(shape,dict) else None
            if (set(desc)<={'name','dtype','shape','layout','attr'} and isinstance(dtype,str) and dtype.startswith('DT_')
                and isinstance(dims,list) and dims and all(type(d) is int and d>=-1 for d in dims)):
                result.append(redact(desc))
        except ValueError:continue
    return result


def read_pbtxt(path):
    """Parse TextFormat fields/messages/lists; never recover by inventing keys."""
    from profile_pbtxt import read
    try:
        with Path(path).open(encoding='utf-8') as stream:document=read(stream)
    except ValueError as error:raise ValueError(str(path)+': '+str(error)) from error
    nodes=[]
    def visit(value):
        if isinstance(value,dict):
            for key,items in valid_items(value):
                if key in ('op','node'):
                    for item in items:
                        if isinstance(item,dict) and 'name' in item:nodes.append(node_summary(item))
                else:visit(items)
        elif isinstance(value,list):
            for item in value:visit(item)
    visit(document)
    if not nodes:raise ValueError('no GE op messages parsed')
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
    def vocabulary_shape(node):
        def dimensions(value):
            if isinstance(value,list):
                return any(dimensions(v) for v in value)
            if isinstance(value,dict):
                for key,item in valid_items(value):
                    if key=='shape':
                        shapes=item if isinstance(item,list) else [item]
                        for shape in shapes:
                            if isinstance(shape,dict) and any(str(d)==str(VOCAB) for d in shape.get('dim',[])):return True
                    elif key not in ('attr','attributes') and dimensions(item):return True
            return False
        return any(dimensions(node.get(key,[])) for key in ('input_desc','output_desc','serialized_descriptors'))
    heads={n['name'] for n in nodes if category(n['type'])=='custom_c' or
           (category(n['type']) in ('matmul','argmax') and vocabulary_shape(n))}
    # A weight descriptor may be on the operand producer rather than MatMul.
    for n in nodes:
        if category(n['type'])=='matmul':
            for edge in n['inputs']:
                producer=table.get(edge_name(edge),{})
                if vocabulary_shape(producer): heads.add(n['name'])
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
