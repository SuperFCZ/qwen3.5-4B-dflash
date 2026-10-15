"""A/C immutable payload checks and TorchAir's convert -> save -> dump boundary."""
from __future__ import annotations
import copy
import hashlib
import inspect
from pathlib import Path
import torch
from .utils import atomic_write_json,sha256_file

CHUNK_BYTES=4*1024*1024


class PayloadError(ValueError):
    def __init__(self,code,detail):
        self.diagnostic=dict(detail,reason_code=code)
        super().__init__(f"{code}: {detail.get('path',detail.get('node'))}")


def tensor_digest(tensor):
    if not tensor.is_contiguous(): raise ValueError('A/C immutable payload requires contiguous storage')
    flat=tensor.detach().view(-1); step=max(1,CHUNK_BYTES//tensor.element_size()); digest=hashlib.sha256()
    for start in range(0,flat.numel(),step):
        chunk=flat[start:start+step].cpu().numpy()
        digest.update(memoryview(chunk).cast('B'))
    return digest.hexdigest()


def node_info(node):
    attrs={}
    for name,value in node.attr.items():
        if name=='value' and value.HasField('t'):
            attrs[name]=dict(tensor_descriptor=str(value.t.desc),protobuf_bytes=value.ByteSize());continue
        attrs[name]=(str(value) if value.ByteSize()<=4096 else
                     dict(protobuf_bytes=value.ByteSize(),fields=[f.name for f,_ in value.ListFields()]))
    return dict(node=node.name,kind=node.type,attributes=attrs,
                output_descriptors=[str(d) for d in node.output_desc])


# Observed in ac-ge-20261014-091937: FileConstant has no inputs and one
# output. These are serialized port-name maps, not arbitrary SDK metadata.
PORT_MAP_KEYS=('_input_name_key','_input_name_value','_output_name_key','_output_name_value')


def valid_port_map(node):
    present=set(PORT_MAP_KEYS).intersection(node.attr)
    if not present: return True
    if present!=set(PORT_MAP_KEYS) or node.input or len(node.output_desc)!=1 or not node.output_desc[0].name:
        return False
    for key in PORT_MAP_KEYS:
        wanted=type(node.attr[key])();wanted.list.SetInParent()
        if key=='_output_name_key': wanted.list.s.append(node.output_desc[0].name.encode('utf-8'))
        if key=='_output_name_value': wanted.list.i.append(0)
        if node.attr[key]!=wanted: return False
    return True


def binding_check(expected,live,stage):
    """Field diff without mutating either node. Only the proven complete map may be added."""
    from google.protobuf.json_format import MessageToDict
    changes=[];allowed=[]
    def change(field,before,after,classification='protected'):
        changes.append(dict(field=field,before=before,after=after,classification=classification))
    if live is None:
        change('node',node_info(expected),None)
    else:
        for key in ('name','type'):
            if getattr(expected,key)!=getattr(live,key): change(key,getattr(expected,key),getattr(live,key))
        if list(expected.input)!=list(live.input): change('input',list(expected.input),list(live.input))
        sdk_addition=(not set(PORT_MAP_KEYS).intersection(expected.attr) and valid_port_map(live)
                      and set(PORT_MAP_KEYS)<=set(live.attr))
        for key in sorted(set(expected.attr)|set(live.attr)):
            left=expected.attr.get(key);right=live.attr.get(key)
            if left!=right:
                accepted=sdk_addition and key in PORT_MAP_KEYS and left is None
                if accepted: allowed.append(key)
                change('attributes.'+key,None if left is None else node_info(expected)['attributes'][key],
                       None if right is None else node_info(live)['attributes'][key],
                       'verified-sdk-port-map-addition' if accepted else 'protected')
        def descriptor_diff(left,right,prefix):
            if isinstance(left,dict) and isinstance(right,dict):
                for key in sorted(set(left)|set(right)): descriptor_diff(left.get(key),right.get(key),prefix+'.'+key)
            elif left!=right: change(prefix,left,right)
        for kind in ('input_desc','output_desc'):
            a,b=getattr(expected,kind),getattr(live,kind)
            if len(a)!=len(b): change(kind+'.count',len(a),len(b))
            for i,(left,right) in enumerate(zip(a,b)):
                descriptor_diff(MessageToDict(left,preserving_proto_field_name=True),
                                MessageToDict(right,preserving_proto_field_name=True),f'{kind}[{i}]')
        if not valid_port_map(expected) or not valid_port_map(live):
            change('port_map_contract','no inputs; one named output at index 0','invalid/incomplete map')
    passed=not any(c['classification']=='protected' for c in changes)
    return dict(node=expected.name,stage=stage,status='PASS' if passed else 'FAIL',
                snapshot=node_info(expected),final=node_info(live) if live is not None else None,
                field_diff=changes,allowed_changes=allowed)


def file_location(node,directory):
    directory=Path(directory); root=directory.resolve()
    detail=dict(node_info(node),directory=str(root),cwd=str(Path.cwd()),offset=node.attr['offset'].i if 'offset' in node.attr else 0)
    if 'file_path' not in node.attr or not node.attr['file_path'].s: raise PayloadError('MISSING_FILE_PATH',detail)
    try: raw=Path(node.attr['file_path'].s.decode('utf-8'))
    except (ValueError,UnicodeError) as e: raise PayloadError('INVALID_FILE_PATH',detail) from e
    if raw.is_absolute(): path=raw; policy='absolute'
    elif not directory.is_absolute() and directory.parts and raw.parts[:len(directory.parts)]==directory.parts:
        path=raw; policy='producer-directory-prefixed-relative'
    else: path=directory/raw; policy='export-directory-relative'
    path=path.resolve();detail.update(file_path=str(raw),path=str(path),path_policy=policy)
    if not path.is_relative_to(root): raise PayloadError('PATH_ESCAPE',detail)
    if 'offset' in node.attr and any(f.name!='i' for f,_ in node.attr['offset'].ListFields()):
        raise PayloadError('INVALID_OFFSET_TYPE',detail)
    if detail['offset']!=0: raise PayloadError('UNSUPPORTED_NONZERO_OFFSET',detail)
    return path,detail


def verify_payload(node,tensor,directory,expected=None):
    actual_tensor=tensor_digest(tensor); expected=expected or actual_tensor
    size=tensor.numel()*tensor.element_size()
    detail=dict(node_info(node),expected_bytes=size,expected_sha256=expected)
    if actual_tensor!=expected: raise PayloadError('IMMUTABLE_TENSOR_CHANGED',detail)
    if node.type=='Const':
        raw=node.attr['value'].t.data; actual=hashlib.sha256(raw).hexdigest()
        detail.update(actual_bytes=len(raw),actual_sha256=actual)
        if len(raw)!=size: raise PayloadError('CONST_SIZE_MISMATCH',detail)
        if actual!=expected: raise PayloadError('CONST_HASH_MISMATCH',detail)
        return dict(kind='Const',sha256=actual,bytes=size)
    path,location=file_location(node,directory);detail.update(location)
    if not path.exists(): raise PayloadError('MISSING_FILE',detail)
    if not path.is_file(): raise PayloadError('NOT_REGULAR_FILE',detail)
    detail['actual_bytes']=path.stat().st_size
    if detail['actual_bytes']!=size: raise PayloadError('FILE_SIZE_MISMATCH',detail)
    actual=sha256_file(path);detail['actual_sha256']=actual
    if actual!=expected: raise PayloadError('FILE_HASH_MISMATCH',detail)
    return dict(kind='FileConstant',path=str(path),offset=0,sha256=actual,bytes=size)


def file_layout(directory):
    root=Path(directory).resolve(); records=[]
    for p in sorted(root.glob('*')):
        children=[p]+(sorted(p.glob('*')) if p.is_dir() and not p.is_symlink() else [])
        for q in children:
            if len(records)>=128: return dict(entries=records,truncated=True)
            records.append(dict(path=str(q.relative_to(root)),symlink=q.is_symlink(),
                                bytes=q.stat().st_size if q.is_file() else None))
    return dict(entries=records,truncated=False)


def callable_info(fn):
    if not callable(fn): return dict(available=False)
    info=dict(available=True)
    try: info['signature']=str(inspect.signature(fn))
    except (ValueError,TypeError): info['signature']='unavailable'
    try:
        source=inspect.getsource(fn);info.update(source=source[:20000],source_truncated=len(source)>20000)
        path=inspect.getsourcefile(fn)
        if path: info.update(source_file=path,source_file_sha256=sha256_file(path))
    except (OSError,TypeError): pass
    return info


class PayloadLifecycle:
    def __init__(self,config,export_utils,original_convert):
        self.config=config; self.original_save=getattr(export_utils,'_save_weight2file',None)
        self.api={name:callable_info(fn) for name,fn in (
            ('_convert_data_to_const',original_convert),('_save_weight2file',self.original_save),
            ('make_export_graph',getattr(export_utils,'make_export_graph',None)))}
        self.pending=False; self.bindings={}; self.nodes={}; self.events=[]; self.binding_checks=[]; self.removed=set()

    def capture(self,graph,immutable,directory,inputs,weight_name):
        from .draft_ac_export import A,C,AUDIT,GE
        self.directory=Path(directory); self.graph=graph; self.converted=False; self.saved=False
        self.input_ids=tuple(map(id,inputs));self.weight_names=dict(weight_name)
        self.used=sum(id(value) in weight_name for value in inputs)
        edges={edge for op in graph.op if op.type in GE.values() for edge in op.input[1:] if edge in immutable}
        atomic_write_json(self.directory/'ac-torchair-fileconstant-api.json',self.api)
        self.bindings={edge:(immutable[edge],tensor_digest(immutable[edge])) for edge in edges}

    def event(self,stage,status,error=None):
        self.events.append(dict(stage=stage,status=status,files=file_layout(self.directory)))
        report=dict(status=status,events=self.events,binding_checks=self.binding_checks,nodes=[node_info(n) for n in self.nodes.values()],
                    original_fileconstants=getattr(self,'original_fileconstants',[]),
                    bindings={e:dict(dtype=str(t.dtype),shape=list(t.shape),bytes=t.numel()*t.element_size(),sha256=h)
                              for e,(t,h) in self.bindings.items()},verified=getattr(self,'verified',{}))
        if error is not None: report.update(error=str(error),failure=getattr(error,'diagnostic',
            dict(reason_code='MATERIALIZER_ERROR' if stage=='save_weight2file' else 'VALIDATION_ERROR',exception_type=type(error).__name__)))
        atomic_write_json(self.directory/'ac-payloads.json',report)
        if status=='FAIL' and hasattr(self,'audit'):
            self.audit['draft_ac']=dict(self.audit.get('draft_ac',{}),status='FAIL',error=str(error),
                                       verification_stage=stage,failure=report.get('failure'))
            atomic_write_json(self.directory/'ac-constants.json',self.audit['draft_ac'])

    def after_convert(self,graph,immutable,directory,result,inputs,weight_name,audit):
        self.graph=graph;self.immutable=immutable;self.audit=audit;self.converted=True
        by_name={n.name:n for n in graph.op}
        self.nodes={edge:(copy.deepcopy(node) if node.type=='FileConstant' else node)
                    for edge in self.bindings for node in [by_name[edge.rpartition(':')[0]]]}
        self.original_fileconstants=[node_info(n) for n in self.nodes.values() if n.type=='FileConstant']
        try:
            if tuple(map(id,inputs))!=self.input_ids or dict(weight_name)!=self.weight_names:
                raise ValueError('A/C immutable input/weight binding changed during conversion')
            for node in self.nodes.values():
                if node.type=='FileConstant':
                    if not valid_port_map(node):
                        raise PayloadError('FILE_BINDING_CHANGED',binding_check(node,node,'after_convert'))
                    file_location(node,self.directory) # Reject escapes BEFORE SDK writes.
            external=(isinstance(result,(tuple,list)) and len(result)==2 and result[0] is True)
            if external and not self.saved and any(n.type=='FileConstant' for n in self.nodes.values()) and callable(self.original_save):
                if result[1]!=self.used: raise ValueError('A/C external weight count differs')
                self.pending=True; self.audit['draft_ac']=dict(status='PENDING_MATERIALIZATION')
                self.event('after_convert_before_save','PENDING_MATERIALIZATION')
            else: self.normalize('after_convert') # Eager/Const producers still require actual verified bytes.
        except Exception as e:
            self.event('after_convert','FAIL',e);raise

    def normalize(self,stage):
        from .draft_ac_constants import normalize_ac_constants,_bound
        from .draft_ac_export import A,GE
        try:
            replaceable={op.input[1].rpartition(':')[0] for op in self.graph.op if op.type==GE[A]}
            self.audit['draft_ac']=normalize_ac_constants(self.graph,self.immutable,self.config,self.directory,
                                                        expected_hashes={e:h for e,(_,h) in self.bindings.items()})
            current={n.name:n for n in self.graph.op}
            for edge,node in list(self.nodes.items()):
                if node.type!='FileConstant': continue
                # Independently reproduce ONLY our verified path and descriptor
                # repair. Never bless all fields by copying the live node first.
                expected=copy.deepcopy(node);tensor,_=self.bindings[edge]
                _bound({expected.name:expected},edge,{edge:tensor},tensor.dtype,list(tensor.shape))
                path,_=file_location(expected,self.directory);expected.attr['file_path'].s=str(path).encode('utf-8')
                repaired=binding_check(node,expected,stage+':immutable-descriptor/path-repair')
                repaired['status']='VERIFIED_REPAIR'
                for change in repaired['field_diff']: change['classification']='verified-immutable-descriptor/path-repair'
                self.binding_checks.append(repaired)
                live=current.get(node.name)
                if live is None and node.name in replaceable:
                    self.removed.add(edge) # Proven A NZ replacement; still hash original Tensor/file at finish.
                else:
                    check=binding_check(expected,live,stage);self.binding_checks.append(check)
                    if check['status']!='PASS': raise PayloadError('FILE_BINDING_CHANGED',check)
                    expected=copy.deepcopy(live) # All fields have now been checked.
                self.nodes[edge]=expected
            self.pending=False; self.event(stage,'PASS')
        except Exception as e:
            self.event(stage,'FAIL',e);raise

    def save(self,inputs,file_path,weight_name,used_weight_num):
        try:
            if (not hasattr(self,'graph') or Path(file_path).resolve()!=self.directory.resolve()
                    or tuple(map(id,inputs))!=self.input_ids or dict(weight_name)!=self.weight_names or used_weight_num!=self.used):
                raise ValueError('A/C materializer input/weight/directory binding differs')
            bound_names={edge.rpartition(':')[0] for edge in self.bindings}
            for node in self.graph.op:
                if node.name in bound_names and node.type=='FileConstant': file_location(node,self.directory)
            for tensor,digest in self.bindings.values():
                if tensor_digest(tensor)!=digest: raise ValueError('immutable Tensor changed before materialization')
            result=self.original_save(inputs,file_path,weight_name,used_weight_num);self.saved=True
            if not self.converted: self.event('save_returned_during_convert','MATERIALIZED_NOT_YET_VERIFIED')
            if self.pending: self.normalize('after_save_before_graph_dump')
            return result
        except Exception as e:
            if hasattr(self,'directory'): self.event('save_weight2file','FAIL',e)
            raise

    def finish(self):
        if not hasattr(self,'graph'): return # Existing hook call-count gate rejects missing conversion.
        try:
            if self.pending:
                raise PayloadError('NOT_MATERIALIZED',dict(node='pending A/C FileConstant',
                    detail='expected _save_weight2file did not complete before export returned'))
            current={n.name:n for n in self.graph.op};self.verified={}
            for edge,(tensor,digest) in self.bindings.items():
                node=self.nodes[edge]; live=current.get(node.name)
                if node.type=='FileConstant' and not (live is None and edge in self.removed):
                    check=binding_check(node,live,'after_export');self.binding_checks.append(check)
                    if check['status']!='PASS': raise PayloadError('FILE_BINDING_CHANGED',check)
                self.verified[edge]=verify_payload(node,tensor,self.directory,digest)
            self.event('after_export','PASS')
        except Exception as e:
            self.event('after_export','FAIL',e);raise
