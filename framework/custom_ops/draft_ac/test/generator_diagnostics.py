"""Read-only tracing evidence. Never changes Dynamo config or suppresses errors."""
import ast
import hashlib
import importlib
import inspect
import json
import linecache
from pathlib import Path
import sys
import traceback
import types
from common import record,write_json


def code_record(code, line=None):
    path=Path(code.co_filename)
    return dict(function=code.co_name,qualname=getattr(code,'co_qualname',code.co_name),file=code.co_filename,
                first_line=code.co_firstlineno,line=line or code.co_firstlineno,
                generator=bool(code.co_flags & inspect.CO_GENERATOR),
                source=linecache.getline(code.co_filename,line or code.co_firstlineno).rstrip(),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None)


def exception_report(error):
    chain=[];seen=set()
    while error is not None and id(error) not in seen:
        seen.add(id(error)); frames=[]; candidates=[];tb=error.__traceback__
        while tb:
            frames.append(code_record(tb.tb_frame.f_code,tb.tb_lineno))
            # Inspect types only: do not repr tensors or invoke user properties.
            for name,value in list(tb.tb_frame.f_locals.items()):
                code=None
                if isinstance(value,types.FrameType): code=value.f_code
                elif isinstance(value,types.CodeType): code=value
                elif isinstance(value,types.FunctionType): code=value.__code__
                if code is not None and code.co_flags & inspect.CO_GENERATOR:
                    candidates.append(dict(local=name,**code_record(code)))
            tb=tb.tb_next
        chain.append(dict(type=type(error).__name__,message=str(error),frames=frames,generator_candidates=candidates))
        error=error.__cause__ or error.__context__
    return chain


def save_exception(error,root):
    write_json(root/'exception.json',dict(status='FAIL',exceptions=exception_report(error)))
    (root/'traceback.txt').write_text(''.join(traceback.format_exception(type(error),error,error.__traceback__)))


def environment_report(root):
    import os
    import torch
    import torch._dynamo
    if torch._dynamo.config.suppress_errors:
        raise ValueError("Dynamo suppress_errors must remain disabled for formal export")
    from full_interface import module_identity
    result=dict(python=sys.version,executable=sys.executable,torch=module_identity(torch),
                environment={k:os.environ.get(k) for k in ('TORCH_LOGS','TORCHDYNAMO_VERBOSE','ASCEND_CUSTOM_OPP_PATH','LD_LIBRARY_PATH')},
                dynamo_config={k:getattr(torch._dynamo.config,k,None) for k in ('suppress_errors','enable_faithful_generator_behavior')})
    for name in ('torch_npu','torchair','transformers'):
        module=importlib.import_module(name);result[name]=module_identity(module)
    # Locate the installed print/fallback branch, not an inferred exception cause.
    module=importlib.import_module('transformers.utils.auto_docstring')
    source=Path(module.__file__).read_text(); lines=source.splitlines();locations=[]
    tree=ast.parse(source)
    for index,line in enumerate(lines,1):
        if 'Config not found' in line:
            functions=[n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.lineno<=index<=n.end_lineno]
            locations.append(dict(line=index,functions=[n.name for n in functions],
                excerpt='\n'.join(f'{i+1}: {lines[i]}' for i in range(max(0,index-9),min(len(lines),index+4)))))
    result['config_message']=dict(module=module_identity(module),locations=locations,
        dflash_alias=str(getattr(module,'HARDCODED_CONFIG_FOR_MODELS',{}).get('dflash-v1')),
        interpretation='documentation config lookup; not established as generator exception cause')
    import native_base
    result['write_guard']=dict(module=module_identity(native_base),source=inspect.getsource(native_base.protect_writes))
    from qwen35_dflash.ascend310p import draft_ac_export_pipeline
    result['export_pipeline']=module_identity(draft_ac_export_pipeline)
    write_json(root/'environment.json',result)
    return result
