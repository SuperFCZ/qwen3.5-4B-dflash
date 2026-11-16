"""Exact source audit of the independent fresh-export worker and export helper."""
import ast
from split_atc import c,HERE


def replaced(text,old,new):
    if text.count(old)!=1:raise ValueError('upstream export source drift: '+old[:80])
    return text.replace(old,new)


def worker_text():
    text=(c.HERE/'worker.py').read_text()
    text=replaced(text,'import gc,os,sys,importlib.util','import gc,os,sys')
    text=replaced(text,'import raw_common as c','from split_atc import SplitAtcRunner,one_capture\nimport raw_common as c')
    text=replaced(text,'    boundary=None','    boundary=None;capture=None\n    result.update(variant=req["variant"],gear=req["gear"],continuation_sources=req["continuation_sources"],private_opp=req["private_opp"])')
    text=replaced(text,'        c.clean();c.verify_tree(req[\'sources\']);c.verify_tree(req[\'evidence\'])','        c.clean();c.verify_tree(req);c.verify_opp(req["private_opp"])')
    old="""        export_spec=importlib.util.spec_from_file_location('draft_a_raw_existing_export',c.AC/'test/run.py')
        exporter=importlib.util.module_from_spec(export_spec);export_spec.loader.exec_module(exporter)
        om,artifacts=exporter.export_one(selected,root,a.atc,draft_only=True)"""
    new="""        from split_export import export_one
        capture=SplitAtcRunner(root/'atc-streams',req['atc_binary'],root,identity_text=req['compiler_identity_text'])
        om,artifacts=export_one(selected,root,a.atc,draft_only=True,runner=capture)"""
    text=replaced(text,old,new)
    text=replaced(text,"        result['atc_command']=deployment['graphs'][0]['atc_command']", """        result['atc_command']=deployment['graphs'][0]['atc_command']
        split=one_capture(capture.records,result['atc_command'])
        result['compiler_stdout_log']=result['atc_log']
        if atc_log.read_bytes()!=c.verify(split['stdout']).read_bytes():raise ValueError('compiler stdout copy differs')
        result['atc_log']=split['stdout'];result['atc_stderr']=split['stderr']""")
    text=replaced(text,"    finally:\n        try:","    finally:\n        result['atc_captures']=capture.records if capture else []\n        try:")
    text=replaced(text,"            c.verify_tree(req['sources']);c.verify_tree(req['evidence'])","            c.verify_tree(req);c.verify_tree(result);c.verify_opp(req['private_opp'])")
    return text


def helper_text():
    text=(c.AC/'test/run.py').read_text();tree=ast.parse(text)
    n=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='export_one')
    fn='\n'.join(text.splitlines()[n.lineno-1:n.end_lineno])+'\n'
    fn=replaced(fn,'*,draft_only=False,air_only=False','*,runner,draft_only=False,air_only=False')
    fn=replaced(fn,"    stages['atc']='RUNNING';write_json(folder/'export-stages.json',stages)","    # A real custom Runner must not enable the compiler's test-double ABI escape.\n    from qwen35_dflash.ascend310p.runtime_input_export import validated_runtime_input_abi\n    for graph in air['graphs']:\n        validated_runtime_input_abi(graph,required=True,allow_test_double=False)\n    stages['atc']='RUNNING';write_json(folder/'export-stages.json',stages)")
    fn=replaced(fn,"extra_args=['--precision_mode=must_keep_origin_dtype','--deterministic=0'])","extra_args=['--precision_mode=must_keep_origin_dtype','--deterministic=0'],runner=runner,atc_identity=runner.identity_text)")
    return '"""A-only copy of export_one with explicit Runner injection; no shared changes."""\nimport gc\nfrom pathlib import Path\nfrom common import write_json,record,checked_file\n\n'+fn


def verify_derivation():
    for name,expected in [('worker.py',worker_text()),('split_export.py',helper_text())]:
        if (HERE/name).read_text()!=expected:raise ValueError('unapproved worker/helper change: '+name)
    return dict(worker_source=c.identity(c.HERE/'worker.py'),helper_source=c.identity(c.AC/'test/run.py'),
        change='Dedicated stream capture and real-ABI precheck only; no model/math/OPP changes')


if __name__=='__main__':verify_derivation()
