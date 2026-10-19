"""Best-effort graph metadata only; identity gates live outside this helper."""
from pathlib import Path
import subprocess
from common import write_json
from c_profile_graph import read_pbtxt, read_om_json, summarize
from fp16_w8_profile import graph_regions


def inspect(case,out,output,*,existing=None,atc=None,env=None):
    nodes=[];origin='UNKNOWN'
    pb=[p for p in case['payloads'] if p.name=='dynamo.pbtxt']
    out['pre_atc_graph']=dict(status='UNPARSED',reason='no unique dynamo.pbtxt')
    if len(pb)==1:
        try:
            parsed=read_pbtxt(pb[0]);summary=summarize(parsed,'pre_atc')
            nodes=parsed;origin='pre_atc';out['pre_atc_graph']=summary
        except Exception as error:
            out['pre_atc_graph']=dict(status='UNPARSED',path=str(pb[0]),error=f'{type(error).__name__}: {error}')
    om=Path(existing)/'om.json' if existing else None
    out['compiled_graph']=dict(status='UNPARSED',reason='no existing compiled metadata; not required for replay')
    if atc:
        om=Path(output)/'om.json';cmd=[atc,'--mode=1','--om='+str(case['om']),'--json='+str(om)]
        write_json(Path(output)/'inspect-command.json',cmd)
        try:
            with (Path(output)/'inspect.log').open('w') as log:
                subprocess.run(cmd,env={**env,'ASCEND_WORK_PATH':str(output)},cwd=output,stdout=log,stderr=subprocess.STDOUT,check=True)
        except Exception as error:
            out['compiled_graph']=dict(status='UNPARSED',error=f'{type(error).__name__}: {error}');om=None
    if om and om.is_file():
        try:
            parsed=read_om_json(om);summary=summarize(parsed,'compiled_om')
            nodes=parsed;origin='compiled_om';out['compiled_graph']=summary
        except Exception as error:
            out['compiled_graph']=dict(status='UNPARSED',path=str(om),error=f'{type(error).__name__}: {error}')
    try:
        result=graph_regions(nodes) if nodes else ({},set(),set())
        out['mapping_source']=origin
        return result
    except Exception as error:
        out['mapping_source']='UNKNOWN';out['mapping_error']=f'{type(error).__name__}: {error}'
        return {},set(),set()
