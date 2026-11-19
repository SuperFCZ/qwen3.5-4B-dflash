"""Read-only admission of the completed Gate8 full benchmark. Never invokes a Gate8 stage."""
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
OPS=HERE.parent
for folder in ('draft_gate_raw_atc8','draft_a_core_scheduler_profile','draft_a_core_scheduler_profile_v3'):
    sys.path.insert(0,str(OPS/folder))
import run_gate8 as gate
import profile_common as legacy
from gate_common import c,flow,require
from collect_profile import run as collect
from run_profile import artifacts
from attribution import recover
import analyze_v3 as deep

__all__=['HERE','admit','artifacts','c','collect','deep','recover','require','sources']


def sources():
    deps=[p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    for folder in ('draft_a_core_scheduler_profile_v2','draft_a_core_scheduler_profile_v3'):
        deps.extend((OPS/folder).glob('*.py'))
    return [c.identity(p) for p in sorted(deps)]+legacy.sources()+gate.sources()


def admit(root,historical_profile):
    file=root/'full-benchmark/report.json';report=c.read(file);refs={}
    state=dict(predecessors=refs,sources=gate.sources(),history_roots=report['history_roots'])
    chain=gate.load_chain(root,'full-benchmark',state)
    gate.stage_status('full-benchmark',report)
    require(report['phase']=='full-benchmark' and report['root']==str(root) and report['predecessors']==refs and report['sources']==state['sources'] and
            report['device_execution']=='COMPLETED' and report['benchmark']=='MEASURED','mixed/incomplete benchmark chain')
    gate.verify_tree(report,{});gate.validate_stage('full-benchmark',report,chain,root)
    refs['full-benchmark']=c.identity(file)
    src=chain['source']['binding'];built=chain['build']['variants'];export=chain['full-export']['models']
    prior=c.read(c.verify(src['host_source']))['binding']['prior']
    previous=c.read(c.verify(src['source_profile']));require(Path(src['source_profile']['path']).parent.parent==historical_profile,'wrong historical Off profile')
    old=c.read(c.verify(previous['predecessors']['admit']))['binding']
    require(old['cann']==src['cann'] and old['history']==src['history'],'wrong historical SDK or artifacts')
    result={}
    for mode in ('off','raw'):
        selected=built[mode];env=flow.build_env(prior,selected);cases={}
        for gear in ('16','64'):
            model=export[mode][gear];spec=src['full'][gear]['io']
            rows=[r for r in report['runs'] if (r['mode'],r['gear'])==(mode,gear)]
            require(len(rows)==3 and all(r['om']==model['om'] and r['runner']==selected['om']['runner'] and r['inputs']==spec['inputs'] and r['golden']==spec['golden'] and r['profiled'] is False for r in rows),'baseline artifacts crossed')
            baseline=report['statistics']['gears'][gear]['variants'][mode]
            require(baseline['count']==90 and baseline['samples_ms']==[v for r in rows for v in r['validation']['timing']['samples_ms']],'not fresh formal baseline')
            cases[gear]=dict(om=model['om'],deployment=model['deployment'],dump=model['dump'],spec=spec,baseline=baseline,
                workspace=rows[0]['validation']['workspace_bytes'],weights=rows[0]['validation']['model_weight_bytes'],
                atc_kernels=model['atc_kernels'],plans=model['plans'],gate_nodes=model['gate_nodes'])
        result[mode]=dict(selected=selected,cases=cases,env=env,cann=src['cann'],extras=src['extras'],atc=old['atc'],
                           msprof=old['msprof'],msprof_alias=old['msprof_alias'])
    require(result['raw']['selected']['opp']['install']!=result['off']['selected']['opp']['install'],'same private OPP')
    return dict(variants=result,reports=refs,flow_source=src['history']['source'],historical_off_profile=src['source_profile'],
                identity_scope='Gate8 manifests/ELFs/DSOs and current full Smoke/Benchmark, no build/export/execute in admit')
