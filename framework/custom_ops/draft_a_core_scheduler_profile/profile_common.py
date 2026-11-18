"""Admission of immutable inputs to an instrumented replay (never imports ACL)."""
from pathlib import Path
import sys
import shutil

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'draft_a_core_scheduler_om_resume'))
import resume
from resume import c,require,flow

FLAGS=('--ascendcl=on','--runtime-api=on','--task-time=on','--aicpu=on','--ai-core=on','--aic-mode=task-based','--aic-metrics=PipeUtilization')


def sources():
    used=[c.AC/'test'/name for name in ('c_profile_graph.py','c_profile_trace.py','profile_device_assignment.py','profile_step_evidence.py','profile_pbtxt.py')]
    return [c.identity(p) for p in sorted(HERE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts]+[c.identity(p) for p in used]+resume.sources()+flow.sources()


def admit(old,continuation):
    stages,history,_=resume.historical(old)
    src=stages['source']['binding'];prior=c.read(c.verify(src['host_source']))['binding']['prior']
    require(stages['export']['attribution']=='ACTUAL_ATC_SCHEDULES_IDENTICAL','different experiment')
    refs={};reports={}
    for phase in resume.PHASES:
        path=continuation/phase/'report.json';r=c.read(path);c.verify_tree(r)
        require(r['old_root']==str(old) and r['history']==history and r['predecessors']==refs and r['sources']==resume.sources() and
                r['phase']==phase and r['status']==resume.STATUS[phase] and not r.get('error') and not r.get('integrity_error'),'resume evidence: '+phase)
        if phase!='admit':flow.inventory(phase,r,stages)
        refs[phase]=c.identity(path);reports[phase]=r
    selected=stages['build']['variants']['A'];env=flow.build_env(prior,selected);cann=prior['current']['cann_root']
    extra=reports['admit']['extra_dsos'];resume.combined(selected,cann,extra,c.read(resume.POLICY))
    for row in extra.values():resume.host.dsos.verify_record(row)
    cases={}
    for gear in ('16','64'):
        spec=prior['current']['evidence']['H']['gears'][gear];model=stages['export']['models']['A/'+gear]['om']
        samples=[]
        for phase in ('full-smoke','benchmark'):
            for row in reports[phase]['runs']:
                if (row['variant'],row['gear'])!=('A',gear):continue
                require(row['om']==model and row['runner']==selected['om']['runner'],'mixed OM/runner')
                folder=c.verify(row['execution']).parent
                for side in ('inputs','golden'):
                    for name,item in spec[side].items():require(c.identity(folder/side/(name+'.bin'))==item,'historical replay data drift')
                validation=flow.verify_run(row,spec,0,phase=='benchmark')
                require(validation==row['validation'],'historical verification drift')
                if phase=='benchmark':samples.extend(validation['timing']['samples_ms'])
        baseline=flow.stats(samples,90)
        require(baseline==reports['benchmark']['statistics'][gear]['A'],'baseline statistics drift')
        cases[gear]=dict(spec=spec,om=model,baseline=baseline,workspace=validation['workspace_bytes'],weights=validation['model_weight_bytes'])
    tool=shutil.which('msprof',path=env['PATH']) or str(Path(cann)/'tools/profiler/bin/msprof')
    require(Path(tool).is_file() and Path(cann).resolve() in Path(tool).resolve().parents,'msprof absent/outside bound toolkit')
    return dict(history=history,resume=refs,selected=selected,extras=extra,cann=cann,env=env,cases=cases,
                msprof=c.identity(tool),msprof_alias=tool,atc=prior['current']['evidence']['tools']['atc'])
