#!/usr/bin/env python3
"""One manually selected stage. Never invokes another stage or falls back to ACLNN."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import traceback
from gate_common import (HERE,STAGES,STATE,GEARS,MODES,LAYERS,SHAPE_KEY,ELIGIBLE,NO_GO,OFF_ELF,API,
                         c,flow,host,resume,require,command,sources,gate_kernel,case_contract,model_inventory,tree)
from source_gate import binding,UnprovenGraph
from export_gate import export_stage,compare_models,verify_compiled_graph
from gate_plan import gate_plan,non_gate,TAG
from measurement import perform,replay_runs,compare_gear,full_stats,inventory,comparison_text


def verify_tree(value,seen):
    """Hash each distinct absolute identity once per verification pass; conflicting records never coalesce."""
    if isinstance(value,dict):
        if value.get('kind')==c.HEAD_ALIAS_KIND:c.verify_head_alias(value);return
        if set(value)=={'path','bytes','sha256'}:
            key=value['path'];signature=(value['bytes'],value['sha256'])
            if key in seen:require(seen[key]==signature,'conflicting file identities: '+key)
            else:c.verify(value);seen[key]=signature
        else:
            for child in value.values():verify_tree(child,seen)
    elif isinstance(value,list):
        for child in value:verify_tree(child,seen)


def verify_build(built,src):
    require(set(built['variants'])==set(MODES),'missing independent Off/Raw package')
    dirs=[]
    for mode,item in built['variants'].items():
        worker=c.read(c.verify(built['workers'][mode]))
        require(worker['status']=='PRIVATE_GATE_PACKAGE_BUILT' and worker['variant']==mode and worker['result']==item,'build worker evidence mismatch')
        c.verify_opp(item['opp']);dirs.append(Path(item['opp']['install']).resolve())
        require(tree(dirs[-1])==item['package_files'],'private OPP paths, links, or payload inventory drift')
        require(item['kernel']==gate_kernel(item['opp']) and item['api']['sha256']==API and item['codec']['method']==TAG,'OPP/API/layout drift')
        require(host.envs.standard_identity(item['env'])==src['standard_env'],'standard execution environment drift')
        resume.combined(item,src['cann'],src['extras'],c.read(resume.POLICY))
        require(item['env']['opp_root']==str(dirs[-1]),'loader/OPP binding')
        require((item['kernel']['elf']['sha256']==OFF_ELF)==(mode=='off') and
                item['compiler']['binary']['sha256']==item['kernel']['elf']['sha256'],'wrong Off/Raw device artifact provenance')
    require(dirs[0]!=dirs[1] and all(dirs[i] not in dirs[1-i].parents for i in (0,1)),'private roots overlap')


def build(root,src,r,source_path):
    prior=c.read(c.verify(src['host_source']))['binding']['prior'];r.update(variants={},workers={})
    for mode in MODES:
        r['active_case']=dict(stage='build',mode=mode);folder=root/mode;package=folder/'package';folder.mkdir()
        env,_=host.envs.for_prior(prior,package/'opp',ready=False)
        req=folder/'request.json';c.write(req,dict(variant=mode,sources=sources(),output=str(package),source_report=c.identity(source_path)))
        command([sys.executable,'-B',HERE/'build_gate.py',req],folder/'builder.log',env)
        info=package/'report.json';child=c.read(info);require(child['status']=='PRIVATE_GATE_PACKAGE_BUILT','private build failure')
        r['workers'][mode]=c.identity(info);r['variants'][mode]=child['result']
    verify_build(r,src);r.pop('active_case')


def verify_models(report,src,built,full):
    models=report['models'];model_inventory(models,full)
    keys=GEARS if full else LAYERS;require(set(report['comparisons'])==set(keys),'missing plan comparisons')
    for key in keys:
        for mode in MODES:
            selected=built['variants'][mode];model=models[mode][key]
            require(model['key']==key and model['mode']==mode and model['opp_root']==selected['opp']['install'] and
                    model['private_kernel']==selected['kernel'],'model bound to a different build')
            worker=c.read(c.verify(model['worker_report']))
            require(worker['scope']==model['scope'] and worker['status']=='GATE8_'+model['scope'].upper()+'_AIR_COMPILED_NOT_EXECUTED' and not worker.get('integrity_error') and
                    all(worker[k]==model[k] for k in ('om','deployment','captures','atc_command')),'Worker/OM capture drift')
            verify_tree(worker,{})
            verify_compiled_graph(model,src,full)
            unsplit={k:flow.lines(c.verify(worker[k])) for k in ('platform','scheduling')}
            ref=src['full'][key if full else '16']['anchor']['plans']
            expected=set(ref) if full else {SHAPE_KEY}
            require(set(model['plans'])==expected,'plan scope changed')
            for tag,entry in model['plans'].items():
                query=entry['query'];p=flow.read_query(c.verify(query['platform_file']),c.verify(query['scheduling_file']),selected['opp'],'ATC','A',list(map(int,tag.split('x'))))
                for field in unsplit:
                    records=[x for x in unsplit[field] if 'x'.join(str(x[k]) for k in ('m','k','n'))==tag]
                    require(records==flow.lines(c.verify(query[field+'_file'])),'shape-split callback no longer matches original stream')
                require(p==query,'ATC Host callback mismatch')
                if tag==SHAPE_KEY:require(gate_plan(p,mode,ref[tag])==entry['contract'],'resource/Cube replay failed')
                else:non_gate(p,ref[tag])
        require(compare_models(models['off'][key],models['raw'][key],full)==report['comparisons'][key],'paired OM replay failed')


def decision(reports):
    return {g:compare_gear(g,reports['c'+g+'-forward'],reports['c'+g+'-reverse']) for g in GEARS}


def eligible(results):
    return set(results)==set(GEARS) and all(results[g]['decision']==ELIGIBLE for g in GEARS)


def stage_status(phase,r):
    status=r.get('status')
    if phase=='compare':require(status==r.get('decision')==ELIGIBLE,'BLOCKED_FULL_DRAFT_NOT_ELIGIBLE')
    else:require(status==STATE[phase],'failed predecessor: '+phase)
    require(not r.get('error') and not r.get('integrity_error'),'invalid predecessor: '+phase)


def validate_stage(phase,r,chain,root):
    if phase=='source':
        src=r['binding'];case_contract(src['cases'])
        historical=c.read(c.verify(src['history']['source']))['binding']
        require(src['cases']=={g:{n:historical['cases'][g+'/'+n] for n in LAYERS} for g in GEARS},'wrong historical case/Golden')
        require(src['off_package']==c.read(c.verify(src['flow_build']))['variants']['A']['package'] and src['host_source']==historical['host_source'],'wrong historical A package/source')
        return
    src=chain['source']['binding']
    if phase=='build':verify_build(r,src);return
    built=chain['build']
    if phase in ('export','full-export'):verify_models(r,src,built,phase=='full-export');return
    if phase=='compare':
        actual=decision(chain)
        require(actual==r['gears'] and r['decision']==(ELIGIBLE if eligible(actual) else NO_GO),'fresh decision drift')
        return
    full=phase.startswith('full-');export=chain['full-export' if full else 'export']
    inventory(r['runs'],phase)
    for row in r['runs']:
        folder=root/phase/str(row['round'])/('c'+row['gear'])/row['layer']/row['mode']
        require(str(folder)==row['folder'] and c.identity(folder/'execution.json')==row['execution'],'historical/synthetic samples not permitted')
    replay_runs(r,src,built,export)
    if not phase.endswith('smoke'):
        smoke=chain['full-smoke' if full else phase[:3]+'-smoke']
        old={(row['gear'],row['layer'],row['mode']):row['validation'] for row in smoke['runs']}
        for row in r['runs']:
            previous=old[row['gear'],row['layer'],row['mode']]
            require(all(row['validation'][k]==previous[k] for k in ('workspace_bytes','model_weight_bytes')),'timed allocation differs from Smoke')
    if phase=='full-benchmark':require(r['statistics']==full_stats(r['runs']),'full sample statistics drift')


def load_chain(root,phase,r):
    chain={};seen={}
    for name in STAGES[:STAGES.index(phase)]:
        path=root/name/'report.json';old=c.read(path)
        stage_status(name,old)
        require(old['phase']==name and old['root']==str(root) and old['history_roots']==r['history_roots'] and
                old['sources']==r['sources'] and old['predecessors']==r['predecessors'],'mixed stage/source chain: '+name)
        verify_tree(old,seen);validate_stage(name,old,chain,root)
        chain[name]=old;r['predecessors'][name]=c.identity(path)
    return chain


def main(args):
    root=args.root.resolve();phase=args.phase
    roots=[p.resolve() for p in (args.flow_root,args.resume_root,args.profile_root)]
    require(all(root!=p and root not in p.parents and p not in root.parents for p in roots),'output and historical roots must be independent')
    if root.exists():
        if phase=='source':require(root.is_dir() and not any(root.iterdir()),'Source requires a new empty experiment root')
        elif any(root.iterdir()):
            previous=root/'source/report.json'
            require(not previous.parent.is_symlink() and previous.is_file() and c.read(previous).get('root')==str(root),
                    'not an isolated Gate8 experiment root')
    out=c.fresh(root/phase,[*roots,Path('/usr/local/Ascend')]);r=dict(status='BLOCKED',error=None,integrity_error=None,phase=phase,root=str(root),
        sources=[],predecessors={},history_roots=list(map(str,roots)),device_execution='NOT_RUN',benchmark='NOT_RUN',
        atc='NOT_RUN',profile='NOT_RUN',production='NOT_MODIFIED')
    try:
        c.clean()
        r['sources']=sources();require(all(p.is_dir() for p in roots),'historical evidence root missing')
        r['checkout']=subprocess.check_output(['git','-C',str(c.REPO),'rev-parse','HEAD'],text=True).strip()
        r['scoped_worktree_status']=subprocess.check_output(['git','-C',str(c.REPO),'status','--short','--',str(HERE.relative_to(c.REPO))],text=True).splitlines()
        chain=load_chain(root,phase,r);src=chain.get('source',{}).get('binding')
        if phase=='source':r['binding']=binding(*roots)
        elif phase=='build':build(out,src,r,root/'source/report.json')
        elif phase=='compare':
            r['gears']=decision(chain);r['decision']=ELIGIBLE if eligible(r['gears']) else NO_GO
            r['comparison_files']={}
            for gear,item in r['gears'].items():
                path=out/('comparison-c'+gear+'.json');c.write(path,item);r['comparison_files'][gear]=c.identity(path)
                path.with_suffix('.md').write_text(comparison_text(item))
        elif phase in ('export','full-export'):export_stage(out,src,chain['build'],r,phase=='full-export')
        else:
            full=phase.startswith('full-');r['device_execution']='ATTEMPTED'
            smoke=None if phase.endswith('smoke') else chain['full-smoke' if full else phase[:3]+'-smoke']
            perform(out,src,chain['build'],chain['full-export' if full else 'export']['models'],phase,r,smoke)
            if phase=='full-benchmark':
                r['statistics']=full_stats(r['runs']);c.write(out/'full-comparison.json',r['statistics']);r['comparison_file']=c.identity(out/'full-comparison.json')
        validate_stage(phase,r,chain,root)
        r['status']=r['decision'] if phase=='compare' else STATE[phase]
    except Exception as exc:
        r['error']=f'{type(exc).__name__}: {exc}'
        if isinstance(exc,UnprovenGraph):r['status']='BLOCKED_UNPROVEN_GRAPH_IDENTITY'
        (out/'traceback.txt').write_text(traceback.format_exc())
    finally:
        try:
            # Keep failed worker logs, received callback records, actual outputs, and binaries at their original fresh paths.
            if r['error']:
                r['retained']=[c.identity(p) for p in sorted(out.rglob('*')) if p.is_file() and not p.is_symlink() and
                               (p.suffix in ('.log','.jsonl','.json','.om','.o','.so','.bin') or p.name=='traceback.txt')]
            verify_tree(r,{});require(sources()==r['sources'],'source identities changed during stage');c.clean()
        except Exception as exc:r.update(status='BLOCKED',integrity_error=repr(exc))
        c.write(out/'report.json',r)
    print(json.dumps({k:r.get(k) for k in ('phase','status','decision','device_execution','atc','active_case','error','integrity_error')},indent=2))
    print('REPORT:',out/'report.json')
    return 3 if r['status']==NO_GO else int(r['status'].startswith('BLOCKED'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False);p.add_argument('phase',choices=STAGES);p.add_argument('--root',type=Path,required=True)
    base=Path('/home/w00949577/qwen35-dflash-run')
    for key,path in [('flow-root','a-core-scheduler-flow-F6ia0q'),('resume-root','a-scheduler-om-resume-0foBgF'),('profile-root','a-only-draft-profile-OC8pCW')]:
        p.add_argument('--'+key,type=Path,default=base/path)
    raise SystemExit(main(p.parse_args()))
