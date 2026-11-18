"""Compile the identical AIR twice. Verify callbacks AND graph-referenced embedded code."""
from pathlib import Path
import sys
from gate_common import (HERE,GATE,SHAPE,SHAPE_KEY,GEARS,MODES,LAYERS,c,command,flow,config_graph,
                         require,model_inventory,gate_kernel,tree,sources)
from gate_plan import gate_plan,non_gate,node_binding,embedded,recheck_embedded
from flow_query import read_query
from platform_contracts import unique_object
from c_profile_graph import read_om_json
import json


def callbacks(folder,report,opp,mode,reference):
    groups={}
    for what in ('platform','scheduling'):
        records={}
        for line in c.verify(report[what]).read_text().splitlines():
            row=json.loads(line,object_pairs_hook=unique_object);tag='x'.join(str(row[k]) for k in ('m','k','n'))
            records.setdefault(tag,[]).append(row)
        require(records.keys()==reference.keys(),'unexpected/missing real ATC shape: '+what)
        groups[what]=records
    results={}
    for tag,want in reference.items():
        shape=list(map(int,tag.split('x')));files=[]
        for what in ('platform','scheduling'):
            path=folder/(tag+'-'+what+'.jsonl');path.write_text(''.join(json.dumps(x)+'\n' for x in groups[what][tag]));files.append(path)
        # Always Scheduler A; "raw" is not scheduler policy B.
        plan=read_query(*files,opp,'ATC','A',shape)
        if shape==SHAPE:proof=gate_plan(plan,mode,want)
        else:non_gate(plan,want);proof=dict(block_dim=plan['platform']['blockDim'],resources=plan['schedule']['resources'],
                                           non_gate='IDENTICAL_TO_ACCEPTED_A_PLAN',parent_sha256=plan['tiling_sha256'])
        results[tag]=dict(query=plan,contract=proof)
    return results


def compile_model(folder,src,selected,mode,key,full):
    prior=c.read(c.verify(src['host_source']))['binding']['prior']
    spec=src['full'][key] if full else None
    frozen=spec['frozen'] if full else src['singles'][key]
    cases=src['cases'][key] if full else {key:src['cases']['16'][key]}
    refs=spec['anchor']['plans'] if full else {SHAPE_KEY:src['full']['16']['anchor']['plans'][SHAPE_KEY]}
    folder.mkdir(parents=True);out=folder/'compiled'
    env=flow.build_env(prior,selected)
    # Limit the op compiler cache to this exact compilation, even across variants sharing an ABI.
    cache=folder/'compiler-cache';cache.mkdir()
    env.update(AI_RUN_DIR=str(folder),ASCEND_WORK_PATH=str(out),ASCEND_CACHE_PATH=str(cache))
    binding=dict(selected['env'],values=dict(selected['env']['values'],ASCEND_CACHE_PATH=str(cache)))
    scope='full' if full else 'projection'
    request=folder/'request.json';c.write(request,dict(output=str(out),scope=scope,flow_sources=sources(),frozen=frozen,evidence={'H':{'state':{'args':prior['current']['evidence']['H']['state']['args']}}},
        opp=selected['opp'],environment=env,runtime_environment=binding))
    # The historical worker accepts a single frozen AIR bundle and only relocates model/output; it does not export from Torch.
    command([sys.executable,'-B',HERE/'gate_atc_worker.py',request],folder/'atc-worker.log',env,cwd=folder)
    child=c.read(out/'report.json')
    require(child['status']=='GATE8_'+scope.upper()+'_AIR_COMPILED_NOT_EXECUTED' and not child.get('error') and not child.get('integrity_error'),'ATC did not complete')
    c.verify_tree(child);c.verify_opp(selected['opp'])
    require(tree(Path(selected['opp']['install']))==selected['package_files'],'ATC private package/source inventory drift')
    deployment=c.read(c.verify(child['deployment']));model=c.verify(child['om']);original=c.read(c.verify(frozen['air']))
    require(deployment['status']=='PASS' and len(deployment['graphs'])==len(original['graphs'])==1,'deployment inventory')
    graph=deployment['graphs'][0]
    require(all(graph[k]==original['graphs'][0][k] for k in ('metadata','runtime_input_abi','custom_op_audit','input_names','output_names','air')) and
            c.identity(c.linked(Path(child['deployment']['path']).parent,graph['om']))==child['om'],'different graph/ABI/payload')
    copied=c.linked(Path(child['deployment']['path']).parent,deployment['air_manifest'])
    require(c.sha(copied)==frozen['air']['sha256'],'ATC used a different AIR manifest')
    gates=config_graph(graph,cases,full)
    plans=callbacks(folder,child,selected['opp'],mode,refs)
    path=folder/'om-graph.json'
    command([c.verify(frozen['atc_binary']),'--mode=1','--om='+str(model),'--json='+str(path)],folder/'graph-inspect.log',env,cwd=folder)
    nodes=[n for n in read_om_json(path) if n['type']==GATE]
    declared={r['name'] for r in graph['runtime_input_abi']['draft_ac']['nodes'] if r['ge_type']==GATE}
    require(len(nodes)==len(declared) and {n['name'] for n in nodes}==declared,'OM dump cannot bind nodes to immutable graph')
    kernels=embedded(model,[node_binding(n) for n in nodes])
    reference_code={}
    if mode=='off':
        for gear in ((key,) if full else GEARS):
            reference_code[gear]={}
            for layer,node in gates.items():
                historical_name=src['full_nodes'][gear][layer]['name']
                before=src['full'][gear]['gate_binaries'][historical_name]
                after=kernels[node['name']]
                reference_code[gear][layer]=dict(old=before,new=after,whole_elf_equal=before['sha256']==after['sha256'],
                    executable_equal=before['executable_sections']==after['executable_sections'])
        c.write(folder/'off-code-comparison.json',reference_code)
        require(all(x['executable_equal'] for rows in reference_code.values() for x in rows.values()),
                'BLOCKED_OFF_ATC_CODE_DRIFT: see off-code-comparison.json; no inherited correctness')
    require(kernel_match(selected),'private OPP Gate pair changed after ATC')
    return dict(mode=mode,key=key,scope='full' if full else 'projection',air=frozen['air'],om=child['om'],deployment=child['deployment'],
                inspection=dict(command=c.identity(folder/'graph-inspect.command.json'),log=c.identity(folder/'graph-inspect.log')),
                off_vs_historical_full=reference_code,
                worker_report=c.identity(out/'report.json'),atc_command=child['atc_command'],captures=child['captures'],dump=c.identity(path),
                opp_root=selected['opp']['install'],private_kernel=selected['kernel'],atc_kernels=kernels,plans=plans,
                gate_nodes={name:n['name'] for name,n in gates.items()},abi=graph['metadata']['tensor_abi'],
                original_gear=None if full else '16',input_gear_policy='BIND_CURRENT_GEAR_CASE; NO_C16_GOLDEN_SUBSTITUTION')


def kernel_match(selected):return gate_kernel(selected['opp'])==selected['kernel']


def verify_compiled_graph(model,src,full):
    key=model['key'];spec=src['full'][key]['frozen'] if full else src['singles'][key]
    require(model['air']==spec['air'],'wrong frozen AIR')
    dep=c.read(c.verify(model['deployment']));actual=c.verify(model['om']);graph=dep['graphs'][0];original=c.read(c.verify(spec['air']))['graphs'][0]
    require(dep['status']=='PASS' and len(dep['graphs'])==1 and c.linked(Path(model['deployment']['path']).parent,graph['om'])==actual and
            all(graph[k]==original[k] for k in ('metadata','runtime_input_abi','custom_op_audit','air','input_names','output_names')),'deployment/immutable graph drift')
    bound=config_graph(graph,src['cases'][key] if full else {key:src['cases']['16'][key]},full)
    require({layer:node['name'] for layer,node in bound.items()}==model['gate_nodes'],'Gate node/weight mapping drift')
    nodes=[node_binding(n) for n in read_om_json(c.verify(model['dump'])) if n['type']==GATE]
    recheck_embedded(actual,nodes,model['atc_kernels'])


def compare_models(left,right,full):
    require((left['mode'],right['mode'])==('off','raw') and left['key']==right['key'] and
            all(left[k]==right[k] for k in ('air','abi','gate_nodes','scope')) and
            left['opp_root']!=right['opp_root'] and left['om']['sha256']!=right['om']['sha256'],'not independent Off/Raw compilations of same graph')
    require(left['private_kernel']['elf']['sha256']!=right['private_kernel']['elf']['sha256'],'OPP Gate ELF did not change')
    require(left['plans'].keys()==right['plans'].keys(),'shape inventory drift')
    for tag,lplan in left['plans'].items():
        if tag==SHAPE_KEY:
            a,b=lplan['contract'],right['plans'][tag]['contract']
            require(a['block_dim']==b['block_dim']==8 and a['cube_sha256']==b['cube_sha256'] and a['cube_fields']==b['cube_fields'] and
                    a['workspace']==b['workspace'] and a['assignments']==b['assignments'],'8-block/Cube/Workspace mismatch')
        else:require(lplan['contract']==right['plans'][tag]['contract'],'non-target Tiling changed')
    expected=set(left['gate_nodes'].values());lk=left['atc_kernels'];rk=right['atc_kernels']
    require(set(lk)==set(rk) and len(lk)==(26 if full else 1) and len(expected)==(5 if full else 1) and expected<=set(lk),'wrong embedded kernel inventory')
    for name in lk:
        equal=lk[name]['executable_sections']==rk[name]['executable_sections']
        if name in expected:require(not equal,'BLOCKED_UNPROVEN_RAW_CODE: selected Gate executable sections identical: '+name)
        else:require(equal,'BLOCKED_FULL_NON_GATE_CODE_DRIFT: '+name)
    return dict(scope=left['scope'],target_nodes=sorted(expected),unchanged_non_gate=len(lk)-len(expected),
                same_air=left['air'],nested_cube='BITWISE_EQUAL',atc_gate_blocks=8,distinct_gate_executable_bytes=True)


def export_stage(root,src,built,r,full):
    keys=GEARS if full else LAYERS;r.update(models={v:{} for v in MODES},comparisons={},atc='ATTEMPTED')
    for key in keys:
        for mode in MODES:
            r['active_case']=dict(stage='ATC',mode=mode,key=key)
            r['models'][mode][key]=compile_model(root/mode/key,src,built['variants'][mode],mode,key,full)
        r['comparisons'][key]=compare_models(r['models']['off'][key],r['models']['raw'][key],full)
    model_inventory(r['models'],full);r['atc']='COMPILED_AND_IDENTIFIED';r.pop('active_case')
