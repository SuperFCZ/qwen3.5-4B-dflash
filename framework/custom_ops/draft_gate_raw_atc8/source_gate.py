"""Re-establish graph, constant, binary and predecessor provenance before any build."""
import struct
from pathlib import Path
from gate_common import (c,flow,host,resume,require,one,check_config,gate_kernel,derive_header,case_contract,config_graph,
                         HEADER,STEM,KERNEL_SOURCE,OFF_OM,OFF_ELF,API,OFF_PARENT,GEARS,LAYERS,SHAPE_KEY,
                         SDK_SCHEMA,SCHEMA_SHA,payloads,option)
from a3_launch import load_build_config
from c_profile_graph import read_om_json
from gate_plan import node_binding,embedded

HISTORICAL_SINGLE_SUITE='6a9ec59e3f6e557ab4ee12aac390dc84dd7f3547f8bb26e733553577bb85301b'


class UnprovenGraph(ValueError):
    pass


def graph_source(aggregate,ref,name,case,full_node,atc):
    try:
        row=one((r for r in aggregate['cases'] if r['name']=='c16-'+name),'single graph '+name)
        require(row['status']=='PASS','historical Gate not accepted')
        folder=c.verify(ref).parent/('a-'+row['name'])
        paths={k:c.linked(folder,row['artifacts'][k]) for k in ('air','deployment','om')}
        require(all(folder in path.parents for path in paths.values()),'escaped historical graph directory')
        air=c.read(paths['air']);dep=c.read(paths['deployment']);files=payloads(paths['air'],air)
        require(dep['status']=='PASS' and len(dep['graphs'])==1 and c.linked(paths['deployment'].parent,dep['air_manifest'])==paths['air'],'unbound AIR/OM manifest')
        graph=air['graphs'][0];compiled=dep['graphs'][0]
        node=config_graph(graph,{name:case})[name]
        require(all(node[k]==full_node[k] for k in ('weight_sha256','nz_sha256','nz_bytes','k','n','scale_gn_sha256','storage_shape','scale_shape')),
                'historical single Gate constants differ from accepted full OM')
        require(graph['metadata']['draft_ac']['projections']==[dict(n=node['n'],k=node['k'],weight_sha256=node['weight_sha256'],scale_gn_sha256=node['scale_gn_sha256'])],
                'unbound logical checkpoint weights')
        require(all(compiled[k]==graph[k] for k in ('metadata','runtime_input_abi','custom_op_audit')) and
                c.linked(paths['deployment'].parent,compiled['om'])==paths['om'],'single projection deployment drift')
        cmd=compiled['atc_command']
        require(c.identity(cmd[0])==atc and option(cmd,'--model')==str(c.linked(paths['air'].parent,graph['air'])) and
                option(cmd,'--output')+'.om'==str(paths['om']),'changed single projection compiler binding')
        for key,value in (('--mode','0'),('--framework','1'),('--soc_version','Ascend310P3'),('--precision_mode','must_keep_origin_dtype'),('--deterministic','0')):
            require(option(cmd,key)==value,'unexpected compilation policy: '+key)
        return dict(air=c.identity(paths['air']),deployment=c.identity(paths['deployment']),om=c.identity(paths['om']),
                    payloads=files,command=cmd,atc_binary=atc,compiler_identity_text=dep['compiler']['identity'])
    except (ValueError,KeyError,IndexError,OSError) as exc:
        raise UnprovenGraph(name+': '+str(exc)) from exc


def binding(old,continuation,profile):
    stages,history,_=resume.historical(old)
    require(stages['export']['attribution']=='ACTUAL_ATC_SCHEDULES_IDENTICAL','different frozen full Draft experiment')
    refs={};reports={};base=stages['source']['binding'];host_source=c.read(c.verify(base['host_source']))['binding'];prior=host_source['prior']
    evidence=prior['current']['evidence'];cann=prior['current']['cann_root']
    sdk=dict(headers=host_source['sdk'],libraries=host_source['sdk_libraries'])
    c.verify_tree(sdk)
    for phase in resume.PHASES:
        path=continuation/phase/'report.json';r=c.read(path);c.verify_tree(r)
        require(r['old_root']==str(old) and r['history']==history and r['predecessors']==refs and r['sources']==resume.sources() and
                r['phase']==phase and r['status']==resume.STATUS[phase] and not r.get('error') and not r.get('integrity_error'),'Resume chain failed: '+phase)
        if phase!='admit':
            flow.inventory(phase,r,stages)
            for row in r['runs']:
                require(flow.verify_run(row,evidence['H']['gears'][row['gear']],0,phase=='benchmark')==row['validation'],'historical Token/KV/measurements drift')
        refs[phase]=c.identity(path);reports[phase]=r
    b=stages['build']['variants']['A'];env=flow.build_env(prior,b);pack=c.read(c.verify(b['package']))
    require(pack['opp']==b['opp'] and pack['variant']=='A' and pack['opp_comparison']['all_non_host_files_equal'] is True and
            pack['opp_comparison']['all_kernel_bytes_equal'] is True,'historical A package differs')
    c.verify_opp(b['opp']);kernel=gate_kernel(b['opp']);require(kernel['elf']['sha256']==OFF_ELF and b['api']['sha256']==API,'wrong Off anchor')
    profile_report=profile/'analyze/profile-report.json';pa=c.read(profile_report);require(pa['status']=='A_OM_PROFILE_ANALYZED','unaccepted profiling input')
    admission=c.read(c.verify(pa['predecessors']['admit']))
    require(admission['status']=='A_OM_PROFILE_ADMITTED' and admission['binding']['history']==history and admission['binding']['resume']==refs,
            'profiling bound to different history')
    require(pa['sources']==admission['sources'],'mixed profiling source seals');c.verify_tree(pa['sources'])
    v3=profile/'analysis-v3/report.json';v=c.read(v3);v2=c.read(c.verify(v['v2']))
    require(v2['status']=='OFFLINE_V2_RECORDED' and v2['input_report']==c.identity(profile_report) and v['status']=='OFFLINE_V3_RECORDED' and
            all(v[k]==history[k] for k in ('source','export')) and v['projection_smoke']==history['projection-smoke'],'wrong v3 provenance')
    bypath={i['path']:i for i in v['files']};cases={g:{name:base['cases'][g+'/'+name] for name in LAYERS} for g in GEARS};case_contract(cases)
    inventory={};full={};proof={};operator_queries={}
    for gear,rows in cases.items():
        for row in rows.values():
            c.verify_tree(row)
            require(all(bits & 0x7c00!=0x7c00 for bits, in struct.iter_unpack('<H',c.verify(row['golden']).read_bytes())),'nonfinite Native Golden')
        model=stages['export']['models']['A/'+gear];spec=evidence['H']['gears'][gear]
        require(c.verify(model['om']) and model['om']['sha256']==OFF_OM[gear] and model['om']==admission['binding']['cases'][gear]['om'],'historical A OM identity')
        dep=c.read(c.verify(model['deployment']));graph=dep['graphs'][0]
        require(graph['metadata']['tensor_abi']==spec['abi'] and c.identity(c.linked(Path(model['deployment']['path']).parent,graph['om']))==model['om'],'unbound full OM')
        inventory[gear]=config_graph(graph,rows,full=True)
        frozen=base['models'][gear]
        original=c.read(c.verify(frozen['air']))['graphs'][0]
        require(all(original[k]==graph[k] for k in ('metadata','runtime_input_abi','custom_op_audit')),'full AIR differs')
        capture=c.read(c.verify(pa['predecessors']['profile-c'+gear]))
        require(capture['status']==f'A_C{gear}_RUNNER_CAPTURE_VERIFIED' and capture['predecessors']['admit']==pa['predecessors']['admit'] and
                capture['sources']==pa['sources'] and capture['execute']=='38_PROFILED_CALLS_GOLDEN_VERIFIED','not the accepted A capture')
        actual_nodes={node['name'] for node in inventory[gear].values()}
        bindings=[node_binding(n) for n in read_om_json(c.verify(capture['graph'])) if n['name'] in actual_nodes]
        require(len(bindings)==5 and {x['node'] for x in bindings}==actual_nodes,'missing historical embedded Gate identities')
        full[gear]=dict(frozen=frozen,anchor=model,compiled_graph=capture['graph'],gate_binaries=embedded(c.verify(model['om']),bindings),
                        io=dict(abi=spec['abi'],inputs=spec['inputs'],golden=spec['golden']))
        require(model['plans'][SHAPE_KEY]['tiling_sha256']==OFF_PARENT,'full Gate Tiling anchor differs')
        proj=bypath[str(profile/'analysis-v3'/('c'+gear)/'projection.json')]
        records=c.read(c.verify(proj))
        operator_queries[gear]={}
        for name,row in rows.items():
            node=inventory[gear][name];match=one((x for x in records if x['name']==node['name'] and x['projection']==name),'missing unique v3 Gate binding')
            require(match['contract_matches']==[name] and match['mkn']==row['shape'] and match['role']=='gate_up' and match['immutable_node']==[node],'v3 Gate node/constant drift')
            q=stages['host-query']['plans']['A/'+gear+'/'+name]
            query=one(stages['host-query']['runs']['A/'+gear+'/'+name],'frozen Gate Host query')
            events=one((item for item in query['files'] if Path(item['path']).name=='events.jsonl'),'missing actual query journal')
            journal=flow.lines(c.verify(events));returned=one((x for x in journal if x['event']=='GetWorkspaceSize'),'Workspace query count')
            require(returned['returncode']==0 and journal[-1]['event']=='query_only' and
                    q['tiling_sha256']==OFF_PARENT and returned['workspace']==q['platform']['workspace']==model['plans'][SHAPE_KEY]['platform']['workspace'],
                    'historical projection/full Gate Tiling or actual operator Workspace drift')
            operator_queries[gear][name]=dict(journal=events,workspace_bytes=returned['workspace'],route='ACLNN_WORKSPACE_ONLY_NOT_ATC_BLOCK_PROOF')
        proof[gear]=proj
    configs={}
    for key,mode in [('B','off'),('R','raw')]:
        child=evidence[key];cfg=load_build_config(c.verify(child['config']))
        require(cfg==child['configuration'],'unbound generated configuration');check_config(cfg,mode)
        for name,digest in cfg['source_sha256'].items():require(c.sha(c.QUANT/name)==digest,'worktree differs from accepted '+mode+' child: '+name)
        configs[mode]=dict(config=child['config'],header=child['header'],kernel_header=child['kernel_header'],configuration=cfg)
    from raw_evidence import configs as verify_pair
    verify_pair(configs['off']['configuration'],configs['raw']['configuration'])
    require(derive_header(c.verify(configs['off']['header']).read_bytes())==c.verify(configs['raw']['header']).read_bytes(),'not the accepted Raw config')
    for sub in ('op_host','op_kernel'):
        ident=pack['project_after'][sub+'/'+HEADER]['file'];require(c.verify(ident).read_bytes()==c.verify(configs['off']['header']).read_bytes(),'wrong historical build header')
    require(c.verify(pack['project_after']['op_host/scheduler_variant.h']['file']).read_text()=='#pragma once\n#define DFLASH_SCHEDULER_POLICY 0\n','unbound Scheduler A selection')
    host_sources={}
    for name in (STEM+'.cpp','scheduler_evidence.h','platform_diag.h'):
        locked=pack['project_after']['op_host/'+name]['file'];local=c.identity(flow.HERE/'op_host'/name)
        require(c.verify(locked) and all(local[k]==locked[k] for k in ('bytes','sha256')),'reviewed scheduler source differs: '+name)
        host_sources[name]=dict(worktree=local,accepted=locked)
    require(pack['project_after']['op_kernel/'+STEM+'.cpp']['file']['sha256']==KERNEL_SOURCE,'wrong original kernel.cpp')
    try:
        pd=c.read(c.verify(host_source['references']['source']))
        single_ref=pd['binding']['historical_fc'];require(single_ref['sha256']==HISTORICAL_SINGLE_SUITE,'historical 52 graph report drift')
        agg=c.read(c.verify(single_ref));require(agg['status']=='PASS','single projection suite not accepted')
        singles={layer:graph_source(agg,single_ref,layer,cases['16'][layer],inventory['16'][layer],evidence['tools']['atc']) for layer in LAYERS}
    except (ValueError,KeyError,OSError) as exc:raise UnprovenGraph(str(exc)) from exc
    require(c.sha(SDK_SCHEMA)==SCHEMA_SHA,'unreviewed TCube schema')
    extras=reports['admit']['extra_dsos'];resume.combined(b,cann,extras,c.read(resume.POLICY))
    for row in extras.values():host.dsos.verify_record(row)
    return dict(cann=cann,sdk=sdk,device=0,history=history,resume=refs,single_suite=single_ref,source_profile=c.identity(profile_report),
                v3=c.identity(v3),v3_projections=proof,flow_build=history['build'],off_package=b['package'],off_kernel=kernel,
                configs=configs,cases=cases,singles=singles,full=full,full_nodes=inventory,standard_manifest=base['standard_manifest'],reviewed_host_sources=host_sources,operator_queries=operator_queries,
                host_source=base['host_source'],extras=extras,standard_env=host.envs.standard_identity(b['env']),
                historical_full_benchmarks=reports['benchmark']['statistics'],private_loader_path=env['ASCEND_CUSTOM_OPP_PATH'])
