"""Manual stages only. No phase dispatches its successor."""
import argparse
import json
import sys
import socket
import subprocess
import traceback
from pathlib import Path
from types import SimpleNamespace
from flow_common import HERE,STAGES,STATUS,admit,build_env,c,fc,host,mapped,require,sources
from fc_contract import lines
from flow_query import read_query,compare_pair
from flow_build import build as build_opp
from reference import compare
from raw_protocol import verify_run,stats,OUTPUTS


def query_key(plan):
    return {key:plan[key] for key in ('platform','schedule','tiling_sha256','host_dso')}


def configure(root,src,target,prior,env,opp):
    c.command(['cmake','-S',HERE/src,'-B',root/'cmake','-DCANN_ROOT='+prior['current']['cann_root'],
               '-DPRIVATE_OPP='+opp['install'],'-DCOMMON_INCLUDE='+str(c.QUANT/'test')],root/'configure.log',env)
    c.command(['cmake','--build',root/'cmake','--target',target],root/'compile.log',env)
    runner=c.identity(root/'cmake'/target)
    return dict(runner=runner,deps=c.dependencies(runner,root/'ldd.log',env))


def build(root,prior,src,r):
    r['variants']={}
    standard=c.read(c.verify(src['standard_manifest']))
    for variant in ('A','B'):
        folder=root/variant;folder.mkdir();pkg=folder/'package';pkg.mkdir()
        pack=build_opp(pkg,prior,variant);c.write(pkg/'report.json',pack);opp=pack['opp']
        env,binding=host.envs.for_prior(prior,opp['install'])
        manifest=host.dsos.with_private(standard,opp,src['standard_manifest']);c.write(folder/'manifest.json',manifest)
        selected=dict(opp=opp,package=c.identity(pkg/'report.json'),env=binding,manifest=c.identity(folder/'manifest.json'))
        r['variants'][variant]=selected
        for source,target in [('runner','projection_runner'),('om','raw_om_runner')]:
            dest=folder/source;dest.mkdir();selected[source]=configure(dest,source,target,prior,env,opp)
        api=c.identity(Path(opp['install'])/'vendors/customize/op_api/lib/libcust_opapi.so')
        require(api in [row['file'] for row in selected['runner']['deps']],'Runner API binding')
        anchor=src['single_fc'][variant]
        require(api['sha256']==anchor['api']['sha256'] and sorted(x['sha256'] for x in opp['preflight']['kernels'])==anchor['kernels'],'device Kernel/API differ from successful FC')
        selected['api']=api
    left,right=[c.read(c.verify(r['variants'][v]['package'])) for v in ('A','B')]
    require(left['base_opp']==right['base_opp'] and left['opp_comparison']['kernels']==right['opp_comparison']['kernels'] and
            r['variants']['A']['api']['sha256']==r['variants']['B']['api']['sha256'],'A/B Kernel/API differ')


def projections(root,prior,src,built,r,expected=None):
    r['plans']={};r['runs']={};cann=prior['current']['cann_root']
    for variant,selected in built['variants'].items():
        base=build_env(prior,selected);binary=selected['runner']
        require(c.dependencies(binary['runner'],root/(variant+'-ldd.log'),base)==binary['deps'],'projection dependencies changed')
        for key,case in src['cases'].items():
            entry=variant+'/'+key;rows=[];r['runs'][entry]=rows;shape=case['shape'];m,k,n=shape
            for repeat in range(2 if expected else 1):
                folder=root/entry/str(repeat);folder.mkdir(parents=True)
                for name,item in case['inputs'].items():(folder/name).symlink_to(c.verify(item))
                mode='execute' if expected else 'query'
                (folder/'shape.txt').write_text(f'{m} {k} {n} {mode}\n')
                platform=folder/'platform.jsonl';sched=folder/'scheduling.jsonl';platform.touch();sched.touch()
                env=dict(base,DFLASH_PLATFORM_DIAG_PATH=str(platform),DFLASH_SCHEDULER_EVIDENCE_PATH=str(sched),DFLASH_PLATFORM_DIAG_ROUTE='ACLNN')
                host.verify_environment(env,selected['env']);row=dict(repeat=repeat,mode=mode);rows.append(row)
                def capture():
                    plan=read_query(platform,sched,selected['opp'],'ACLNN',variant,shape)
                    if key=='16/fc':
                        anchor=src['single_fc'][variant]['query']
                        require(plan['schedule']==anchor['schedule'] and all(plan['platform'][field]==value for field,value in anchor['platform'].items() if field!='host_dso'),'generic Host/accepted FC drift')
                    libs=mapped(folder/'pre-maps.txt',selected,cann)
                    require(all(item in [lib['file'] for lib in libs] for item in (selected['api'],plan['host_dso'])),'missing bound API/Host')
                    row['pre_maps']=c.identity(folder/'pre-maps.txt')
                    if expected:require(query_key(plan)==query_key(expected[entry]),'launch query changed')
                    r['plans'][entry]=plan
                try:
                    if expected:
                        workspace=expected[entry]['schedule']['resources']['workspace']
                        fc.invoke(c.verify(binary['runner']),folder,repeat,env,workspace,capture)
                        mapped(folder/'post-maps.txt',selected,cann)
                        row['actual']=c.identity(folder/'actual.bin')
                        data=c.verify(row['actual']).read_bytes()
                        row['numerics']=compare(c.verify(case['golden']).read_bytes(),data,SimpleNamespace(m=m,n=n))
                        require(row['numerics']['finite'] and row['numerics']['bitwise_equal'],'Native bits differ')
                        require(not repeat or rows[0]['actual']['sha256']==row['actual']['sha256'],'Repeat differs')
                        safety(folder,repeat,workspace)
                    else:
                        c.command([c.verify(binary['runner']),folder,0,-1,0],folder/'runner.log',env);capture()
                        events=lines(folder/'events.jsonl')
                        require([x['event'] for x in events]==['aclInit','aclrtSetDevice','aclrtGetDevice','context','stream','GetWorkspaceSize','query_only'] and
                                all(x['returncode']==0 for x in events) and events[-2]['workspace']==r['plans'][entry]['platform']['workspace'],'Host-only journal')
                finally:row['files']=[c.identity(p) for p in sorted(folder.iterdir()) if p.is_file() and not p.is_symlink()]
            if variant=='B':
                compare_pair(r['plans']['A/'+key],r['plans'][entry])
                if expected:
                    require([x['actual']['sha256'] for x in rows]==[x['actual']['sha256'] for x in r['runs']['A/'+key]],'A/B projection outputs differ')
    require(len(r['plans'])==104,'projection inventory incomplete')


def safety(folder,repeat,workspace):
    events=lines(folder/'events.jsonl')
    names=['aclInit','aclrtSetDevice','aclrtGetDevice','context','stream','GetWorkspaceSize','precheck','execute_attempt','aclnnDFlashGroupQuantLinear','synchronize','readonly','complete']
    require([x['event'] for x in events]==names and all(x['returncode']==0 for x in events) and events[2]['device']==0 and
            events[3]['present'] and events[4]['present'] and events[5]['workspace']==workspace and
            events[6]['poison_byte']==(127,255)[repeat] and all(events[-2][key]==1 for key in ('x','w','s')),'runtime safety')
    for when in ('pre','post'):
        require(lines(folder/(when+'-guards.jsonl'))==[dict(buffer=name,side=side,bytes=512,expected=165,changes=[]) for name in
            ('x','w','s','y','workspace') for side in ('prefix','suffix')],'GM guard evidence')


def export(root,prior,src,built,queries,r):
    r['models']={};r['atc']='ATTEMPTED'
    for variant,selected in built['variants'].items():
        for gear,frozen in src['models'].items():
            key=variant+'/'+gear;folder=root/key;folder.mkdir(parents=True);out=folder/'compiled'
            env=build_env(prior,selected);env.update(AI_RUN_DIR=str(folder),ASCEND_WORK_PATH=str(out))
            request=folder/'request.json'
            c.write(request,dict(output=str(out),frozen=frozen,evidence={'H':prior['current']['evidence']['H']},opp=selected['opp'],environment=env,runtime_environment=selected['env']))
            c.command([sys.executable,'-B',HERE/'export_worker.py',request],folder/'worker.log',env)
            report=c.read(out/'report.json');require(report['status']=='FLOW_FULL_A_ONLY_ATC_COMPILED' and not report.get('integrity_error'),'ATC worker blocked')
            graph=c.read(c.verify(report['deployment']))['graphs'][0];original=c.read(c.verify(frozen['deployment']))['graphs'][0]
            require(all(graph[k]==original[k] for k in ('metadata','runtime_input_abi','custom_op_audit')),'full graph drift')
            shapes={tuple(case['shape']) for name,case in src['cases'].items() if name.startswith(gear+'/')}
            records={}
            for field in ('platform','scheduling'):
                grouped={}
                for record in lines(c.verify(report[field])):grouped.setdefault(tuple(record[k] for k in ('m','k','n')),[]).append(record)
                require(set(grouped)==shapes,'ATC shape inventory differs')
                records[field]=grouped
            plans={}
            for shape in sorted(shapes):
                tag='x'.join(map(str,shape));paths=[]
                for field in ('platform','scheduling'):
                    path=folder/(tag+'-'+field+'.jsonl');path.write_text(''.join(json.dumps(x)+'\n' for x in records[field][shape]));paths.append(path)
                plan=read_query(*paths,selected['opp'],'ATC',variant,shape);plans[tag]=plan
                for name,case in src['cases'].items():
                    if name.startswith(gear+'/') and tuple(case['shape'])==shape:
                        runtime=queries[variant+'/'+name]
                        require(plan['schedule']['resources']==runtime['schedule']['resources'] and plan['tiling_sha256']==runtime['tiling_sha256'] and
                                plan['platform']['capacity']==runtime['platform']['capacity'],'ATC/Runtime resource drift')
            r['models'][key]=dict(worker=c.identity(out/'report.json'),om=report['om'],deployment=report['deployment'],plans=plans)
    diffs={}
    for gear in src['models']:
        left,right=[r['models'][v+'/'+gear]['plans'] for v in ('A','B')]
        diffs[gear]={tag:compare_pair(left[tag],right[tag]) for tag in left}
    r['atc_pairs']=diffs
    r['attribution']='ACTUAL_ATC_SCHEDULES_IDENTICAL' if all(x['A_block_dim']==x['B_block_dim'] for rows in diffs.values() for x in rows.values()) else 'ACTUAL_ATC_SCHEDULES_DIFFER'
    r['atc']='FOUR_FULL_OMS_COMPILED'


def invoke_om(argv,log,env,admit):
    parent,child=socket.socketpair()
    with parent,child,log.open('wb') as stream:
        command=list(map(str,argv))+[str(child.fileno())]
        c.write(log.with_suffix('.command.json'),dict(argv=command,env=env))
        with subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT,pass_fds=(child.fileno(),)) as proc:
            child.close();parent.settimeout(600)
            try:
                require(parent.recv(1)==b'?', 'OM process exited before launch admission')
                admit();parent.sendall(b'G');parent.close()
                require(proc.wait(timeout=600)==0,'OM Runner failed: '+str(log))
            finally:
                parent.close()
                if proc.poll() is None:
                    try:proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait()


def full(root,prior,built,exported,r,smoke=None):
    r['runs']=[];timed=smoke is not None;r['attribution']=exported['attribution']
    for iteration in range(3 if timed else 1):
        for gear in ('16','64'):
            for variant in ('AB' if (iteration+(gear=='64'))%2==0 else 'BA'):
                selected=built['variants'][variant];key=variant+'/'+gear
                spec=prior['current']['evidence']['H']['gears'][gear];folder=root/str(iteration)/key;folder.mkdir(parents=True)
                for side in ('inputs','outputs'):
                    (folder/(side+'.tsv')).write_text(''.join(x['name']+' '+x['dtype']+' '+' '.join(map(str,x['shape']))+'\n' for x in spec['abi'][side]))
                for side in ('inputs','golden'):
                    dest=folder/side;dest.mkdir()
                    for name,item in spec[side].items():(dest/(name+'.bin')).symlink_to(c.verify(item))
                model=exported['models'][key]['om'];binary=selected['om'];env=build_env(prior,selected)
                require(c.dependencies(binary['runner'],folder/'ldd.log',env)==binary['deps'],'OM loader drift')
                row=dict(variant=variant,gear=gear,round=iteration,om=model,runner=binary['runner'],profiled=False);r['runs'].append(row)
                def admit_maps():mapped(folder/'pre-maps.txt',selected,prior['current']['cann_root'],om=True)
                try:invoke_om([c.verify(binary['runner']),0,c.verify(model),folder,folder/'inputs',folder/'golden','benchmark' if timed else 'smoke'],folder/'runner.log',env,admit_maps)
                finally:row['files']=[c.identity(p) for p in sorted(folder.iterdir()) if p.is_file()]
                for when in ('pre','post'):mapped(folder/(when+'-maps.txt'),selected,prior['current']['cann_root'],om=True)
                row['execution']=c.identity(folder/'execution.json')
                suffixes=['actual-0.bin','actual-1.bin']+(['benchmark-last.bin','postcheck.bin'] if timed else [])
                row['outputs']={name:[c.identity(folder/(name+'-'+suffix)) for suffix in suffixes] for name in OUTPUTS}
                row['validation']=verify_run(row,spec,0,timed)
                if timed:
                    old=next(x for x in smoke['runs'] if x['variant']==variant and x['gear']==gear)
                    require(all(old['validation'][k]==row['validation'][k] for k in ('workspace_bytes','model_weight_bytes')),'OM allocation drift')
    if timed:
        r['statistics']={}
        for gear in ('16','64'):
            data={v:[x['validation']['timing']['samples_ms'] for x in r['runs'] if x['gear']==gear and x['variant']==v] for v in 'AB'}
            result={v:stats(sum(data[v],[]),90) for v in 'AB'}
            result['round_A_over_B']=[stats(a,30)['median_ms']/stats(b,30)['median_ms'] for a,b in zip(data['A'],data['B'])]
            result['A_over_B']=result['A']['median_ms']/result['B']['median_ms'];r['statistics'][gear]=result


def inventory(phase,r,stages):
    if phase in ('host-query','projection-smoke'):
        keys={v+'/'+name for v in 'AB' for name in stages['source']['binding']['cases']}
        require(len(keys)==104 and set(r['plans'])==set(r['runs'])==keys,'partial or mixed projection report')
        for key in keys:
            rows=r['runs'][key];executed=phase=='projection-smoke'
            require([row['repeat'] for row in rows]==([0,1] if executed else [0]),'incomplete repeat inventory')
            if executed:
                golden=stages['source']['binding']['cases'][key.split('/',1)[1]]['golden']
                require(all(row['actual']['sha256']==golden['sha256'] and row['actual']['bytes']==golden['bytes'] for row in rows),'unaccepted projection output')
    elif phase=='export':
        require(set(r['models'])=={'A/16','B/16','A/64','B/64'} and set(r['atc_pairs'])=={'16','64'},'missing OM')
    elif phase in ('full-smoke','benchmark'):
        timed=phase=='benchmark'
        wanted=[(i,g,v) for i in range(3 if timed else 1) for g in ('16','64') for v in ('AB' if (i+(g=='64'))%2==0 else 'BA')]
        require([(x['round'],x['gear'],x['variant']) for x in r['runs']]==wanted,'unexpected OM run order/count')
        for row in r['runs']:
            require(row['om']==stages['export']['models'][row['variant']+'/'+row['gear']]['om'] and row['validation']['status']=='ALL_TOKEN_KV_BITS_AND_SAFETY_MATCH','unaccepted OM run')


def main(args):
    phase=args.phase;base=args.root.resolve();root=c.fresh(base/phase,[args.host_root,args.smoke_root])
    report=dict(phase=phase,status='BLOCKED',sources=sources(),predecessors={},device_execution='NOT_RUN',benchmark='NOT_RUN')
    try:
        stages={}
        for prev in STAGES[:STAGES.index(phase)]:
            path=base/prev/'report.json';old=c.read(path);c.verify_tree(old)
            require(old['phase']==prev and old['status']==STATUS[prev] and old['sources']==report['sources'] and
                    old['predecessors']==report['predecessors'] and not old.get('integrity_error'),'unsealed/mixed stage: '+prev)
            inventory(prev,old,stages)
            stages[prev]=old;report['predecessors'][prev]=c.identity(path)
        if phase=='source':report['binding']=admit(args.host_root.resolve(),args.smoke_root.resolve())
        else:
            src=stages['source']['binding'];prior=c.read(c.verify(src['host_source']))['binding']['prior']
            if phase=='build':build(root,prior,src,report)
            else:
                built=stages['build']
                for selected in built['variants'].values():c.verify_opp(selected['opp'])
                if phase in ('host-query','projection-smoke'):
                    if phase=='projection-smoke':report['device_execution']='ATTEMPTED'
                    projections(root,prior,src,built,report,stages['host-query']['plans'] if phase=='projection-smoke' else None)
                elif phase=='export':export(root,prior,src,built,stages['host-query']['plans'],report)
                else:
                    report['device_execution']='ATTEMPTED'
                    if phase=='benchmark':report['benchmark']='ATTEMPTED'
                    full(root,prior,built,stages['export'],report,stages.get('full-smoke'))
        if phase in ('full-smoke','projection-smoke','benchmark'):report['device_execution']='COMPLETED'
        if phase=='benchmark':report['benchmark']='MEASURED'
        inventory(phase,report,stages);require(report['sources']==sources(),'source drift');report['status']=STATUS[phase]
    except Exception as err:report['error']=repr(err);(root/'traceback.txt').write_text(traceback.format_exc())
    finally:
        try:c.verify_tree(report);c.clean()
        except Exception as err:report.update(status='BLOCKED',integrity_error=str(err))
        c.write(root/'report.json',report)
    print(report['status'],report.get('error'),report.get('integrity_error'),root/'report.json');return int(report['status']=='BLOCKED')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=STAGES);parser.add_argument('--root',required=True,type=Path)
    prefix=Path('/home/w00949577/qwen35-dflash-run')
    parser.add_argument('--host-root',type=Path,default=prefix/'a-core-scheduler-host-MzJtLl')
    parser.add_argument('--smoke-root',type=Path,default=prefix/'a-core-scheduler-fc-smoke-bWW88T')
    raise SystemExit(main(parser.parse_args()))
