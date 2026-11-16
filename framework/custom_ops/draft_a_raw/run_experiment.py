#!/usr/bin/env python3
"""Explicit stages only. Precheck never compiles, imports Torch or runs a device."""
import argparse,sys
import hashlib
from pathlib import Path
import raw_common as c
from raw_protocol import OUTPUTS,verify_run,order,summarize

STATES=dict(precheck='A_RAW_IDENTITIES_VERIFIED_NO_DEVICE',build='A_RAW_PRIVATE_PACKAGES_BUILT',probes='A_RAW_52_PROJECTIONS_PER_VARIANT_MATCH',
    export='A_RAW_C16_C64_OMS_COMPILED',smoke='A_RAW_ALL_TOKEN_KV_MATCH',benchmark='A_RAW_FULL_DRAFT_PAIRED_MEASURED')


def verify_build(r):
    c.verify_tree(r)
    for value in r['variants'].values():
        if value['status']!='PRIVATE_OPP_COMPILED_NOT_NPU_VALIDATED':raise ValueError('private build incomplete')
        c.verify_opp(value['opp'])
    if r['variants']['B']['opp']['c_config']!=r['variants']['R']['opp']['c_config']:raise ValueError('C registration configuration changed')


def verify_export(r):
    from raw_resources import compare,accepted_gate
    if set(r['models'])!={'B','R'}:raise ValueError('export variants missing')
    for v in ('B','R'):
        if set(r['models'][v])!={'16','64'}:raise ValueError('export gears missing')
        for gear,model in r['models'][v].items():
            if model['status']!='A_ONLY_OM_COMPILED_NOT_EXECUTED' or model.get('error') or model.get('integrity_error'):raise ValueError('failed export worker')
            accepted_gate(model['host_resources']['plans']['16,2560,19456'],r['evidence'],v)
            child=c.read(c.verify(model['worker_report']))
            if child!={k:x for k,x in model.items() if k not in ('host_resources','worker_report')}:raise ValueError('worker export report differs')
            if model['counts']!={'DFlashGroupQuantLinear':26,'DFlashDraftLmHeadTop1':0,'DFlashDraftLmHeadTop1Audit':0}:raise ValueError('full A-only node count differs')
            if model['abi']!=r['evidence']['H']['gears'][gear]['abi']:raise ValueError('public ABI changed')
            ir=c.verify(model['graph']).read_bytes().replace(model['output_root'].encode(),b'<VARIANT_ROOT>')
            if hashlib.sha256(ir).hexdigest()!=model['logical_ir_sha256']:raise ValueError('logical graph changed')
    for gear in ('16','64'):
        b=r['models']['B'][gear];a=r['models']['R'][gear]
        if any(b[k]!=a[k] for k in ('abi','counts','weight_bindings','head_sha256','logical_ir_sha256')):raise ValueError('non-Raw graph change')
        compare(b['host_resources'],a['host_resources'])


def prior(path,stage,current):
    old=c.read(path);c.verify_tree(old)
    if old.get('status')!=STATES[stage] or old.get('error') or old.get('integrity_error'):raise ValueError('failed predecessor: '+stage)
    for key in ('sources','device','cann_root'):
        if old[key]!=current[key]:raise ValueError('predecessor identity drift: '+key)
    if stage=='build':verify_build(old)
    if stage=='export':verify_export(old)
    if stage=='probes':
        from raw_probes import validate,compare
        wanted=[(v,g,x['name']) for v in ('B','R') for g in (16,64) for x in current['evidence']['projections'][str(g)]]
        if [(x['variant'],x['gear'],x['name']) for x in old['runs']]!=wanted:raise ValueError('projection coverage missing')
        for row in old['runs']:
            case=next(x for x in current['evidence']['projections'][str(row['gear'])] if x['name']==row['name'])
            from raw_resources import accepted_gate
            if row['shape']==[16,2560,19456]:accepted_gate(row['validation']['plan'],current['evidence'],row['variant'])
            if row['validation']!=validate(row,case,current['evidence'][row['variant']]['configuration'],current['device']):raise ValueError('projection evidence changed')
        for b,v in zip(old['runs'][:52],old['runs'][52:]):compare(b,v)
    return old


def run_om(a,root,r,variant,gear,timed=False):
    e=r['evidence'];spec=e['H']['gears'][str(gear)];build=r['build']
    if variant=='H':om=spec['om'];opp=e['H']['opp_root']
    else:om=r['export']['models'][variant][str(gear)]['om'];opp=build['variants'][variant]['opp']['install']
    c.verify_opp(e['H']['opp'] if variant=='H' else build['variants'][variant]['opp'])
    root.mkdir(parents=True,exist_ok=False);inp=root/'inputs';gold=root/'golden';inp.mkdir();gold.mkdir()
    env=c.selected_env(e,opp)
    if c.dependencies(build['runner'],root/'runner-ldd.log',env)!=build['runner_dependencies']:raise ValueError('Runner loaded dependency drift')
    for name,item in spec['inputs'].items():(inp/(name+'.bin')).symlink_to(c.verify(item))
    for name,item in spec['golden'].items():(gold/(name+'.bin')).symlink_to(c.verify(item))
    for side in ('inputs','outputs'):
        (root/(side+'.tsv')).write_text(''.join(t['name']+' '+t['dtype']+' '+' '.join(map(str,t['shape']))+'\n' for t in spec['abi'][side]))
    row=dict(variant=variant,gear=gear,profiled=False,om=om,runner=build['runner'],inputs=spec['inputs'],golden=spec['golden'],opp=str(opp))
    r.setdefault('runs',[]).append(row)
    try:c.command([c.verify(build['runner']),a.device,c.verify(om),root,inp,gold,'benchmark' if timed else 'smoke'],root/'runner.log',env)
    finally:row['retained_files']=[c.identity(p) for p in sorted(root.iterdir()) if p.is_file()]
    row['execution']=c.identity(root/'execution.json')
    suffixes=['actual-0.bin','actual-1.bin']+(['benchmark-last.bin','postcheck.bin'] if timed else [])
    row['outputs']={n:[c.identity(root/(n+'-'+s)) for s in suffixes] for n in OUTPUTS}
    row['validation']=verify_run(row,spec,a.device,timed)
    if timed:
        previous=next(x for x in r['smoke']['runs'] if (x['gear'],x['variant'])==(gear,variant))['validation']
        if any(previous[k]!=row['validation'][k] for k in ('workspace_bytes','model_weight_bytes')):raise ValueError('model allocation drift since Smoke')
    return row


def main(a):
    protected=[a.history,a.forward,a.reverse]+[p.parent for p in (a.precheck_report,a.build_report,a.probes_report,a.export_report,a.smoke_report) if p]
    root=c.fresh(a.output_dir,protected);r=dict(status='BLOCKED',phase=a.phase,sources=c.sources(),device=a.device,cann_root=str(a.cann_root),
        npu='NOT_RUN',compile='NOT_RUN',benchmark='NOT_RUN',profile='NOT_RUN',adoption='NOT_DECIDED',production='NOT_MODIFIED')
    try:
        c.clean()
        if a.phase=='precheck':
            from raw_evidence import check
            r['evidence']=check(a,root)
            if r['evidence']['H']['environment']['device_id']!=a.device or Path(r['evidence']['H']['environment']['cann_root']).resolve()!=a.cann_root:raise ValueError('historical Device/CANN root differs')
        else:
            links={}
            for stage in list(STATES)[:list(STATES).index(a.phase)]:
                p=getattr(a,stage+'_report');old=prior(p,stage,r)
                if any(old.get(k+'_report')!=v for k,v in links.items()):raise ValueError('mixed predecessor chain')
                links[stage]=c.identity(p);r[stage+'_report']=links[stage]
                if stage=='precheck':r['evidence']=old['evidence']
                else:
                    if old['evidence']!=r['evidence']:raise ValueError('H/B/R evidence drift')
                    r[stage]=old
            if a.phase=='build':
                r['compile']='ATTEMPTED';r['variants']={}
                for v in ('B','R'):
                    req=root/(v+'-request.json');dest=root/v
                    c.write(req,dict(evidence=r['evidence'],sources=r['sources'],variant=v,output=str(dest)))
                    c.command([sys.executable,'-B',c.HERE/'build.py',req],root/(v+'-build.log'),c.selected_env(r['evidence'],dest/'joint/opp'))
                    r['variants'][v]=c.read(dest/'report.json')
                    env=c.selected_env(r['evidence'],r['variants'][v]['opp']['install'])
                    c.command(['cmake','-S',c.QUANT/'test','-B',dest/'projection-runner','-DCANN_ROOT='+str(a.cann_root),'-DASCEND_OPP_PATH='+r['variants'][v]['opp']['install'],'-DDFLASH_K_TILE=256'],root/(v+'-probe-configure.log'),env)
                    c.command(['cmake','--build',dest/'projection-runner','--target','dflash_group_quant_linear_test','--parallel','2'],root/(v+'-probe-build.log'),env)
                    r['variants'][v]['projection_runner']=c.identity(dest/'projection-runner/dflash_group_quant_linear_test')
                verify_build(r)
                c.command(['cmake','-S',c.HERE/'runner','-B',root/'runner-build','-DCANN_ROOT='+str(a.cann_root),'-DCOMMON_INCLUDE='+str(c.QUANT/'test')],root/'runner-configure.log')
                c.command(['cmake','--build',root/'runner-build','--parallel','2'],root/'runner-build.log')
                r['runner']=c.identity(root/'runner-build/raw_om_runner')
                r['runner_dependencies']=c.dependencies(r['runner'],root/'runner-ldd.log',c.selected_env(r['evidence'],r['variants']['B']['opp']['install']))
                r['compile']='COMPLETED_NOT_DEVICE_EXECUTION'
            elif a.phase=='probes':
                from raw_probes import validate,compare
                r['npu']='ATTEMPTED';r['runs']=[]
                for v in ('B','R'):
                    build=r['build']['variants'][v]
                    for gear in (16,64):
                        for case in r['evidence']['projections'][str(gear)]:
                            dest=root/v/f'c{gear}'/case['name'];dest.mkdir(parents=True)
                            for name,item in case['inputs'].items():(dest/name).symlink_to(c.verify(item))
                            row=dict(variant=v,gear=gear,name=case['name'],shape=case['shape'],runner=build['projection_runner'])
                            r['runs'].append(row)
                            try:c.command([c.verify(build['projection_runner']),a.device,dest,*case['shape']],dest/'runner.log',c.selected_env(r['evidence'],build['opp']['install']))
                            finally:row['retained_files']=[c.identity(p) for p in sorted(dest.iterdir()) if p.is_file()]
                            row.update(execution=c.identity(dest/'execution.json'),log=c.identity(dest/'runner.log'),outputs=[c.identity(dest/f'actual-{i}.bin') for i in (0,1)])
                            row['validation']=validate(row,case,r['evidence'][v]['configuration'],a.device)
                            from raw_resources import accepted_gate
                            if case['shape']==[16,2560,19456]:accepted_gate(row['validation']['plan'],r['evidence'],v)
                            c.write(root/'report.json',r)
                for b,v in zip(r['runs'][:52],r['runs'][52:]):compare(b,v)
                r['npu']='COMPLETED'
            elif a.phase=='export':
                from raw_resources import collect,compare
                verify_build(r['build']);r['models']={};r['compile']='ATTEMPTED'
                for v in ('B','R'):
                    r['models'][v]={}
                    for gear in (16,64):
                        dest=root/v/f'c{gear}';req=root/f'{v}-c{gear}-request.json'
                        c.write(req,dict(evidence=r['evidence'],sources=r['sources'],variant=v,gear=gear,device=a.device,output=str(dest)))
                        log=root/f'{v}-c{gear}-export.log'
                        env=c.selected_env(r['evidence'],r['build']['variants'][v]['opp']['install']);env['AI_RUN_DIR']=str(root);env['ASCEND_WORK_PATH']=str(dest)
                        c.command([sys.executable,'-B',c.HERE/'worker.py',req],log,env)
                        model=c.read(dest/'report.json')
                        if model['status']!='A_ONLY_OM_COMPILED_NOT_EXECUTED' or model.get('error') or model.get('integrity_error'):raise ValueError('export worker failed')
                        model['host_resources']=collect(c.verify(model['atc_log']),dest/'host-plans',gear,r['evidence'][v]['configuration'])
                        model['worker_report']=c.identity(dest/'report.json')
                        r['models'][v][str(gear)]=model
                for gear in ('16','64'):
                    b=r['models']['B'][gear];v=r['models']['R'][gear]
                    if any(b[k]!=v[k] for k in ('abi','counts','weight_bindings','head_sha256','logical_ir_sha256')):raise ValueError('non-Raw logical graph/weights changed')
                    compare(b['host_resources'],v['host_resources'])
                verify_export(r)
                r['compile']='COMPLETED';r['npu']='DEVICE_TENSOR_PREPARATION_ONLY_DRAFT_NOT_EXECUTED'
            elif a.phase=='smoke':
                r['npu']='ATTEMPTED'
                for gear in (16,64):
                    # New Runner revalidates H first, then B, then R against the
                    # same original Native Golden. Equality also binds H/B/R.
                    for v in ('H','B','R'):run_om(a,root/f'c{gear}'/v,r,v,gear)
                r['equality_proof']={str(g):dict(comparisons=['H=Native','B=Native','B=H','R=Native','R=B'],
                    method='All 11 output buffers independently compared byte-for-byte with the same frozen Native Golden, twice per variant; transitive equality.',token_ids=15,complete_kv_buffers=10,untouched_kv_regions_included=True) for g in (16,64)}
                r['npu']='COMPLETED'
            elif a.phase=='benchmark':
                smoke=r['smoke']
                if [(x['gear'],x['variant']) for x in smoke['runs']]!=[(g,v) for g in (16,64) for v in ('H','B','R')]:raise ValueError('full Smoke coverage missing')
                for row in smoke['runs']:
                    if row['validation']!=verify_run(row,r['evidence']['H']['gears'][str(row['gear'])],a.device,False):raise ValueError('Smoke changed')
                r['npu']='ATTEMPTED';r['benchmark']='ATTEMPTED'
                for i in range(3):
                    for gear in (16,64):
                        for j,v in enumerate(order(i,gear)):
                            row=run_om(a,root/f'round-{i}'/f'c{gear}'/v,r,v,gear,True);row.update(round=i,position=j)
                            c.write(root/'report.json',r)
                r['comparison']=summarize(r['runs']);c.write(root/'comparison.json',r['comparison']);r['benchmark']='MEASURED';r['npu']='COMPLETED'
        r['status']=STATES[a.phase]
    except Exception as e:r['error']=f'{type(e).__name__}: {e}';r['status']='BLOCKED'
    finally:
        try:
            c.verify_tree(r)
            if 'evidence' in r:c.verify_opp(r['evidence']['H']['opp'])
            for built in (r.get('build'),r if a.phase=='build' and 'variants' in r else None):
                if built:
                    for variant in built['variants'].values():c.verify_opp(variant['opp'])
            if r['sources']!=c.sources():raise ValueError('source changed during stage')
            c.clean()
        except Exception as e:r.update(status='BLOCKED',integrity_error=str(e))
        c.write(root/'report.json',r)
    print(r['status'],root/'report.json');return int(r['status']=='BLOCKED')


def parse(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('phase',choices=STATES);p.add_argument('--output-dir',type=Path,required=True)
    base=Path('/home/w00949577/qwen35-dflash-run')
    p.add_argument('--history',type=Path,default=base/'ac-draft-full-20261016-071408')
    p.add_argument('--forward',type=Path,default=base/'gate-raw-prefetch-five-20261116-015738')
    p.add_argument('--reverse',type=Path,default=base/'gate-raw-prefetch-five-reverse-20261116-022749')
    p.add_argument('--cann-root',type=Path,default=Path('/usr/local/Ascend/cann-9.0.0'));p.add_argument('--device',type=int,default=0)
    for name in list(STATES)[:-1]:p.add_argument('--'+name+'-report',type=Path)
    a=p.parse_args(argv)
    for stage in list(STATES)[:list(STATES).index(a.phase)]:
        if getattr(a,stage+'_report') is None:p.error('--'+stage+'-report required')
    if a.device<0:p.error('negative device')
    for k,v in vars(a).items():
        if isinstance(v,Path):setattr(a,k,v.resolve())
    return a


if __name__=='__main__':raise SystemExit(main(parse()))
