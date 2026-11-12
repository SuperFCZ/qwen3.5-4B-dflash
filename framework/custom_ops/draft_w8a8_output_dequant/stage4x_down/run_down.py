#!/usr/bin/env python3
"""Gated real-data Down experiment; never modifies or times a historical OPP."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import x_common as c
from x_transform import verify_repository,materialize
from reference_x import prepare,expected
from protocol_x import validate,require_numeric,fault_expected
from run_stage2b import command

STATES=dict(source='STAGE4X_SOURCE_REVIEWED',prepare='STAGE4X_CPU_PREPARED',hostplan='STAGE4X_HOST_RESOURCE_FEASIBLE',
    build='STAGE4X_BUILT_NPU_NOT_RUN',host='STAGE4X_FORMAL_HOST_ADMITTED',tile0='STAGE4X_TILE0_SAFETY_OBSERVED',
    smoke='STAGE4X_FULL_SAFETY_OBSERVED',fault='STAGE4X_GUARD_COVERAGE_MATCH',benchmark='STAGE4X_DOWN_RESIDENT_MEASURED',mlp='STAGE4X_FROZEN_BOUNDARY_MLP_RECORDED')


def schedule(phase,cases):
    if phase=='host':return [(cases[0],kind,0,0,-1) for kind in ('candidate','fault')]
    if phase=='tile0':return [(x,'candidate',0,0,0) for x in cases]
    if phase=='smoke':return [(x,'candidate',0,limit,-1) for x in cases for limit in (0,2)]
    if phase=='fault':return [(cases[0],'fault',mode,1,0) for mode in range(2,8)]
    raise ValueError('stage scope')


def previous(path,phase,r):
    old=c.read(path);c.verify_tree(old)
    if old.get('status')!=STATES[phase] or old.get('error') or old.get('integrity_error'):raise ValueError('previous stage failed: '+phase)
    for key in ('sources','evidence','device','cann_root'):
        if old[key]!=r[key]:raise ValueError('predecessor identity differs: '+key)
    c.w.pe.compare_machine(old['machine'],r['machine'])
    if phase=='build':c.verify_build(old)
    if phase=='hostplan':
        from resources_x import validate as resource_check
        from down_dependency_identity import verify_dependency_binding
        verify_dependency_binding(old['sdk']['dependency_binding'])
        raw=c.read(c.verify(old['host_query']));row=raw['cases'][0]
        if resource_check(raw,row)!=old['resources'] or old['resources']['status']!='HOST_RESOURCE_CHECKS_PASS':raise ValueError('resource proof changed')
        if old['t1']!=dict(fields={k:row['fields'][k] for k in c.p.FIELDS},capacity=raw['capacity'],system_workspace_bytes=raw['system_workspace_bytes'],scratch=202240):raise ValueError('T1 not exact queried fields')
    if phase in ('host','tile0','smoke','fault'):
        want=[(x['id'],k,m,l,t) for x,k,m,l,t in schedule(phase,r['data']['cases'])]
        if [(x['case'],x['variant'],x['mode'],x['limit'],x['tile']) for x in old['runs']]!=want:raise ValueError('device coverage missing')
        for row in old['runs']:
            case=next(x for x in r['data']['cases'] if x['id']==row['case'])
            if row['runner']!=r['build']['variants'][row['variant']]['runner']:raise ValueError('Runner identity mismatch')
            if row['validation']!=validate(row,case,r['evidence'],r['t1'],r['device'],host=phase=='host'):raise ValueError('prior safety/numerics drift')
    return old


def invoke(a,dest,r,case,kind,mode,limit,tile,host=False,timed=False,anchor=None):
    c.verify_build(r['build']);dest.mkdir(parents=True)
    build=r['build']['variants'][kind]
    for name,item in [('a_i8.bin',case['files']['a_i8.bin']),('b_i8.bin',r['data']['packed_b']),('quant_u64_doc_candidate.bin',case['files']['quant_u64_doc_candidate.bin'])]:
        (dest/name).symlink_to(c.verify(item))
    if not host:
        for i in range(2):(dest/f'expected-{i}.bin').write_bytes(fault_expected(mode,i) if kind=='fault' else expected(case,tile,i))
    if timed:
        if kind!='candidate' or mode or limit or tile!=-1:raise ValueError('only normal full Auto can time')
        (dest/'admitted_full.bin').symlink_to(c.verify(anchor))
    row=dict(case=case['id'],variant=kind,mode=mode,limit=limit,tile=tile,runner=build['runner'],profiled=False)
    r.setdefault('runs',[]).append(row)
    args=[c.verify(build['runner']),a.device,dest,mode,limit,tile]+(['--host-only'] if host else ['--benchmark'] if timed else [])
    try:command(args,dest/'runner.log',c.env(a.cann_root,build['opp'],r['evidence']['formal_core_policy'],r['t1'],limit),r,str(dest.relative_to(a.output_dir)),1800)
    finally:row['retained_files']=[c.identity(p) for p in sorted(dest.iterdir()) if p.is_file()]
    row.update(execution=c.identity(dest/'execution.json'),log=c.identity(dest/'runner.log'),admission=c.identity(dest/'resource-admission.json'))
    if not host:
        names=c.w.pe.nz.OUTPUTS[:4 if timed else 2]
        row['outputs']=[c.identity(dest/n) for n in names]
        if kind!='fault':row['numeric']=[c.identity(dest/(n+'.numeric.json')) for n in names]
    row['validation']=validate(row,case,r['evidence'],r['t1'],a.device,host,timed,anchor)
    c.write(dest/'validation.json',row['validation']);return row


def main(a):
    root=c.w.fresh(a.output_dir)
    r=dict(status='NOT_RUN',phase=a.phase,sources=c.sources(),device=a.device,cann_root=str(a.cann_root),
        npu='NOT_RUN',benchmark='NOT_RUN',quality='UNRESOLVED',production='NOT_APPROVED',profile='NOT_RUN')
    try:
        verify_repository();r['evidence']=c.bind(a.stage4w_root,a.device)
        if r['evidence']['cann_root']!=str(a.cann_root):raise ValueError('selected SDK differs from W')
        g=c.read(c.verify(r['evidence']['gate']))
        bound=c.wg.history(SimpleNamespace(paired_report=c.verify(g['paired_report']),cann_root=a.cann_root,device=a.device),root,r)
        if bound!=g['evidence']:raise ValueError('accepted Gate SDK/environment changed')
        r['sdk_layout']=c.sdk_layout(a.cann_root)
        links={}
        for phase in list(STATES)[:list(STATES).index(a.phase)]:
            if phase=='benchmark' and a.phase=='mlp':continue
            path=getattr(a,phase+'_report');old=previous(path,phase,r)
            if any(old.get(k+'_report')!=v for k,v in links.items()):raise ValueError('cross-phase chain drift')
            links[phase]=c.identity(path);r[phase+'_report']=links[phase]
            if phase=='prepare':r['data']=old['data']
            if phase=='hostplan':r['t1']=old['t1']
            if phase=='build':r['build']=old
            if phase=='smoke':smoke=old
        if a.phase=='prepare':r['data']=prepare(root/'data',r['evidence'])
        elif a.phase=='hostplan':
            from host_x import query
            query(a,root,r)
        elif a.phase=='build':
            from build_x import build
            r['variants']={}
            for kind in ('candidate','fault'):
                src=root/('source-'+kind);audit=materialize(src,kind,c.identity,c.write)
                dest=root/kind;dest.mkdir()
                item=dict(evidence=dict(preflight=r['evidence']['baseline']['original_evidence']['preflight'],t1=r['t1']),experiment_evidence=r['evidence'],audit=audit)
                r['variants'][kind]=item
                try:build(a,dest,item,src,kind)
                finally:c.write(dest/'build-report.json',item)
            c.verify_build(r)
        elif a.phase in ('host','tile0','smoke','fault'):
            if a.phase!='host':r['npu']='ATTEMPTED'
            for i,(case,kind,mode,limit,tile) in enumerate(schedule(a.phase,r['data']['cases'])):
                invoke(a,root/f'run-{i}',r,case,kind,mode,limit,tile,host=a.phase=='host');c.write(root/'report.json',r)
            if a.phase!='host':r['npu']='COMPLETED'
            if a.phase in ('tile0','smoke'):r['numeric_status']='BITWISE_REFERENCE_OBSERVATION' if all(x['validation']['status']=='DOWN_GOLDEN_AND_SAFETY_MATCH' for x in r['runs']) else 'NUMERIC_MISMATCH_UNRESOLVED'
        elif a.phase=='benchmark':
            require_numeric(smoke['runs']);r['npu']='ATTEMPTED';r['benchmark']='ATTEMPTED'
            for rnd in range(3):
                cases=r['data']['cases'] if rnd%2==0 else list(reversed(r['data']['cases']))
                for case in cases:
                    anchor=next(x['outputs'][0] for x in smoke['runs'] if x['case']==case['id'] and x['limit']==0)
                    row=invoke(a,root/f'round-{rnd}'/case['id'],r,case,'candidate',0,0,-1,timed=True,anchor=anchor);row['round']=rnd
                    c.write(root/'report.json',r)
            r['statistics']={case['id']:{k:c.w.stats([v for x in r['runs'] if x['case']==case['id'] for v in x['validation']['timing'][k]['samples_ms']],90) for k in ('prepare','execute_sync')} for case in r['data']['cases']}
            r.update(npu='COMPLETED',benchmark='RESIDENT_DOWN_MEASURED',device_time='UNAVAILABLE_NO_DEVICE_TIMESTAMPS')
        elif a.phase=='mlp':
            r['composition']=dict(scope='INDEPENDENT_NPU_BOUNDARY_COMPOSITION_NOT_ONE_RESIDENT_LAUNCH',
                gate_and_swiglu=c.read(c.verify(r['evidence']['mlp']))['cases'],down=smoke['runs'],
                real_input_source='NPU Gate -> NPU SiLU -> NPU Multiply, exact bytes from W',
                activation_quant='shared16 rows CPU FP64/RNE; H2D outside resident timing',
                integrated_latency='NOT_RUN_DO_NOT_ADD_MEDIANS',online_npu_preprocessing='NOT_RUN',quality='UNRESOLVED')
            r['numeric_status']=smoke['numeric_status'];c.write(root/'single-layer-numerics.json',r['composition'])
        r['status']=STATES[a.phase]
    except Exception as exc:r.update(status='BLOCKED',error=f'{type(exc).__name__}: {exc}')
    finally:
        try:
            c.verify_tree(r)
            if r['sources']!=c.sources():raise ValueError('Stage4X source changed')
        except Exception as exc:r.update(status='BLOCKED',integrity_error=str(exc))
        c.write(root/'report.json',r)
    print(r['status'],root/'report.json');return int(r['status']=='BLOCKED')


def parse(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('phase',choices=STATES);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--stage4w-root',type=Path,default=Path('/home/w00949577/qwen35-dflash-run/real-draft-mlp-mountfix-pqbjQq'))
    p.add_argument('--cann-root',type=Path,default=Path('/usr/local/Ascend/cann-9.0.0'));p.add_argument('--device',type=int,default=0)
    for phase in list(STATES)[:-1]:p.add_argument('--'+phase+'-report',type=Path)
    a=p.parse_args(argv)
    for phase in list(STATES)[:list(STATES).index(a.phase)]:
        if phase=='benchmark' and a.phase=='mlp':continue
        if getattr(a,phase+'_report') is None:p.error('--'+phase+'-report required')
    if a.device<0:p.error('negative device')
    for key,value in vars(a).items():
        if isinstance(value,Path):setattr(a,key,value.resolve())
    return a


if __name__=='__main__':raise SystemExit(main(parse()))
