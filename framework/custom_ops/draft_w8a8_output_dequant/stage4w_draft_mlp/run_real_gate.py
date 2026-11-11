#!/usr/bin/env python3
"""New real activations, immutable accepted B-NZ OPP/Runner; no Kernel compilation."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import real_mlp_common as c
import run_pair as pair
from pair_contract import comparison, validate_row
from run_stage2b import command

STATES = dict(source='REAL_MLP_SOURCE_BOUND', build='REAL_MLP_RUNNERS_REVERIFIED',
    host='REAL_MLP_HOST_MATCH', smoke='REAL_GATE_SAFETY_COMPLETE_QUALITY_UNRESOLVED',
    benchmark='REAL_GATE_RESIDENT_MEASURED_QUALITY_UNRESOLVED')


def history(a, root, r):
    old=c.read(a.paired_report); c.verify_tree(old)
    if old.get('status') != pair.STATES['benchmark'] or old.get('error') or old.get('integrity_error'):
        raise ValueError('accepted paired benchmark missing')
    br=c.verify(old['evidence']['bnz']['reports']['benchmark'])
    h=SimpleNamespace(bnz_root=br.parent.parent,cann_root=a.cann_root,device=a.device)
    bound=c.pe.bind(h,root,r)
    expected=dict(sources=c.pe.sources(),evidence=bound,device=a.device,cann_root=str(a.cann_root),machine=r['machine'])
    if any(old[k]!=expected[k] for k in ('sources','evidence','device','cann_root')):
        raise ValueError('paired evidence/environment changed')
    c.pe.compare_machine(old['machine'],r['machine'])
    for stage in ('source','build','smoke'):
        key=stage+'_report'
        value=pair.prior(c.verify(old[key]),stage,expected)
        expected[key]=old[key]
        if stage=='smoke':anchor=value['native_repeat_anchor']
    for row in old['runs']:
        if row['validation'] != validate_row(row,bound,a.device,True,anchor if row['scheme']=='FP16' else None):
            raise ValueError('historical paired safety/timing changed')
    comp=comparison(old['runs'],bound['bnz']['packing'])
    if any(old['comparison'][k]!=v for k,v in comp.items()):
        raise ValueError('paired summary differs from raw samples')
    return bound


def prior(path, stage, r):
    old=c.read(path); c.verify_tree(old)
    if old.get('status')!=STATES[stage] or old.get('error') or old.get('integrity_error'):
        raise ValueError('predecessor failed: '+stage)
    for k in ('sources','prepare_report','paired_report','evidence','device','cann_root'):
        if old[k]!=r[k]:raise ValueError('predecessor identity differs: '+k)
    c.pe.compare_machine(old['machine'],r['machine'])
    if stage=='build':pair.verify_build(old['runtime'],r['evidence'],Path(r['cann_root']))
    return old


def validate(row, case, bound, device, host=False, timed=False, anchor=None):
    b=bound['bnz']; ex=c.read(c.verify(row['execution']))
    policy=b['evidence']['baseline']['policy']
    pl=c.pe.c.old_protocol.plan(c.verify(row['log']).read_text(),0,0,-1,b['t1'],policy)
    if pl['blocks']!=7:raise ValueError('formal accepted7 Block policy required, not Manager8')
    adm=c.read(c.verify(row['admission']))
    if ex['workspace_bytes']!=pl['total'] or adm!=dict(status='ADMITTED_BEFORE_EXECUTION',workspace_bytes=pl['total'],declared_total=pl['total'],blocks=pl['blocks'],cores=pl['cores'],core_policy=pl['core_policy'],host_proof_checked=not host):
        raise ValueError('actual Workspace admission changed')
    if row['runner']!=b['build']['variants']['candidate']['build']['runner'] or row.get('profiled') is not False:
        raise ValueError('normal Runner identity/profile mismatch')
    if host:
        if ex!=dict(status='HOST_TILING_ONLY',npu='NOT_RUN',device=device,workspace_bytes=pl['total'],output_mode=0,core_limit=0,tile_id=-1):
            raise ValueError('Host contract differs')
        return dict(status='FORMAL_HOST_ADMITTED',plan=pl)
    required=dict(status='EXPLORATORY_EXECUTION_SAFE',op=b['operator']['op'],soc='Ascend310P3',shape=[16,2560,19456],
        tile_count=76,group_count=1,guards=True,readonly=True,repeat_equal=True,repetitions=2,device=device,
        cpu_fallback=False,output_mode=0,core_limit=0,tile_id=-1,guard_coverage='FULL_K_TILE_384_WORD_THREE_BOUNDARIES',
        prepare_calls=38 if timed else 2,verify_calls=4 if timed else 2,poison_verified=True,poison_patterns=[127,255],
        timed_prepare_io=False,timed_log_stable=True,timed_log_checked=timed,sync_completed=True,
        resource_admission='BEFORE_EXECUTION',ub_guard_protocol='UB_AND_CORE_ARENA_POISON_ON_FAILURE',
        layout='nd',input_count=3,finite_active_output=True,inactive_poison_intact=True,active_coverage_checked=True,
        guard_coverage_unchanged=True,arena_detector='XOR_OR_U32',reference_kind='CPU_FULL_K_REFERENCE',
        kernel_guards_unchanged=False,strict_golden_match=False,production='NOT_APPROVED')
    if any(type(ex.get(k))!=type(v) or ex[k]!=v for k,v in required.items()):
        raise ValueError('device safety contract incomplete')
    timing=c.timing(ex,timed)
    raw=[c.verify(x).read_bytes() for x in row['outputs']]
    if len(raw)!=(4 if timed else 2) or any(x!=raw[0] for x in raw):raise ValueError('Repeat/tail/postcheck changed')
    if anchor is not None and any(x!=c.verify(anchor).read_bytes() for x in raw):raise ValueError('fresh independently observed repeat anchor changed')
    golden=c.verify(case['files']['cpu_fullk_fp16.bin']).read_bytes()
    observations=[c.observe(x,golden,-1,min(i,1),raw[0] if i else None) for i,x in enumerate(raw)]
    if observations!=[c.read(c.verify(x)) for x in row['numeric']]:raise ValueError('full numeric difference inventory changed')
    expected_numeric='NUMERIC_MISMATCH_UNRESOLVED' if observations[0]['mismatch_count'] else 'BITWISE_REFERENCE_OBSERVATION'
    if ex['numeric_status']!=expected_numeric:raise ValueError('numeric status drift')
    y=np.frombuffer(raw[0],'<f2').reshape(16,19456)
    ref=c.array(case['files']['gate_up_reference_f64.bin'],'<f8',(16,19456))
    native=c.array(case['files']['gate_up_w8a16.bin'],'<f2',(16,19456))
    return dict(status='REAL_INPUT_SAFETY_AND_REPEAT_OBSERVED',quality='UNRESOLVED',
        numeric_status=expected_numeric,numeric=observations,plan=pl,timing=timing,
        vs_effective_fp64={name:c.errors(ref[:,s],y[:,s]) for name,s in [('full',slice(None)),('gate',slice(0,9728)),('up',slice(9728,None))]},
        vs_original_w8a16=c.errors(native,y),historical_201_mismatches_reused=False)


def invoke(a, dest, r, case, prep, host=False, timed=False, anchor=None):
    dest.mkdir(parents=True,exist_ok=False)
    bound=r['evidence']; b=bound['bnz']; wb=b['build']['variants']['candidate']['build']
    pair.verify_build(r['runtime'],bound,a.cann_root)
    files=dict(a_i8=case['files']['a_i8.bin'], b_i8=prep['weights']['gate_up']['packed_b'],
        quant_u64_doc_candidate=case['files']['quant_u64_doc_candidate.bin'])
    for name,item in files.items():(dest/(name+'.bin')).symlink_to(c.verify(item))
    if not host:
        for i in range(2):(dest/f'expected-{i}.bin').symlink_to(c.verify(case['files']['cpu_fullk_fp16.bin']))
    if timed:(dest/'admitted_full.bin').symlink_to(c.verify(anchor))
    env=c.pe.nz.env(b['evidence'],a.cann_root,wb['opp'])
    system=b['t1']['system_workspace_bytes']; blocks=b['evidence']['baseline']['policy']['selected']
    env['FULLK_EXPLORE_WORKSPACE_BYTES']=str(system+blocks*((system+511)//512*512+1024))
    row=dict(case=case['id'],runner=wb['runner'],profiled=False,timed=timed,inputs=files)
    r.setdefault('runs',[]).append(row)
    argv=[c.verify(wb['runner']),a.device,dest,0,0,-1]+(['--host-only'] if host else ['--benchmark'] if timed else [])
    try:command(argv,dest/'runner.log',env,r,'run_'+str(dest.relative_to(a.output_dir)),900)
    finally:row['retained_files']=[c.identity(x) for x in sorted(dest.iterdir()) if x.is_file()]
    row.update(execution=c.identity(dest/'execution.json'),log=c.identity(dest/'runner.log'),admission=c.identity(dest/'resource-admission.json'))
    if not host:
        names=c.pe.nz.OUTPUTS[:4 if timed else 2]
        row['outputs']=[c.identity(dest/n) for n in names]
        row['numeric']=[c.identity(dest/(n+'.numeric.json')) for n in names]
    row['validation']=validate(row,case,bound,a.device,host,timed,anchor)
    return row


def main(a):
    root=c.fresh(a.output_dir)
    r=dict(status='NOT_RUN',phase=a.phase,sources=c.sources(),device=a.device,cann_root=str(a.cann_root),
        prepare_report=c.identity(a.prepare_report),paired_report=c.identity(a.paired_report),
        npu='NOT_RUN',benchmark='NOT_RUN',quality='UNRESOLVED',production='NOT_APPROVED',
        full_w8a8_mlp='NOT_RUN_DOWN_KERNEL_UNSUPPORTED')
    try:
        prep=c.prepared(a.prepare_report)
        r['evidence']=history(a,root,r)
        predecessors={}
        for stage in list(STATES)[:list(STATES).index(a.phase)]:
            path=getattr(a,stage+'_report'); old=prior(path,stage,r)
            for key,value in predecessors.items():
                if old.get(key+'_report')!=value:raise ValueError('cross-phase predecessor chain differs')
            r[stage+'_report']=c.identity(path); predecessors[stage]=r[stage+'_report']
            if stage=='build':r['runtime']=old['runtime']
            if stage in ('host','smoke'):
                if [v['case'] for v in old['runs']]!=[v['id'] for v in prep['cases']]:raise ValueError('sample coverage changed')
                for case,row in zip(prep['cases'],old['runs']):
                    if row['validation']!=validate(row,case,r['evidence'],a.device,host=stage=='host'):
                        raise ValueError('prior sample safety changed')
                if stage=='smoke':smoke=old
        if a.phase=='build':
            runtime=dict(evidence=r['evidence'])
            pair.build_bind(a,root,runtime); runtime['status']=pair.STATES['build']
            pair.verify_build(runtime,r['evidence'],a.cann_root); r['runtime']=runtime
            r['build_kind']='FROZEN_NORMAL_RUNNER_OPP_REVERIFICATION_NO_RECOMPILE'
        elif a.phase in ('host','smoke','benchmark'):
            if a.phase!='host':r['npu']='ATTEMPTED'
            rounds=range(3) if a.phase=='benchmark' else range(1)
            for rnd in rounds:
                cases=prep['cases'] if rnd%2==0 else list(reversed(prep['cases']))
                for case in cases:
                    anchor=next(x['outputs'][0] for x in smoke['runs'] if x['case']==case['id']) if a.phase=='benchmark' else None
                    row=invoke(a,root/f'round-{rnd}'/case['id'],r,case,prep,a.phase=='host',a.phase=='benchmark',anchor)
                    row['round']=rnd
                    c.write(root/'report.json',r)
            if a.phase!='host':r['npu']='COMPLETED'
            if a.phase=='benchmark':
                r['statistics']={case['id']:{k:c.stats([v for row in r['runs'] if row['case']==case['id'] for v in row['validation']['timing'][k]['samples_ms']],90) for k in ('prepare','execute_sync')} for case in prep['cases']}
                r['benchmark']='MEASURED_RESIDENT_GATE_ONLY'
                r['device_kernel_time']='UNAVAILABLE_NO_DEVICE_TIMESTAMPS'
                r['historical_0_472195_ms']='REFERENCE_ONLY_NO_EXTRAPOLATION_OR_FRESH_RATIO'
        r['status']=STATES[a.phase]
    except Exception as exc:r.update(status='BLOCKED',error=f'{type(exc).__name__}: {exc}')
    finally:
        try:
            c.verify_tree(r); c.prepared(a.prepare_report)
            if r['sources']!=c.sources():raise ValueError('source changed')
            if 'evidence' in r:c.pe.unchanged(r['evidence'],a.cann_root)
        except Exception as exc:r.update(status='BLOCKED',integrity_error=str(exc))
        c.write(root/'report.json',r)
    print(r['status']);return int(r['status']=='BLOCKED')


def parse(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('phase',choices=STATES)
    for key in ('output-dir','prepare-report','paired-report'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--cann-root',type=Path,default=Path('/usr/local/Ascend/cann-9.0.0'))
    p.add_argument('--device',type=int,default=0)
    for stage in list(STATES)[:-1]:p.add_argument('--'+stage+'-report',type=Path)
    a=p.parse_args(argv)
    for stage in list(STATES)[:list(STATES).index(a.phase)]:
        if getattr(a,stage+'_report') is None:p.error('--'+stage+'-report is required')
    if a.device<0:p.error('negative device')
    for key,value in vars(a).items():
        if isinstance(value,Path):setattr(a,key,value.resolve())
    return a


if __name__=='__main__':raise SystemExit(main(parse()))
