#!/usr/bin/env python3
"""Independent N128 T1 correctness. No benchmark/profile entry point."""
import argparse
import os
from pathlib import Path
from j_common import (SETTINGS,BOUNDED_TILES,common,identity,verify,verify_tree,write,fresh,read,
                      sources,evidence,plan_from_log,check_execution)
from j_reference import prepare,verify_case,compare
from build_n128 import build,installed
from source_audit import audit
from run_stage2b import command

BIND_KEYS=('settings','sources','evidence','device','prepare_report','case')
STATES={'host-only':'STAGE4J_HOST_TILING_ONLY','bounded':'STAGE4J_BOUNDED_MATCH','full':'STAGE4J_FULL_GATE_MATCH'}

def same(r,b,keys=BIND_KEYS):
    if any(b.get(k)!=r[k] for k in keys):raise ValueError('stage evidence/source/device binding differs')

def prepared(path,r):
    if path is None:raise ValueError('successful --prepare-report required')
    p=read(path);same(r,p,('settings','sources','evidence','device'))
    if p.get('status')!='STAGE4J_PREPARED_NPU_NOT_RUN' or p.get('phase')!='complete' or p.get('npu')!='NOT_RUN':raise ValueError('incomplete CPU preparation')
    case=read(verify(p['case']));verify_case(case,numerical=True)
    if case['stage4b_case']!=r['evidence']['stage4b_case']:raise ValueError('case provenance differs')
    r.update(prepare_report=identity(path),case=p['case']);return case

def admission_expected(ex,plan,host_only):
    return dict(status='ADMITTED_BEFORE_EXECUTION',workspace_bytes=ex['workspace_bytes'],declared_total=plan['total'],
                blocks=plan['blocks'],cores=plan['cores'],core_policy=plan['core_policy'],host_proof_checked=not host_only)

def sequence(stage):
    if stage=='host-only':return [(1,21,-1)]
    if stage=='bounded':return [(0,m,t) for t in BOUNDED_TILES for m in (1,20,0)]
    if stage=='full':return [(l,m,-1) for l in (0,2) for m in (21,0)]
    raise ValueError('unknown device phase')

def evaluate_row(r,row,maps,host_only):
    mode,limit,tile=row['mode'],row['core_limit'],row['tile_id'];ex=read(verify(row['execution']))
    plan=plan_from_log(verify(row['log']).read_text(),mode,limit,tile,r['evidence']['t1'])
    if plan['core_policy']!=r['evidence']['formal_core_policy']:raise ValueError('formal TilingContext Core policy differs from accepted Stage4B; review required')
    if ex['workspace_bytes']<plan['total'] or read(verify(row['resource_admission']))!=admission_expected(ex,plan,host_only):raise ValueError('workspace admission mismatch')
    if host_only:
        if ex.get('status')!='HOST_TILING_ONLY' or ex.get('npu')!='NOT_RUN' or ex.get('host_only') is not True or ex.get('device')!=r['device'] or ex.get('core_limit')!=1 or ex.get('output_mode')!=21:raise ValueError('Host-only scope differs')
        return plan,dict(status='HOST_TILING_ONLY')
    check_execution(ex,r['device'],mode,limit,tile)
    result=compare(maps,mode,plan['blocks'],tile,[verify(x).read_bytes() for x in row['outputs']])
    return plan,result

def validate_proof(path,r,maps,stage):
    if path is None:raise ValueError('required prior '+stage+' proof missing')
    p=read(path);same(r,p)
    if p.get('status')!=STATES[stage] or p.get('phase')!='complete' or p.get('build_report')!=r['build_report']:raise ValueError('wrong/incomplete '+stage+' proof')
    rows=p.get('runs',[])
    if [(x['core_limit'],x['mode'],x['tile_id']) for x in rows]!=sequence(stage):raise ValueError('incomplete '+stage+' run coverage')
    policies=[]
    for row in rows:
        plan,result=evaluate_row(p,row,maps,stage=='host-only')
        if row['plan']!=plan or row['comparison']!=result or result['status'] not in ('HOST_TILING_ONLY','BITWISE_MATCH') or row['status']!=result['status']:raise ValueError('prior proof mismatch')
        policies.append(plan['core_policy'])
    if any(p!=policies[0] for p in policies) or p['core_policy']!=policies[0]:raise ValueError('prior Core policy drift')
    if stage=='host-only' and p.get('npu')!='NOT_RUN':raise ValueError('Host-only claimed device execution')
    return policies[0]

def smoke(args,root,r,case):
    b=read(args.build_report);same(r,b)
    if b.get('status')!='STAGE4J_BUILT_NPU_NOT_RUN' or b.get('phase')!='complete':raise ValueError('independent completed Build required')
    if installed(Path(b['opp']))!=b['installed']:raise ValueError('private OPP changed')
    maps=verify_case(case);host_only=args.stage=='host-only'
    r.update(build_report=identity(args.build_report),cann_compile=b['cann_compile'],installation=b['installation'],runs=[])
    policy=None
    if not host_only:
        policy=validate_proof(args.host_report,r,maps,'host-only');r['host_report']=identity(args.host_report)
        if policy['selected']<2:raise ValueError('independent 2-Core control unavailable')
    if args.stage=='full':
        if validate_proof(args.bounded_report,r,maps,'bounded')!=policy:raise ValueError('bounded/Host Core policy differs')
        previous=read(args.bounded_report)
        if previous.get('host_report')!=r['host_report']:raise ValueError('bounded Host proof changed')
        r['bounded_report']=identity(args.bounded_report)
    for limit,mode,tile in sequence(args.stage):
        folder=root/'execute'/f'limit-{limit}'/f'tile-{tile}'/f'mode-{mode}';folder.mkdir(parents=True)
        for name in ('a_i8.bin','b_i8.bin','quant_u64_doc_candidate.bin'):(folder/name).symlink_to(verify(case['files'][name]))
        row=dict(core_limit=limit,mode=mode,tile_id=tile,status='RUNNING');r['runs'].append(row)
        key=f'l{limit}_t{tile}_m{mode}';r.update(phase=key,npu='NOT_RUN' if host_only else 'RUNNING');write(root/'report.json',r)
        environment=common.env(args.cann_root,Path(b['opp']));environment.pop('DFLASH_STAGE4J_EXPECTED_CORE_POLICY',None)
        if policy:environment['DFLASH_STAGE4J_EXPECTED_CORE_POLICY']=','.join(str(policy[k]) for k in ('aic','aiv','all','selected','arch'))
        command([verify(b['runner']),args.device,folder,mode,limit,tile]+(['--host-only'] if host_only else []),folder/'runner.log',environment,r,key,600)
        row.update(execution=identity(folder/'execution.json'),log=identity(folder/'runner.log'),resource_admission=identity(folder/'resource-admission.json'))
        if not host_only:row['outputs']=[identity(folder/f'actual-{i}.bin') for i in range(2)]
        plan,result=evaluate_row(r,row,maps,host_only);row.update(plan=plan,comparison=result,status=result['status']);write(root/'report.json',r)
        if policy is not None and plan['core_policy']!=policy:raise ValueError('TilingContext policy changed')
        if host_only:policy=plan['core_policy']
        if row['status'] not in ('HOST_TILING_ONLY','BITWISE_MATCH'):raise ValueError('numerical/Poison/Repeat failure; no next phase')
    if args.stage=='full':
        finals=[verify(x['outputs'][0]).read_bytes() for x in r['runs'] if x['mode']==0]
        if len(finals)!=2 or finals[0]!=finals[1]:raise ValueError('Auto/2-Core outputs differ')
    if installed(Path(b['opp']))!=b['installed']:raise ValueError('private OPP changed after execution')
    verify_case(case)
    r.update(status=STATES[args.stage],phase='complete',core_policy=policy,npu='NOT_RUN' if host_only else 'COMPLETED',
        scope='HOST_ONLY' if host_only else ('BOUNDED_FOUR_SEPARATE_TILES' if args.stage=='bounded' else 'FULL_311296_ELEMENTS_AUTO_AND_2_CORE'),
        summary=dict(runs=len(r['runs']),repetitions=0 if host_only else 2,complete_gate_pass=args.stage=='full'))

def main(args):
    if args.stage in ('host-only','bounded','full') and args.prepare_report is None:
        args.prepare_report=verify(read(args.build_report)['prepare_report'])
    protected=[args.cann_root,args.stage4i_root,args.stage4b_report.parent.parent]
    protected += [p.parent for p in (args.prepare_report,args.build_report,args.host_report,args.bounded_report) if p]
    root=fresh(args.output_dir,protected)
    r=dict(stage='4J',status='NOT_RUN',phase='admission',settings=SETTINGS,sources=sources(),device=args.device,
        cann_compile='NOT_RUN',installation='NOT_RUN',npu='NOT_RUN',benchmark='NOT_RUN',profile='NOT_RUN',performance='NOT_RUN',draft='NOT_RUN',online_quantization='NOT_RUN')
    try:
        source_review,diff=audit();write(root/'source-audit.json',source_review);(root/'source.diff').write_text(diff)
        r['source_audit']=identity(root/'source-audit.json');r['source_diff']=identity(root/'source.diff')
        r['evidence']=evidence(args)
        if args.stage=='prepare':
            r['phase']='cpu_packing';write(root/'report.json',r)
            case=prepare(root/'data',r['evidence']);verify_case(case)
            r.update(case=identity(root/'data/case.json'),status='STAGE4J_PREPARED_NPU_NOT_RUN',phase='complete',validated_tile_groups=3040)
        else:
            case=prepared(args.prepare_report,r);write(root/'report.json',r)
            if args.stage=='build':build(args,root,r)
            else:smoke(args,root,r,case)
    except Exception as error:
        r.update(status='FAILED',error=f'{type(error).__name__}: {error}')
        for k in ('cann_compile','installation','npu'):
            if r[k]=='RUNNING':r[k]='FAILED_OR_INCOMPLETE'
    finally:
        try:
            for x in r['sources']:verify(x)
            if 'evidence' in r:verify_tree(r['evidence'])
            if 'case' in r:verify_case(read(verify(r['case'])))
        except Exception as error:r.update(status='FAILED',integrity_error=str(error))
        if r['status']=='FAILED':r['failure_artifacts']=[identity(p) for p in sorted(root.rglob('*')) if p.is_file() and p.suffix in ('.log','.bin') and not p.is_symlink()]
        write(root/'report.json',r)
    print(r['status']+': '+str(root/'report.json'));return int(r['status']=='FAILED')

def parse():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('prepare','build','host-only','bounded','full'))
    p.add_argument('--output-dir',type=Path,required=True)
    for name in ('prepare-report','build-report','host-report','bounded-report'):p.add_argument('--'+name,type=Path)
    base=Path('/home/w00949577/qwen35-dflash-run')
    p.add_argument('--stage4b-report',type=Path,default=base/'w8a8-full-gate-20261027-093617/smoke/report.json')
    p.add_argument('--stage4i-root',type=Path,default=base/'w8a8-n128-host-shareub-20261030-083337')
    p.add_argument('--cann-root',type=Path,default=Path(os.environ.get('CANN_ROOT','/usr/local/Ascend/cann-9.0.0')))
    p.add_argument('--device',type=int,default=0);a=p.parse_args();a.cann_root=a.cann_root.resolve()
    if a.device<0:p.error('negative device')
    required={'build':['prepare_report'],'host-only':['build_report'],'bounded':['build_report','host_report'],'full':['build_report','host_report','bounded_report']}
    if any(getattr(a,k) is None for k in required.get(a.stage,[])):p.error('missing prior phase report')
    return a

if __name__=='__main__':raise SystemExit(main(parse()))
