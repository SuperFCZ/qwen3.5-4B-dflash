#!/usr/bin/env python3
"""Four private variants; only move SetLocalWorkspace to tile entry."""
import argparse,json,re,shlex,shutil,statistics
from pathlib import Path
from f_common import (c4,identity,write,verify,verify_tree,POLICIES,COVERAGE,
    check_execution,baseline,full,stage4b,validate_build,ORDER,SMOKE_ORDER,summarize,fresh,sources,evidence,sdk_binding,binary_summary)
from workspace_variant import verify_audit
from workspace_variant import materialize
from f_build import build_variant,installed
from profile_metrics import collect,numeric,norm
from run_stage2b import command


def env(args,proof,opp):return c4.environment(dict(policy=proof['policy'],opp=str(opp)),args.cann_root)


def build(args,root,r):
    audit=materialize(root/'sources',identity,write,r['evidence']['stage4d_b_kernel']);r['source_audit']=audit;r['variants']={}
    accepted_sources={item['path']:item for item in r['evidence']['stage4b_sources']}
    if any(accepted_sources.get(item['path'])!=item for item in audit['originals']):raise ValueError('materialized inputs differ from accepted Stage4B source identities')
    for v in POLICIES:
        folder=root/v;folder.mkdir();item=dict(status='NOT_RUN',evidence=r['evidence'],device=args.device,variant=v,safety_protocol=POLICIES[v])
        r['variants'][v]=item;r['phase']='build_'+v;write(root/'build-report.json',r)
        try:build_variant(args,folder,item,Path(audit['variants'][v]['source_dir']))
        finally:write(folder/'build-report.json',item)
        verify_audit(audit,verify)
        # Verify compiled project inputs as well as the materialized audit snapshots.
        for record in item['generated_sources']:verify(record)
        for sub in ('op_host','op_kernel'):
            for p in (Path(audit['variants'][v]['source_dir'])/sub).iterdir():
                if p.is_file() and p.read_bytes()!=(folder/'CustomOp'/sub/p.name).read_bytes():raise ValueError('compiled source no longer equals audit')
    r['binary_comparison']=binary_summary(r['variants'])
    r.update(status='STAGE4F_BUILT_NPU_NOT_RUN',phase='complete')


def accept_build(args,r):
    if args.pair_build_report is None:raise ValueError('--pair-build-report required')
    b=json.loads(args.pair_build_report.read_text());validate_build(b,r,installed)
    r['pair_build_report']=identity(args.pair_build_report);r['source_audit']=b['source_audit'];r['binary_comparison']=b['binary_comparison']
    return b


def invoke(args,folder,r,b,v,timed,profile=False,limit=0):
    folder.mkdir(parents=True);proof=r['evidence'];variant=b['variants'][v]
    for name in ('a_i8.bin','b_i8.bin','quant_u64_doc_candidate.bin','golden_fp16.bin'):(folder/name).symlink_to(verify(proof['files'][name]))
    argv=[verify(variant['runner']),args.device,folder,limit,v]+(['--benchmark'] if timed else [])
    environment=env(args,proof,variant['opp'])
    profiler_id=None;profile_argv=None
    if profile:
        profiler=shutil.which(args.msprof)
        if not profiler:raise ValueError('msprof unavailable')
        profiler_id=identity(profiler);launcher=folder.parent/'launch.sh'
        launcher.write_text('#!/usr/bin/env bash\nset -euo pipefail\nexec '+shlex.join(list(map(str,argv)))+' > '+shlex.quote(str(folder/'runner.log'))+' 2>&1\n')
        profile_argv=[profiler,'--ai-core=on','--aic-mode=task-based','--aic-metrics=PipeUtilization',
            '--output='+str(folder.parent/'profiler'),'--application='+shlex.join(['bash',str(launcher)])]
        command(profile_argv,folder.parent/'profile.log',environment,r,'profile_'+v,1800)
    else:command(argv,folder/'runner.log',environment,r,str(folder.relative_to(Path(r['output_dir']))),600)
    execution=identity(folder/'execution.json');ex=json.loads(verify(execution).read_text());stats=check_execution(ex,args.device,v,timed,limit)
    outputs=baseline.verify_outputs(proof,folder,timed)
    plan=full.plan_from_log((folder/'runner.log').read_text(),0,limit)
    configurations={line[line.index('STAGE4B_PLAN'):].strip() for line in (folder/'runner.log').read_text().splitlines() if 'STAGE4B_PLAN' in line}
    if len(configurations)!=1:raise ValueError('missing/drifting complete Host Matmul resource configuration')
    if plan['core_policy']!=proof['policy']:raise ValueError('Core policy drift')
    admission=identity(folder/'resource-admission.json')
    if json.loads(verify(admission).read_text())!=stage4b.admission_expected(ex,plan,False):raise ValueError('resource admission differs')
    if installed(Path(variant['opp']))!=variant['installed']:raise ValueError('private variant OPP changed')
    return dict(status='SCOPED_GOLDEN_MATCH',variant=v,core_limit=limit,safety_protocol=POLICIES[v],execution=execution,outputs=outputs,
        plan=plan,kernel_configuration=configurations.pop(),admission=admission,workspace_bytes=ex['workspace_bytes'],timing=stats,profiled=profile,
        formal_latency_evidence=timed and not profile,profiler=profiler_id,profiler_command=profile_argv,
        safety_equivalence_to_stage4b=('SAME_SOURCE_GUARD_PROTOCOL' if v.startswith('A') else 'NOT_CLAIMED'),guard_coverage=COVERAGE[v])

def same_resources(runs):
    if not runs:raise ValueError('no resource evidence')
    if len({row['core_limit'] for row in runs})>1:
        for limit in (0,2):
            subset=[row for row in runs if row['core_limit']==limit]
            if subset:same_resources(subset)
        return
    for key in ('plan','workspace_bytes','kernel_configuration'):
        if any(row[key]!=runs[0][key] for row in runs):raise ValueError('A0/A1/B0/B1 resource/configuration differs: '+key)


def accepted_report(path,status,r,b):
    if path is None:raise ValueError('required prior scoped report missing: '+status)
    p=json.loads(path.read_text());verify_tree(p)
    if p.get('status')!=status or p.get('phase')!='complete' or any(p.get(k)!=r[k] for k in ('sources','evidence','device','pair_build_report','sdk_binding')):raise ValueError('prior A0/A1/B0/B1 report binding differs')
    timed=status=='STAGE4F_CONTRASTS_MEASURED_DIAGNOSTIC_ONLY'
    expected=[(v,0) for row in ORDER for v in row] if timed else SMOKE_ORDER
    if [(e['variant'],e['core_limit']) for e in p.get('runs',[])]!=expected:raise ValueError('incomplete paired run coverage')
    for e in p['runs']:
        v=e['variant'];limit=e['core_limit'];folder=Path(e['execution']['path']).parent
        if e['status']!='SCOPED_GOLDEN_MATCH' or e['safety_protocol']!=POLICIES[v] or e['profiled'] is not False or e['formal_latency_evidence'] is not timed:raise ValueError('variant safety/profile scope differs')
        if e.get('guard_coverage')!=COVERAGE[v] or e.get('safety_equivalence_to_stage4b')!=('SAME_SOURCE_GUARD_PROTOCOL' if v.startswith('A') else 'NOT_CLAIMED'):raise ValueError('incorrect guard coverage claim')
        ex=json.loads(verify(e['execution']).read_text())
        if check_execution(ex,r['device'],v,timed,limit)!=e['timing'] or baseline.verify_outputs(r['evidence'],folder,timed)!=e['outputs']:raise ValueError('prior numeric/timing evidence differs')
        plan=full.plan_from_log((folder/'runner.log').read_text(),0,limit)
        if plan!=e['plan'] or plan['core_policy']!=r['evidence']['policy'] or json.loads(verify(e['admission']).read_text())!=stage4b.admission_expected(ex,plan,False):raise ValueError('prior resources differ')
        configurations={line[line.index('STAGE4B_PLAN'):].strip() for line in (folder/'runner.log').read_text().splitlines() if 'STAGE4B_PLAN' in line}
        if configurations!={e['kernel_configuration']} or e['workspace_bytes']!=ex['workspace_bytes']:raise ValueError('prior complete resource configuration differs')
    same_resources(p['runs'])
    if timed and summarize(p['runs'])!=p['comparison']:raise ValueError('paired summary differs from raw samples')
    validate_build(b,r,installed);return p


def profile_delta(profiles):
    def tables(data):
        result={}
        for row in data['counter_rows']:
            kind=re.sub(r'\d{8,}', '<timestamp>',Path(row['file']['path']).name)
            for key,value in row['fields'].items():
                if key.lower().startswith('machine') or not any(word in key.lower() for word in ('duration','scalar','mac','cube','vec','mte')) or not numeric(value):continue
                result.setdefault((kind,key),[]).append(float(value.strip().rstrip('%')))
        return result
    tables_by_variant={v:tables(profiles[v]['raw']) for v in POLICIES};rows=[]
    for base,v in (('A0','A1'),('B0','B1')):
        a,b=tables_by_variant[base],tables_by_variant[v]
        for key in sorted(set(a)|set(b)):
            x,y=a.get(key,[]),b.get(key,[])
            row=dict(baseline=base,variant=v,file_kind=key[0],column=key[1],baseline_count=len(x),variant_count=len(y),unit='as declared by original column')
            if not x or not y or len(x)!=len(y):row['status']='UNRESOLVED_MISSING_OR_UNEQUAL_ROW_COVERAGE'
            else:
                av,bv=statistics.median(x),statistics.median(y)
                row.update(status='RAW_SAME_COLUMN_COMPARISON',baseline_median=av,variant_median=bv,delta_baseline_minus_candidate=av-bv)
            rows.append(row)
    required={'task_duration':('duration',),'scalar_time':('scalartime',),'scalar_ratio':('scalarratio',),
        'mte1_time':('mte1time',),'mte2_time':('mte2time',),'mte3_time':('mte3time',),
        'vector_time':('vectime','vectortime'),'vector_ratio':('vecratio','vectorratio'),
        'mac_time':('mactime',),'mac_ratio':('macratio',),'cube_utilization':('cubeutilization',)}
    availability={v:{name:('OBSERVED_RAW_COLUMNS' if any(any(token in norm(key[1]) for token in tokens) for key in table) else 'UNAVAILABLE') for name,tokens in required.items()} for v,table in tables_by_variant.items()}
    return dict(columns=rows,availability=availability,
        scope='Profile rows include precheck/warmup/sample/postcheck; same CSV kind and exact column only; not formal latency.',
        interpretation='No pipe sums. Only compare A0/A1 or B0/B1. Scalar differences are not wall-time savings.',
        critical_path='UNRESOLVED',non_guard_scalar_time='UNRESOLVED',further_overlap='NOT_RUN')


def main(args):
    if args.stage=='profile' and args.benchmark_report and args.pair_build_report is None:
        args.pair_build_report=verify(json.loads(args.benchmark_report.read_text())['pair_build_report'])
    protected=[args.cann_root,args.stage4e_root,args.stage4d_root,args.stage4c_root,args.stage4b_root]
    protected += [p.parent for p in (args.pair_build_report,args.smoke_report,args.benchmark_report) if p]
    root=fresh(args.output_dir,protected)
    r=dict(stage='4F',status='NOT_RUN',phase='evidence',device=args.device,sources=sources(),output_dir=str(root),
        npu='NOT_RUN',performance='NOT_RUN',safety_protocols=POLICIES,adoption='NOT_DECIDED; B0/B1 permanently diagnostic-only',
        further_overlap='NOT_RUN',partial_device_diagnostics='NOT_RUN; full-output Smoke only',equivalent_vector_guard='NOT_RUN',draft='NOT_RUN',online_preprocessing='ONLINE_PREPROCESSING_NOT_RUN')
    try:
        r['evidence']=evidence(args);r['sdk_binding']=sdk_binding(args.cann_root)
        r['machine']=c4.machine(root,dict(**r['evidence'],opp=r['evidence']['stage4b']['opp']),args.cann_root,r)
        if args.stage=='build':build(args,root,r)
        else:
            b=accept_build(args,r);r['runs']=[]
            if args.stage=='smoke':
                for v,limit in SMOKE_ORDER:
                    r.update(phase=f'smoke_{v}_core{limit}',npu='RUNNING');write(root/'smoke-report.json',r)
                    r['runs'].append(invoke(args,root/v/f'core-{limit}'/'case',r,b,v,False,limit=limit))
                same_resources(r['runs'])
                r.update(status='STAGE4F_SCOPED_CORRECTNESS_MATCH',phase='complete',npu='NUMERICAL_MATCH_UNDER_DIFFERENT_GUARD_PROTOCOLS')
            elif args.stage=='benchmark':
                prior_smoke=accepted_report(args.smoke_report,'STAGE4F_SCOPED_CORRECTNESS_MATCH',r,b);r['smoke_report']=identity(args.smoke_report)
                plans=[]
                for i,order in enumerate(ORDER):
                    for position,v in enumerate(order):
                        r.update(phase=f'round{i}_{v}',npu='RUNNING');write(root/'benchmark-report.json',r)
                        row=invoke(args,root/f'round-{i}'/v/'case',r,b,v,True);row.update(round=i,position=position)
                        plans.append(row['plan']);r['runs'].append(row)
                if any(p!=plans[0] for p in plans):raise ValueError('A0/A1/B0/B1 Core/Tiling/workspace plan differs')
                same_resources(r['runs']+prior_smoke['runs'])
                r.update(status='STAGE4F_CONTRASTS_MEASURED_DIAGNOSTIC_ONLY',phase='complete',comparison=summarize(r['runs']),
                    npu='SCOPED_GOLDEN_MATCH',performance='FRESH_UNPROFILED_SEPARATE_A_B_PAIRS_5_30_THREE_ROUNDS')
            else:
                prior_benchmark=accepted_report(args.benchmark_report,'STAGE4F_CONTRASTS_MEASURED_DIAGNOSTIC_ONLY',r,b)
                r['benchmark_report']=identity(args.benchmark_report);r['profiles']={}
                for v in POLICIES:
                    r.update(phase='profile_'+v,npu='RUNNING');write(root/'profile-report.json',r)
                    row=invoke(args,root/v/'case',r,b,v,True,True);raw=collect(root/v/'profiler')
                    r['profiles'][v]=dict(execution=row,raw=raw,safety_protocol=POLICIES[v],formal_latency_evidence=False)
                if len({p['execution']['profiler']['sha256'] for p in r['profiles'].values()})!=1:raise ValueError('profiler tool changed between variants')
                same_resources([p['execution'] for p in r['profiles'].values()]+prior_benchmark['runs'])
                r.update(status='STAGE4F_PROFILES_COLLECTED_DIAGNOSTIC_ONLY' if all(p['raw']['status']=='RAW_OPERATOR_METRICS_FOUND' for p in r['profiles'].values()) else 'PROFILE_METRICS_UNAVAILABLE',
                    phase='complete',profile_comparison=profile_delta(r['profiles']),npu='PROFILE_SCOPED_GOLDEN_MATCH',performance='PROFILE_EXCLUDED_FROM_FORMAL_AB')
    except Exception as error:
        r.update(status='FAILED',error=f'{type(error).__name__}: {error}')
        if r['npu']=='RUNNING':r['npu']='FAILED_OR_INCOMPLETE'
    finally:
        try:
            for item in r['sources']:verify(item)
            if 'evidence' in r:verify_tree(r['evidence'])
            if 'sdk_binding' in r:verify_tree(r['sdk_binding'])
            if 'source_audit' in r:verify_audit(r['source_audit'],verify)
            if args.pair_build_report:validate_build(json.loads(args.pair_build_report.read_text()),r,installed)
            for b in r.get('variants',{}).values():
                if b.get('opp') and installed(Path(b['opp']))!=b['installed']:raise ValueError('new private OPP changed')
            for item in r.get('machine',{}).get('version_files',[]):verify(item['file'])
        except Exception as error:r.update(status='FAILED',integrity_error=str(error))
        r['artifacts']=[identity(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name in ('execution.json','runner.log','actual-0.bin','actual-1.bin','benchmark-last.bin','postcheck.bin')]
        write(root/(args.stage+'-report.json'),r)
        write(root/'guard-coverage.json',dict(protocols=POLICIES,coverage=COVERAGE,adoption=r['adoption'],source_audit=r.get('source_audit'),scope='No safety equivalence or production promotion'))
    print(r['status']+': '+str(root/(args.stage+'-report.json')));return int(r['status']=='FAILED')


def parse():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('build','smoke','benchmark','profile'))
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--pair-build-report',type=Path);p.add_argument('--smoke-report',type=Path);p.add_argument('--benchmark-report',type=Path);p.add_argument('--msprof',default='msprof')
    for name,default in [('stage4e','w8a8-tile-guard-20261029-020949'),('stage4d','w8a8-guard-ab-20261029-012635'),('stage4c','w8a8-baseline-20261027-111748'),('stage4b','w8a8-full-gate-20261027-093617'),('stage4a','w8a8-core-policy-20261027-083223'),('stage3b4','w8a8-twenty-group-20261027-055348'),('stage3b3','w8a8-two-group-20261027-042044'),('stage3b2','w8a8-local-nz-20261027-031856'),('stage3b1','w8a8-stage3b-20261027-024844'),('stage3a','w8a8-stage3a-20261027-011523'),('stage2b','w8a8-stage2b-20261026-100533')]:
        p.add_argument('--'+name+'-root',type=Path,default=Path('/home/w00949577/qwen35-dflash-run')/default)
    for name in ('build','basic','zero','nonpower'):p.add_argument('--'+name+'-report',type=Path)
    p.add_argument('--stage2c-report',type=Path,default=Path('/home/w00949577/qwen35-dflash-run/w8a8-stage2c-20261026-104943/report.json'))
    p.add_argument('--device',type=int,default=0);p.add_argument('--cann-root',type=Path,default=Path('/usr/local/Ascend/cann-9.0.0'))
    args=p.parse_args();args.cann_root=args.cann_root.resolve()
    for name in ('build','basic','zero','nonpower'):
        if getattr(args,name+'_report') is None:setattr(args,name+'_report',args.stage2b_root/name/'report.json')
    if args.device<0:p.error('negative device')
    return args

if __name__=='__main__':raise SystemExit(main(parse()))
