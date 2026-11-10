#!/usr/bin/env python3
import argparse
from pathlib import Path
import nz_common as s
import nz_transform as t
from nz_build import build
from nz_protocol import validate,paired,fault_expected,ORDER
from run_stage2b import command
STATES=dict(source='B_NZ_SOURCE_REVIEWED',prepare='B_NZ_PREPACK_VERIFIED',hostplan='B_NZ_STANDALONE_RESOURCE_FEASIBLE',build='B_NZ_BUILT_NPU_NOT_RUN',host='B_NZ_FORMAL_HOST_MATCH',tile0='B_NZ_TILE0_MATCH',full='B_NZ_FULL_MATCH',fault='B_NZ_FAULT_COVERAGE_MATCH',benchmark='B_NZ_PAIRED_MEASURED')


def verify_repository():t.verify_repository()

def prior(path,stage,r):
 if path is None:raise ValueError('missing '+stage+' report')
 old=s.read(path);s.verify_tree(old)
 if old.get('status')!=STATES[stage] or old.get('error') or old.get('integrity_error') or any(old.get(k)!=r[k] for k in ('sources','device','evidence','cann_root')):raise ValueError('predecessor identity/state differs')
 for key in ('os','version_files'):
  if old['machine'][key]!=r['machine'][key]:raise ValueError('machine drift')
 if stage=='prepare':s.verify_prepack(old['packing'],r['evidence'])
 if stage=='hostplan':
  from nz_resources import validate as resource_check
  raw=s.read(s.verify(old['host_query']))
  if old['t1']!={'fields':{k:raw['cases'][0]['fields'][k] for k in s.p.FIELDS},'capacity':raw['capacity'],'system_workspace_bytes':raw['system_workspace_bytes'],'scratch':202240}:raise ValueError('Host plan T1 differs from actual query')
  from dependency_identity_v import verify_dependency_binding
  verify_dependency_binding(old['sdk']['dependency_binding'])
  if resource_check(raw,raw['cases'][0])!=old['resources'] or old['resources']['status']!='HOST_RESOURCE_CHECKS_PASS':raise ValueError('NZ Host plan changed')
 if stage=='build':s.verify_build(old)
 if stage not in ('source','prepare') and 'prepare_report' in r and old.get('prepare_report')!=r['prepare_report']:raise ValueError('different prepack gate')
 schedules=dict(host=[('candidate',0,0,-1),('fault',0,0,-1)],tile0=[('candidate',0,0,0)],full=[('candidate',0,0,-1)],fault=[('fault',m,1,0) for m in range(2,8)])
 if stage in schedules:
  if [(x['variant'],x['mode'],x['limit'],x['tile']) for x in old['runs']]!=schedules[stage]:raise ValueError('coverage incomplete')
  for row in old['runs']:
   if row['variant']!='baseline' and (row['accepted_t1']!=old['t1'] or row['b_input']!=old['packing']['packed_b']):raise ValueError('row T1/packed B changed')
   if row['validation']!=validate(row,r['evidence'],r['device'],host=stage=='host'):raise ValueError('prior validation drift')
   b=s.read(s.verify(old['build_report']))
   if row['runner']!=b['variants'][row['variant']]['build']['runner']:raise ValueError('wrong Runner variant')
 return old


def invoke(a,root,r,b,kind,mode=0,limit=0,tile=-1,host=False,timed=False):
 if timed and (kind=='fault' or mode or limit or tile!=-1):raise ValueError('only normal full Auto may time')
 s.verify_bound(r['evidence']);s.verify_build(b)
 root.mkdir(parents=True);proof=r['evidence']['baseline'];case=s.read(s.verify(proof['case']));maps=s.c.frozen.ref.verify_case(case)
 s.verify_prepack(r['packing'],r['evidence'])
 for n in ('a_i8.bin','b_i8.bin','quant_u64_doc_candidate.bin'):
  item=r['packing']['packed_b'] if kind!='baseline' and n=='b_i8.bin' else case['files'][n]
  (root/n).symlink_to(s.verify(item))
 if not host:
  for i in range(2):(root/f'expected-{i}.bin').write_bytes(fault_expected(mode,i) if kind=='fault' else s.c.old_protocol.expected(maps,mode,proof['policy']['selected'],tile,i))
 if timed:(root/'admitted_full.bin').symlink_to(s.verify(r['evidence']['anchor']))
 if kind=='baseline':runner=r['evidence']['baseline_runner'];opp=proof['normal']['opp']
 else:runner=b['variants'][kind]['build']['runner'];opp=b['variants'][kind]['build']['opp']
 row=dict(variant=kind,mode=mode,limit=limit,tile=tile,runner=runner,profiled=False,accepted_t1=r['t1'],b_input=r['packing']['packed_b'] if kind!='baseline' else case['files']['b_i8.bin'])
 r.setdefault('runs',[]).append(row)
 argv=[s.verify(runner),a.device,root,mode,limit,tile]+(['--host-only'] if host else ['--benchmark'] if timed else [])
 env=s.env(r['evidence'],a.cann_root,opp)
 if kind!='baseline':
  system=r['t1']['system_workspace_bytes'];blocks=limit or proof['policy']['selected']
  env['FULLK_EXPLORE_WORKSPACE_BYTES']=str(system+blocks*((system+511)//512*512+1024))
 try:command(argv,root/'runner.log',env,r,'run_'+root.name,1800 if kind=='fault' else 900)
 finally:row['retained_files']=[s.identity(p) for p in sorted(root.iterdir()) if p.is_file()]
 row.update(execution=s.identity(root/'execution.json'),log=s.identity(root/'runner.log'),admission=s.identity(root/'resource-admission.json'))
 if not host:
  row['outputs']=[s.identity(root/n) for n in s.OUTPUTS[:4 if timed else 2]]
  if kind!='fault':row['numeric']=[s.identity(root/(n+'.numeric.json')) for n in s.OUTPUTS[:4 if timed else 2]]
 row['validation']=validate(row,r['evidence'],a.device,host,timed);s.write(root/'validation.json',row['validation']);return row


def main(a):
 root=s.fresh(a.output_dir,[s.HERE,a.cann_root,a.experiment_root,a.arena_root,a.profile_report.parent]+[p.parent for p in (a.source_report,a.prepare_report,a.hostplan_report,a.build_report,a.host_report,a.tile_report,a.full_report,a.fault_report) if p])
 r=dict(status='NOT_RUN',phase=a.phase,sources=s.sources(),device=a.device,cann_root=str(a.cann_root),npu='NOT_RUN',benchmark='NOT_RUN',numeric_status='NUMERIC_MISMATCH_UNRESOLVED',production='NOT_APPROVED')
 try:
  verify_repository();r['evidence']=s.bind(a,root,r)
  if a.phase!='source':prior(a.source_report,'source',r);r['source_report']=s.identity(a.source_report)
  if a.phase=='prepare':
   from nz_layout import prepack
   source=s.verify(s.read(s.verify(r['evidence']['baseline']['case']))['files']['b_i8.bin'])
   packed=root/'b_nz.bin';layout=prepack(source,packed)
   r['packing']=dict(source_b=s.identity(source),packed_b=s.identity(packed),layout=layout)
   s.verify_prepack(r['packing'],r['evidence'])
  elif a.phase not in ('source','prepare'):
   prep=prior(a.prepare_report,'prepare',r);r['prepare_report']=s.identity(a.prepare_report);r['packing']=prep['packing']
   if prep['source_report']!=r['source_report']:raise ValueError('prepack Source chain differs')
  if a.phase=='hostplan':
   from host_nz import query
   query(a,root,r)
  elif a.phase not in ('source','prepare','hostplan'):
   plan=prior(a.hostplan_report,'hostplan',r);r['hostplan_report']=s.identity(a.hostplan_report);r['t1']=plan['t1']
   if plan['prepare_report']!=r['prepare_report'] or plan['source_report']!=r['source_report']:raise ValueError('Host plan/prepack chain differs')
  if a.phase=='build':
   r['variants']={}
   for kind in ('candidate','fault'):
    src=root/('source-'+kind);audit=t.materialize(src,kind,s.identity,s.write);dest=root/kind;dest.mkdir()
    item=dict(evidence=dict(r['evidence']['baseline']['original_evidence'],t1=r['t1']),experiment_evidence=r['evidence'])
    r['variants'][kind]=dict(audit=audit,build=item)
    try:build(a,dest,item,src,kind)
    finally:s.write(dest/'build-report.json',item)
    r['variants'][kind]['compiled_sources']=item['generated_sources']
   s.verify_build(r)
  elif a.phase not in ('source','prepare','hostplan','build'):
   b=prior(a.build_report,'build',r);r['build_report']=s.identity(a.build_report)
   if any(b[k]!=r[k] for k in ('source_report','prepare_report','hostplan_report','t1')):raise ValueError('Build source chain differs')
   previous=[('host_report','host'),('tile_report','tile0'),('full_report','full'),('fault_report','fault')]
   required={'host':0,'tile0':1,'full':2,'fault':3,'benchmark':4}[a.phase]
   for attr,stage in previous[:required]:
    value=prior(getattr(a,attr),stage,r)
    if value['build_report']!=r['build_report']:raise ValueError('gate from different Build')
    for oldkey,_ in previous[:previous.index((attr,stage))]:
     if value[oldkey]!=r[oldkey]:raise ValueError('gate chain changed')
    r[attr]=s.identity(getattr(a,attr))
   r['npu']='NOT_RUN' if a.phase=='host' else 'ATTEMPTED'
   if a.phase=='host':
    for kind in ('candidate','fault'):invoke(a,root/kind,r,b,kind,host=True)
   elif a.phase in ('tile0','full'):invoke(a,root/'run',r,b,'candidate',tile=0 if a.phase=='tile0' else -1)
   elif a.phase=='fault':
    for mode in range(2,8):invoke(a,root/f'mode-{mode}',r,b,'fault',mode,1,0);s.write(root/'report.json',r)
    r['fault_coverage']=dict(ub_words=384,ub_checkpoints=3,arena_words_per_side=128,arena_sides=2,arena_checkpoints=2,ub_injections_per_repeat=36864,arena_gm_injections_per_repeat=16384,total_two_repeat_injections=106496,abort_witnesses='3 UB plus2 Arena; each repeated twice',tile=0,blocks=1,outer_gm_guards='UNCHANGED_RUNNER_CHECKS_NOT_EXHAUSTIVE_DEVICE_BIT_INJECTION',performance_eligible=False)
   else:
    # Fresh untimed baseline + candidate checks before any formal timing.
    for kind in ('baseline','candidate'):invoke(a,root/('precheck-'+kind),r,b,kind)
    r['prechecks']=r.pop('runs');r['runs']=[];r['benchmark']='ATTEMPTED'
    for i,order in enumerate(ORDER):
     for pos,kind in enumerate(order):
      row=invoke(a,root/f'round-{i}'/kind,r,b,kind,timed=True);row.update(round=i,position=pos);s.write(root/'report.json',r)
    r['comparison']=paired(r['runs'])
    from nz_layout import amortization
    stats=r['comparison']['statistics'];base_ms=stats['baseline']['execute_sync']['median_ms'];candidate_ms=stats['candidate']['execute_sync']['median_ms']
    r['comparison']['prepack_cost']=r['packing']['layout']
    r['comparison']['workspace_by_variant']={n:next(x['validation']['plan']['total'] for x in r['runs'] if x['variant']==n) for n in ('baseline','candidate')}
    r['comparison']['b_physical_bytes_by_variant']={'baseline':49807360,'candidate':49807360}
    r['comparison']['amortization_permutation_only']=amortization(r['packing']['layout']['permutation_ms'],base_ms,candidate_ms)
    r['comparison']['amortization_complete_prepare']=amortization(r['packing']['layout']['prepare_wall_ms'],base_ms,candidate_ms)
    r['benchmark']='COMPLETED'
   if a.phase!='host':r['npu']='COMPLETED'
  r['status']=STATES[a.phase]
 except Exception as x:r.update(status='BLOCKED',error=f'{type(x).__name__}: {x}')
 finally:
  try:
   verify_repository();s.verify_tree(r)
   if r['sources']!=s.sources():raise ValueError('sources changed')
   if 'evidence' in r:s.verify_bound(r['evidence'])
   if 'packing' in r:s.verify_prepack(r['packing'],r['evidence'])
   if a.build_report:s.verify_build(s.read(a.build_report))
  except Exception as x:r.update(status='BLOCKED',integrity_error=str(x))
  if r['status']=='BLOCKED':r.pop('comparison',None)
  s.write(root/'report.json',r)
  if a.phase=='benchmark':s.write(root/'benchmark-report.json',r);s.write(root/'comparison.json',r.get('comparison',dict(status='NOT_RUN_OR_BLOCKED')))
 print(r['status']+': '+str(root/'report.json'));return int(r['status']=='BLOCKED')


def parse(argv=None):
 p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False);p.add_argument('phase',choices=STATES);p.add_argument('--output-dir',type=Path,required=True)
 p.add_argument('--cann-root',type=Path,default=Path('/usr/local/Ascend/cann-9.0.0'));p.add_argument('--device',type=int,default=0)
 p.add_argument('--experiment-root',type=Path,default=Path('/home/w00949577/qwen35-dflash-run/w8a8-fullk-perf-explore-20261110-022343'))
 p.add_argument('--arena-root',type=Path,default=Path('/home/w00949577/qwen35-dflash-run/w8a8-fullk-arena-ab-20261110-054934'))
 p.add_argument('--profile-report',type=Path,default=Path('/home/w00949577/qwen35-dflash-run/w8a8-fullk-profile-20261110-030853/collect/profile-report.json'))
 for x in ('source','prepare','hostplan','build','host','tile','full','fault'):p.add_argument('--'+x+'-report',type=Path)
 a=p.parse_args(argv);stages=list(STATES);i=stages.index(a.phase)
 required=['source','prepare','hostplan','build','host','tile','full','fault'][:i]
 for x in required:
  if getattr(a,x+'_report') is None:p.error('--'+x+'-report required')
 if a.device<0:p.error('negative device')
 for k,v in vars(a).items():
  if isinstance(v,Path):setattr(a,k,v.resolve())
 return a
if __name__=='__main__':raise SystemExit(main(parse()))
