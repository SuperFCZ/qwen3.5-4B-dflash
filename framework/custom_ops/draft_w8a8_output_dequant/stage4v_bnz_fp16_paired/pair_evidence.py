"""Reverify accepted B-NZ and native FP16 evidence, without rebuilding either."""
from pathlib import Path
from types import SimpleNamespace
import re,sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'stage4v_fullk_b_nz'))
import nz_common as nz
import run_b_nz as old
import nz_protocol as npcol
from nz_layout import amortization
c=nz.c;rr=nz.v.rr
identity,verify,read,write,fresh,verify_tree=nz.identity,nz.verify,nz.read,nz.write,nz.fresh,nz.verify_tree
__all__=['amortization']


def sources():return [identity(p) for p in sorted(HERE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts]


def stage_path(root,item):
 p=verify(item)
 if root not in p.parents:raise ValueError('predecessor outside selected experiment')
 return p


def resolve_operator(build):
 installed=build['variants']['candidate']['build']['installed'];header=verify(installed['header'])
 text=header.read_text();names=re.findall(r'\b(aclnn\w+GetWorkspaceSize)\s*\(',text)
 if len(set(names))!=1:raise ValueError('ambiguous installed normal API symbol')
 prepare=names[0];execute=prepare[:-len('GetWorkspaceSize')]
 if not re.search(r'\b'+execute+r'\s*\(',text):raise ValueError('installed execute symbol absent')
 name=execute[len('aclnn'):]
 definition=Path(build['variants']['candidate']['build']['opp']).parent/'operator.json'
 op=read(definition)
 if len(op)!=1 or op[0]['op']!=name or name!=nz.NAMES['candidate'][0]:raise ValueError('installed definition/header/source Op identity disagree')
 return dict(op=name,prepare_symbol=prepare,execute_symbol=execute,header=installed['header'],definition=identity(definition),
  kernel_identities=installed['kernels'],api=installed['api'],host=installed['host'])


def compare_machine(a,b):
 if any(a[k]!=b[k] for k in ('os','version_files')):raise ValueError('machine/driver/version identity drift')


def bind(a,root,r):
 root_old=a.bnz_root.resolve(strict=True);br=read(root_old/'benchmark/benchmark-report.json');verify_tree(br)
 if br!=read(root_old/'benchmark/report.json') or br['comparison']!=read(root_old/'benchmark/comparison.json'):raise ValueError('B-NZ benchmark copies differ')
 # Reconstruct only existing parent evidence locations, never guess an OPP install path.
 ev=br['evidence'];arena=Path(verify(ev['arena_benchmark'])).parent.parent
 full_exp=Path(verify(ev['arena_evidence']['history']['reports']['benchmark'])).parent.parent
 args=SimpleNamespace(arena_root=arena,experiment_root=full_exp,profile_report=verify(ev['arena_evidence']['profile_report']),cann_root=a.cann_root,device=a.device)
 current=dict(sources=nz.sources(),device=a.device,cann_root=str(a.cann_root))
 current['evidence']=nz.bind(args,root,current);r['machine']=current['machine']
 records={};stages={}
 for stage,key in [('source','source_report'),('prepare','prepare_report'),('hostplan','hostplan_report'),('build','build_report'),('host','host_report'),('tile0','tile_report'),('full','full_report'),('fault','fault_report')]:
  stages[stage]=old.prior(stage_path(root_old,br[key]),stage,current);records[stage]=br[key]
 if br.get('status')!='B_NZ_PAIRED_MEASURED' or br.get('benchmark')!='COMPLETED' or br.get('npu')!='COMPLETED' or br.get('error') or br.get('integrity_error'):raise ValueError('B-NZ benchmark incomplete')
 if any(br[k]!=current[k] for k in ('sources','device','cann_root','evidence')):raise ValueError('B-NZ benchmark source/evidence differs')
 compare_machine(br['machine'],current['machine'])
 links=['source_report','prepare_report','hostplan_report','build_report','host_report','tile_report','full_report','fault_report']
 order=['source','prepare','hostplan','build','host','tile0','full','fault','benchmark']
 for i,stage in enumerate(order[1:],1):
  rec=br if stage=='benchmark' else stages[stage]
  if any(rec[k]!=br[k] for k in links[:i]):raise ValueError('B-NZ predecessor chain mismatch '+stage)
 if br['packing']!=stages['prepare']['packing'] or br['t1']!=stages['hostplan']['t1']:raise ValueError('B-NZ packing/T1 proof changed')
 for stage in ('build','host','tile0','full','fault'):
  if stages[stage]['packing']!=br['packing'] or stages[stage]['t1']!=br['t1']:raise ValueError('B-NZ phase packing/T1 drift')
 if [x['variant'] for x in br['prechecks']]!=['baseline','candidate']:raise ValueError('B-NZ prechecks incomplete')
 build=stages['build'];nz.verify_build(build);nz.verify_prepack(br['packing'],ev)
 for row in br['prechecks']+br['runs']:
  timed='round' in row
  if npcol.validate(row,ev,a.device,timed=timed)!=row['validation']:raise ValueError('B-NZ historical outputs/timing changed')
  runner=ev['baseline_runner'] if row['variant']=='baseline' else build['variants']['candidate']['build']['runner']
  if row['runner']!=runner or row.get('profiled') is not False:raise ValueError('wrong historical Runner/profile scope')
  if row['variant']=='candidate' and (row['b_input']!=br['packing']['packed_b'] or row['accepted_t1']!=br['t1']):raise ValueError('B-NZ row input/T1 changed')
 expected=npcol.paired(br['runs']);ss=expected['statistics'];bm=ss['baseline']['execute_sync']['median_ms'];cm=ss['candidate']['execute_sync']['median_ms']
 expected.update(prepack_cost=br['packing']['layout'],workspace_by_variant={n:next(x['validation']['plan']['total'] for x in br['runs'] if x['variant']==n) for n in ('baseline','candidate')},b_physical_bytes_by_variant={'baseline':49807360,'candidate':49807360},
  amortization_permutation_only=amortization(br['packing']['layout']['permutation_ms'],bm,cm),amortization_complete_prepare=amortization(br['packing']['layout']['prepare_wall_ms'],bm,cm))
 if br['comparison']!=expected:raise ValueError('B-NZ aggregate/cost summary differs')
 records['benchmark']=identity(root_old/'benchmark/benchmark-report.json')
 original=ev['baseline']['original_evidence'];native_br=read(verify(original['stage4r_benchmark']));verify_tree(native_br)
 nr=dict(sources=rr.sources(),evidence=original['qproof'],device=a.device,candidate=native_br['candidate'],policy=rr.POLICY,machine=current['machine'])
 rr.prior(verify(original['stage4r_benchmark']),'benchmark',nr)
 nprep=rr.prior(verify(native_br['prepare_report']),'prepare',nr);nbuild=rr.prior(verify(native_br['build_report']),'build',nr)
 if nbuild['prepare_report']!=native_br['prepare_report'] or nprep['reference']!=original['native_reference'] or nbuild['native']!=original['native_build']:raise ValueError('FP16 historical source/reference/Runner differs')
 compare_machine(nbuild['machine'],current['machine']);compare_machine(native_br['machine'],current['machine'])
 rr.verify_binding(nbuild['native']['header_binding'],a.cann_root)
 native_data=read(verify(nprep['reference']));verify_tree(native_data)
 if native_data['policy']!=rr.POLICY:raise ValueError('FP16 numerical policy drift')
 nr.update(prepare_report=native_br['prepare_report'],build_report=native_br['build_report'])
 smoke=rr.validate_smoke(verify(native_br['smoke_report']),nr,nbuild,native_data)
 if any(x.get('error') or x.get('integrity_error') for x in (native_br,nprep,nbuild,smoke)):raise ValueError('Native historical stage carries failure')
 if native_data.get('status')!='STAGE4R_CPU_REFERENCE_PREPARED' or native_data['weights']['status']!='CPU_EFFECTIVE_WEIGHT_VERIFIED':raise ValueError('Native independent reference proof missing')
 if smoke['frozen_native']!=original['native_frozen']:raise ValueError('FP16 historical repeat anchor changed')
 from run_stage4v import validate_native
 for row in native_br['runs']:
  if row['scheme']=='FP16':
   shadow=dict(row,frozen_native=smoke['frozen_native'])
   if validate_native(shadow,dict(device=a.device,evidence=original),True)!=row['timing']:raise ValueError('historical FP16 sample/validation changed')
 case=read(verify(ev['baseline']['case']))
 if case['native_reference']!=nprep['reference']:raise ValueError('W8 and Native do not descend from the same effective FP16 weights/X')
 resolved=resolve_operator(build)
 return dict(status='BNZ_AND_NATIVE_HISTORIES_REVERIFIED',bnz=dict(reports=records,build=build,evidence=ev,packing=br['packing'],t1=br['t1'],full_anchor=ev['anchor'],operator=resolved),
  native=dict(prepare=native_br['prepare_report'],build_report=native_br['build_report'],smoke_report=native_br['smoke_report'],benchmark=original['stage4r_benchmark'],build=nbuild['native'],reference=nprep['reference'],data=native_data,policy=rr.POLICY,historical_anchor=smoke['frozen_native']),
  source_relation='Same frozen X and RN16 Group128 effective weights as provenance; different activation and Full-K weight quantization',
  historical_numbers_not_paired=dict(bnz=br['comparison']['statistics']['candidate'],native_fp16_median_ms=.78698,native_fp16_p95_ms=.79286))


def unchanged(e,cann):
 verify_tree(e);nz.verify_bound(e['bnz']['evidence']);nz.verify_build(e['bnz']['build']);nz.verify_prepack(e['bnz']['packing'],e['bnz']['evidence'])
 rr.verify_binding(e['native']['build']['header_binding'],cann)
 if resolve_operator(e['bnz']['build'])!=e['bnz']['operator']:raise ValueError('normal operator identity changed')
