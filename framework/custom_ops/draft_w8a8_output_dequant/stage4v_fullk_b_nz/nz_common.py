import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent
sys.path.insert(0,str(BASE/'stage4v_fullk_scalar_guard_ab'))
import sg_common as sg
import run_guard_ab as arena_run
import sg_protocol as arena_protocol
c=sg.c;v=sg.v;p=v.p;q=v.q;common=v.common
identity,verify,read,write,fresh,verify_tree=sg.identity,sg.verify,sg.read,sg.write,sg.fresh,sg.verify_tree
from nz_transform import NAMES,verify_audit
from nz_layout import verify_pack
OUTPUTS=sg.OUTPUTS
__all__=['NAMES','p','q','common','OUTPUTS']


def sources():return [identity(x) for x in sorted(HERE.rglob('*')) if x.is_file() and '__pycache__' not in x.parts and '.pytest_cache' not in x.parts]


def bind(a,root,r):
 original=sg.bind(a,root,r);bench_path=a.arena_root/'benchmark/report.json';br=read(bench_path);verify_tree(br)
 current=dict(sources=sg.sources(),evidence=original,device=a.device,cann_root=str(a.cann_root),machine=r['machine'])
 states={};records={}
 for stage,key in [('source','source_report'),('build','build_report'),('host','host_report'),('tile0','tile_report'),('full','full_report'),('fault','fault_report')]:
  path=verify(br[key]);states[stage]=arena_run.prior(path,stage,current);records[stage]=br[key]
  if a.arena_root not in path.parents:raise ValueError('baseline stage outside selected root')
 if br.get('status')!='GUARD_AB_PAIRED_MEASURED' or br.get('npu')!='COMPLETED' or br.get('benchmark')!='COMPLETED' or br.get('error') or br.get('integrity_error'):raise ValueError('Arena benchmark incomplete')
 if any(br[k]!=current[k] for k in ('sources','evidence','device','cann_root')):raise ValueError('Arena benchmark identity differs')
 if br!=read(a.arena_root/'benchmark/benchmark-report.json') or br['comparison']!=read(a.arena_root/'benchmark/comparison.json'):raise ValueError('Arena benchmark copies differ')
 if any(br['machine'][k]!=r['machine'][k] for k in ('os','version_files')):raise ValueError('Arena benchmark machine differs')
 for stage in ('build','host','tile0','full','fault'):
  if states[stage]['source_report']!=records['source']:raise ValueError('baseline Source chain differs')
 for stage in ('host','tile0','full','fault'):
  if states[stage]['build_report']!=records['build']:raise ValueError('baseline Build chain differs')
 for stage in ('tile0','full','fault'):
  if states[stage]['host_report']!=records['host']:raise ValueError('baseline Host chain differs')
 if states['fault']['full_report']!=records['full'] or states['fault']['tile_report']!=records['tile0'] or states['full']['tile_report']!=records['tile0']:raise ValueError('baseline safety chain differs')
 if len(br['prechecks'])!=2:raise ValueError('baseline fresh prechecks absent')
 for row in br['prechecks']+br['runs']:
  timed='round' in row
  if arena_protocol.validate(row,original,a.device,timed=timed)!=row['validation']:raise ValueError('baseline benchmark output/safety changed')
 if arena_protocol.paired(br['runs'])!=br['comparison']:raise ValueError('Arena paired summary differs')
 build=states['build'];sg.verify_build(build);selected=build['variants']['candidate']['build']
 for row in br['prechecks']+br['runs']:
  wanted=original['baseline_runner'] if row['variant']=='baseline' else selected['runner']
  if row['runner']!=wanted:raise ValueError('baseline runner identity differs')
 base=dict(original['baseline'],normal=selected)
 sdk=[]
 for item in read(HERE/'sdk-review.json')['sources']:
  f=identity(a.cann_root/item['relative_path'])
  if a.cann_root not in Path(f['path']).parents or f['sha256']!=item['sha256'] or f['bytes']!=item['bytes']:raise ValueError('B NZ SDK identity differs')
  sdk.append(dict(file=f,lines=item['lines']))
 return dict(arena_evidence=original,arena_build=records['build'],arena_reports=records,arena_benchmark=identity(bench_path),baseline=base,baseline_runner=selected['runner'],anchor=original['anchor'],
  formal_core_policy=base['policy'],sdk=sdk,preflight=base['original_evidence']['preflight'],baseline_stats=br['comparison']['statistics']['candidate'])


def env(evidence,cann,opp=None):
 p0=dict(evidence['baseline'])
 if opp is not None:p0['normal']=dict(p0['normal'],opp=str(opp))
 return c.environment(p0,cann)


def verify_bound(e):
 verify_tree(e);sg.verify_bound(e['arena_evidence']);sg.verify_build(read(verify(e['arena_build'])))


def verify_prepack(record,e):
 verify_tree(record)
 source=verify(read(verify(e['baseline']['case']))['files']['b_i8.bin'])
 if record['source_b']!=identity(source):raise ValueError('different source B')
 verify_pack(source,verify(record['packed_b']),record['layout'])


def verify_build(b):
 from nz_build import installed
 for kind in ('candidate','fault'):
  x=b['variants'][kind];verify_audit(x['audit'],verify);item=x['build'];verify_tree(item)
  if item.get('status')!='GUARD_AB_PRIVATE_BUILT' or any(z['returncode']!=0 for z in item['commands'].values()):raise ValueError('new private build incomplete')
  if installed(Path(item['opp']),kind)!=item['installed'] or item['linked_libraries']['LIB_custom']!=item['installed']['api']:raise ValueError('candidate installed identity differs')
  want={Path(f['path']).parent.name+'/'+Path(f['path']).name:verify(f).read_bytes() for f in x['audit']['files'] if Path(f['path']).suffix in ('.cpp','.h')}
  want['op_host/t1_contract.h']=v.contract_header(b['t1']).encode()
  got={Path(f['path']).parent.name+'/'+Path(f['path']).name:verify(f).read_bytes() for f in item['generated_sources']}
  if want!=got:raise ValueError('compiled source/T1 differs')
  for d in item['dependencies'].values():c.verify_runtime(d)
