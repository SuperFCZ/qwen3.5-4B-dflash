"""Unique Formal CO1 Host experiment; shared history is read-only."""
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'stage4v_external_co1_review'))
import co1_contract as review
from co1_audit import sdk_binding,prior_evidence
from vdeq_sdk import bind as vdeq_bind,schema_names
from raw_probe_runtime_identity import collect as runtime_binding,history_records,verify_runtime
from run_stage2b import command
__all__=['command','runtime_binding','history_records','verify_runtime']
v=review.prev.raw.v
identity,verify,read,write,fresh=review.identity,review.verify,review.read,review.write,review.fresh
env=review.prev.raw.common.env
NAME='DFlashExternalCo1HostProbe';SNAKE='d_flash_external_co1_host_probe'


def sources():return [identity(p) for p in sorted(HERE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts]


def bind(args):
 old=read(args.review_report);sdk=sdk_binding(args.cann_root)
 if old['status']!='BLOCKED' or not old['audit_completed'] or old.get('error') or old.get('integrity_error') or old['sources']!=review.sources() or old['sdk']!=sdk:raise ValueError('accepted CO1 source review identity required')
 prior=old['prior'];tiling=read(verify(prior['reports']['tiling']))
 root=Path(prior['reports']['review']['path']).parent.parent
 if prior_evidence(root,sdk)!=prior:raise ValueError('prior VDEQ evidence changed')
 raw_host=read(verify(tiling['evidence']['raw_host']))
 if raw_host['status']!='FORMAL_HOST_FEASIBLE' or raw_host['device']!=args.device:raise ValueError('reference device/Formal policy missing')
 r=dict(review=identity(args.review_report),sdk=sdk,codec_and_schema_sdk=vdeq_bind(args.cann_root),
  reviewed_platform=raw_host['evidence']['reviewed_platform'],preflight=raw_host['evidence']['preflight'],frozen=raw_host['evidence']['frozen'],
  policy=raw_host['evidence']['policy'],reference_raw_host=tiling['evidence']['raw_host'],
  schema_identity=read(root/'review/report.json')['serializer']['schema'],
  schema=read(verify(read(root/'review/report.json')['serializer']['schema'])),
  original_formal_unknown=prior['formal_missing'])
 if len(r['original_formal_unknown'])!=23:raise ValueError('unexpected historical formal coverage')
 r['field_names']=schema_names(r['codec_and_schema_sdk'])
 from co1_serialization import bind_layout
 r['serialization_binding']=bind_layout(r)
 return r


def capture_header(names):
 if len(names)!=50 or len(set(names))!=50:raise ValueError('50 SDK fields required')
 lines=['#pragma once','template<class T> void Co1Fields(std::ostream& out,T& p){out<<"{";']
 for i,k in enumerate(names):
  if not k.isidentifier():raise ValueError('bad field identifier')
  lines.append('out<<"'+(',' if i else '')+'\\"'+k+'\\":"<<p.get_'+k+'();')
 return '\n'.join(lines+['out<<"}";}', ''])

BATCH=('batchM','batchN','singleBatchM','singleBatchN','BatchNum')
AUX_ZERO=('depthAL1CacheUB','depthBL1CacheUB','mxTypePara')+tuple(x for prefix,ends in [('A',('B','S','N','G','D')),('B',('B','S','N','G','D')),('C',('B','S1','N','G','S2'))] for x in [prefix+'LayoutInfo'+e for e in ends])


def generated_header(names):
 from co1_serialization import SCHEMA_SHA
 text=capture_header(names)
 text+='static const char* Co1SchemaSha="'+SCHEMA_SHA+'";\n'
 text+='template<class T> void Co1MarkerFields(T& p){'+''.join('p.set_'+n+'('+str(0x13570000+i)+');' for i,n in enumerate(names))+'}\n'
 text+='template<class T> bool Co1Aux(T& p){return '+ ' && '.join([f'p.get_{n}()==0' for n in AUX_ZERO]+[f'(p.get_{n}()==0||p.get_{n}()==1)' for n in BATCH])+';}\n'
 if set(names)!=set(review.prev.raw.v.p.FIELDS)|set(BATCH)|set(AUX_ZERO):raise ValueError('unaccounted schema field')
 return text


def validate_callback(row,op_binary,proof,execution):
 from co1_serialization import validate_serialization
 names=proof['field_names'];p=row.get('fields')
 if not isinstance(p,dict) or set(p)!=set(names) or any(type(x)is not int for x in p.values()):raise ValueError('50 actual fields required')
 validate_serialization(row,op_binary,proof)
 if row['status']!='FORMAL_CALLBACK_ACCEPTED' or row['context']!='REGISTERED_FORMAL_TILING_CALLBACK' or type(row['get_tiling_rc'])is not int or row['get_tiling_rc']!=0 or row['error']:raise ValueError('callback failed')
 setters=('SetAType','SetBType','SetCType','SetBiasType','SetBias','SetShape','SetOrgShape','SetFixSplit','SetBufferSpace')
 if set(row['setters'])!=set(setters) or any(x is not None and (type(x)is not int or x<0) for x in row['setters'].values()):raise ValueError('setup contract differs')
 if row['core']!=proof['policy'] or row['core']['arch']!=2002 or type(row['block_dim'])is not int or row['block_dim']!=1:raise ValueError('formal Core policy differs from accepted device')
 result=review.conditional_budget(p,row['capacity'],row['system_workspace_bytes'])
 if result['status']!='CONDITIONAL_ARITHMETIC_CHECKS_PASS' or row['capacity']['L0B']!=65536:raise ValueError('CO1 resource rejection')
 if any(p[k]!=0 for k in AUX_ZERO) or any(p[k] not in (0,1) for k in BATCH):raise ValueError('unaccounted auxiliary layout/cache/batch')
 expected_checks={'shape','scope','steps','depths','L1','db','L0AB','external_L0C','UB','scratch','auxiliary_consumers','Workspace'}
 if set(row['checks'])!=expected_checks or any(v is not True for v in row['checks'].values()):raise ValueError('Host resource checks incomplete')
 if row['ub']!=dict(scratch=202240,echo=16384,scale=2048,fp16=8192,guards=2048,reserve=8192,total=239104,effective_shared=0):raise ValueError('UB plan changed')
 if row['external_l0c']!=dict(bytes=16384,start=0,alignment=1024,effective_shared=0,allocation='PLANNED_NOT_EXECUTED'):raise ValueError('external CO1 plan changed')
 system=row['system_workspace_bytes'];arena=(system+511)//512*512;total=system+arena+1024
 if type(execution['workspace_bytes'])is not int or row['arena_bytes']!=arena or row['requested_workspace_bytes']!=total or not total<=execution['workspace_bytes']<=512*1024*1024:raise ValueError('actual ACLNN Workspace request not admitted')
 if row['npu']!='NOT_RUN' or row['kernel']!='COMPILE_ONLY':raise ValueError('not a Host-only callback')
 return result
