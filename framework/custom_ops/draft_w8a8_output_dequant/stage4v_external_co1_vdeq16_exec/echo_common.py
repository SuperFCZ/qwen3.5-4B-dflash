"""Echo first. VDEQ16 implementation/execution awaits a real Echo PASS."""
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'stage4v_external_co1_host'))
import co1_host_common as old
import run_formal_host as old_run
from co1_serialization import verify_serialization_record as old_serial_verify
from vdeq_evidence import load as load_raw
from raw_probe_runtime_identity import collect as runtime_binding,history_records,verify_runtime
__all__=['runtime_binding','history_records','verify_runtime']
v=old.v;identity,verify,read,write,fresh,env=old.identity,old.verify,old.read,old.write,old.fresh,old.env
NAME='DFlashExternalCo1EchoProbe';SNAKE='d_flash_external_co1_echo_probe'


def sources():return [identity(p) for p in sorted(HERE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts]


def bind(a):
 h=read(a.co1_root/'host/formal-host-report.json')
 if h['status']!='FORMAL_CO1_HOST_FEASIBLE' or h['phase']!='complete' or h['sources']!=old.sources() or h['device']!=a.device or h['npu']!='NOT_RUN':raise ValueError('accepted complete CO1 Formal Host required')
 args=SimpleNamespace(review_report=verify(h['evidence']['review']),cann_root=a.cann_root,device=a.device)
 if old.bind(args)!=h['evidence']:raise ValueError('CO1 source/SDK evidence changed')
 old_run.prior(verify(h['source_report']),'SOURCE_REVIEWED',h);build=old_run.prior(verify(h['build_report']),'PRIVATE_OPP_COMPILED',h)
 if build['artifacts']!=h['artifacts'] or build['source_report']!=h['source_report']:raise ValueError('CO1 build binding')
 old_run.artifact_check(h['artifacts'],h['evidence'])
 for row in h['callbacks']:
  old_serial_verify(row['serialization'],h['evidence'])
  old.validate_callback(row['data'],verify(row['op_serialization']).read_bytes(),h['evidence'],read(verify(h['execution'])))
 if not h['callbacks'] or h['field_count']!=50 or h['workspace_bytes']!=4195328 or h['formal_fields']!=h['callbacks'][0]['data']['fields']:raise ValueError('CO1 accepted scope')
 raw=load_raw(a.frozen_root,a.raw_root,a.cann_root)
 if raw['raw']['device']!=a.device or raw['raw_host']['evidence']['policy']!=h['evidence']['policy']:raise ValueError('Raw device/policy changed')
 p=dict(h['evidence']);p.update(accepted=h['callbacks'][0]['data'],accepted_workspace=h['workspace_bytes'],accepted_host=identity(a.co1_root/'host/formal-host-report.json'),raw_evidence=raw['identities'])
 from echo_serialization import bind_layout
 p['serialization_binding']=bind_layout(p)
 p['extra_sdk']=[]
 for rel,sha in [('impl/basic_api/kernel_tensor_impl.h','1e5d6669d91e5687db76fcace1cb1b2a9af0dfde92c4b5ee65dd2b23b99b9e0a'),('impl/basic_api/kernel_tbuf_impl.h','e838500d23ab3c5269fbcc41c64de1fb674782657727a4a0dbb4d1bf8706abf0')]:
  f=identity(a.cann_root/'aarch64-linux/asc'/rel)
  if f['sha256']!=sha or a.cann_root not in Path(f['path']).parents:raise ValueError('physical address SDK mismatch')
  p['extra_sdk'].append(dict(file=f,lines=[[646,668],[864,866]] if 'tensor_impl' in rel else [[20,65],[68,81]],scope='LOCAL_HASH_BOUND_TO_CURRENT_INSTALL_AT_SOURCE_GATE'))
 return p,raw


def generated_header(names,accepted=None):
 s=old.generated_header(names)
 if accepted is None:raise ValueError('actual accepted Formal contract required')
 p=accepted['fields'];cap=accepted['capacity'];core=accepted['core']
 s+='static const unsigned long long EchoExpectedWorkspace='+str(accepted['requested_workspace_bytes'])+'ULL;\n'
 s+='template<class T,class C> bool EchoAccepted(T& p,const C& c,unsigned selected,unsigned long long ub,unsigned long long l1,unsigned long long l0a,unsigned long long l0b,unsigned long long l0c,unsigned long long system){return '
 clauses=[f'p.get_{k}()=={p[k]}' for k in names]
 clauses += [f'{var}=={cap[k]}ULL' for var,k in [('ub','UB'),('l1','L1'),('l0a','L0A'),('l0b','L0B'),('l0c','L0C')]]
 clauses += [f'c.{k}=={core[k]}' for k in ('aic','aiv','all','arch')]+[f'selected=={core["selected"]}',f'system=={accepted["system_workspace_bytes"]}ULL']
 s+=' && '.join(clauses)+';}\n'
 s+='template<class T> bool EchoWireMatches(T& p,const unsigned char* b){int32_t x=0;'
 for i,k in enumerate(names):s+='std::memcpy(&x,b+'+str(48+4*i)+',4);if(x!=p.get_'+k+'())return false;'
 s+='return true;}\n'
 return s


def validate_callback(row,op,proof,execution):
 from echo_serialization import validate_serialization
 validate_serialization(row,op,proof)
 if row['fields']!=proof['accepted']['fields'] or any(row[k]!=proof['accepted'][k] for k in ('capacity','core','block_dim','system_workspace_bytes','arena_bytes','requested_workspace_bytes','ub','external_l0c')):raise ValueError('new formal resources drifted')
 if row['kernel']!='ECHO_ONLY' or row['status']!='FORMAL_CALLBACK_ACCEPTED' or row['get_tiling_rc']!=0 or row['error'] or row['npu']!='NOT_RUN' or row['checks'].get('admitted_formal_contract') is not True or not all(row['checks'].values()):raise ValueError('new callback rejected')
 # Keep inherited setup, exact UB and resource gates, with only the provenance label adapted.
 normalized=dict(row,kernel='COMPILE_ONLY',checks={k:x for k,x in row['checks'].items() if k!='admitted_formal_contract'})
 # Old source binding is not used here: both generated/compiled new sources are bound above.
 from co1_contract import conditional_budget
 result=conditional_budget(row['fields'],row['capacity'],row['system_workspace_bytes'])
 if result['status']!='CONDITIONAL_ARITHMETIC_CHECKS_PASS' or execution['workspace_bytes']!=proof['accepted_workspace']:raise ValueError('budget/workspace mismatch')
 if normalized['setters']!=proof['accepted']['setters']:raise ValueError('setup changed')
 return result


def prepare(folder,raw):
 folder.mkdir();a=verify(raw['raw']['data']['a_i8.bin']).read_bytes();b=verify(raw['raw']['data']['b_i8.bin']).read_bytes()
 words=raw['original']['words'].astype('<u8').tobytes();expected=verify(raw['raw']['outputs'][0]).read_bytes()
 result={}
 for n,data in [('a.bin',a),('b.bin',b),('scale.bin',words),('expected.bin',expected)]:
  f=folder/n;f.write_bytes(data);result[n]=identity(f)
 return result


def compare_echo(expected,outputs):
 return old.review.prev.raw.compare(np.frombuffer(expected,'<i4').reshape(16,256).astype(np.int64),outputs)
