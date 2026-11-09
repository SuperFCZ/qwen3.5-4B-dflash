#!/usr/bin/env python3
"""Independent CO1 registered Host callback: source -> private build -> Host query only."""
import argparse,os
from pathlib import Path
import co1_host_common as c
import co1_build as builder
from co1_host_dependencies import verify_binding
from raw_probe_runtime_identity import verify_runtime
from co1_serialization import bind_capture,compare_callbacks,verify_serialization_record

STATUS={'source':'SOURCE_REVIEWED','build':'PRIVATE_OPP_COMPILED','host':'FORMAL_CO1_HOST_FEASIBLE'}


def artifact_check(item,proof):
 c.v.verify_tree(item);verify_binding(item['core_sdk_binding']);verify_runtime(item['core_sdk_binding'])
 recorded={x['file']['path']:x['file'] for x in item['core_sdk_binding']['dependencies']}
 for entry in proof['serialization_binding']['sdk_sources']:
  if recorded.get(entry['file']['path'])!=entry['file']:raise ValueError('compiled serializer SDK identity differs')
 if item['status']!='PRIVATE_OPP_COMPILED' or item['kernel_role']!='COMPILE_ONLY' or item['npu']!='NOT_RUN' or builder.installed(Path(item['opp']))!=item['installed']:raise ValueError('private Host-only artifacts changed')
 want={sub+'/'+p.name:p.read_bytes() for sub in ('op_host','op_kernel') for p in (c.HERE/sub).iterdir() if p.is_file()}
 want['op_host/co1_fields.h']=c.generated_header(proof['field_names']).encode()
 got={Path(x['path']).parent.name+'/'+Path(x['path']).name:c.verify(x).read_bytes() for x in item['generated_sources']}
 if want!=got or (Path(item['opp']).parent/'operator.json').read_bytes()!=(c.HERE/'operator.json').read_bytes():raise ValueError('generated Host/placeholder/ABI differs')
 if item['linked_libraries']['LIB_custom']!=item['installed']['api']:raise ValueError('Runner links another Op')


def prior(path,status,r):
 old=c.read(path)
 if old['status']!=status or old['phase']!='complete' or old['sources']!=r['sources'] or old['evidence']!=r['evidence'] or old['device']!=r['device'] or old['npu']!='NOT_RUN':raise ValueError('predecessor status/identity mismatch')
 return old


def host_query(a,root,r):
 artifact_check(r['artifacts'],r['evidence']);folder=root/'callback';folder.mkdir()
 env=c.env(a.cann_root,Path(r['artifacts']['opp']))
 try:c.command([c.verify(r['artifacts']['runner']),a.device,folder],folder/'runner.log',env,r,'formal_host',600)
 finally:
  r['query_files']=[c.identity(p) for p in sorted(folder.iterdir()) if p.is_file()]
  rows=[]
  for p in sorted(folder.glob('callback-*.json')):
   row=c.read(p);rows.append(dict(json=c.identity(p),data=row,cube=c.identity(p.with_name(p.stem+'-cube.bin')) if p.with_name(p.stem+'-cube.bin').is_file() else None))
  r['callbacks']=rows
  for row in rows:
   path=Path(row['json']['path']);op=path.with_name(path.stem+'-op.bin')
   row['op_serialization']=c.identity(op) if op.is_file() else None
   payload=path.with_name(path.stem+'-cube-payload.bin');row['cube_payload']=c.identity(payload) if payload.is_file() else None
 if not rows:raise ValueError('actual registered callback did not capture evidence')
 ex=c.read(folder/'execution.json')
 if ex.get('status')!='HOST_QUERY_RETURNED_ZERO' or ex.get('returncode')!=0 or ex.get('device')!=a.device or ex.get('soc')!='Ascend310P3' or ex.get('executor_returned') is not True or ex.get('execute_api_calls')!=0 or ex.get('npu')!='NOT_RUN' or ex.get('input_payload')!='NOT_UPLOADED_NOT_READ':raise ValueError('not a valid Host-only receiver result')
 compare_callbacks(rows)
 checks=[]
 for row in rows:
  if row['cube'] is None or row['op_serialization'] is None or row['op_serialization']['bytes']<200:raise ValueError('missing actual cube/Op serialization')
  resources,serialization=bind_capture(row,r['evidence'],ex);checks.append(resources)
  serialization['callback_binary']=r['artifacts']['installed']['host']
  serialization['compiled_callback_sources']=r['artifacts']['generated_sources']
  serialization['sdk_runtime_libraries']=r['artifacts']['core_sdk_binding']['sdk_libraries']
  verify_serialization_record(serialization,r['evidence']);row['serialization']=serialization
 r['execution']=c.identity(folder/'execution.json');r['resource_checks']=checks
 r['workspace_bytes']=ex['workspace_bytes'];r['workspace_accounting']=dict(callback_requested=rows[0]['data']['requested_workspace_bytes'],aclnn_returned=ex['workspace_bytes'],framework_extra=ex['workspace_bytes']-rows[0]['data']['requested_workspace_bytes'],allocation='NOT_RUN')
 r['formal_fields']=rows[0]['data']['fields']
 r['formal_cube_binary']=rows[0]['cube_payload'];r['serialization']=rows[0]['serialization'];r['field_count']=50
 r.update({k:r['serialization'][k] for k in ('cube_payload_offset','cube_payload_bytes','cube_payload_source','cube_payload_sha256','op_serialization_sha256','schema_sha256')})
 r['events']=dict(status='DESIGN_RECORDED_NOT_RUN',allocation='AllocEventID/ReleaseEventID, actual IDs checked at future device runtime',dependencies=c.review.dependencies())
 r['historical_formal_fields']=dict(fp16='23 UNKNOWN',raw='23 UNKNOWN',missing=r['evidence']['original_formal_unknown'],not_filled_from_co1=True)


def main(a):
 root=c.fresh(a.output_dir,[a.cann_root,a.review_report.parent]);r=dict(stage=a.stage,status='NOT_RUN',phase='source',sources=c.sources(),device=a.device,npu='NOT_RUN',benchmark='NOT_RUN',events='NOT_RUN',kernel_role='COMPILE_ONLY')
 try:
  r['evidence']=c.bind(a);r['source_review']='SOURCE_REVIEWED'
  if a.stage!='source':
   prior(a.source_report,'SOURCE_REVIEWED',r);r['source_report']=c.identity(a.source_report)
   if a.stage=='build':
    r['phase']='private_build';product=root/'private';product.mkdir();item=dict(evidence=r['evidence'],npu='NOT_RUN');r['artifacts']=item
    builder.build(a,product,item,c.HERE);artifact_check(item,r['evidence'])
   else:
    old=prior(a.build_report,'PRIVATE_OPP_COMPILED',r)
    if old['source_report']!=r['source_report']:raise ValueError('Build/source binding differs')
    r['build_report']=c.identity(a.build_report);r['artifacts']=old['artifacts'];r['phase']='formal_callback';host_query(a,root,r)
  r.update(status=STATUS[a.stage],phase='complete')
 except Exception as e:r.update(status='BLOCKED',error=f'{type(e).__name__}: {e}')
 finally:
  try:
   c.v.verify_tree(r)
   for row in r.get('callbacks',[]):
    if 'serialization' in row:verify_serialization_record(row['serialization'],r['evidence'])
   if 'serialization' in r:
    for key in ('cube_payload_offset','cube_payload_bytes','cube_payload_source','cube_payload_sha256','op_serialization_sha256','schema_sha256'):
     if r[key]!=r['serialization'][key]:raise ValueError('top-level serialization identity differs')
   if r['sources']!=c.sources() or 'evidence' in r and c.bind(a)!=r['evidence']:raise ValueError('source/SDK/history drift')
   if r.get('artifacts',{}).get('status')=='PRIVATE_OPP_COMPILED':artifact_check(r['artifacts'],r['evidence'])
  except Exception as e:r.update(status='BLOCKED',integrity_error=str(e))
  c.write(root/'formal-host-report.json',r)
 print(r['status']+': '+str(root/'formal-host-report.json'));return int(r['status']=='BLOCKED')


def parse(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=STATUS)
 for flag in ('output-dir','review-report'):p.add_argument('--'+flag,type=Path,required=True)
 for flag in ('source-report','build-report'):p.add_argument('--'+flag,type=Path)
 p.add_argument('--device',type=int,default=0);p.add_argument('--cann-root',type=Path,default=Path(os.environ.get('CANN_ROOT','/usr/local/Ascend/cann-9.0.0')))
 a=p.parse_args(argv)
 if a.device<0 or a.stage!='source' and not a.source_report or a.stage=='host' and not a.build_report:p.error('device/predecessor report missing')
 if a.stage=='source' and (a.source_report or a.build_report) or a.stage=='build' and a.build_report:p.error('unexpected predecessor report')
 for k in ('output_dir','review_report','source_report','build_report','cann_root'):
  if getattr(a,k) is not None:setattr(a,k,getattr(a,k).resolve())
 return a

if __name__=='__main__':raise SystemExit(main(parse()))
