#!/usr/bin/env python3
import argparse,os,sys,shutil,struct
from pathlib import Path
import echo_common as c
import echo_build as build
from echo_dependencies import verify_binding
from echo_execution_contract import validate_execution
from echo_serialization import bind_capture,compare_callbacks,verify_serialization_record
STATUS=dict(source='SOURCE_REVIEWED',cpu='CPU_TESTS_PASSED',build='PRIVATE_OPP_BUILT',host='FORMAL_HOST_FEASIBLE',echo='ECHO_NPU_MATCH')


def artifacts(item,proof):
 c.v.verify_tree(item);verify_binding(item['core_sdk_binding']);c.verify_runtime(item['core_sdk_binding'])
 if item['status']!='PRIVATE_OPP_BUILT' or item['kernel_role']!='ECHO_ONLY' or build.installed(Path(item['opp']))!=item['installed']:raise ValueError('wrong private Echo build')
 want={sub+'/'+p.name:p.read_bytes() for sub in ('op_host','op_kernel') for p in (c.HERE/sub).iterdir() if p.is_file()}
 want['op_host/co1_fields.h']=c.generated_header(proof['field_names'],proof['accepted']).encode()
 got={Path(x['path']).parent.name+'/'+Path(x['path']).name:c.verify(x).read_bytes() for x in item['generated_sources']}
 if want!=got or (Path(item['opp']).parent/'operator.json').read_bytes()!=(c.HERE/'operator.json').read_bytes() or item['linked_libraries']['LIB_custom']!=item['installed']['api']:raise ValueError('compiled source/ABI/OPP changed')


def prior(path,status,r):
 p=c.read(path)
 if p['status']!=status or p['phase']!='complete' or any(p[k]!=r[k] for k in ('sources','evidence','device')):raise ValueError('stage/source/evidence gate differs')
 if status!='ECHO_NPU_MATCH' and p.get('npu')!='NOT_RUN':raise ValueError('pre-execution gate must be NPU NOT_RUN')
 return p


def verify_host(h):
 files={Path(f['path']).name:c.verify(f) for f in h['files']}
 ex=c.read(files['execution.json'])
 validate_execution(ex,h['device'],h['evidence']['accepted_workspace'],host=True)
 compare_callbacks(h['callbacks'])
 for row in h['callbacks']:
  if c.read(c.verify(row['json']))!=row['data']:raise ValueError('Host callback JSON drift')
  c.validate_callback(row['data'],c.verify(row['op_serialization']).read_bytes(),h['evidence'],ex)
  verify_serialization_record(row['serialization'],h['evidence'])


def witness(raw):
 if len(raw)!=256:raise ValueError('witness size')
 w=struct.unpack('<32Q',raw)
 ok=w[:8]==(0x434f314543484f31,1,0,0,16384,230912,1,0) and all(x<8 for x in w[8:16]) and all(x>0 for x in w[16:24]) and not any(w[24:])
 return dict(status='VALID' if ok else 'FAILED',failure_code=w[2],diagnostic_scope='RECORDED' if w[0]==0x434f314543484f31 else 'BEFORE_WITNESS_OR_DAMAGED_WITNESS',words=list(w),l0c_guard='RANGE_ONLY_NO_CANARIES',event_ids=dict(zip(('M_V','V_M','V_S','MTE2_V','V_MTE3','MTE3_MTE2','MTE2_S','S_MTE3'),w[8:16])))


def execute(a,root,r,host):
 item=r['artifacts'];artifacts(item,r['evidence']);d=root/'run';d.mkdir()
 for name,f in r['data'].items():shutil.copyfile(c.verify(f),d/name)
 r['run_inputs']=[c.identity(d/name) for name in r['data']]
 try:build.command([c.verify(item['runner']),a.device,d,'host' if host else 'echo'],d/'runner.log',c.env(a.cann_root,Path(item['opp'])),r,'query' if host else 'echo',1800)
 finally:
  r['files']=[c.identity(p) for p in sorted(d.iterdir()) if p.is_file()]
  if (d/'execution.json').is_file():r['npu']=c.read(d/'execution.json').get('npu','UNRESOLVED')
  outputs=[d/f'actual-{i}.bin' for i in range(2) if (d/f'actual-{i}.bin').is_file()]
  if outputs:
   r['outputs']=[c.identity(p) for p in outputs];r['comparison']=c.compare_echo(c.verify(r['data']['expected.bin']).read_bytes(),[p.read_bytes() for p in outputs]);c.write(root/'echo-comparison.json',r['comparison'])
  r['witnesses']=[dict(file=c.identity(p),analysis=witness(p.read_bytes())) for p in sorted(d.glob('witness-*.bin'))]
 ex=c.read(d/'execution.json')
 validate_execution(ex,a.device,r['evidence']['accepted_workspace'],host=host)
 rows=[]
 for p in sorted(d.glob('callback-*.json')):
  row=dict(json=c.identity(p),data=c.read(p),cube=c.identity(p.with_name(p.stem+'-cube.bin')),op_serialization=c.identity(p.with_name(p.stem+'-op.bin')),cube_payload=c.identity(p.with_name(p.stem+'-cube-payload.bin')))
  resources,serial=bind_capture(row,r['evidence'],ex);verify_serialization_record(serial,r['evidence']);row.update(resources=resources,serialization=serial);rows.append(row)
 compare_callbacks(rows);r['callbacks']=rows
 if not host:
  if r['comparison']['status']!='MATCH' or len(r['witnesses'])!=2 or any(x['analysis']['status']!='VALID' for x in r['witnesses']):raise ValueError('Echo numerical/witness failure')


def main(a):
 root=c.fresh(a.output_dir,[a.cann_root,a.co1_root,a.raw_root,a.frozen_root]);r=dict(status='NOT_RUN',phase='evidence',stage=a.stage,sources=c.sources(),device=a.device,npu='NOT_RUN',vdeq16='NOT_IMPLEMENTED_PENDING_ECHO_PASS',benchmark='PROHIBITED',profile='NOT_RUN')
 try:
  proof,raw=c.bind(a);r['evidence']=proof
  if a.stage=='vdeq':
   if a.echo_report:prior(a.echo_report,'ECHO_NPU_MATCH',r)
   raise ValueError('VDEQ16 development/execution gate closed: independent device Echo PASS and a subsequent implementation review are required')
  r['data']=c.prepare(root/'data',raw)
  if a.stage!='source':
   prior(a.source_report,STATUS['source'],r);r['source_report']=c.identity(a.source_report)
  if a.stage=='cpu':
   r['phase']='cpu_regression';build.command([sys.executable,'-B','-m','pytest','-q','-p','no:cacheprovider',c.HERE/'test'],root/'cpu.log',dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),r,'cpu',300)
  if a.stage in ('build','host','echo'):
   cp=prior(a.cpu_report,STATUS['cpu'],r)
   if cp['source_report']!=r['source_report']:raise ValueError('CPU/source mismatch')
   r['cpu_report']=c.identity(a.cpu_report)
   if a.stage=='build':
    r['phase']='build';p=root/'private';p.mkdir();r['artifacts']=dict(evidence=proof,npu='NOT_RUN');build.build(a,p,r['artifacts'],c.HERE)
   else:
    b=prior(a.build_report,STATUS['build'],r)
    if any(b[k]!=r[k] for k in ('source_report','cpu_report')):raise ValueError('Build gate differs')
    r['build_report']=c.identity(a.build_report);r['artifacts']=b['artifacts']
    if a.stage=='echo':
     h=prior(a.host_report,STATUS['host'],r)
     if h['artifacts']!=r['artifacts'] or h['build_report']!=r['build_report']:raise ValueError('Host/build differs')
     verify_host(h);r['host_report']=c.identity(a.host_report)
    r['phase']=a.stage;execute(a,root,r,a.stage=='host')
  r.update(status=STATUS[a.stage],phase='complete')
 except Exception as e:r.update(status='BLOCKED',error=f'{type(e).__name__}: {e}')
 finally:
  try:
   c.v.verify_tree(r)
   if r['sources']!=c.sources() or 'evidence' in r and c.bind(a)[0]!=r['evidence']:raise ValueError('evidence changed')
   if r.get('artifacts',{}).get('status')=='PRIVATE_OPP_BUILT':artifacts(r['artifacts'],r['evidence'])
  except Exception as e:r.update(status='BLOCKED',integrity_error=str(e))
  c.write(root/'report.json',r)
 print(r['status']+': '+str(root/'report.json'));return int(r['status']=='BLOCKED')


def parse(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=list(STATUS)+['vdeq'])
 for flag in ('output-dir','co1-root','raw-root','frozen-root'):p.add_argument('--'+flag,type=Path,required=True)
 for flag in ('source-report','cpu-report','build-report','host-report','echo-report'):p.add_argument('--'+flag,type=Path)
 p.add_argument('--device',type=int,default=0);p.add_argument('--cann-root',type=Path,default=Path(os.environ.get('CANN_ROOT','/usr/local/Ascend/cann-9.0.0')))
 a=p.parse_args(argv)
 needed={'source':[],'cpu':['source_report'],'build':['source_report','cpu_report'],'host':['source_report','cpu_report','build_report'],'echo':['source_report','cpu_report','build_report','host_report'],'vdeq':[]}[a.stage]
 if a.device<0 or any(getattr(a,k) is None for k in needed):p.error('missing predecessor/device')
 for k,v in vars(a).items():
  if isinstance(v,Path):setattr(a,k,v.resolve())
 return a

if __name__=='__main__':raise SystemExit(main(parse()))
