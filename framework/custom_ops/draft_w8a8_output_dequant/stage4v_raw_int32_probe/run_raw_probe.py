#!/usr/bin/env python3
"""Private Tile0 INT32 diagnostic; Host query -> artifact seal -> two untimed runs."""
import argparse,os,re,shutil
from pathlib import Path
import numpy as np
import raw_probe_common as c
from build_raw_probe import build,installed,command,env

STATUS={'host':'FORMAL_HOST_FEASIBLE','build':'BUILT_NPU_NOT_RUN','bounded':'RAW_INT32_NPU_MATCH'}

def plan(text,t1,policy):
 def one(pattern):
  rows=re.findall(pattern,text)
  if not rows or any(row!=rows[0] for row in rows):raise ValueError('missing/drifting Host evidence: '+pattern)
  return rows[0]
 if tuple(map(int,one(r'STAGE4RAW_CORE_POLICY aic=(\d+) aiv=(\d+) all=(\d+) selected=(\d+)')))!=tuple(policy[k] for k in ('aic','aiv','all','selected')) or one(r'STAGE4RAW_ARCH_PROBE arch=(\S+)')!='2002':raise ValueError('formal Core policy changed')
 for k,value in t1['fields'].items():
  if int(one(r'STAGE4RAW_T1 '+k+r'=(-?\d+)'))!=value:raise ValueError('formal Tiling changed '+k)
 if one(r'STAGE4RAW_T1_CONTRACT match=(\d+) effectiveSharedUb=0')!='1':raise ValueError('formal T1 rejected')
 vals=list(map(int,one(r'STAGE4RAW_SCHEDULE coreLimit=(\d+) blocks=(\d+) cores=(\d+) system=(\d+) arena=(\d+) stride=(\d+) user=(\d+) total=(\d+) mode=(\d+)')))
 system=t1['system_workspace_bytes'];arena=(system+511)//512*512;stride=arena+1024
 if vals!=[1,1,policy['selected'],system,arena,stride,stride,system+stride,0]:raise ValueError('workspace formula or launch scope changed')
 return dict(blocks=1,usedCoreNum=t1['fields']['usedCoreNum'],available_cores=policy['selected'],system=system,arena=arena,stride=stride,total=system+stride)

def verify_artifacts(item,proof):
 c.v.verify_tree(item)
 from raw_probe_dependencies import verify_binding
 verify_binding(item['core_sdk_binding'])
 if item['status']!='STAGE4RAW_PRIVATE_BUILT' or item['cann_compile']!='COMPLETED' or installed(Path(item['opp']))!=item['installed']:raise ValueError('private build changed/incomplete')
 want={sub+'/'+p.name:p.read_bytes() for sub in ('op_host','op_kernel') for p in (c.HERE/sub).iterdir() if p.is_file()}
 want['op_host/t1_contract.h']=c.contract_header(proof['t1']).encode()
 got={Path(f['path']).parent.name+'/'+Path(f['path']).name:c.verify(f).read_bytes() for f in item['generated_sources']}
 if got!=want or (Path(item['opp']).parent/'operator.json').read_bytes()!=(c.HERE/'operator.json').read_bytes():raise ValueError('compiled sources/ABI drift')
 if item['linked_libraries']['LIB_custom']!=item['installed']['api']:raise ValueError('Runner private API identity differs')

def verify_gate(report,args,proof,status):
 if report['status']!=status or report['phase']!='complete' or report['sources']!=c.sources() or report['device']!=args.device or report['evidence']!=proof:raise ValueError('predecessor status/source/device/evidence mismatch')
 c.v.verify_tree(report)
 if report.get('source_review')!='SOURCE_REVIEWED' or report.get('formal_host')!='FORMAL_HOST_FEASIBLE' or report.get('npu')!='NOT_RUN':raise ValueError('predecessor must be Host/Build only')
 if status==STATUS['host']:
  files={Path(f['path']).name:c.verify(f) for f in report['receiver_files']}
  actual_plan=plan(files['runner.log'].read_text(),proof['t1'],proof['policy'])
  ex=c.read(files['execution.json']);ad=c.read(files['resource-admission.json'])
  expected=dict(status='HOST_TILING_ONLY',npu='NOT_RUN',device=args.device,workspace_bytes=report['workspace_bytes'])
  admission=dict(status='ADMITTED_BEFORE_EXECUTION',workspace_bytes=report['workspace_bytes'],declared_total=actual_plan['total'],blocks=1,cores=proof['policy']['selected'],core_policy=proof['policy'],host_proof_checked=False)
  if ex!=expected or ad!=admission or actual_plan!=report['formal_plan'] or report['workspace_bytes']<actual_plan['total']:raise ValueError('formal Host evidence differs')
 verify_artifacts(report['artifacts'],proof)
 return report

def data(root,a,b,oracle):
 root.mkdir();files={}
 for name,raw in [('a_i8.bin',a),('b_i8.bin',b),('oracle_i32.bin',oracle.astype('<i4').tobytes()),('oracle_i64.bin',oracle.astype('<i8').tobytes())]:
  p=root/name;p.write_bytes(raw);files[name]=c.identity(p)
 return files

def run_receiver(args,root,r,host,expected_workspace=None):
 folder=root/'execution';folder.mkdir()
 for name in ('a_i8.bin','b_i8.bin','oracle_i32.bin'):shutil.copyfile(c.verify(r['data'][name]),folder/name)
 r['execution_inputs']=[c.identity(folder/name) for name in ('a_i8.bin','b_i8.bin','oracle_i32.bin')]
 item=r['artifacts'];e=env(args.cann_root,Path(item['opp']));policy=r['evidence']['policy']
 e['DFLASH_STAGE4RAW_EXPECTED_CORE_POLICY']=','.join(str(policy[k]) for k in ('aic','aiv','all','selected','arch'))
 if expected_workspace is not None:e['DFLASH_STAGE4RAW_EXPECTED_WORKSPACE']=str(expected_workspace)
 argv=[c.verify(item['runner']),args.device,folder]+(['--host-only'] if host else [])
 r['npu']='NOT_RUN'
 try:command(argv,folder/'runner.log',e,r,'host_query' if host else 'bounded',1800)
 finally:
  r['receiver_files']=[c.identity(p) for p in sorted(folder.iterdir()) if p.is_file()]
  if (folder/'execution.json').is_file():r['npu']=c.read(folder/'execution.json').get('npu','UNRESOLVED')
  outputs=[folder/('actual-'+str(i)+'.bin') for i in (0,1)]
  available=[p for p in outputs if p.is_file()]
  if available:
   oracle=np.fromfile(c.verify(r['data']['oracle_i64.bin']),'<i8').reshape(16,256)
   r['integer_comparison']=c.compare(oracle,[p.read_bytes() for p in available])
   r['outputs']=[c.identity(p) for p in available];c.write(root/'integer-comparison.json',r['integer_comparison'])
 ex=c.read(folder/'execution.json');ad=c.read(folder/'resource-admission.json')
 pl=plan((folder/'runner.log').read_text(),r['evidence']['t1'],policy)
 want=dict(status='ADMITTED_BEFORE_EXECUTION',workspace_bytes=ex['workspace_bytes'],declared_total=pl['total'],blocks=1,cores=policy['selected'],core_policy=policy,host_proof_checked=not host)
 if ad!=want or ex['workspace_bytes']<pl['total']:raise ValueError('receiver resource admission differs')
 if host:
  if ex!=dict(status='HOST_TILING_ONLY',npu='NOT_RUN',device=args.device,workspace_bytes=ex['workspace_bytes']):raise ValueError('Host-only execution differs')
 else:
  want=dict(status='RAW_INT32_EXECUTION_MATCH',npu='COMPLETED',device=args.device,workspace_bytes=expected_workspace,shape=[16,2560,256],elements=4096,repetitions=2,guards=True,readonly=True,poison=True,poison_patterns=[127,128],repeat_equal=True,cpu_fallback=False,benchmark='NOT_RUN')
  if ex!=want or r['integer_comparison']['status']!='MATCH':raise ValueError('integer/safety/Repeat contract failed')
 r['formal_plan']=pl;r['workspace_bytes']=ex['workspace_bytes']
 r['formal_host']='FORMAL_HOST_FEASIBLE';r['npu']='NOT_RUN' if host else 'RAW_INT32_NPU_MATCH'


def main(args):
 root=c.fresh(args.output_dir,[args.frozen_root,args.cann_root,args.raw_host_report.parent])
 r=dict(status='NOT_RUN',phase='evidence',stage=args.stage,sources=c.sources(),device=args.device,source_review='NOT_RUN',formal_host='NOT_RUN',cann_compile='NOT_RUN',npu='NOT_RUN',benchmark='NOT_RUN',fault='NOT_RUN',cause='UNRESOLVED',commands={})
 try:
  proof,a,b,oracle=c.evidence(args);r['evidence']=proof;r['source_review']='SOURCE_REVIEWED'
  r['data']=data(root/'data',a,b,oracle)
  r['source_audit']=c.audit();r['phase']='formal_host' if args.stage=='host' else args.stage
  if args.stage=='host':
   product=root/'private';product.mkdir();item=dict(evidence=proof,commands={},npu='NOT_RUN')
   r['artifacts']=item
   build(args,product,item,c.HERE);verify_artifacts(item,proof);r['cann_compile']='COMPLETED'
   run_receiver(args,root,r,True)
  else:
   host=verify_gate(c.read(args.host_report),args,proof,STATUS['host']);r['host_report']=c.identity(args.host_report)
   r.update(artifacts=host['artifacts'],formal_plan=host['formal_plan'],workspace_bytes=host['workspace_bytes'],formal_host='FORMAL_HOST_FEASIBLE',cann_compile='COMPLETED')
   if args.stage=='bounded':
    seal=verify_gate(c.read(args.build_report),args,proof,STATUS['build'])
    if seal['host_report']!=r['host_report'] or seal['artifacts']!=r['artifacts'] or seal['workspace_bytes']!=r['workspace_bytes']:raise ValueError('sealed Build does not bind this formal Host')
    r['build_report']=c.identity(args.build_report)
    run_receiver(args,root,r,False,host['workspace_bytes'])
  r.update(status=STATUS[args.stage],phase='complete')
 except Exception as error:r.update(status='BLOCKED',error=f'{type(error).__name__}: {error}')
 finally:
  try:
   c.v.verify_tree(r)
   if r['sources']!=c.sources():raise ValueError('source inventory changed')
   if 'evidence' in r and c.evidence(args)[0]!=r['evidence']:raise ValueError('frozen/SDK/Host evidence changed during phase')
  except Exception as error:r.update(status='BLOCKED',integrity_error=str(error))
  c.write(root/'report.json',r)
 print(r['status']+': '+str(root/'report.json'));return int(r['status']=='BLOCKED')


def parse(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=STATUS)
 for flag in ('output-dir','frozen-root','raw-host-report'):p.add_argument('--'+flag,type=Path,required=True)
 p.add_argument('--host-report',type=Path);p.add_argument('--build-report',type=Path)
 p.add_argument('--cann-root',type=Path,default=Path(os.environ.get('CANN_ROOT','/usr/local/Ascend/cann-9.0.0')))
 p.add_argument('--device',type=int,default=0);a=p.parse_args(argv)
 if a.device<0:p.error('negative device')
 if a.stage!='host' and not a.host_report:p.error('--host-report required')
 if a.stage=='bounded' and not a.build_report:p.error('--build-report required')
 if a.stage=='host' and (a.host_report or a.build_report) or a.stage=='build' and a.build_report:p.error('unexpected predecessor report')
 for k in ('output_dir','frozen_root','raw_host_report','host_report','build_report','cann_root'):
  if getattr(a,k) is not None:setattr(a,k,getattr(a,k).resolve())
 a.runner_kind='runner';return a

if __name__=='__main__':raise SystemExit(main(parse()))
