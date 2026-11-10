#!/usr/bin/env python3
"""Read-only admission and CPU checks. Device gates are closed, not dummy passes."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import perf_contract as c
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'stage4v_external_co1_vdeq16_phase2'))
import audit_phase2 as prior_audit
import phase2_evidence as e


def sources():
 return [e.identity(p) for p in sorted(HERE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts]


def binding(root):
 root=root.resolve(strict=True)
 prior=prior_audit.sdk(root)
 spec=json.loads((HERE/'normal-sdk-review.json').read_text());rows=[]
 for s in spec['sources']:
  f=e.identity(root/s['relative_path']);path=Path(f['path'])
  if root not in path.parents or f['sha256']!=s['sha256'] or f['bytes']!=s['bytes']:raise ValueError('normal-path SDK identity differs: '+s['relative_path'])
  count=len(path.read_text().splitlines())
  if any(not 1<=lo<=hi<=count for lo,hi in s['lines']):raise ValueError('source range outside pinned file')
  rows.append(dict(file=f,lines=s['lines']))
 return dict(status='FIXED_FILES_MATCHED_NOT_NORMAL_PATH_PROOF',base=prior,normal=rows,manifest=e.identity(HERE/'normal-sdk-review.json'))


def parse(argv=None):
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('phase',choices=('review','cpu','host','build','smoke','benchmark'))
 p.add_argument('--output-dir',type=Path,required=True)
 p.add_argument('--cann-root',type=Path,default=Path('/usr/local/Ascend/cann-9.0.0'))
 for k in ('echo-root','co1-root','raw-root','frozen-root','review-root'):p.add_argument('--'+k,type=Path)
 p.add_argument('--device',type=int,default=0)
 a=p.parse_args(argv);keys=('echo_root','co1_root','raw_root','frozen_root','review_root')
 if a.device<0 or any(getattr(a,k) for k in keys) and not all(getattr(a,k) for k in keys):p.error('all five historical roots or none required')
 for k,v in vars(a).items():
  if isinstance(v,Path):setattr(a,k,v.resolve())
 return a


def main(a):
 root=e.fresh(a.output_dir,[HERE,a.cann_root]+[getattr(a,k) for k in ('echo_root','co1_root','raw_root','frozen_root','review_root') if getattr(a,k)])
 r=dict(c.admission(),requested_phase=a.phase,sources=sources(),audit_completed=False)
 try:
  if a.phase in ('host','build','smoke','benchmark'):
   r['reason']='No executable implementation is admitted. Interface syntax or altered report cannot open this gate.'
  elif a.phase=='review':
   r['sdk']=binding(a.cann_root)
   if a.echo_root:r['history'],_=e.bind(a)
   else:r['history_status']='NOT_READ_SERVER_FILES_UNAVAILABLE'
   r['audit_completed']=True
  elif a.phase=='cpu':
   cmd=[sys.executable,'-B','-m','pytest','-q','-p','no:cacheprovider',str(HERE/'test')]
   with (root/'cpu.log').open('w') as log:result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=False)
   r['cpu']=dict(command=cmd,returncode=result.returncode,log=e.identity(root/'cpu.log'))
   if result.returncode:raise ValueError('CPU regression failed')
   r['cpu_status']='CPU_CONTRACT_TESTS_PASSED_NOT_DEVICE_PROOF';r['audit_completed']=True
 except Exception as x:r['error']=f'{type(x).__name__}: {x}'
 finally:
  try:
   e.echo.v.verify_tree(r)
   if r['sources']!=sources():raise ValueError('sources changed during audit')
   if 'sdk' in r and r['sdk']!=binding(a.cann_root):raise ValueError('SDK changed during audit')
  except Exception as x:r['integrity_error']=str(x);r['audit_completed']=False
  e.write(root/'report.json',r)
 print(r['status']+': '+str(root/'report.json'))
 return 1 if r.get('error') or r.get('integrity_error') else 2

if __name__=='__main__':raise SystemExit(main(parse()))
