#!/usr/bin/env python3
"""Source-only External CO1 feasibility review. Never initializes ACL or launches a Kernel."""
import argparse,os
from pathlib import Path
import co1_contract as c
from co1_audit import sdk_binding,formal_capability,prior_evidence


def main(a):
 root=c.fresh(a.output_dir,[a.cann_root]+([a.evidence_root] if a.evidence_root else []))
 r=dict(status='NOT_RUN',sources=c.sources(),npu='NOT_RUN')
 try:
  r['sdk']=sdk_binding(a.cann_root);r['formal_50']=formal_capability();r['dependencies']=c.dependencies();r['resource_design']=c.resource_design()
  if a.evidence_root:r['prior']=prior_evidence(a.evidence_root,r['sdk'])
  else:r['prior']=dict(status='NOT_READ_LOCAL_SERVER_PATHS_UNAVAILABLE')
  r.update(c.decision());r['audit_completed']=True
 except Exception as e:r.update(status='BLOCKED',audit_completed=False,error=f'{type(e).__name__}: {e}')
 finally:
  try:
   c.prev.verify_tree(r)
   if r['sources']!=c.sources():raise ValueError('source inventory changed')
   if 'sdk' in r and r['sdk']!=sdk_binding(a.cann_root):raise ValueError('SDK changed')
  except Exception as e:r.update(status='BLOCKED',audit_completed=False,integrity_error=str(e))
  c.write(root/'review-report.json',r)
  c.write(root/'formal-tiling-feasibility.json',r.get('formal_50',dict(status='NOT_RUN')))
  c.write(root/'dependency-report.json',dict(edges=r.get('dependencies',[]),npu='NOT_RUN'))
  c.write(root/'resource-design.json',r.get('resource_design',dict(status='NOT_RUN')))
 print(r['status']+': '+str(root/'review-report.json'))
 # 2 means completed review with a closed execution gate; 1 means evidence/audit failure.
 return 2 if r.get('audit_completed') else 1


def parse(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True)
 p.add_argument('--cann-root',type=Path,default=Path(os.environ.get('CANN_ROOT','/usr/local/Ascend/cann-9.0.0')))
 p.add_argument('--evidence-root',type=Path)
 a=p.parse_args(argv)
 for k in ('output_dir','cann_root','evidence_root'):
  if getattr(a,k) is not None:setattr(a,k,getattr(a,k).resolve())
 return a

if __name__=='__main__':raise SystemExit(main(parse()))
