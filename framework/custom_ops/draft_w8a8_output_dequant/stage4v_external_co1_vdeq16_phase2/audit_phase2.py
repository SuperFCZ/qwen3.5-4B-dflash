#!/usr/bin/env python3
"""Read-only Phase2 design audit. No compilation, OPP, ACLNN call or NPU launch."""
import argparse,os
from pathlib import Path
import phase2_evidence as e
import phase2_contract as contract
import phase21_contract as event_contract


def sdk(cann):
 cann=cann.resolve(strict=True);spec=e.read(e.HERE/'sdk-review.json');rows=[]
 release=e.echo.old.review.prev.raw.v.p.host_o.bind(cann)
 for s in spec['sources']:
  f=e.identity(cann/s['relative_path'])
  if cann not in Path(f['path']).parents or f['sha256']!=s['sha256'] or f['bytes']!=s['bytes']:raise ValueError('SDK identity mismatch '+s['id'])
  count=len(e.verify(f).read_text().splitlines())
  if not all(1<=lo<=hi<=count for lo,hi in s['lines']):raise ValueError('SDK range mismatch')
  rows.append(dict(id=s['id'],file=f,lines=s['lines'],interface=s['interface']))
 focus=e.read(e.HERE/'phase21-sdk-review.json');focused=[]
 baseline={s['relative_path']:s for s in spec['sources']}
 for s in focus['sources']:
  original=baseline[s['relative_path']]
  if any(s[k]!=original[k] for k in ('id','sha256','bytes')):raise ValueError('Phase2.1 changed SDK identity')
  f=e.identity(cann/s['relative_path']);count=len(e.verify(f).read_text().splitlines())
  if f['sha256']!=s['sha256'] or f['bytes']!=s['bytes'] or not all(1<=lo<=hi<=count for lo,hi in s['lines']):raise ValueError('Phase2.1 SDK binding/range differs')
  focused.append(dict(id=s['id'],file=f,lines=s['lines']))
 if len(focused)!=4 or len({s['id'] for s in focused})!=4:raise ValueError('expected four distinct focused sources')
 return dict(status='CURRENT_INSTALL_MATCHES_REVIEWED_9_0_0_HASHES',files=rows,release=release,release_usage='VERSION_IDENTITY_ONLY_NO_PLATFORM_QUERY',manifest=e.identity(e.HERE/'sdk-review.json'),
  focused_review=dict(manifest=e.identity(e.HERE/'phase21-sdk-review.json'),files=focused))


def decision(history_checked):
 return dict(event_contract.outcome(),design='B_PREFERRED_FOR_INDEPENDENT_REVIEW_NOT_IMPLEMENTATION',
  historical_files='SHA_REVERIFIED' if history_checked else 'NOT_READ_LOCAL_SERVER_FILES_UNAVAILABLE',
  required_later=['new three-output ABI Formal Host/resource verification','real Echo + VDEQ16 repeated device observation'],
  unknown=['VDEQ16 internal multiply precision and tie-breaking','equivalence to original FP16 Matmul live L0C','Phase2 device outcomes'],
  golden='UNCHANGED')


def main(a):
 root=e.fresh(a.output_dir,[a.cann_root]+[p for p in (a.echo_root,a.co1_root,a.raw_root,a.frozen_root,a.review_root) if p])
 r=dict(status='NOT_RUN',sources=e.sources(),npu='NOT_RUN')
 try:
  r['sdk']=sdk(a.cann_root)
  if a.echo_root:
   r['history'],original=e.bind(a)
   prior_review=e.read(a.review_root/'review/report.json');tiling=e.read(a.review_root/'tiling/report.json');analysis=e.read(a.review_root/'analysis/report.json')
   if prior_review['status']!='SOURCE_REVIEWED' or tiling['status']!='SAVED_TILING_COMPARED_FORMAL_SCOPE_PARTIAL' or analysis['status']!='FROZEN_RAW_AND_FP16_ANALYZED_CAUSE_UNRESOLVED':raise ValueError('historical VDEQ offline review incomplete')
   for item in (tiling,analysis):
    if item['review_report']!=e.identity(a.review_root/'review/report.json') or item['evidence']['raw_bounded']!=original['evidence']['raw_evidence']['raw_bounded']:raise ValueError('VDEQ review/Raw evidence link differs')
   r['offline_history']={s:e.identity(a.review_root/s/'report.json') for s in ('review','tiling','analysis')}
  r['geometry']=contract.validate_geometry(contract.geometry());r['resource_design']=event_contract.layout_b()
  r['alternative_resource_design_A']=contract.layout()
  r['event_ownership_model']=dict(scope='PURE_CPU_CONDITIONAL_MODEL_NOT_DEVICE_EVIDENCE',
    sv={plan:event_contract.sv_trace(plan) for plan in ('A','B')},manual_B=event_contract.manual_pool_trace())
  r.update(decision('history' in r));r['audit_completed']=True
 except Exception as x:r.update(status='BLOCKED',audit_completed=False,error=f'{type(x).__name__}: {x}')
 finally:
  try:
   e.echo.v.verify_tree(r)
   if r['sources']!=e.sources() or 'sdk' in r and sdk(a.cann_root)!=r['sdk']:raise ValueError('source/SDK changed')
  except Exception as x:r.update(status='BLOCKED',audit_completed=False,integrity_error=str(x))
  e.write(root/'design-review-report.json',r)
 print(r['status']+': '+str(root/'design-review-report.json'));return 2 if r.get('audit_completed') else 1


def parse(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True)
 p.add_argument('--cann-root',type=Path,default=Path(os.environ.get('CANN_ROOT','/usr/local/Ascend/cann-9.0.0')))
 for k in ('echo-root','co1-root','raw-root','frozen-root','review-root'):p.add_argument('--'+k,type=Path)
 p.add_argument('--device',type=int,default=0);a=p.parse_args(argv)
 keys=('echo_root','co1_root','raw_root','frozen_root','review_root')
 if a.device<0 or any(getattr(a,k) for k in keys) and not all(getattr(a,k) for k in keys):p.error('supply all five evidence roots, or none for local SDK-only review')
 for k,v in vars(a).items():
  if isinstance(v,Path):setattr(a,k,v.resolve())
 return a

if __name__=='__main__':raise SystemExit(main(parse()))
